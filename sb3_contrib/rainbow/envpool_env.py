"""EnvPool-backed Atari vector env for Rainbow.

Replaces the gymnasium + ``SubprocVecEnv`` stack. EnvPool runs the ALE in a C++
thread pool inside the calling process, so there are no worker subprocesses to
spawn, pipe to or join - which is what made the periodic evaluation runs hang
(each evaluation spawned a process that itself spawned a ``SubprocVecEnv``).

EnvPool already performs every step the old wrapper chain did - grayscale,
84x84 resize, frame skip, frame stack, noop reset, fire reset, optional
life-loss termination and optional reward clipping - so the wrappers are gone.
What is left here is the adaptation of EnvPool's vector API to SB3's
:class:`~stable_baselines3.common.vec_env.VecEnv` contract, which differs in
three ways that matter:

* EnvPool auto-resets *lazily*: the step that ends an episode returns the
  terminal observation, and the reset observation only arrives on the following
  step (which silently discards its action). SB3 expects the reset observation
  on the done step itself, with the terminal one in ``info["terminal_observation"]``.
  We drive the reset ourselves on the done step so no transition is wasted.
* EnvPool has no ``Monitor``, so raw (unclipped) episode returns are tracked
  here from ``info["reward"]`` and published as ``info["episode"]``.
* With ``episodic_life=True`` a life loss is reported as ``terminated`` even
  though the ALE keeps running. Calling ``reset()`` there would restart the
  game and refill the lives, so life losses instead get a partial noop step,
  which is how EnvPool applies its soft reset. ``info["terminated"]``
  distinguishes a real game over from a life loss.
"""

import logging
import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

try:
    import envpool
except ImportError as exc:  # pragma: no cover - exercised only without envpool
    envpool = None
    _ENVPOOL_IMPORT_ERROR = exc
else:
    _ENVPOOL_IMPORT_ERROR = None

# The gymnasium ALE/<game>-v5 spec caps an episode at 108_000 frames; at a frame
# skip of 4 that is 27_000 agent steps. EnvPool's default is effectively
# unbounded, so it has to be set explicitly to keep the same protocol.
ATARI_MAX_EPISODE_STEPS = 27_000


class EnvPoolAtariVecEnv(VecEnv):
    """Adapts an EnvPool Atari pool to SB3's :class:`VecEnv` interface.

    :param pool: a pool built by :func:`envpool.make_gymnasium`, synchronous
        (``batch_size == num_envs``) so that batch position ``i`` is always
        env id ``i``.
    :param episodic_life: whether the pool was created with ``episodic_life=True``.
        Only used to document intent; the real decision is driven by
        ``info["terminated"]``.
    """

    def __init__(self, pool, episodic_life: bool):
        if pool.is_async:
            raise ValueError("EnvPoolAtariVecEnv requires a synchronous pool (batch_size == num_envs)")

        self.pool = pool
        self.episodic_life = episodic_life

        super().__init__(pool.config["num_envs"], pool.observation_space, pool.action_space)
        self.render_mode = None

        self._env_ids = np.arange(self.num_envs, dtype=np.int32)
        self._actions = None
        # raw, unclipped score of the episode currently in flight, per env
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._start_time = time.time()

    # -- VecEnv API ---------------------------------------------------------

    def reset(self):
        obs, _ = self.pool.reset()
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int32).reshape(self.num_envs)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.pool.step(self._actions)

        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        # info["terminated"] is the real ALE game over and ignores life losses,
        # so it stays correct whether or not episodic_life is enabled.
        game_over = np.asarray(info["terminated"], dtype=bool)

        self._episode_returns += info["reward"]
        self._episode_lengths += 1

        dones = terminated | truncated
        # A real episode end needs a real reset. A life loss ends the SB3
        # episode but must leave the ALE running - resetting it there would
        # restart the game and hand back a full set of lives.
        needs_restart = game_over | truncated
        life_lost = dones & ~needs_restart

        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = obs[i].copy()
            if truncated[i] and not game_over[i]:
                # PER strips this component so truncation still bootstraps
                infos[i]["TimeLimit.truncated"] = True

        for i in np.flatnonzero(needs_restart):
            infos[i]["episode"] = {
                "r": float(self._episode_returns[i]),
                "l": int(self._episode_lengths[i]),
                "t": round(time.time() - self._start_time, 6),
            }
            self._episode_returns[i] = 0.0
            self._episode_lengths[i] = 0

        if needs_restart.any():
            restart_ids = self._env_ids[needs_restart]
            restart_obs, _ = self.pool.reset(restart_ids)
            obs[restart_ids] = restart_obs

        if life_lost.any():
            # EnvPool has a soft reset pending on these envs and will spend the
            # next step performing it, silently dropping that step's action. Do
            # it here with a noop so the caller never loses a transition. The
            # game clock does not advance; this is exactly what SB3's
            # EpisodicLifeEnv + FireResetEnv do on reset.
            life_ids = self._env_ids[life_lost]
            soft_obs = self.pool.step(np.zeros(life_ids.size, dtype=np.int32), life_ids)[0]
            obs[life_ids] = soft_obs

        return obs, np.asarray(rewards, dtype=np.float32), dones, infos

    def close(self):
        # EnvPool pools have no explicit close; drop the reference so the C++
        # thread pool is torn down with the object.
        self.pool = None

    def seed(self, seed=None):
        # EnvPool seeds at construction time only.
        logger.warning("EnvPoolAtariVecEnv.seed() is a no-op; pass seed= to make_atari_envpool instead")
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        value = getattr(self, attr_name, None)
        if value is None:
            value = getattr(self.pool, attr_name)
        return [value] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("EnvPool envs do not expose per-env attributes")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("EnvPool envs do not expose per-env methods")

    def env_is_wrapped(self, wrapper_class, indices=None):
        # Nothing is a gym.Wrapper: EnvPool applies the preprocessing internally.
        return [False] * len(self._get_indices(indices))

    def get_images(self):
        raise NotImplementedError("EnvPool Atari pools do not expose RGB renders")


def make_atari_envpool(
    num_envs,
    game,
    framestack=4,
    repeat_probs=0.0,
    terminal_on_life_loss=True,
    clip_rewards=True,
    seed=42,
    num_threads=None,
):
    """Build an EnvPool Atari pool wrapped as an SB3 ``VecEnv``.

    Mirrors the old gymnasium chain: ``AtariPreprocessing`` (grayscale, 84x84,
    frame skip 4, up to 30 noops) plus fire-reset, ``ClipRewardEnv`` and
    ``FrameStackObservation``. Observations come out as uint8
    ``(framestack, 84, 84)``, i.e. already channel-first, so SB3 does not add a
    ``VecTransposeImage``.

    :param num_envs: number of parallel ALE instances.
    :param game: bare game name, e.g. ``"NameThisGame"`` (no ``ALE/`` prefix).
    :param framestack: number of stacked frames.
    :param repeat_probs: sticky-action probability.
    :param terminal_on_life_loss: report a life loss as an episode end (training).
    :param clip_rewards: clip step rewards to their sign (training). Raw scores
        are still reported through ``info["episode"]["r"]``.
    :param seed: EnvPool seed; envs are seeded ``seed, seed + 1, ...``.
    :param num_threads: size of EnvPool's worker thread pool. Defaults to
        ``num_envs``, which keeps a small evaluation pool from grabbing every
        core while training is running.
    """
    if envpool is None:
        raise ImportError(
            "envpool is required for the Rainbow Atari envs. Install it with `pip install envpool`."
        ) from _ENVPOOL_IMPORT_ERROR

    logger.debug("Creating %d envpool environments for %s.", num_envs, game)
    pool = envpool.make_gymnasium(
        f"{game}-v5",
        num_envs=num_envs,
        num_threads=num_envs if num_threads is None else num_threads,
        seed=seed,
        stack_num=framestack,
        frame_skip=4,
        noop_max=30,
        img_height=84,
        img_width=84,
        gray_scale=True,
        use_fire_reset=True,
        episodic_life=terminal_on_life_loss,
        reward_clip=clip_rewards,
        repeat_action_probability=repeat_probs,
        max_episode_steps=ATARI_MAX_EPISODE_STEPS,
    )
    logger.debug("Environments created.")
    return EnvPoolAtariVecEnv(pool, episodic_life=terminal_on_life_loss)
