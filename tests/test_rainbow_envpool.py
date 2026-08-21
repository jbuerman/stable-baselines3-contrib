"""Tests for the EnvPool-backed Atari vector env used by Rainbow.

These cover the three places where EnvPool's vector API and SB3's VecEnv
contract disagree, since that is where the adapter can silently corrupt the
replay buffer: lazy auto-reset, missing episode statistics, and life losses
being reported as terminations.
"""

import numpy as np
import pytest

envpool = pytest.importorskip("envpool")

from sb3_contrib.rainbow.envpool_env import EnvPoolAtariVecEnv, make_atari_envpool  # noqa: E402

GAME = "Breakout"
N_ENVS = 2


def _rollout(env, n_steps, rng, action=None):
    """Step env, yielding (obs, rewards, dones, infos) tuples."""
    for _ in range(n_steps):
        if action is None:
            actions = rng.integers(0, env.action_space.n, size=env.num_envs)
        else:
            actions = np.full(env.num_envs, action)
        yield env.step(actions)


def test_spaces_and_reset():
    env = make_atari_envpool(N_ENVS, GAME, framestack=4, seed=7)
    try:
        assert isinstance(env, EnvPoolAtariVecEnv)
        assert env.num_envs == N_ENVS
        # channel-first uint8 so SB3 does not insert a VecTransposeImage
        assert env.observation_space.shape == (4, 84, 84)
        assert env.observation_space.dtype == np.uint8

        obs = env.reset()
        assert obs.shape == (N_ENVS, 4, 84, 84)
        assert obs.dtype == np.uint8
    finally:
        env.close()


def test_rewards_are_clipped_only_when_asked():
    rng = np.random.default_rng(0)
    for clip_rewards, expected in [(True, {-1.0, 0.0, 1.0}), (False, None)]:
        env = make_atari_envpool(N_ENVS, GAME, clip_rewards=clip_rewards, seed=7)
        try:
            env.reset()
            seen = set()
            for _, rewards, _, _ in _rollout(env, 500, rng):
                assert rewards.dtype == np.float32
                seen.update(np.unique(rewards).tolist())
            if expected is not None:
                assert seen <= expected
        finally:
            env.close()


def test_life_loss_terminates_without_restarting_the_game():
    """episodic_life dones must not reset the ALE, or every life would restart it."""
    rng = np.random.default_rng(0)
    env = make_atari_envpool(N_ENVS, GAME, terminal_on_life_loss=True, seed=7)
    try:
        env.reset()
        life_losses = game_overs = 0
        for _, _, dones, infos in _rollout(env, 3000, rng):
            for i in np.flatnonzero(dones):
                assert "terminal_observation" in infos[i]
                # episode stats are published on a real game over only
                if "episode" in infos[i]:
                    game_overs += 1
                else:
                    life_losses += 1
            if game_overs >= 2 and life_losses >= 4:
                break
        assert life_losses > game_overs, (life_losses, game_overs)
    finally:
        env.close()


def test_episode_info_reports_raw_score():
    """info["episode"]["r"] must be the unclipped game score, as Monitor used to give."""
    rng = np.random.default_rng(1)
    env = make_atari_envpool(N_ENVS, GAME, terminal_on_life_loss=False, clip_rewards=False, seed=11)
    try:
        env.reset()
        totals = np.zeros(N_ENVS)
        lengths = np.zeros(N_ENVS, dtype=int)
        checked = 0
        for _, rewards, _, infos in _rollout(env, 8000, rng):
            totals += rewards
            lengths += 1
            for i in range(N_ENVS):
                if "episode" in infos[i]:
                    assert infos[i]["episode"]["r"] == pytest.approx(totals[i])
                    assert infos[i]["episode"]["l"] == lengths[i]
                    totals[i], lengths[i] = 0.0, 0
                    checked += 1
            if checked >= 2:
                break
        assert checked >= 2
    finally:
        env.close()


def test_game_over_returns_reset_obs_without_wasting_a_step():
    """EnvPool auto-resets lazily; the adapter must force the reset on the done step."""
    rng = np.random.default_rng(2)
    env = make_atari_envpool(1, GAME, terminal_on_life_loss=False, seed=5)
    try:
        env.reset()
        for _, _, dones, _ in _rollout(env, 8000, rng):
            if dones[0]:
                break
        else:
            pytest.fail("no game over within the step budget")

        env.step(np.array([1]))
        # if the reset had been left to EnvPool, that step would have been
        # swallowed by the reset and elapsed_step would still be 0
        assert env.pool.step(np.zeros(1, dtype=np.int32))[4]["elapsed_step"][0] == 2
    finally:
        env.close()


def test_truncation_is_flagged_separately(monkeypatch):
    """PER strips the truncation component of done, so it must be reported."""
    monkeypatch.setattr("sb3_contrib.rainbow.envpool_env.ATARI_MAX_EPISODE_STEPS", 50)
    env = make_atari_envpool(1, GAME, terminal_on_life_loss=True, seed=3)
    try:
        env.reset()
        for _, _, dones, infos in _rollout(env, 200, None, action=1):
            if dones[0] and infos[0].get("TimeLimit.truncated"):
                assert "terminal_observation" in infos[0]
                # the absorbed soft resets must not inflate the episode length
                assert infos[0]["episode"]["l"] == 50
                break
        else:
            pytest.fail("truncation never fired")
    finally:
        env.close()


def test_life_loss_does_not_waste_a_step():
    """EnvPool spends the step after a life loss on its soft reset; the adapter absorbs it."""
    rng = np.random.default_rng(3)
    env = make_atari_envpool(N_ENVS, GAME, terminal_on_life_loss=True, seed=7)
    try:
        env.reset()
        for _, _, dones, infos in _rollout(env, 3000, rng):
            if dones.any() and "episode" not in infos[int(np.argmax(dones))]:
                break
        else:
            pytest.fail("no life loss within the step budget")

        # both envs must still be on the same game clock: if the soft reset had
        # been left to EnvPool, the env that lost a life would lag by one step
        before = env.pool.step(np.zeros(N_ENVS, dtype=np.int32))[4]["elapsed_step"]
        after = env.pool.step(np.zeros(N_ENVS, dtype=np.int32))[4]["elapsed_step"]
        assert len(set(before.tolist())) == 1, before
        assert (after == before + 1).all(), (before, after)
    finally:
        env.close()


def test_per_framestack_overlap_invariant():
    """PER stores only the newest frame of next_obs and reuses obs's frame pointers.

    That is only valid while consecutive observations overlap by framestack-1
    frames, so a mid-episode stack refresh would silently corrupt every
    reconstructed state.
    """
    rng = np.random.default_rng(0)
    n_envs = 4
    env = make_atari_envpool(n_envs, GAME, framestack=4, terminal_on_life_loss=True, seed=7)
    try:
        last_obs = env.reset()
        last_terminal = np.ones(n_envs, dtype=bool)  # PER starts each stream as terminal
        checked = violations = 0
        for new_obs, _, dones, infos in _rollout(env, 3000, rng):
            # exactly what SB3's _store_transition hands to PER.add
            next_obs = new_obs.copy()
            for i in np.flatnonzero(dones):
                next_obs[i] = infos[i]["terminal_observation"]
            for i in range(n_envs):
                if not last_terminal[i]:
                    checked += 1
                    if not np.array_equal(last_obs[i][1:], next_obs[i][:3]):
                        violations += 1
            last_terminal = dones.copy()
            last_obs = new_obs
        assert checked > 1000
        assert violations == 0, f"{violations}/{checked} transitions would reconstruct corrupted states"
    finally:
        env.close()


def test_async_pool_is_rejected():
    pool = envpool.make_gymnasium(f"{GAME}-v5", num_envs=4, batch_size=2)
    with pytest.raises(ValueError, match="synchronous"):
        EnvPoolAtariVecEnv(pool, episodic_life=True)
