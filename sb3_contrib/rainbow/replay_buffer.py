import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        alpha=0.5,
        beta=0.4,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            **kwargs,
        )

        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.sum_tree = SumTree(buffer_size)
        self.max_priority = 1.0

        self._last_weights: th.Tensor | None = None
        self._last_indices: np.ndarray | None = None

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)

        # Add max priority for new transition
        idx = (self.pos - 1) % self.buffer_size
        self.sum_tree.append(self.max_priority**self.alpha)

    def sample(self, batch_size, env=None):
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        samples = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            samples.append(s)

        samples = np.array(samples)

        prios, data_indices, tree_indices = self.sum_tree.find(samples)

        # Use SB3 to fetch actual data
        replay_data = super().sample(batch_size, env)

        probs = prios / p_total
        weights = (self.size() * probs) ** (-self.beta)
        weights /= weights.max()

        weights = th.tensor(weights, dtype=th.float32, device=self.device)

        self._last_weights = weights
        self._last_indices = tree_indices

        return replay_data

    def get_per_data(self):
        if self._last_weights is None or self._last_indices is None:
            raise RuntimeError("PER sampling data not available")

        return self._last_weights, self._last_indices

    def update_priorities(self, tree_indices, priorities):
        priorities = priorities + self.eps

        self.max_priority = max(self.max_priority, np.max(priorities))

        self.sum_tree.update(tree_indices, priorities**self.alpha)


class SumTree:
    """SumTree

    A binary tree data structure where the parent's value is the sum of its children
    """

    def __init__(self, size, procgen=False):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, value):
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    def total(self):
        return self.sum_tree[0]
