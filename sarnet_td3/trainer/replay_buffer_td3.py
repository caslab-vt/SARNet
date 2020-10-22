import numpy as np
import random

from sarnet_td3.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, arglist):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(arglist.buffer_size)
        self._next_idx = 0
        self.args = arglist

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, h_t, c_t, memory_t, q1_gru_t, q2_gru_t, action, reward, obs_tp1, h_t1, c_t1, memory_t1, q1_gru_t1, q2_gru_t1, done):
        # Convert all data to a reduced value, to keep in check the RAM usage
        data = (obs_t, np.expand_dims(h_t, axis=1), np.expand_dims(c_t, axis=1), np.expand_dims(memory_t, axis=1), np.expand_dims(q1_gru_t, axis=1),
                np.expand_dims(q2_gru_t, axis=1), action, reward, obs_tp1, np.expand_dims(h_t1, axis=1), np.expand_dims(c_t1, axis=1), np.expand_dims(memory_t1, axis=1),
                np.expand_dims(q1_gru_t1, axis=1), np.expand_dims(q2_gru_t1, axis=1), done)

        # data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, hses_t, cses_t, memories_t, q1_grus_t, q2_grus_t, actions, rewards, obses_tp1, hses_t1, cses_t1, memories_t1, q1_grus_t1, q2_grus_t1, dones = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, h_t, c_t, memory_t, q1_gru_t, q2_gru_t, action, reward, obs_tp1, h_t1, c_t1, memory_t1, q1_gru_t1, q2_gru_t1, done = data

            obses_t.append(np.array(obs_t, copy=False))
            hses_t.append(np.array(h_t, copy=False))
            cses_t.append(np.array(c_t, copy=False))
            memories_t.append(np.array(memory_t, copy=False))
            q1_grus_t.append(np.array(q1_gru_t, copy=False))
            q2_grus_t.append(np.array(q2_gru_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            hses_t1.append(np.array(h_t1, copy=False))
            cses_t1.append(np.array(c_t1, copy=False))
            memories_t1.append(np.array(memory_t1, copy=False))
            q1_grus_t1.append(np.array(q1_gru_t1, copy=False))
            q2_grus_t1.append(np.array(q2_gru_t1, copy=False))
            dones.append(done)


        return (np.array(obses_t), np.array(hses_t), np.array(cses_t), np.array(memories_t), np.array(q1_grus_t),
                np.array(q2_grus_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(hses_t1),
                np.array(cses_t1), np.array(memories_t1), np.array(q1_grus_t1), np.array(q2_grus_t1), np.array(dones))

    def sample(self, batch_size, beta, p_index):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def clear(self):
        self._storage = []
        self._next_idx = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, arglist, num_agents=1):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(arglist)
        assert arglist.alpha >= 0
        self._alpha = arglist.alpha
        self.num_agents = num_agents
        it_capacity = 1
        while it_capacity < arglist.buffer_size:
            it_capacity *= 2

        self._it_sum = []
        self._it_min = []

        for i in range(self.num_agents):
            self._it_sum.append(SumSegmentTree(it_capacity))
            self._it_min.append(MinSegmentTree(it_capacity))
        self._max_priority = [1.0 for _ in range(self.num_agents)]

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        for p_index in range(self.num_agents):
            self._it_sum[p_index][idx] = self._max_priority[p_index] ** self._alpha
            self._it_min[p_index][idx] = self._max_priority[p_index] ** self._alpha

    def _sample_proportional(self, p_index, batch_size):
        res = []
        p_total = self._it_sum[p_index].sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum[p_index].find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta, p_index):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(p_index, batch_size)

        weights = []
        p_min = self._it_min[p_index].min() / self._it_sum[p_index].sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[p_index][idx] / self._it_sum[p_index].sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return (encoded_sample, weights, idxes)

    def update_priorities(self, idxes, priorities, p_index):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[p_index][idx] = priority ** self._alpha
            self._it_min[p_index][idx] = priority ** self._alpha

            self._max_priority[p_index] = max(self._max_priority[p_index], priority)