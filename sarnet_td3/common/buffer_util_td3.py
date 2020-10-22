import numpy as np


# Retrieve batch/trajectories from the buffer for updates
class BufferOp(object):
    def __init__(self, args, num_agents):
        self.args = args
        self.beta = 0.6

        if self.args.policy_grad == "maddpg" and not self.args.PER_sampling:
            from sarnet_td3.trainer.replay_buffer_td3 import ReplayBuffer
            self.buffer = ReplayBuffer(self.args)
        elif self.args.policy_grad == "maddpg" and self.args.PER_sampling:
            from sarnet_td3.trainer.replay_buffer_td3 import PrioritizedReplayBuffer
            self.buffer = PrioritizedReplayBuffer(self.args, num_agents)

    def set_priorities(self, idxes, priorities, p_index):
        self.buffer.update_priorities(idxes, priorities, p_index)

    def return_exp(self, p_index):
        # Receive index of form - [Env [# Num of traj, # Batch size]]
        samples = self.buffer.sample(self.args.batch_size, self.beta, p_index)

        importance = None
        index = None
        if self.args.PER_sampling:
            samples, importance, index = samples

        obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, act_n_t, rew_n_t, obs_n_t1, h_n_t1, c_n_t1, \
        mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t = samples

        q1_h_n_t = np.moveaxis(q1_h_n_t, 0, -2)
        q1_h_n_t1 = np.moveaxis(q1_h_n_t1, 0, -2)
        if self.args.td3:
            q2_h_n_t = np.moveaxis(q2_h_n_t, 0, -2)
            q2_h_n_t1 = np.moveaxis(q2_h_n_t1, 0, -2)
        else:
            q2_h_n_t = q1_h_n_t
            q2_h_n_t1 = q1_h_n_t1

        # Input: [batch, agent, traj, dim] or [batch, traj, agent]
        # Output: [agent, traj, batch, dim] or [agent, traj, batch]
        # Compress all batches from each environment to a single array

        obs_n_t = np.moveaxis(obs_n_t, 0, -2)
        h_n_t = np.moveaxis(h_n_t, 0, -2)
        h_n_t1 = np.moveaxis(h_n_t1, 0, -2)
        if self.args.encoder_model in {"LSTM"}:
            c_n_t = np.moveaxis(c_n_t, 0, -2)
            c_n_t1 = np.moveaxis(c_n_t1, 0, -2)
        else:
            c_n_t = h_n_t
            c_n_t1 = h_n_t1
        mem_n_t = np.moveaxis(mem_n_t, 0, -2)
        act_n_t = np.moveaxis(act_n_t, 0, -2)
        obs_n_t1 = np.moveaxis(obs_n_t1, 0, -2)
        mem_n_t1 = np.moveaxis(mem_n_t1, 0, -2)
        # Input dim [batch, traj, agent]
        # Output dim [agent, traj, batch]
        done_n_t = np.swapaxes(done_n_t, -1, 0)
        rew_n_t = np.swapaxes(rew_n_t, -1, 0)

        data = (obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, act_n_t, rew_n_t, obs_n_t1, h_n_t1,
                c_n_t1, mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t)

        return data, index, importance  # Returns a list of tuples of batches for each step in the trajectory

    # Save experiences in the buffer
    def collect_exp(self, exp):
        if self.args.benchmark or self.args.display:
            return
        else:
            obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, action_n_t, rew_n_t, obs_n_t1, h_n_t1, c_n_t1, \
            mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t = self.reshape_action_to_buffer(exp)
            # [num_env, traj, agent, dim]
            for j in range(self.args.num_env):
                self.buffer.add(obs_n_t[j], h_n_t[j,:,0,:], c_n_t[j,:,0,:], mem_n_t[j,:,0,:], q1_h_n_t[j,:,0,:], q2_h_n_t[j,:,0,:], action_n_t[j], rew_n_t[j], obs_n_t1[j],
                              h_n_t1[j,:,0,:], c_n_t1[j,:,0,:], mem_n_t1[j,:,0,:], q1_h_n_t1[j,:,0,:], q2_h_n_t1[j,:,0,:], done_n_t[j])

            return "buffer_added"

    # Reshapes Input to [batch, agent, traj, dim]
    def reshape_action_to_buffer(self, exp):
        obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, action_n_t, rew_n_t, obs_n_t1, h_n_t1, c_n_t1, mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t = exp
        action_n_t = np.swapaxes(np.asarray(action_n_t), 0, -2)
        obs_n_t = np.swapaxes(np.asarray(obs_n_t), 0, -2)
        obs_n_t1 = np.swapaxes(np.asarray(obs_n_t1), 0, -2)
        h_n_t = np.swapaxes(np.asarray(h_n_t), 0, -2)
        h_n_t1 = np.swapaxes(np.asarray(h_n_t1), 0, -2)
        mem_n_t = np.swapaxes(np.asarray(mem_n_t), 0, -2)
        mem_n_t1 = np.swapaxes(np.asarray(mem_n_t1), 0, -2)
        if self.args.encoder_model in {"LSTM"}:
            c_n_t = np.swapaxes(np.asarray(c_n_t), 0, -2)
            c_n_t1 = np.swapaxes(np.asarray(c_n_t1), 0, -2)
        else:
            c_n_t = h_n_t
            c_n_t1 = h_n_t1
        q1_h_n_t = np.swapaxes(np.asarray(q1_h_n_t), 0, -2)
        q1_h_n_t1 = np.swapaxes(np.asarray(q1_h_n_t1), 0, -2)
        if self.args.td3:
            q2_h_n_t = np.swapaxes(np.asarray(q2_h_n_t), 0, -2)
            q2_h_n_t1 = np.swapaxes(np.asarray(q2_h_n_t1), 0, -2)
        else:
            q2_h_n_t = q1_h_n_t
            q2_h_n_t1 = q1_h_n_t1
        rew_n_t = np.swapaxes(np.asarray(rew_n_t), 0, -2)  # .astype(np.float16) # [Batch, traj, #agent]
        done_n_t = np.swapaxes(done_n_t, 0, 1)  # [Batch, traj, #agent]

        return obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, action_n_t, rew_n_t, obs_n_t1, h_n_t1, c_n_t1, mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t