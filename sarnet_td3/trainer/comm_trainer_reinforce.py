import numpy as np
import tensorflow as tf

import sarnet_td3.common.tf_util as U
import sarnet_td3.common.buffer_util_td3 as butil

from sarnet_td3 import MAgentTrainer
from sarnet_td3.common.distributions import make_pdtype

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals, polyak):
    polyak = 1.0 - polyak
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def create_placeholder_vpg(obs_shape_n, act_space_n, num_agents, args):
    # Create placeholders
    with tf.name_scope("placeholders"):
        obs_ph_n = []
        memory_ph_n = []
        h_ph_n = []
        c_ph_n = []
        return_ph_n = []

        for i in range(num_agents):
            if args.env_type == "mpe":
                obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i), traj=True).get())
            else:
                obs_ph_n.append(U.BatchInput((obs_shape_n[i],), name="observation" + str(i), traj=True).get())
            h_ph_n.append(U.BatchInput((args.gru_units,), name="gru_ph1" + str(i)).get())
            c_ph_n.append(U.BatchInput((args.gru_units,), name="gru_ph2" + str(i)).get())
            memory_ph_n.append(U.BatchInput((args.value_units,), name="memory_ph" + str(i)).get())
            return_ph_n.append(tf.compat.v1.placeholder(tf.float32, [None, None], name="returns" + str(i)))

        act_pdtype_n = [make_pdtype(act_space, args.env_type) for act_space in act_space_n]
        act_ph_n = [tf.compat.v1.placeholder(tf.int32, [None, None], name="act_one_hot" + str(i)) for i in range(len(act_space_n))]

        return obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, act_ph_n, act_space_n, return_ph_n


class CommAgentTrainerVPG(MAgentTrainer):
    def __init__(self, name, p_model, obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, act_ph_n,
                 action_space_n, return_in_ph, args, p_index, num_env=1, is_train=False):
        self.name = name
        self.args = args
        self.p_index = p_index
        self.reuse = False
        self.num_adv = self.args.num_adversaries
        self.n = len(obs_ph_n)  # Total number of agents
        self.n_start = 0
        self.n_end = self.num_adv
        self.comm_type = self.args.adv_test
        # Update at these many number of steps
        self.step_update_time = 10

        if self.args.optimizer == "RMSProp":
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.args.actor_lr, decay=0.97, epsilon=1e-6)
        else:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.actor_lr)

        # Setup weight sharing for first initialization of adv/good policy
        if not(self.p_index == 0 or self.p_index == self.num_adv): self.reuse = True
        # Prepare indexing parameters
        if self.name == "good_agent":
            self.comm_type = self.args.good_test
            self.n_start = self.num_adv
            self.n_end = self.n

        # Batch size and number of agents/environments
        self.num_env = num_env

        # Initialize actor network for communication
        actor_net = p_model(is_train, self.args, reuse=self.reuse)
        pMA_model = self.agent_model(self.comm_type, actor_net)

        self.max_replay_buffer_len = self.args.update_lag

        self.act, self.p_train, self.v_train = self._pMA_VPG_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            make_memory_ph_n=memory_ph_n,
            make_h_ph_n=h_ph_n,
            make_c_ph_n=c_ph_n,
            make_act_ph_n=act_ph_n,
            action_space_n=action_space_n,
            make_return_ph_n=return_in_ph,
            p_func=pMA_model,
            grad_norm_clipping=0.5,
            reuse=self.reuse,
        )

    def agent_model(self, comm_type, p_model):
        if comm_type == "SARNET":
            return p_model.sarnet
        elif comm_type == "TARMAC":
            return p_model.tarmac
        elif comm_type == "COMMNET":
            return p_model.commnet
        elif comm_type == "DDPG":
            return p_model.ddpg
        elif comm_type == "IC3NET":
            return p_model.ic3net

    def _p_setup_placeholder(self, obs_ph_n, h_ph_n, c_ph_n, memory_ph_n):
        p_input = [None] * int(self.n * 4)
        for i in range(self.n):
            p_input[i] = obs_ph_n[i]
            p_input[i + self.n] = h_ph_n[i]
            p_input[i + int(2 * self.n)] = c_ph_n[i]
            p_input[i + int(3 * self.n)] = memory_ph_n[i]
        return p_input

    def _pMA_VPG_train(self, make_obs_ph_n, make_memory_ph_n, make_h_ph_n, make_c_ph_n, make_act_ph_n, action_space_n, make_return_ph_n, p_func, grad_norm_clipping=None, scope="agent", reuse=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # create distributions
            act_pdtype_n = [make_pdtype(act_space, self.args.env_type) for act_space in action_space_n]

            # set up placeholders
            obs_ph_n = make_obs_ph_n
            memory_ph_n = make_memory_ph_n
            h_ph_n = make_h_ph_n
            c_ph_n = make_c_ph_n
            act_onehot_ph = make_act_ph_n[self.p_index]
            return_ph = make_return_ph_n[self.p_index]

            # Feed all inputs. Let the model decide what to choose.
            p_input = self._p_setup_placeholder(obs_ph_n, h_ph_n, c_ph_n, memory_ph_n)
            p, enc_state, memory_state, attention, value = p_func(p_input, int(act_pdtype_n[self.p_index].param_shape()[0]), self.p_index, self.n, self.n_start, self.n_end, scope="p_func", reuse=reuse)

            # wrap parameters in distribution and sample
            act_pd = act_pdtype_n[self.p_index].pdfromflat(p)
            act_soft_sample = act_pd.sample(noise=False, onehot=True)
            # print(act_soft_sample)
            act_onehot = tf.multinomial(act_soft_sample[-1,:,:], 1)
            # print(act_onehot)
            value_out = tf.squeeze(value, axis=0)  # remove the time dimension from the output for storing in the buffer

            return_ph_expd = tf.expand_dims(return_ph, axis=-1)
            # Value Network Optimization
            # value = tf.squeeze(value, axis=-1)  # remove the last single out dim, to align with return (#trajlen, #batch)
            target = return_ph_expd - value
            loss_v = tf.reduce_mean(tf.math.squared_difference(value, return_ph_expd))
            optim_v = self.optimizer.minimize(loss_v, name='adam_optim_v')

            # Policy Network Optimization
            # print(act_soft_sample)
            target_pi = tf.squeeze(target, axis=-1)
            loss_pi = tf.reduce_mean(tf.stop_gradient(target_pi) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=p, labels=act_onehot_ph), name='loss_pi')
            optim_pi = self.optimizer.minimize(loss_pi, name='adam_optim_pi')

            # Create callable functions
            # policy network
            # Use sess.run to the feed the dictionary, since we are not calling it anywhere else, simi
            update_pi = optim_pi
            update_v = optim_v
            train_v = U.function(inputs=p_input + [return_ph], outputs=update_v)
            train_pi = U.function(inputs=p_input + [act_onehot_ph] + [return_ph], outputs=update_pi)
            act = U.function(inputs=p_input, outputs=[act_onehot, act_soft_sample, enc_state, memory_state, attention, value_out])

            return act, train_pi, train_v

    def prep_input(self, obs, h, c, memory, is_train=True):
        input = [None] * int(self.n * 4)
        for i in range(self.n):
            input[i] = obs[i]
            input[i + self.n] = h[i]
            input[i + int(2 * self.n)] = c[i]
            input[i + int(3 * self.n)] = memory[i]

        return input

    def action(self, input, is_train=False):
        return self.act(*input)

    def sample_experience(self, bufferop):
        # Receive all the data for the sampled trajectories
        data, index, importance = bufferop.return_exp()

        return data, index, importance

    def update(self, agents, buffer_data, t):
        # Check if an update is needed
        # if not (t % self.step_update_time == 0):  # only update every 10 steps for policy, 5 for critic
        #     return "no_update"

        # Get mini-batch of trajectories
        # Returns the following indexing scheme

        # Shape of the trajectory is [# numtraj, [agent, trajlen, numenv,  dim] or
        # [numtraj [agent, trajlen, num_env]] for rew/done
        obs_n_buffer, h_n_buffer, c_n_buffer, memory_n_buffer, action_n_buffer, action_n_logits_buffer, rew_n_buffer, \
        value_n_buffer, done_n_buffer = buffer_data

        """ Prepare Inputs for network feed """
        # Receives [batch_size, [trajlen, numenv, agent]] -> concat [trajlen, batch x numenv, agent]
        # Reshape to - [agent, trajlen, batchsize x num_env]]
        rew_n_buffer = np.transpose(np.concatenate(rew_n_buffer, axis=1), (2, 0, 1))
        # done_n_buffer = np.transpose(np.concatenate(done_n_buffer, axis=1), (2, 0, 1))
        # Receives [batch_size, [trajlen, agent, numenv]] -> concat [trajlen, agent, batch x numenv]
        # Reshape to - [agent, trajlen, batchsize x num_env]]
        # value_n_buffer = np.transpose(np.concatenate(value_n_buffer, axis=-1), (2, 0, 1))
        # Receives [batch, [traj, agent, numevn, dim]] -> [traj, agent, numenv x batch, dim]
        # Reshape to [agent, trajlen, numenv x batch, dim]
        obs_n_buffer = np.swapaxes(np.concatenate(obs_n_buffer, axis=-2), 1, 0)
        action_n_buffer = np.squeeze(np.swapaxes(np.concatenate(action_n_buffer, axis=-2), 1, 0))
        # For hidden states we only feed the start (i.e. no trajlen)
        h_n_buffer = np.swapaxes(np.concatenate(h_n_buffer, axis=-2), 1, 0)
        h_n_buffer = h_n_buffer[:, 0, :, :]
        c_n_buffer = np.swapaxes(np.concatenate(c_n_buffer, axis=-2), 1, 0)
        c_n_buffer = c_n_buffer[:, 0, :, :]
        memory_n_buffer = np.swapaxes(np.concatenate(memory_n_buffer, axis=-2), 1, 0)
        memory_n_buffer = memory_n_buffer[:, 0, :, :]

        returns = []
        advantages = []
        # Calculate returns
        return_so_far = np.zeros(np.shape(rew_n_buffer[self.p_index, 0, :]))

        # Get trajectory length to compute the returns in reverse
        traj_len, _ = rew_n_buffer[self.p_index].shape
        # Do returns calculation for individual agent
        for traj_idx in reversed(range(traj_len)):
            return_so_far = self.args.gamma * return_so_far + rew_n_buffer[self.p_index, traj_idx, :]
            returns.append(return_so_far)

        # Returns is of the form [trajlen, dim]
        # We need first indexes as agents for easier data manipulation
        # returns = np.stack(returns, axis=0)
        train_input = self.prep_input(obs_n_buffer, h_n_buffer, c_n_buffer, memory_n_buffer)
        #for i in range(5):
        _ = self.v_train(*(train_input + [returns]))
        _ = self.p_train(*(train_input + [action_n_buffer[self.p_index]] + [returns]))

        return "update done"
