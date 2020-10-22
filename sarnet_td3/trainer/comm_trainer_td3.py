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
        try:
            expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
        except:
            print(var_target)
            print(var)
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def create_placeholder_td3(obs_shape_n, act_space_n, num_agents, args):
    # Create placeholders
    with tf.name_scope("placeholders"):
        obs_ph_n = []
        memory_ph_n = []
        gru1_ph_n = []
        gru2_ph_n = []
        q_gru_ph_n = []
        target_ph = []
        for i in range(num_agents):
            # Generate [Traj, Batch, Dimension] placeholder
            if args.env_type == "mpe":
                obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i), traj=True).get())
            else:
                obs_ph_n.append(U.BatchInput((obs_shape_n[i],), name="observation" + str(i), traj=True).get())
            gru1_ph_n.append(U.BatchInput((args.gru_units,), name="gru_ph1" + str(i)).get())
            gru2_ph_n.append(U.BatchInput((args.gru_units,), name="gru_ph2" + str(i)).get())
            q_gru_ph_n.append(U.BatchInput((args.critic_units,), name="q_gru_ph" + str(i)).get())
            memory_ph_n.append(U.BatchInput((args.value_units,), name="memory_ph" + str(i)).get())

        act_pdtype_n = [make_pdtype(act_space, args.env_type) for act_space in act_space_n]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i), traj=True) for i in range(len(act_space_n))]

        critic_ct = 1
        if args.td3:
            critic_ct += 1
        for i in range(critic_ct):
            target_ph.append(tf.compat.v1.placeholder(tf.float32, [None, None], name="target" + str(critic_ct)))

        importance_in_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])

        return obs_ph_n, gru1_ph_n, gru2_ph_n, memory_ph_n, act_ph_n, act_space_n, target_ph, q_gru_ph_n, importance_in_ph


class CommAgentTrainerTD3(MAgentTrainer):
    def __init__(self, name, p_model, critic_model, obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, act_ph_n,
                 action_space_n, target_ph, q_gru_ph_n, importance_in_ph, args, p_index, num_env=1, is_train=False):
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
        if self.args.td3:
            self.step_update_time = 5

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

        # TD3 Updates
        self.polyak = self.args.polyak / (self.n_end - self.n_start)
        self.p_flag = 0
        self.update_flag = 0
        if self.args.td3:
            self.update_flag = 1

        # Initialize actor network for communication
        actor_net = p_model(is_train, self.args, reuse=self.reuse)
        pMA_model = self.agent_model(self.comm_type, actor_net)

        self.min_replay_buffer_len = self.args.update_lag

        # Relaxed One Hot for discrete grid world environments
        self.tau0 = 1.0
        self.temperature = self.tau0
        self.ANNEAL_RATE = 1e-5
        self.MIN_TEMP = 1e-4

        # Create all the functions necessary to train the model
        self.q1_train, self.q1_update, self.q1_debug = self._qMA_train(
            critic_index=1,
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            make_q_gru_ph_n=q_gru_ph_n,
            make_act_ph_n=act_ph_n,
            make_target_ph=target_ph[0],
            importance_in=importance_in_ph,
            q_func=critic_model,
            optimizer=self.optimizer,
            grad_norm_clipping=0.5,
            reuse=False
        )

        if self.args.td3:
            self.q2_train, self.q2_update, self.q2_debug = self._qMA_train(
                critic_index=2,
                scope=self.name,
                make_obs_ph_n=obs_ph_n,
                make_q_gru_ph_n=q_gru_ph_n,
                make_act_ph_n=act_ph_n,
                make_target_ph=target_ph[1],
                importance_in=importance_in_ph,
                q_func=critic_model,
                optimizer=self.optimizer,
                grad_norm_clipping=0.5,
                reuse=False
            )

        self.act, self.p_train, self.p_update, self.p_debug = self._pMA_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            make_memory_ph_n=memory_ph_n,
            make_q_gru_ph_n=q_gru_ph_n,
            make_h_ph_n=h_ph_n,
            make_c_ph_n=c_ph_n,
            make_act_ph_n=act_ph_n,
            action_space_n=action_space_n,
            importance_in=importance_in_ph,
            p_func=pMA_model,
            q_func=critic_model,
            optimizer=self.optimizer,
            grad_norm_clipping=0.5,
            reuse=self.reuse,
        )

    def set_temp(self, train_step):
        self.temperature = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * train_step), self.MIN_TEMP)

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

    def _p_setup_placeholder(self, obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, q_gru_ph):
        p_input = [None] * (int(self.n * 4) + 1)
        for i in range(self.n):
            p_input[i] = obs_ph_n[i]
            p_input[i + self.n] = h_ph_n[i]
            p_input[i + int(2 * self.n)] = c_ph_n[i]
            p_input[i + int(3 * self.n)] = memory_ph_n[i]
        p_input[int(self.n * 4)] = q_gru_ph

        return p_input

    def _qp_setup_placeholder(self, p_input, action_ph_n):
        q_input = [None] * (self.n * 2 + 1)
        for i in range(self.n):
            q_input[i] = p_input[i]
            q_input[self.n + i] = action_ph_n[i]
        q_input[int(self.n * 2)] = p_input[self.n * 4]

        return q_input

    def _q_setup_placeholder(self, obs_ph_n, q_gru_ph, action_ph_n):
        q_input = [None] * (self.n * 2 + 1)
        for i in range(self.n):
            q_input[i] = obs_ph_n[i]
            q_input[self.n + i] = action_ph_n[i]
        q_input[int(self.n * 2)] = q_gru_ph

        return q_input

    def _pMA_train(self, make_obs_ph_n, make_memory_ph_n, make_q_gru_ph_n, make_h_ph_n, make_c_ph_n, make_act_ph_n, action_space_n, importance_in, p_func, q_func, optimizer,
                   grad_norm_clipping=None, scope="agent", reuse=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # create distributions
            act_pdtype_n = [make_pdtype(act_space, self.args.env_type) for act_space in action_space_n]

            # set up placeholders
            obs_ph_n = make_obs_ph_n
            memory_ph_n = make_memory_ph_n
            h_ph_n = make_h_ph_n
            c_ph_n = make_c_ph_n
            act_ph_n = make_act_ph_n
            q_gru_ph = make_q_gru_ph_n[self.p_index]

            # Feed all inputs. Let the model decide what to choose.
            p_input = self._p_setup_placeholder(obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, q_gru_ph)
            p, enc_state, memory_state, attention = p_func(p_input, int(act_pdtype_n[self.p_index].param_shape()[0]), self.p_index, self.n, self.n_start, self.n_end, scope="p_func", reuse=reuse)
            # Get parent/relative scope of the policy function
            p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

            # wrap parameters in distribution
            act_pd = act_pdtype_n[self.p_index].pdfromflat(p)

            if not (self.args.benchmark or self.args.display):
                act_sample = act_pd.sample()  # Add gumbel noise to prediction for regularization
            else:
                act_sample = act_pd.sample(noise=False)  # only softmax, no noise

            # Calculate loss
            # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
            p_reg_t = tf.reduce_sum(tf.square(act_pd.flatparam()), axis=1) / tf.to_float(self.args.len_traj_update)
            p_reg = tf.reduce_sum(p_reg_t, axis=0) / tf.to_float(self.args.batch_size)

            act_input_n = act_ph_n + []
            # Use Gumbel Out for calculating policy loss
            act_input_n[self.p_index] = act_pd.sample()
            q_input = self._qp_setup_placeholder(p_input, act_input_n)

            q, state = q_func(q_input, self.n, self.args, scope="q_func" + str(self.p_index) + "1", reuse=True, p_index=self.p_index)
            q = q[:, :, 0]
            # Calculate policy loss
            # pg_loss = -tf.reduce_mean(q)
            pg_loss_t = tf.reduce_sum(q, axis=1) / tf.to_float(self.args.len_traj_update)
            pg_loss = - tf.reduce_sum(pg_loss_t, axis=0) / tf.to_float(self.args.batch_size)
            loss = pg_loss + p_reg * 1e-3
            optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

            # Create callable functions
            # policy network
            train = U.function(inputs=p_input + act_ph_n, outputs=loss, updates=[optimize_expr])
            act = U.function(inputs=p_input, outputs=[act_sample, enc_state, memory_state, attention])
            p_values = U.function(p_input, p)

            # target network (Use one hot for discrete)
            target_p, t_enc_state, target_memory, _ = p_func(p_input, int(act_pdtype_n[self.p_index].param_shape()[0]), self.p_index, self.n, self.n_start, self.n_end, scope="target_p_func", reuse=reuse)
            target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
            update_target_p = make_update_exp(p_func_vars, target_p_func_vars, self.polyak)

            # if self.args.env_type == "ic3net": noise_target = False
            target_act_sample = act_pdtype_n[self.p_index].pdfromflat(target_p).sample(noise=True)
            target_act = U.function(inputs=p_input, outputs=[target_act_sample, t_enc_state, target_memory])

            return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

    def _qMA_train(self, critic_index, make_obs_ph_n, make_q_gru_ph_n, make_act_ph_n, make_target_ph, importance_in, q_func, optimizer, grad_norm_clipping=None,
                  scope="trainer", reuse=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # set up placeholders
            obs_ph_n = make_obs_ph_n
            q_gru_ph_n = make_q_gru_ph_n
            act_ph_n = make_act_ph_n
            target_ph = make_target_ph

            q_input = self._q_setup_placeholder(obs_ph_n, q_gru_ph_n[self.p_index], act_ph_n)

            q, q_gru_state = q_func(q_input, self.n, self.args, scope="q_func" + str(self.p_index) + str(critic_index), reuse=reuse, p_index=self.p_index)
            q_func_vars = U.scope_vars(U.absolute_scope_name("q_func" + str(self.p_index) + str(critic_index)))
            q = q[:, :, 0]

            q_error = q - target_ph
            if self.args.PER_sampling:
                q_loss = tf.reduce_mean(tf.multiply(tf.square(q_error), importance_in))
            else:
                q_loss_t = tf.reduce_sum(tf.square(q_error), axis=1) / tf.to_float(self.args.len_traj_update)
                q_loss = tf.reduce_sum(q_loss_t, axis=0) / tf.to_float(self.args.batch_size)
            # viscosity solution to Bellman differential equation in place of an initial condition
            q_reg = tf.reduce_mean(tf.square(q))
            loss = q_loss + 1e-3 * q_reg

            optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

            # Create callable functions
            if self.args.PER_sampling:
                train = U.function(inputs=q_input + [importance_in] + [target_ph], outputs=[loss, q_error], updates=[optimize_expr])
            else:
                train = U.function(inputs=q_input + [target_ph], outputs=[loss, q_error], updates=[optimize_expr])
            q_values = U.function(q_input, [q, q_gru_state])

            # target network
            target_q, t_q_gru_state = q_func(q_input, self.n, self.args, scope="target_q_func" + str(self.p_index) + str(critic_index), reuse=reuse, p_index=self.p_index)
            target_q = target_q[:, :, 0]
            target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func" + str(self.p_index) + str(critic_index)))
            update_target_q = make_update_exp(q_func_vars, target_q_func_vars, self.args.polyak)
            target_q_values = U.function(q_input, [target_q, t_q_gru_state])

            return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

    def prep_input(self, obs, h, c, memory, q_h, is_train):
        input = [None] * int(self.n * 4 + 1)
        for i in range(self.n):
            input[i] = obs[i]
            input[i + self.n] = h[i]
            input[i + int(2 * self.n)] = c[i]
            input[i + int(3 * self.n)] = memory[i]
        input[self.n * 4] = q_h

        return input

    def prep_q_input(self, obs_n, action_n, q_gru):
        input = [None] * (int(self.n * 2) + 1)
        for i in range(self.n):
            input[i] = obs_n[i]
            input[self.n + i] = action_n[i]
        input[int(self.n * 2)] = q_gru

        return input

    def action(self, input, is_train=False):
        return self.act(*input)

    def sample_experience(self, bufferop):
        # Receive all the data for the sampled trajectories
        data, index, importance = bufferop.return_exp(self.p_index)

        return data, index, importance

    def set_priority(self, buffer, indices, errors):
        # errors are of the shape [#trajlen, batch]
        # Compute mean over trajectory length
        traj_error_mean = self.args.alpha * np.amax(errors, axis=0) + (1 - self.args.alpha) * np.mean(errors, axis=0)
        traj_error_mean = np.clip(traj_error_mean, 0.01, 2)
        buffer.set_priorities(indices, traj_error_mean, self.p_index)

    def target_update_rmaddpg(self, agents, obs_n_t1, h_n_t1, c_n_t1, mem_n_t1, q1_h_n_t1, q2_h_n_t1):
        # data, index, importance = self.sample_experience(buffer)

        t_act_next_n = []
        # Get the samples for a particular step: Shape [agents, num_env, dim]
        for i, agent in enumerate(agents):
            t_input = agent.prep_input(obs_n_t1, h_n_t1, c_n_t1, mem_n_t1, q1_h_n_t1[self.p_index], is_train=True)  # q1 gru is dummy
            t_act_next, _, _ = agent.p_debug['target_act'](*t_input)
            t_act_next_n.append(t_act_next)

        t_q1_input = self.prep_q_input(obs_n_t1, t_act_next_n, q1_h_n_t1[self.p_index])
        t_q1_next, _ = self.q1_debug['target_q_values'](*t_q1_input)

        if self.args.td3:
            t_q2_input = self.prep_q_input(obs_n_t1, t_act_next_n, q2_h_n_t1[self.p_index])
            t_q2_next, _ = self.q2_debug['target_q_values'](*t_q2_input)
            t_q1_next = np.minimum(t_q1_next, t_q2_next)

        return t_q1_next

    def update(self, agents, bufferop, t):
        # Check if an update is needed
        if len(bufferop.buffer) < self.min_replay_buffer_len:  # replay buffer is not large enough
            return "no_update"
        if not (t % self.step_update_time == 0):  # only update every 10 steps for policy, 5 for critic
            return "no_update"

        # print("update happened at time step: {}".format(t))

        # if t % 1000 == 0:
        #     self.set_temp(t)

        # Get target actions
        # data is of shape [agent, traj, batch, dim]
        data, indexes, importance = self.sample_experience(bufferop)
        # Stack all inputs as a trajectory with [agent, time, batchsize, dim]

        obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_n_t, q2_h_n_t, action_n_t, rew_n_t, obs_n_t1, \
        h_n_t1, c_n_t1, mem_n_t1, q1_h_n_t1, q2_h_n_t1, done_n_t = data

        target_q_next = self.target_update_rmaddpg(agents, obs_n_t1, h_n_t1[:, 0, :, :], c_n_t1[:, 0, :, :], mem_n_t1[:, 0, :, :], q1_h_n_t1[:, 0, :, :], q2_h_n_t1[:, 0, :, :])

        rew = np.array(rew_n_t)[self.p_index]
        if self.args.env_type == "mpe":
            done_n_t = np.reshape(np.array(done_n_t)[self.p_index], target_q_next.shape)
        else:
            done_n_t = np.reshape(np.array(done_n_t)[:], target_q_next.shape)
        ep_target_q = rew + self.args.gamma * target_q_next #* (1 - done_n_t)

        t_q1_input = self.prep_q_input(obs_n_t, action_n_t, q1_h_n_t[self.p_index, 0, :, :])
        if self.args.PER_sampling:
            q1_loss, q1_error = self.q1_train(*(t_q1_input + [importance] + [ep_target_q]))
        else:
            q1_loss, q1_error = self.q1_train(*(t_q1_input + [ep_target_q]))

        if self.args.td3:
            t_q2_input = self.prep_q_input(obs_n_t, action_n_t, q2_h_n_t[self.p_index, 0, :, :])
            if self.args.PER_sampling:
                q2_loss, q2_error = self.q2_train(*(t_q2_input + [importance] + [ep_target_q]))
            else:
                q2_loss, q2_error = self.q2_train(*(t_q2_input + [ep_target_q]))

        q_error = q1_error
        if self.args.PER_sampling:
            if self.args.td3:
                q_error = np.minimum(q_error, q2_error)
            self.set_priority(bufferop, indexes, q_error)

        # end = time.time()
        # print("Priority Setting Took %f ms" % ((end - start) * 1000.0))
        p_loss = None
        if self.p_flag == self.update_flag:
            p_input = self.prep_input(obs_n_t, h_n_t[:, 0, :, :], c_n_t[:, 0, :, :],
                                      mem_n_t[:, 0, :, :], q1_h_n_t[self.p_index, 0, :, :], is_train=True)
            act = []
            for i in range(len(agents)):
                act.append(action_n_t[i])
            p_loss = self.p_train(*(p_input + act))
            self.p_update()
            self.p_flag = 0
        else:
            self.p_flag += 1

        self.q1_update()
        if self.args.td3:
            self.q2_update()

        return [q1_loss, p_loss, np.mean(ep_target_q), np.mean(rew), np.mean(target_q_next)]
