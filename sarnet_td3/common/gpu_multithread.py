import threading, queue, time, os, pickle
# from queue import Queue
import numpy as np
import tensorflow as tf
import sarnet_td3.common.tf_util as U
from tensorflow.python.keras.backend import set_session
lock = threading.Lock()


class MultiTrainTD3(threading.Thread):
    def __init__(self, input_queue, output_queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True
        self.trainers = args[0]
        self.args = args[1]
        self.buffer_op = args[2]
        self.num_env = args[3]
        self.sess = args[4]
        self.num_agents = args[5]
        self.num_adversaries = args[6]
        self.ep_rewards = [[0.0] for _ in range(self.num_env)]
        self.ep_end_rewards = [[0.0] for _ in range(self.num_env)]
        self.ep_success = [[0.0] for _ in range(self.num_env)]
        self.agent_rewards = [[[0.0] for _ in range(self.num_agents)] for _ in range(self.num_env)]
        self.agent_info = [[[[]] for i in range(self.num_agents)] for _ in range(self.num_env)]
        # self.agent_info = [[[[]]] for _ in range(self.num_env)]
        self.final_ep_rewards = []  # Shape: (batch, #) sum of rewards for training curve
        self.final_ep_end_rewards = []
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.save_rate = self.args.max_episode_len * 100
        self.save_n_ep = self.num_env * 10
        self.print_step = -int(self.save_n_ep / self.num_env)

        self.q_h_init = np.zeros(shape=(self.num_env, self.args.critic_units))
        self.mem_init = np.zeros(shape=(self.num_env, self.args.value_units))
        self.time_prev = time.time()

    def run(self):
        # print(threading.currentThread().getName(), self.receive_messages)
        with self.sess.as_default():
            # Freeze graph to avoid memory leaks
            # self.sess.graph.finalize()
            while True:
                try:
                    action, p_index, data = self.input_queue.get()
                    if action is "None":  # If you send `None`, the thread will exit.
                        return
                    elif action is "get_action":
                        out = self.get_action(data, p_index)
                        self.output_queue.put(out)
                    elif action is "get_qdebug":
                        out = self.get_qdebug(data, p_index)
                        self.output_queue.put(out)
                    elif action is "get_loss":
                        out = self.get_loss(data, p_index)
                        self.output_queue.put(out)
                    elif action is "write_tboard":
                        self.write_tboard(data)
                    elif action is "add_to_buffer":
                        self.buffer_op.collect_exp(data)
                    elif action is "save_rew_info":
                        self.save_rew_info(data)
                    elif action is "save_benchmark":
                        out = self.save_benchmark(data)
                        self.output_queue.put(out)
                    elif action is "reset_rew_info":
                        self.reset_rew_info()
                    elif action is "save_model_rew":
                        if not (self.args.benchmark or self.args.display):
                            self.save_model(data)
                            self.plot_rewards(data)
                except queue.Empty:
                    continue

    def get_action(self, data, p_index):
        with lock:
            agent = self.trainers[p_index]
            obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_t, is_train = data
            obs_n_t = np.stack(obs_n_t, axis=-2)  # This returns [agent, batch, dim]
            obs_n_t = np.expand_dims(obs_n_t, axis=1)  # This adds [agent, time, batch, dim]
            p_input_j = agent.prep_input(obs_n_t, h_n_t, c_n_t, mem_n_t, q1_h_t[p_index], is_train)
            # print(np.shape(obs_n_t))
            act_j_t, state_j_t1, mem_j_t1, attn_j_t = agent.action(p_input_j, is_train)

            if self.args.encoder_model == "LSTM" or self.args.encoder_model != "DDPG":
                c_j_t1, h_j_t1 = state_j_t1
            else:
                h_j_t1 = state_j_t1
                c_j_t1 = state_j_t1

            if agent.comm_type in {"DDPG", "COMMNET", "IC3NET"}:
                mem_j_t1 = np.zeros(shape=(self.num_env, self.args.value_units))

            return act_j_t, h_j_t1, c_j_t1, mem_j_t1, attn_j_t

    def get_qdebug(self, data, p_index):
        with lock:
            # with sess.as_default():
            agent = self.trainers[p_index]
            obs_n_t, action_n_t, q1_h_n_t, q2_h_n_t = data
            obs_n_t = np.stack(obs_n_t, axis=-2)  # This returns [agent, batch, dim]
            obs_n_t = np.expand_dims(obs_n_t, axis=1)  # This adds [agent, time, batch, dim]
            q1_j_input = agent.prep_q_input(obs_n_t, action_n_t, q1_h_n_t[p_index])
            _, q1_h_j_t1 = agent.q1_debug['q_values'](*(q1_j_input))

            if self.args.td3:
                q2_input = agent.prep_q_input(obs_n_t, action_n_t, q2_h_n_t[p_index])
                _, q2_h_j_t1 = agent.q2_debug['q_values'](*(q2_input))
            else:
                q2_h_j_t1 = []

            return q1_h_j_t1, q2_h_j_t1

    def get_loss(self, data, p_index):
        with lock:
            # with sess.as_default():
            agent = self.trainers[p_index]
            train_step = data
            loss = agent.update(self.trainers, self.buffer_op, train_step)

            return loss

    def write_tboard(self, data):
        with lock:
            loss, train_step, writer, summary_ops, summary_vars, num_agents = data
            # Tensorboard
            episode_b_rewards = []
            for j in range(self.num_env):
                if self.args.env_type == "mpe":
                    episode_b_rewards.append(np.mean(self.ep_rewards[j][self.print_step:]))
                else:
                    episode_b_rewards.append(np.mean(self.ep_success[j][self.print_step:]))
            episode_b_rewards = np.mean(np.array(episode_b_rewards))
            num_steps = train_step * self.num_env

            # Add to tensorboard only when actor agent is updated
            if loss[0][1] is not None:
                fd = {}
                for i, key in enumerate(summary_vars):
                    if i == 0:
                        fd[key] = episode_b_rewards
                    else:
                        agnt_idx = int((i - 1) / 5)
                        if agnt_idx == num_agents: agnt_idx -= 1
                        if loss[agnt_idx] is not None:
                            fd[key] = loss[agnt_idx][int((i - 1) % 5)]

                summary_str = U.get_session().run(summary_ops, feed_dict=fd)
                writer.add_summary(summary_str, num_steps)
                writer.flush()

    def save_rew_info(self, data):
        with lock:
            rew_n, info_n, ep_step = data
            # rew_n (num_env, num_agents)
            if self.args.env_type == "mpe":
                for j in range(self.num_env):
                    for i, rew in enumerate(rew_n[j]):
                        if ep_step >= self.args.max_episode_len - 10:  # Compute only last 10 episode step rewards
                            self.ep_end_rewards[j][-1] += rew
                        self.ep_rewards[j][-1] += rew
                        self.agent_rewards[j][i][-1] += rew
            elif self.args.env_type == "ic3net":
                for j in range(self.num_env):
                    self.ep_success[j][-1] += info_n[j]

            if self.args.benchmark and self.args.env_type == "mpe":
                for j in range(self.num_env):
                    for i, info in enumerate(info_n[j]):
                        self.agent_info[j][i][-1].append(info)

    def reset_rew_info(self):
        with lock:
            for j in range(self.num_env):
                self.ep_rewards[j].append(0)
                self.ep_success[j].append(0)
                self.ep_end_rewards[j].append(0)
                for i in range(self.num_agents):
                    self.agent_rewards[j][i].append(0)

            if self.args.benchmark:
                for j in range(self.num_env):
                    for i in range(self.num_agents):
                        self.agent_info[j][i].append([[]])

    def save_benchmark(self, data):
        with lock:
            exp_name, exp_itr = data
            benchmark_dir = os.path.join('./exp_data', exp_name, exp_itr, self.args.benchmark_dir)
            if not os.path.exists(benchmark_dir):
                os.mkdir(benchmark_dir)
            file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.benchmark_dir + '/' + exp_name + '.pkl'
            print('Finished benchmarking, now saving...')
            # pickle_info = [self.agent_info[j] for j in range(self.num_env)]
            with open(file_name, 'wb') as fp:
                # Dump files as [num_env, [# agents, [#ep, [#stps, [dim]]]]
                pickle.dump(self.agent_info, fp)
            return "bench_saved"

    def save_model(self, data):
        with lock:
            # train_step = t_step * num_env
            train_step, num_episodes, time_taken, exp_name, exp_itr, data_file, saver = data
            # Policy File
            if num_episodes % (self.save_n_ep) == 0:
                save_dir = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.save_dir + str(train_step)
                U.save_state(save_dir, self.sess, saver=saver)
                # episode_rewards, agent_rewards, final_ep_rewards, final_ep_ag_rewards = rewards
                if self.args.env_type == "mpe":
                    # print statement depends on whether or not there are adversaries
                    if self.num_adversaries == 0:
                        episode_b_rewards = []
                        ep_end_b_rewards = []
                        ep_ag_b_rewards = []
                        for j in range(self.num_env):
                            episode_b_rewards.append(np.mean(self.ep_rewards[j][self.print_step:]))
                            ep_end_b_rewards.append(np.mean(self.ep_end_rewards[j][self.print_step:]))
                        episode_b_rewards = np.mean(np.array(episode_b_rewards))
                        ep_end_b_rewards = np.mean(ep_end_b_rewards) / 10.
                        for i in range(self.num_agents):
                            temp_ag_reward = []
                            for j in range(self.num_env):
                                temp_ag_reward.append(np.mean(self.agent_rewards[j][i][self.print_step:]))
                            ep_ag_b_rewards.append(np.mean(np.array(temp_ag_reward)))
                        print("steps: {}, episodes: {}, mean episode reward: {}, mean end rewards: {}, time: {}".format(
                            train_step, num_episodes, episode_b_rewards, ep_end_b_rewards, round(time.time() - self.time_prev, 3)))
                        with open(data_file, "a+") as f:
                            f.write("\n" + "steps: {}, episodes: {}, mean episode reward: {}, mean end rewards: {}, time: {}".format(
                            train_step, num_episodes, episode_b_rewards, ep_end_b_rewards, round(time.time() - self.time_prev, 3)) + "\n")
                    else:
                        episode_b_rewards = []
                        ep_end_b_rewards = []
                        ep_ag_b_rewards = []
                        for j in range(self.num_env):
                            episode_b_rewards.append(np.mean(self.ep_rewards[j][self.print_step:]))
                            ep_end_b_rewards.append(np.mean(self.ep_end_rewards[j][self.print_step:]))
                        episode_b_rewards = np.mean(np.array(episode_b_rewards))
                        ep_end_b_rewards = np.mean(ep_end_b_rewards)
                        for i in range(self.num_agents):
                            temp_ag_reward = []
                            for j in range(self.num_env):
                                temp_ag_reward.append(np.mean(self.agent_rewards[j][i][self.print_step:]))
                            ep_ag_b_rewards.append(np.mean(np.array(temp_ag_reward)))

                        print("steps: {}, episodes: {}, mean episode reward: {}, mean end rewards: {}, agent episode reward: {}, time: {}".format(
                            train_step, num_episodes, episode_b_rewards, ep_end_b_rewards, [rew for rew in ep_ag_b_rewards],
                            round(time.time() - self.time_prev, 3)) + "\n")
                        with open(data_file, "a+") as f:
                            f.write("\n" + "steps: {}, episodes: {}, mean episode reward: {}, mean end rewards: {}, agent episode reward: {}, time: {}".format(
                            train_step, num_episodes, episode_b_rewards, ep_end_b_rewards, [rew for rew in ep_ag_b_rewards],
                            round(time.time() - self.time_prev, 3)) + "\n")

                    # Keep track of final episode reward
                    self.final_ep_rewards.append(episode_b_rewards)
                    self.final_ep_end_rewards.append(ep_end_b_rewards)
                    for rew in ep_ag_b_rewards:
                            self.final_ep_ag_rewards.append(rew)
                self.time_prev = time.time()

    def plot_rewards(self, data):
        with lock:
            train_step, num_episodes, t_start, exp_name, exp_itr, data_file, saver = data
            plot_dir = os.path.join('./exp_data', exp_name, exp_itr, self.args.plots_dir)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            rew_file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.plots_dir + '/' + exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(self.final_ep_rewards, fp)
            rew_ep_end_file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.plots_dir + '/' + exp_name + '_rewards_ep_end.pkl'
            with open(rew_ep_end_file_name, 'wb') as fp:
                pickle.dump(self.final_ep_end_rewards, fp)
            agrew_file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.plots_dir + '/' + exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(self.final_ep_ag_rewards, fp)


"""
REINFORCE Threads
"""


class MultiTrainVPG(threading.Thread):
    def __init__(self, input_queue, output_queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True
        self.trainers = args[0]
        self.args = args[1]
        self.buffer_op = args[2]
        self.num_env = args[3]
        self.sess = args[4]
        self.num_agents = args[5]
        self.num_adversaries = args[6]
        self.ep_rewards = [[0.0] for _ in range(self.num_env)]
        self.ep_success = [[0.0] for _ in range(self.num_env)]
        self.agent_rewards = [[[0.0] for _ in range(self.num_agents)] for _ in range(self.num_env)]
        self.agent_info = [[[[]]] for _ in range(self.num_env)]
        self.final_ep_rewards = []  # Shape: (batch, #) sum of rewards for training curve
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.save_rate = self.args.max_episode_len * 100
        if self.args.env_type == "mpe":
            self.print_step = -int(self.save_rate / self.num_env)
        else:  # print for episode end only (success rate)
            self.print_step = -int(self.save_rate / (self.num_env * self.args.max_episode_len))

        self.q_h_init = np.zeros(shape=(self.num_env, self.args.critic_units))
        self.mem_init = np.zeros(shape=(self.num_env, self.args.value_units))

        self.time_prev = time.time()

    def run(self):
        # print(threading.currentThread().getName(), self.receive_messages)
        with self.sess.as_default():
            # Freeze graph to avoid memory leaks
            # self.sess.graph.finalize()
            while True:
                try:
                    action, p_index, data = self.input_queue.get()
                    if action is "None":  # If you send `None`, the thread will exit.
                        return
                    elif action is "get_action":
                        out = self.get_action(data, p_index)
                        self.output_queue.put(out)
                    elif action is "get_loss":
                        out = self.get_loss(data, p_index)
                        self.output_queue.put(out)
                    elif action is "write_tboard":
                        self.write_tboard(data)
                    elif action is "add_to_buffer":
                        self.buffer_op.collect_exp(data)
                    elif action is "add_to_buffer_reinforce":
                        self.buffer_op.collect_exp(data)
                    elif action is "save_rew_info":
                        self.save_rew_info(data)
                    elif action is "save_benchmark":
                        out = self.save_benchmark(data)
                        self.output_queue.put(out)
                    elif action is "reset_rew_info":
                        self.reset_rew_info()
                    elif action is "save_model_rew":
                        if not (self.args.benchmark or self.args.display):
                            self.save_model(data)
                            self.plot_rewards(data)
                except queue.Empty:
                    continue

    def get_action(self, data, p_index):
        with lock:
            agent = self.trainers[p_index]
            obs_n_t, h_n_t, c_n_t, mem_n_t, is_train = data
            obs_n_t = np.stack(obs_n_t, axis=-2)
            obs_n_t = np.expand_dims(obs_n_t, axis=1)  # This adds [agent, time, batch, dim]
            p_input_j = agent.prep_input(obs_n_t, h_n_t, c_n_t, mem_n_t, is_train)
            act_j_t, act_soft_j_t, state_j_t1, mem_j_t1, attn_j_t, value_j_t = agent.action(p_input_j, is_train)

            if self.args.encoder_model == "LSTM":
                c_j_t1, h_j_t1 = state_j_t1
            else:
                h_j_t1 = state_j_t1
                c_j_t1 = state_j_t1

            if agent.comm_type in {"DDPG", "COMMNET", "IC3NET"}:
                mem_j_t1 = np.zeros(shape=(self.num_env, self.args.value_units))

            return act_j_t, act_soft_j_t, h_j_t1, c_j_t1, mem_j_t1, attn_j_t, value_j_t

    def get_loss(self, data, p_index):
        with lock:
            # with sess.as_default():
            train_step, buffer_data = data
            agent = self.trainers[p_index]
            loss = agent.update(self.trainers, buffer_data, train_step)

            return loss

    def write_tboard(self, data):
        with lock:
            loss, train_step, writer, summary_ops, summary_vars, num_agents = data
            # Tensorboard
            episode_b_rewards = []
            for j in range(self.num_env):
                if self.args.env_type == "mpe":
                    episode_b_rewards.append(np.mean(self.ep_rewards[j][self.print_step:]))
                else:
                    episode_b_rewards.append(np.mean(self.ep_success[j][self.print_step:]))
            episode_b_rewards = np.mean(np.array(episode_b_rewards))
            num_steps = train_step * self.num_env

            # Add to tensorboard only when actor agent is updated
            if loss[0][1] is not None:
                fd = {}
                for i, key in enumerate(summary_vars):
                    if i == 0:
                        fd[key] = episode_b_rewards
                    else:
                        agnt_idx = int((i - 1) / 5)
                        if agnt_idx == num_agents: agnt_idx -= 1
                        if loss[agnt_idx] is not None:
                            fd[key] = loss[agnt_idx][int((i - 1) % 5)]

                summary_str = U.get_session().run(summary_ops, feed_dict=fd)
                writer.add_summary(summary_str, num_steps)
                writer.flush()

    def save_rew_info(self, data):
        with lock:
            rew_n, info_n, terminal = data
            if self.args.env_type == "mpe":
                for j in range(self.num_env):
                    for i, rew in enumerate(rew_n[j]):
                        self.ep_rewards[j][-1] += rew
                        self.agent_rewards[j][i][-1] += rew
            elif self.args.env_type == "ic3net":
                for j in range(self.num_env):
                    self.ep_success[j][-1] += info_n[j]

            if self.args.benchmark and self.args.env_type == "mpe":
                for j in range(self.num_env):
                    for i, info in enumerate(info_n[j]):
                        self.agent_info[-1][i].append(info_n[0]['n'])

    def reset_rew_info(self):
        with lock:
            for j in range(self.num_env):
                self.ep_rewards[j].append(0)
                self.ep_success[j].append(0)
                for i in range(self.num_agents):
                    self.agent_rewards[j][i].append(0)
            if self.args.benchmark:
                for j in range(self.num_env):
                    self.agent_info[j].append([[]])

    def save_benchmark(self, data):
        with lock:
            exp_name, exp_itr = data
            benchmark_dir = os.path.join('./exp_data', exp_name, exp_itr, self.args.benchmark_dir)
            if not os.path.exists(benchmark_dir):
                os.mkdir(benchmark_dir)
            file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.benchmark_dir + '/' + exp_name + '.pkl'
            print('Finished benchmarking, now saving...')
            with open(file_name, 'wb') as fp:
                pickle.dump(self.ep_success, fp)
            return "bench_saved"

    def save_model(self, data):
        with lock:
            # train_step = t_step * num_env
            train_step, num_episodes, time_taken, exp_name, exp_itr, data_file, saver = data
            # Policy File
            save_dir = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.save_dir + str(train_step)
            U.save_state(save_dir, self.sess, saver=saver)
            episode_b_success = []
            for j in range(self.num_env):
                episode_b_success.append(np.mean(self.ep_success[j][self.print_step:]))
            episode_b_success = np.mean(np.array(episode_b_success)) / self.args.max_episode_len
            print("steps: {}, episodes: {}, mean episode success: {}, time: {}".format(
                train_step, num_episodes, episode_b_success, round(time.time() - self.time_prev, 3)) + "\n")
            with open(data_file, "a+") as f:
                f.write("\n" + "steps: {}, episodes: {}, mean episode success: {}, time: {}".format(
                    train_step, num_episodes, episode_b_success, round(time.time() - self.time_prev, 3)) + "\n")
                self.final_ep_rewards.append(episode_b_success)

    def plot_rewards(self, data):
        with lock:
            train_step, num_episodes, t_start, exp_name, exp_itr, data_file, saver = data
            plot_dir = os.path.join('./exp_data', exp_name, exp_itr, self.args.plots_dir)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            rew_file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + self.args.plots_dir + '/' + exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(self.final_ep_rewards, fp)


def get_gputhreads(trainers, args, buffer_op, num_env, num_agents, num_adv):
    threads = []
    sess = tf.compat.v1.get_default_session()
    for t in range(args.num_gpu_threads):
        input_q = queue.Queue()
        output_q = queue.Queue()
        if args.policy_grad == "maddpg":
            threads.append(MultiTrainTD3(input_q, output_q, args=(trainers, args, buffer_op, num_env, sess, num_agents, num_adv)))
        elif args.policy_grad == "reinforce":
            threads.append(
                MultiTrainVPG(input_q, output_q, args=(trainers, args, buffer_op, num_env, sess, num_agents, num_adv)))
        threads[t].start()
        time.sleep(1)

    return threads


def close_gputhreads(threads):
    for t in threads:
        t.input_queue.put(("None", None, None))

    for t in threads:
        t.join()

    print('GPU trainers cancelled')

    return
