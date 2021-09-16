import queue
import time, os, pickle
import numpy as np
import tensorflow as tf
import sarnet_td3.common.tf_util as U
from sarnet_td3.common.np_utils import create_dir
from experiments.config_args import build_summaries
from experiments.benchmark import benchmark


""" REINFORCE Update Utilities"""
class ActionOPVPG(object):
    def __init__(self, trainers, args, num_env, num_agents, env_proc, train_thread, is_train):
        self.trainers = trainers
        self.args = args
        self.num_env = num_env
        self.num_agents = num_agents
        self.env_proc = env_proc
        self.train_thread = train_thread
        self.is_train = is_train
        self.episode_step = 0
        self.train_step = 0
        self.train_epoch = 0
        self.save_rate = self.args.max_episode_len * 10
        if self.args.restore:
            self.train_step = int(args.policy_file)

        """ Setup one time hidden state/memory initialization parameters """
        # Initial hidden states for actor Networks
        self.new_memory_n_init = [np.zeros(shape=(self.num_env, args.value_units)) for _ in range(self.num_agents)]
        self.new_h_n_init = [np.zeros(shape=(self.num_env, args.gru_units)) for _ in range(self.num_agents)]
        self.new_c_n_init = []
        if self.args.encoder_model in {"LSTM"}:
            self.new_c_n_init = [np.zeros(shape=(self.num_env, args.gru_units)) for _ in range(self.num_agents)]

        """ Input parameters for feeding network and buffer at every timestep
        Dimensions: [Agent, [Batch/env, dim]], Ready to be fed to network
        """
        # Actor Hidden States
        self.h_n_t = self.new_h_n_init
        self.c_n_t = self.new_c_n_init
        # Communication
        self.memory_n_t = self.new_memory_n_init
        # Temporary trajectory buffer
        self.buffer_size = 0
        self.reset_temp_traj_buffer()
        self.reset_main_buffer_batch()
        self.terminal = False

        # Create directories
        self.exp_name, self.exp_itr, self.tboard_dir, self.data_file = create_dir(self.args)
        self.create_tensorboard()
        self.t_start = time.time()

        # Allocate Threads for each operation
        self.actor_critic_thread = [p_index % (self.args.num_gpu_threads - 1) for p_index in range(self.num_agents)]
        self.cpu_rew_thread = self.args.num_gpu_threads - 1

        # Benchmarking things to track attention and other metrics
        self.attn_benchmark = []
        self.mem_benchmark = []

    """
    -----------------------------------------------------------------------
    Action Prediction and Critic Updates
    -----------------------------------------------------------------------
    """
    # Stores actions and hidden states received in action_n_t and memory/h/c values
    def queue_recv_actor(self):
        self.action_n_t = [None] * len(self.trainers)
        self.action_n_onehot_t = [None] * len(self.trainers)
        self.memory_n_t1 = [None] * len(self.trainers)
        self.h_n_t1 = [None] * len(self.trainers)
        self.c_n_t1 = [None] * len(self.trainers)
        self.attn_n_t = [None] * len(self.trainers)
        self.value_n_t = [None] * len(self.trainers)

        """ Actor Output """
        # print("Shape of obs", np.shape(self.obs_n_t))
        p_data = (self.obs_n_t, self.h_n_t, self.c_n_t, self.memory_n_t, self.is_train)
        for p_index in range(self.num_agents):
            thread_idx = self.actor_critic_thread[p_index]
            self.train_thread[thread_idx].input_queue.put(("get_action", p_index, p_data))

        # ToDo optimize the way the threads wait for the results
        done_thread = [None] * int(self.num_agents)
        while not all(done_thread):
            for p_index in range(self.num_agents):
                thread_idx = self.actor_critic_thread[p_index]
                try:
                    if self.action_n_t[p_index] is None:
                        self.action_n_onehot_t[p_index], act_p_index_t, self.h_n_t1[p_index], self.c_n_t1[p_index], \
                        self.memory_n_t1[p_index], self.attn_n_t[p_index], self.value_n_t[p_index] = \
                                                            self.train_thread[thread_idx].output_queue.get()
                        # print(self.action_n_onehot_t[p_index])
                        # Remove the time dimension from axis 0
                        self.action_n_t[p_index] = np.squeeze(act_p_index_t, axis=0)
                    else:
                        done_thread[p_index] = True
                except self.train_thread[thread_idx].output_queue.empty:
                    if all(done_thread):
                        break
                    else:
                        continue

        if self.args.benchmark:
            self.attn_benchmark.append(self.attn_n_t)
            self.mem_benchmark.append(self.memory_n_t1)

    def update_states(self):
        if self.terminal:

            self.episode_step = 0
            self.reset_states()
        else:
            self.obs_n_t = self.obs_n_t1
            self.h_n_t = self.h_n_t1
            self.c_n_t = self.c_n_t1
            self.memory_n_t = self.memory_n_t1

    """
    -----------------------------------------------------------------------
    Buffer Manipulation
    -----------------------------------------------------------------------
    """

    def reset_temp_traj_buffer(self):
        self.obs_n_traj, self.h_n_traj, self.c_n_traj, self.memory_n_traj, self.value_n_traj, self.action_n_logits_traj, \
        self.action_n_traj, self.rew_n_traj, self.done_n_traj = [], [], [], [], [], [], [], [], []

    def add_traj(self):
        # network outs are - [#agent, [batch, dim]] and the fed as [#numtraj, [agents, [batch, dim]]]
        self.obs_n_traj.append(np.stack(self.obs_n_t, axis=-2))  # Reshape each time step obs to [agent, batch, dim]
        self.action_n_traj.append(self.action_n_onehot_t)
        self.action_n_logits_traj.append(self.action_n_t)
        self.done_n_traj.append(self.done_n_t)
        self.rew_n_traj.append(self.rew_n_t)
        self.h_n_traj.append(self.h_n_t)
        self.c_n_traj.append(self.c_n_t)
        self.memory_n_traj.append(self.memory_n_t)
        # This [agent, batch] -> [trajlen, [agent, batch]]
        self.value_n_traj.append(self.value_n_t)

    def reset_main_buffer_batch(self):
        self.obs_n_buffer = []
        self.h_n_buffer = []
        self.c_n_buffer = []
        self.memory_n_buffer = []
        self.action_n_buffer = []
        self.action_n_logits_buffer = []
        self.rew_n_buffer = []
        self.value_n_buffer = []
        self.done_n_buffer = []
        self.buffer_size = 0

    def save_buffer(self):
        self.terminal = (self.episode_step >= self.args.max_episode_len)
        if self.args.benchmark or self.args.display:
            return True

        if len(self.obs_n_traj) < self.args.len_traj_update:
            self.add_traj()
            return False
        else:
            # Input data is to each buffer in is [traj, agent, numenv, dim]
            self.obs_n_buffer.append(np.asarray(self.obs_n_traj))
            self.h_n_buffer.append(np.asarray(self.h_n_traj))
            self.c_n_buffer.append(np.asarray(self.c_n_traj))
            self.memory_n_buffer.append(np.asarray(self.memory_n_traj))
            self.action_n_buffer.append(np.asarray(self.action_n_traj))
            self.action_n_logits_buffer.append(np.asarray(self.action_n_logits_traj))
            # Store it as [trajlen, numenv, #agent] -> [batch, [trajlen, numenv, agent]]
            self.rew_n_buffer.append(np.asarray(self.rew_n_traj))
            self.done_n_buffer.append(np.asarray(self.done_n_traj))
            # This is from the network [batch, [trajlen, agent, numenv]]
            self.value_n_buffer.append(np.asarray(self.value_n_traj))
            self.buffer_size += 1
            # Reset the temporary trajectory buffer and then add to it
            self.reset_temp_traj_buffer()
            self.add_traj()

            return True

    def return_buffer(self):
        # Returns [batch, [trajlen, agent, numenv, dim]] or [batch, [trajlen, numenv, #agent]]
        return (self.obs_n_buffer, self.h_n_buffer, self.c_n_buffer, self.memory_n_buffer, self.action_n_buffer,
                self.action_n_logits_buffer, self.rew_n_buffer, self.value_n_buffer, self.done_n_buffer)

    """
    -----------------------------------------------------------------------
    Environment computations
    -----------------------------------------------------------------------
    """
    def get_env_act(self):
        # print(np.shape(np.stack(self.action_n_onehot_t, axis=-2)))
        # print(np.shape(self.action_n_onehot_t))
        # print(np.shape(np.stack(self.action_n_onehot_t, axis=-2)))
        self.obs_n_t1, self.rew_n_t, self.done_n_t, self.info_n_t = self.env_proc.step(np.stack(self.action_n_onehot_t, axis=-2))
        # print(self.info_n_t)
        # Increment counters
        self.episode_step += 1
        self.train_step += 1

    def reset_states(self):
        # print("resetting state \n")
        # Actor Hidden States
        self.h_n_t = self.new_h_n_init
        self.c_n_t = self.new_c_n_init
        # Communication
        self.memory_n_t = self.new_memory_n_init
        epoch = int((self.train_step / self.args.max_episode_len) // 10) # 10 is the number of updates before an epoch is consumed
        self.obs_n_t = self.env_proc.reset(epoch)
        # Reward
        # self.train_thread[self.cpu_rew_thread].input_queue.put(("reset_rew_info", 0, None))

    def display_env(self):
        if self.args.display:
            time.sleep(0.1)
            self.env_proc.render()

    """
    -----------------------------------------------------------------------
    Reward and training data storage and manipulation
    -----------------------------------------------------------------------
    """
    def reset_rew_info(self):
        # Reward
        self.train_thread[self.cpu_rew_thread].input_queue.put(("reset_rew_info", 0, None))

    def save_rew_info(self):
        self.train_thread[self.args.num_gpu_threads - 1].input_queue.put(("save_rew_info", 0, (self.rew_n_t, self.info_n_t, self.episode_step)))

    def save_benchmark(self):
        if self.args.benchmark_iters > self.train_step * self.num_env:
            return None
        if self.terminal:
            self.train_thread[self.cpu_rew_thread].input_queue.put(("save_benchmark", 0, (self.exp_name, self.exp_itr)))
            bench_status = None
            while not(bench_status):
                try:
                    if bench_status is not "bench_saved":
                        bench_status = self.train_thread[self.cpu_rew_thread].output_queue.get()
                except self.train_thread[self.cpu_rew_thread].output_queue.empty:
                    if bench_status is "bench_saved":
                        break
                    else:
                        continue

            file_name = './exp_data/' + self.exp_name + '/' + self.exp_itr + '/' + self.args.benchmark_dir + '/' + self.exp_name + "_attn" + '.pkl'
            with open(file_name, 'wb') as fp:
                pickle.dump(self.attn_benchmark, fp)
            fp.close()
            print("Saving memory")
            file_name = './exp_data/' + self.exp_name + '/' + self.exp_itr + '/' + self.args.benchmark_dir + '/' + self.exp_name + "_mem" + '.pkl'
            with open(file_name, 'wb') as fp:
                pickle.dump(self.mem_benchmark, fp)
            fp.close()
            benchmark(self.args)
            return True
        else:
            return None

    def create_tensorboard(self):
        """ Setup tensorboard """
        if not (self.args.benchmark or self.args.display):
            self.writer = tf.compat.v1.summary.FileWriter(self.tboard_dir, U.get_session().graph)
            self.writer.flush()
            self.summary_ops, self.summary_vars = build_summaries(self.num_agents, self.args)

    def save_model_rew_disk(self, saver, time_taken):
        data = (self.train_step * self.num_env, self.train_step * self.num_env / self.args.max_episode_len, time_taken, self.exp_name, self.exp_itr, self.data_file, saver)
        self.train_thread[self.cpu_rew_thread].input_queue.put(("save_model_rew", 0, data))

    """
    -----------------------------------------------------------------------
    Training Steps and Calls in the Main Loop
    -----------------------------------------------------------------------
    """
    def get_loss(self):
        loss = [None] * self.num_agents
        if self.buffer_size * self.args.len_traj_update * self.num_env < self.args.batch_size:
            return
        else:
            # print(np.shape(np.asarray(self.value_n_buffer)))
            # print(np.shape(self.value_n_buffer[-1]))
            traj_data = self.return_buffer()
            self.reset_temp_traj_buffer()
            self.reset_main_buffer_batch()
            for p_index in range(self.num_agents):
                thread_idx = self.actor_critic_thread[p_index]
                self.train_thread[thread_idx].input_queue.put(("get_loss", p_index, (self.train_step, traj_data)))

            done_thread = [None] * self.num_agents
            while not all(done_thread):
                for p_index in range(self.num_agents):
                    thread_idx = self.actor_critic_thread[p_index]
                    try:
                        if loss[p_index] is None:
                            loss[p_index] = self.train_thread[thread_idx].output_queue.get()
                        else:
                            done_thread[p_index] = True
                    except self.train_thread[thread_idx].output_queue.empty:
                        if all(done_thread):
                            break
                        else:
                            continue

            # Write data to Tensor board
            if not (self.args.benchmark or self.args.display):
                tboard_data = (loss, self.train_step, self.writer, self.summary_ops, self.summary_vars, self.num_agents)
                self.train_thread[self.cpu_rew_thread].input_queue.put(("write_tboard", 0, tboard_data))
