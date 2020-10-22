import multiprocessing as mp
import numpy as np
import time

def mpe_pipe_worker(pipe, args, env_idx):
    if args.env_type == "mpe":
        from sarnet_td3.common.env_setup import make_mpe_env
        env = make_mpe_env(args)
        env.seed(args.random_seed + env_idx)
        np.random.seed(args.random_seed + env_idx)
    elif args.env_type == "ic3net":
        from sarnet_td3.common.ic3_env_setup import make_ic3_env
        env = make_ic3_env(args)
        env.seed(args.random_seed + env_idx)
        np.random.seed(args.random_seed + env_idx)
    else:
        assert "invalid environment name"
        return

    while True:
        action = pipe.recv()
        if action == "None":
            break

        elif 'reset' in action:
            _, epoch = action
            if epoch is None:
                pipe.send(env.reset())
            else:
                pipe.send(env.reset(epoch=epoch))

        elif action == 'get_obs_shape':
            if args.env_type == "mpe":
                obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            else:
                obs_shape_n = [env.observation_dim for _ in range(args.num_adversaries)]

            pipe.send(obs_shape_n)

        elif action == 'get_action_shape':
            if args.env_type == "mpe":
                pipe.send(env.action_space)
            else:
                pipe.send([env.num_actions for _ in range(args.num_adversaries)])

        elif action == 'get_num_agents':
            if args.env_type == "mpe":
                pipe.send(env.n)
            else:
                pipe.send(args.num_adversaries)

        elif action == 'render':
            time.sleep(0.1)
            env.render()
            continue

        else:
            obs, reward, done, info = env.step(action)
            pipe.send((obs, reward, done, info))


class MultiEnv(object):
    """Create multiple environments on CPU threads"""

    def __init__(self, args, num_envs):
        self.parent_pipes, self.child_pipes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.workers = []
        self.arglist = args

        for idx, child_pipe in enumerate(self.child_pipes):
            process = mp.Process(target=mpe_pipe_worker, args=(child_pipe, self.arglist, idx), daemon=True)
            self.workers.append(process)
            process.start()

        print("There are {} environment workers".format(len(self.workers)))

    def get_obs_shape(self):
        for pipe in self.parent_pipes:
            pipe.send('get_obs_shape')
            if self.arglist.same_env:
                return pipe.recv()

    def get_num_agents(self):
        for pipe in self.parent_pipes:
            pipe.send('get_num_agents')
            if self.arglist.same_env:
                return pipe.recv()

    def get_action_space(self):
        for pipe in self.parent_pipes:
            pipe.send('get_action_shape')
            if self.arglist.same_env:
                return pipe.recv()

    def stop_env(self):
        for pipe in self.parent_pipes:
            pipe.send('None')

    # Add neighbors here
    def reset(self, epoch=None):
        new_obs = []

        for pipe in self.parent_pipes:
            pipe.send(('reset', epoch))

        for pipe in self.parent_pipes:
            obs_a = pipe.recv()
            new_obs.append(obs_a)
        return new_obs

    def cancel(self):
        for pipe in self.parent_pipes:
            pipe.send("None")

        for worker in self.workers:
            worker.join()
        print('workers cancelled')

    def render(self):
        for pipe in self.parent_pipes:
            pipe.send("render")

    def step(self, actions):
        new_obs = []
        rewards = []
        dones = []
        infos = []

        for action, pipe in zip(actions, self.parent_pipes):
            pipe.send(action)

        for pipe in self.parent_pipes:
            obs, reward, done, info = pipe.recv()
            # [# Env, [# Agent, Dim]]
            new_obs.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return new_obs, rewards, dones, infos


