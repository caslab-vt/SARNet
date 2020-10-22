import gym
import argparse
import sys
import ic3net_envs

from sarnet_td3.common.env_multiproc import *


def create_env(args):
    num_env = args.num_env
    if args.display:
        num_env = 1

    # Initialize the environments, generally to the number of threads available
    train_envs = MultiEnv(args, num_env)  # Create multiple instances of environment
    obs_shape_n = train_envs.get_obs_shape()  # Get observation shape of the environment
    action_space = train_envs.get_action_space()  # action space of the environment
    num_agents = train_envs.get_num_agents()  # Get number of agents in the environment
    num_adversaries = min(num_agents, args.num_adversaries)
    print("Scenario: ", args.scenario)
    print("# Agents: ", num_agents, " Num_adv: ", num_adversaries)
    print("Experiment Name {}".format(args.exp_name))
    print('Using good policy {} and adv policy {}'.format(args.good_test, args.adv_test))
    print('Random Seed {}'.format(args.random_seed))

    return train_envs, num_env, num_agents, num_adversaries, obs_shape_n, action_space


def make_mpe_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(args.scenario + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    if args.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env
