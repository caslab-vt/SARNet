import gym
import ic3net_envs
from experiments.config_args import parse_args
from sarnet_td3.common.env_wrapper import GymWrapper
import argparse
import sys

def init_args_for_env(parser):
    env_dict = {
        'levers': 'Levers-v0',
        'number_pairs': 'NumberPairs-v0',
        'predator_prey': 'PredatorPrey-v0',
        'traffic_junction': 'TrafficJunction-v0',
        'starcraft': 'StarCraftWrapper-v0'
    }

    args = sys.argv
    # args_env = parser.parse_args()
    # env_name = args_env.scenario

    env_name = None
    for index, item in enumerate(args):
        if item == '--scenario':
            env_name = args[index + 1]

    if not env_name or env_name not in env_dict:
        return
    import gym
    import ic3net_envs
    if env_name == 'starcraft':
        import gym_starcraft
    env = gym.make(env_dict[env_name])
    env.init_args(parser)

def init(args, final_init=True):
    if args.scenario == 'levers':
        env = gym.make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif args.scenario == 'number_pairs':
        env = gym.make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif args.scenario == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif args.scenario == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif args.scenario == 'starcraft':
        env = gym.make('StarCraftWrapper-v0')
        env.multi_agent_init(args, final_init)
        env = GymWrapper(env.env)

    else:
        raise RuntimeError("wrong env name")

    return env

def ic3_parser_args(main_args):
    ic3_parser = main_args.add_argument_group('IC3 Env')
    # environment
    ic3_parser.add_argument('--nactions', default='1', type=str,
                            help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')

    ic3_parser.add_argument('--random', action='store_true', default=False,
                            help="enable random model")
    init_args_for_env(main_args)

    return ic3_parser

def make_ic3_env(main_args):
    env = init(main_args, False)

    return env