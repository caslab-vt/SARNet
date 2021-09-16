import argparse
import tensorflow as tf


def build_summaries(num_agents, args):
    # Define parameters to be logged
    if args.write_tb_loss:
        exp_performance = [tf.Variable(0.) for _ in range(int(num_agents * 5) + 1)]
        for i in range(num_agents):
            tf.summary.scalar("Reward" + str(i), exp_performance[1 + int(i * 5)])
            tf.summary.scalar("P-loss" + str(i), exp_performance[2 + int(i * 5)])
            tf.summary.scalar("TargetQ" + str(i), exp_performance[3 + int(i * 5)])
            tf.summary.scalar("Rew_update" + str(i), exp_performance[4 + int(i * 5)])
            tf.summary.scalar("TargetQNxt" + str(i), exp_performance[5 + int(i * 5)])
    else:
        exp_performance = [tf.Variable(0.) for _ in range(1)]
        tf.compat.v1.summary.scalar("Reward", exp_performance[0])

    summary_vars = exp_performance
    summary_ops = tf.compat.v1.summary.merge_all()
    return summary_ops, summary_vars


def parse_args():
    parser = argparse.ArgumentParser("SARNet")
    # Environment
    parser.add_argument("--env-type", type=str, default="mpe", choices=["mpe", "ic3net"], help="name of the environment script")
    parser.add_argument("--scenario", type=str, default="simple_spread_10", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=24000, help="number of episodes")
    parser.add_argument("--num-total-frames", type=int, default=5e6, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=10, help="number of adversaries/agents")
    parser.add_argument("--policy-grad", type=str, default="maddpg", choices=["maddpg", "reinforce"], help="policy for good agents")
    parser.add_argument("--random-seed", type=int, default=123, help="Random Seed")
    parser.add_argument("--buffer-size", type=int, default=1e5, help="Replay buffer size")
    parser.add_argument("--PER-sampling", action="store_true", default=False, help="use Prioritized Experience Replay")
    parser.add_argument("--alpha", type=int, default=0.8, help="How much prioritization is used")

    # Test Parameters
    parser.add_argument("--adv-test", type=str, default="SARNET", choices=["SARNET", "TARMAC", "COMMNET", "IC3NET", "DDPG"], help="Adversarial Agent Type")
    parser.add_argument("--good-test", type=str, default="COMMNET", choices=["SARNET", "TARMAC", "COMMNET", "IC3NET", "DDPG"], help="Good Agent Type")
    parser.add_argument("--encoder-model", default="LSTM", choices=["GRU", "LSTM", "MLP"], type=str, help="Type of actor encoder to use")
    parser.add_argument("--adv-critic-model", default="GRU", choices=["MLP", "GRU", "MAAC"], type=str, help="Type of critic to use")
    parser.add_argument("--gd-critic-model", default="GRU", choices=["MLP", "GRU", "MAAC"], type=str, help="Type of critic to use")
    parser.add_argument("--td3", action="store_true", default=True, help="use TD3 for updates, else DDPG")
    parser.add_argument("--upd-seq-agent", action="store_true", default=False, help="Updates agents at the same time-step if False")

    # Sampling for update parameters
    parser.add_argument("--num-env", type=int, default=200, help="Number of environments/threads in CPU")
    parser.add_argument("--num-gpu-threads", type=int, default=7, help="Number of environments/threads in CPU")
    parser.add_argument("--timeout", type=int, default=1, help="Thread Timeout")
    parser.add_argument("--update-lag", type=int, default=10000, help="number of steps to complete before starting update")
    parser.add_argument("--num-updates", type=int, default=1, help="Updates per 100 episodes")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size per environment instance to optimize at the same time")
    parser.add_argument("--zero-state-update", action="store_true", default=False, help="Use zero state for updates")
    parser.add_argument("--len-traj-update", type=int, default=10, help="Length of trajectory to optimize")

    #Device Parameters
    parser.add_argument("--same-env", action="store_true", default=True)
    parser.add_argument("--gpu-device", type=str, default="0", help="Which GPU to select")

    # Core training parameters
    parser.add_argument("--hidden-state-Q", action="store_true", default=False, help="use hidden states for critic updates")
    parser.add_argument("--zero-state-init", action="store_true", default=True, help="Use zeros for initializing the hidden states")
    parser.add_argument("--optimizer", default="ADAM", choices=["RMSProp", "ADAM"], type=str, help="Type of optimizer to use")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="learning rate for Actor Adam optimizer")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for  Critic  Adam optimizer")
    parser.add_argument("--polyak", type=float, default=5e-2, help="cumulative polyak for all similar agents")
    parser.add_argument("--policy-reg", action="store_true", default=True, help="Regularize the p_loss")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount factor")

    # Network parameters
    parser.add_argument("--pre-encoder", action="store_true", default=True, help="encode obs  with MLP before parsing through gru")
    parser.add_argument("--recurrent", action="store_true", default=True, help="use GRU instead of MLP for observation encoding")
    parser.add_argument("--gru-units", type=int, default=128, help="number of units in the rnn")
    parser.add_argument("--encoder-units", type=int, default=128, help="number of units for encoding before gru")
    parser.add_argument("--action-units", type=int, default=128, help="number of units for action selection")
    parser.add_argument("--critic-units", type=int, default=256, help="number of critic units")

    # Communication parameters
    parser.add_argument("--query-units", type=int, default=32, help="number of control units for reasoning")
    parser.add_argument("--key-units", type=int, default=32, help="number of key units for reasoning")
    parser.add_argument("--value-units", type=int, default=32, help="number of value/message units for reasoning")
    parser.add_argument("--nheads", type=int, default=1, help="number of value/message units for reasoning")
    parser.add_argument("--QKV-act", action="store_true", default=False, help="ReLU activation of QKV projections, else linear projection")
    # ablation
    parser.add_argument("--sar-attn", action="store_true", default=False, help="ReLU activation of QKV projections, else linear projection")
    parser.add_argument("--tar-attn", action="store_true", default=False, help="ReLU activation of QKV projections, else linear projection")

    # SARNET parameters
    parser.add_argument("--bNorm-state", action="store_true", default=False, help="batch norm state for SARNET specifically")
    parser.add_argument("--FeedOldMemory", action="store_true", default=False, help="compare old memory to generate new memory")
    parser.add_argument("--FeedOldMemoryToObsEnc", action="store_true", default=False, help="use mem from previous time step for encoding")
    parser.add_argument("--FeedMsgToValueProj", action="store_true", default=False, help="use mem from previous time step for encoding")
    parser.add_argument("--FeedInteractions", action="store_true", default=True, help="Perform a linear projection after computing (mem * val) + val")
    parser.add_argument("--TwoLayerEncodeSarnet", action="store_true", default=True, help="Use a GRU for action proj")
    parser.add_argument("--SARplusIC3", action="store_true", default=False, help="Use IC3Net as part of SARNET")

    # MAAC parameters


    # TARMAC parameters
    parser.add_argument("--TwoLayerEncodeTarmac", action="store_true", default=False, help="Perform a linear projection after computing (mem * val) + val")
    parser.add_argument("--TARplusIC3", action="store_true", default=False, help="Use IC3Net as part of TARMAC")

    # Batch Normalization parameters
    parser.add_argument("--memoryBN", action="store_true", default=False, help="use batch normalization on the recurrent memory")
    parser.add_argument("--stemBN", action="store_true", help="use batch normalization in the output unit (stem)")
    parser.add_argument("--outputBN", action="store_true", help="use batch normalization in the output unit")
    parser.add_argument("--bnDecay", default=0.999, type=float, help="batch norm decay rate")
    parser.add_argument("--bnCenter", action="store_true", help="batch norm with centering")
    parser.add_argument("--bnScale", action="store_true", help="batch norm with scaling")

    """"----------Dropouts----------"""
    parser.add_argument("--memory_dropout",  default=0.85, type=float,    help="dropout on the recurrent memory")
    parser.add_argument("--read_dropout",    default=0.85, type=float,    help="dropout of the read unit")
    parser.add_argument("--write_dropout",   default=1.0, type=float,    help="dropout of the write unit")
    parser.add_argument("--output_dropout",  default=0.85, type=float,   help="dropout of the output unit")

    # nonlinearities
    parser.add_argument("--relu", default="STD", choices=["STD", "PRM", "ELU", "LKY", "SELU"], type=str, help="type of ReLU to use: standard, parametric, ELU, or leaky")
    # parser.add_argument("--reluAlpha",  default = 0.2, type = float,    help = "alpha value for the leaky ReLU")

    parser.add_argument("--mulBias", default=0.0, type=float, help="bias to add in multiplications (x + b) * (y + b) for better training")  #

    """Call Parameters"""
    parser.add_argument("---queryInputAct", type=str, default="NON", choices=["NON", "RELU", "TANH"], help="Activation function for query in to the call")

    """Action Calculation"""
    parser.add_argument("---feedMemObsAction", action="store_true", default=True, help="Feed memory and observation for action projection")

    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="SAR-TJ6-AddRate", help="name of the experiment")
    parser.add_argument("--exp-itr", type=str, default="0", help="name of the experiment")
    parser.add_argument("--policy-file", type=str, default="642600", help="name of policy itr to use for benchmark")
    parser.add_argument("--save-dir", type=str, default="/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--write-tb-loss", action="store_true", default=False)

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=80000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="bench-std", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves", help="directory where plot data is saved")
    # 80000
    """IC3Net Parameters"""
    from sarnet_td3.common.ic3_env_setup import ic3_parser_args
    ic3_parser_args(parser)

    return parser.parse_args()
