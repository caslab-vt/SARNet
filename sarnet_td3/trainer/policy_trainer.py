import sarnet_td3.common.tf_util as U
import tensorflow as tf

from sarnet_td3.models.critic_model import mlp_model, rnn_model, maac_rnn_model
from sarnet_td3.models.comm_policy import CommActorNetwork
from sarnet_td3.models.comm_policy_td3 import CommActorNetworkTD3
from sarnet_td3.models.comm_policy_vpg import CommActorNetworkVPG

from sarnet_td3.trainer.comm_trainer_td3 import CommAgentTrainerTD3, create_placeholder_td3
from sarnet_td3.trainer.comm_trainer_reinforce import CommAgentTrainerVPG, create_placeholder_vpg


def get_trainers_td3(num_agents, obs_shape_n, action_space, args, num_env, is_train=False):
    obs_ph_n, gru1_ph_n, gru2_ph_n, memory_ph_n, act_ph_n, action_space_n, target_ph, q_gru_ph_n, importance_in_ph = create_placeholder_td3(obs_shape_n, action_space, num_agents, args)
    model = mlp_model
    if args.recurrent and args.adv_critic_model == "MAAC":
        model = maac_rnn_model
    elif args.adv_critic_model == "GRU":
        model = rnn_model
    trainers = []
    with tf.compat.v1.variable_scope(args.exp_name + "/" + args.adv_test + "_ADV", reuse=False):
        for i in range(args.num_adversaries):
            trainers.append(CommAgentTrainerTD3("adv_agent", CommActorNetworkTD3, model, obs_ph_n, gru1_ph_n, gru2_ph_n, memory_ph_n, act_ph_n, action_space_n, target_ph, q_gru_ph_n, importance_in_ph, args, i,
                                                num_env, is_train))

    model = mlp_model
    if args.recurrent and args.gd_critic_model == "MAAC":
        model = maac_rnn_model
    elif args.gd_critic_model == "GRU":
        model = rnn_model

    with tf.compat.v1.variable_scope(args.exp_name + "/" + args.good_test + "_GD", reuse=False):
        for i in range(args.num_adversaries, num_agents):
            trainers.append(CommAgentTrainerTD3("good_agent", CommActorNetworkTD3, model, obs_ph_n, gru1_ph_n, gru2_ph_n, memory_ph_n, act_ph_n, action_space_n, target_ph, q_gru_ph_n, importance_in_ph, args, i,
                                                num_env, is_train))

    return trainers

def get_trainers_vpg(num_agents, obs_shape_n, action_space, args, num_env, is_train=False):
    obs_ph_n, h_ph_n, c_ph_n, memory_ph_n, act_ph_n, action_space_n, return_in_ph = create_placeholder_vpg(obs_shape_n, action_space, num_agents, args)
    trainers = []
    with tf.compat.v1.variable_scope(args.exp_name + "/" + args.adv_test + "_ADV", reuse=False):
        for i in range(args.num_adversaries):
            trainers.append(CommAgentTrainerVPG("adv_agent", CommActorNetworkVPG, obs_ph_n, h_ph_n, c_ph_n, memory_ph_n,
                                                act_ph_n, action_space_n, return_in_ph, args, i, num_env, is_train))
    return trainers

def load_model(num_agents, obs_shape_n, action_space, arglist, num_env, is_train):
    sess = U.inter_gpu_session(arglist)
    if arglist.policy_grad == "reinforce":
        return get_trainers_vpg(num_agents, obs_shape_n, action_space, arglist, num_env, is_train), sess
    elif arglist.policy_grad == "maddpg":
        return get_trainers_td3(num_agents, obs_shape_n, action_space, arglist, num_env, is_train), sess
    else:
        assert "Incorrect Policy Gradient Model Chosen"
