import numpy as np
import os
import json

"""
Output: [traj, [agent, batch, dim]
"""
def reshape_data(data_ep):
    ep_obs_n, ep_gru1_n, ep_gru2_n, ep_memory_n, ep_q1_gru_n, ep_q2_gru_n, ep_action_n, ep_rew_n, ep_new_obs_n,\
        ep_new_gru1_n, ep_new_gru2_n, ep_new_memory_n, ep_new_q1_gru_n, ep_new_q2_gru_n, ep_done_n = \
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for ep_idx in range(len(data_ep)):
        obs_n, gru1_n, gru2_n, memory_n, q1_gru_n, q2_gru_n, action_n, rew_n, new_obs_n, new_gru1_n, new_gru2_n, new_memory_n, new_q1_gru_n, new_q2_gru_n, done_n = data_ep[ep_idx]
        ep_obs_n.append(obs_n)
        ep_gru1_n.append(gru1_n)
        ep_gru2_n.append(gru2_n)
        ep_memory_n.append(memory_n)
        ep_q1_gru_n.append(q1_gru_n)
        ep_q2_gru_n.append(q2_gru_n)
        ep_action_n.append(action_n)
        ep_rew_n.append(rew_n)
        ep_new_obs_n.append(new_obs_n)
        ep_new_gru1_n.append(new_gru1_n)
        ep_new_gru2_n.append(new_gru2_n)
        ep_new_memory_n.append(new_memory_n)
        ep_new_q1_gru_n.append(new_q1_gru_n)
        ep_new_q2_gru_n.append(new_q2_gru_n)
        # (traj, agent, ) for mpe and (#traj, ) for ic3
        ep_done_n.append(done_n)

    return (ep_obs_n, ep_gru1_n, ep_gru2_n, ep_memory_n, ep_q1_gru_n, ep_q2_gru_n, ep_action_n, ep_rew_n, ep_new_obs_n, ep_new_gru1_n, ep_new_gru2_n, ep_new_memory_n, ep_new_q1_gru_n, ep_new_q2_gru_n, ep_done_n)


# Input: [agents, traj, batch, dim], or [traj, [agent, batch, dim] if traj_major=True
# Output: [agents, [traj x batch, dim]]
def merge_ep_data(a_ep, traj_major=False):
    if traj_major:
        # Convert to [agents, traj, batch, dim]
        a_ep = pack_ep_agent(a_ep)

    agents, _, _, dim = np.shape(np.array(a_ep))
    ep_data = []
    for i in range(agents):
        ep_data.append(np.reshape(a_ep[i], (-1, dim)))

    return ep_data


def pack_ep_agent(a_ep):
    # [traj, [agent, batch, dim]] -> [agent, traj, batch, dim]
    list_a = np.stack(a_ep, axis=1)

    return list_a


""" 
Directory Generation
"""

def create_dir(args):
    # Create exp type directory
    exp_name = args.exp_name
    exp_dir = os.path.join('./exp_data', exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # Create
    if not (args.benchmark or args.display or args.restore):
        exp_itr = None
        for file in sorted(os.listdir(exp_dir)):
            exp_itr = file
        if exp_itr and not args.restore:
            exp_itr = str(int(exp_itr) + 1)
        else:
            exp_itr = '0'

        os.mkdir(os.path.join('./exp_data/' + exp_name + '/' + exp_itr))
    else:
        exp_itr = args.exp_itr

    tensorboard_dir = None

    data_file = os.path.join('./exp_data', exp_name, exp_itr, "values.txt")

    if not (args.benchmark or args.display or args.restore):
        # Save configuration files
        with open('./exp_data/' + exp_name + '/' + exp_itr + '/args.txt', 'w') as fp:
            json.dump(args.__dict__, fp, indent=2)

        # Create experiment folder
        tensorboard_dir = os.path.join('./exp_data', exp_name, exp_itr, 'tensorboard')
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)

    return exp_name, exp_itr, tensorboard_dir, data_file

