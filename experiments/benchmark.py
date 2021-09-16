import pickle
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import os

from experiments.config_args import parse_args


def simple_spread(arglist, bench_val):
    collision = []
    min_dist = []
    reward = []
    occ_land = []

    for env_bench in bench_val:
        for agnt_idx, agent_bench in enumerate(env_bench):
            for ep_bench in agent_bench:
                collision_ep = 0
                min_dist_ep = 0
                reward_ep = 0
                occ_land_ep = 0
                for stps_bench in ep_bench:
                    if agnt_idx < arglist.num_adversaries:
                        try:
                            collision_ep += int(stps_bench['collisions'][0])
                            reward_ep += stps_bench['rew'][0]
                            min_dist_ep += stps_bench['min_dists'][0]
                            occ_land_ep += int(stps_bench['occ_land'][0])
                        except:
                            continue
                collision.append(collision_ep)
                min_dist.append(min_dist_ep)
                reward.append(reward_ep)
                occ_land.append(occ_land_ep)

    reward_mn = np.mean(reward)
    reward_var = np.std(reward)
    collision_mn = np.mean(collision)
    collision_var = np.std(collision)
    min_dist_mn = np.mean(min_dist)
    min_dist_var = np.std(min_dist)
    occ_land_mn = np.mean(occ_land)
    occ_land_var = np.std(occ_land)

    print("Average Reward: ", reward_mn, " Std Reward: ", reward_var)
    print("Average Collisions: ", collision_mn, " Std Coll: ", collision_var)
    print("Average Dist: ", min_dist_mn, " Std Dist: ", min_dist_var)
    print("Average Occupied Landmarks: ", occ_land_mn, "Std Occ Land: ", occ_land_var)

    save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Reward: %f" % reward_mn + "\n\n")
        bench_file.write("Average Collisions: %f" % collision_mn + "\n\n")
        bench_file.write("Average Dist: %f" % min_dist_mn + "\n\n")
        bench_file.write("Average Occupied Landmarks: %d" % occ_land_mn + "\n\n")
        bench_file.write("Std Reward: %f" % reward_var + "\n\n")
        bench_file.write("Std Collisions: %f" % collision_var + "\n\n")
        bench_file.write("Std Dist: %f" % min_dist_var + "\n\n")
        bench_file.write("Std Occupied Landmarks: %d" % occ_land_var + "\n\n")
        # bench_file.write("Average Steps: %d" % num_steps)


def simple_tag(arglist, bench_val):
    collision = []

    for env_bench in bench_val:
        for agnt_idx, agent_bench in enumerate(env_bench):
            for ep_bench in agent_bench:
                collision_ep = 0
                for stps_bench in ep_bench:
                    # print(np.shape(stps_bench))
                    if agnt_idx < arglist.num_adversaries:
                        # print(stps_bench['collisions'][0])
                        try:
                            collision_ep += int(stps_bench['collisions'][0])
                        except:
                            continue
                collision.append(collision_ep)

    collision_mn = np.mean(collision)
    collision_var = np.std(collision)
    print("Average Collisions: ", collision_mn, " Std Collision: ", collision_var)
    save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Collisions: {}, Std Collisions: {}".format(collision_mn, collision_var ))


def simple_adv(arglist):
    exp_name = 'PD5'
    arch_name = 'IC3'
    bench_name = 'bench3'
    bench_file = os.path.join(dirname, 'exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt')
    bench_load = open(bench_file, 'rb')
    bench_val = pickle.load(bench_load)
    bench_load.close()
    arglist.scenario = "simple_adversary_4"
    adv_collision = 0
    good_collision = 0
    num_steps = arglist.benchmark_iters / arglist.max_episode_len
    agnt_adv_success = []
    agnt_gd_success = []
    total_ep = 0
    for env_bench in bench_val:
        for agnt_idx, agent_bench in enumerate(env_bench):
            success_gd = 0
            success_adv = 0
            for ep_bench in agent_bench:
                adv_succ = False
                gd_succ = False
                success_gd = 0
                success_adv = 0
                for stps_bench in ep_bench:
                    # print(np.shape(stps_bench))
                    if agnt_idx < 4:
                        # print(stps_bench['collisions'][0])
                        try:
                            adv_collision += int(stps_bench['collisions'][0])
                            if int(stps_bench['collisions'][0]) == 1:
                                adv_succ = True
                        except:
                            continue
                    else:
                        try:
                            good_collision += int(stps_bench['collisions'][0])
                            if int(stps_bench['collisions'][0]) == 1:
                                gd_succ = True
                        except:
                            continue
                    if adv_succ:
                        success_adv += 1
                    if gd_succ:
                        success_gd += 1

            agnt_adv_success.append(success_adv)
            agnt_gd_success.append(success_gd)

    success_gd_ep_mn = np.mean(agnt_gd_success)
    success_gd_ep_std = np.std(agnt_gd_success)
    success_adv_ep_mn = np.mean(agnt_adv_success)
    success_adv_ep_std = np.std(agnt_adv_success)
    print("Average Adv Success: {}, Std Adv Success: {}".format(success_adv_ep_mn, success_adv_ep_std))
    print("Average Good Success: {}, Std Good Success: {}".format(success_gd_ep_mn, success_gd_ep_std))
    save_info_dir = './exp_data/' + '/' + exp_name + '/' + arch_name + '/' + bench_name + '/' + arch_name + '-' + exp_name + '.txt'
    # save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Adv Success: {}, Std Adv Success: {}".format(success_adv_ep_mn, success_adv_ep_std))
        bench_file.write("Average Good Success: {}, Std Good Success: {}".format(success_gd_ep_mn, success_gd_ep_std))


def traffic_junction(arglist, bench_val):
    success_rate = 0
    mean_success_rate = np.mean(bench_val)
    std_success_rate = np.std(bench_val)
    success_rate_ep = np.concatenate(bench_val)
    success_rate_ep = np.count_nonzero(success_rate_ep == arglist.max_episode_len) / len(success_rate_ep)
    print("Average Success Rate: {}, Std. Success Rate: {}".format(mean_success_rate, std_success_rate))
    dirname = os.path.dirname(__file__)
    save_info_dir = os.path.join(dirname, 'exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt')
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Success Rate: {}, Std. Success Rate: {}, Flawless Episodes: {}".format(mean_success_rate, std_success_rate, success_rate_ep))


def benchmark(arglist):
    dirname = os.path.dirname(__file__)
    bench_file = os.path.join(dirname, 'exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.pkl')
    bench_load = open(bench_file, 'rb')
    bench_val = pickle.load(bench_load)
    bench_load.close()

    if arglist.scenario == "simple_spread_3" or arglist.scenario == "simple_spread_6" or arglist.scenario == "simple_spread_10" or arglist.scenario == "simple_spread_20":
        simple_spread(arglist, bench_val)

    if arglist.scenario == "simple_tag_3" or arglist.scenario == "simple_tag_6" or arglist.scenario == "simple_tag_12" or arglist.scenario == "simple_tag_15":
        simple_tag(arglist, bench_val)

    if arglist.scenario == "simple_adversary_4" or arglist.scenario == "simple_adversary_6":
        simple_adv(arglist)

    if arglist.scenario == "traffic_junction":
        traffic_junction(arglist, bench_val)


if __name__ == '__main__':
    arglist = parse_args()
    benchmark(arglist)