import pickle
import numpy as np
import matplotlib as mpl
mpl.use('pdf')

from experiments.config_args import parse_args

def simple_spread(arglist, bench_val):
    collision = []
    min_dist = []
    reward = []
    occ_land = []

    num_steps = arglist.benchmark_iters / arglist.max_episode_len
    # for ep in range(len(bench_val)):  # number of episodes in benchmark
    #     for steps in range(len(bench_val[ep][0])):  # number of steps in episode
    #         for agent in range(len(bench_val[ep][0][steps])):  # number of agents in step
    #             reward += bench_val[ep][0][steps][agent][0]
    #             collision += bench_val[ep][0][steps][agent][1]
    #             min_dist += bench_val[ep][0][steps][agent][2]
    #             num_steps += 1

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
                            reward_ep += int(stps_bench['rew'][0])
                            min_dist_ep += int(stps_bench['min_dists'][0])
                            occ_land_ep += int(stps_bench['occ_land'][0])
                        except:
                            continue
                    collision.append(np.mean(collision_ep))
                    min_dist.append(np.mean(min_dist_ep))
                    reward.append(np.mean(reward_ep))
                    occ_land.append(np.mean(occ_land_ep))
    #
    # Average
    # Reward: -2012.0
    # Average
    # Collisions: 768.5
    # Average
    # Dist: 1243.5
    # Average
    # Occupied
    # Landmarks: 198.0

    reward_mn = np.mean(reward)
    reward_var = np.std(reward)
    collision_mn = np.mean(collision)
    collision_var = np.std(collision)
    min_dist_mn = np.mean(min_dist)
    min_dist_var = np.std(min_dist)
    occ_land_mn = np.mean(occ_land)
    occ_land_var = np.std(occ_land)

    # reward = reward / num_steps
    # collision = collision / num_steps
    # min_dist = min_dist / num_steps
    # occ_land = occ_land / num_steps

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
    collision = 0
    num_steps = arglist.benchmark_iters / arglist.max_episode_len
    # Metrics
    # Number of Collisions
    # print("Length of bench:", len(bench_val))
    # print(np.shape(bench_val))
    # print(np.shape(np.asarray(bench_val)))
    # print(np.shape(bench_val[0]))
    # print(np.shape(bench_val[0][0]))
    for env_bench in bench_val:
        for agnt_idx, agent_bench in enumerate(env_bench):
            for ep_bench in agent_bench:
                for stps_bench in ep_bench:
                    # print(np.shape(stps_bench))
                    if agnt_idx < arglist.num_adversaries:
                        # print(stps_bench['collisions'][0])
                        try:
                            collision += int(stps_bench['collisions'][0])
                        except:
                            continue

    # # for ep in range(len(bench_val)):  # number of episodes in benchmark
    # #     for agent in range(len(bench_val[0])):
    # #         for
    # for ep in range(len(bench_val)):  # number of episodes in benchmark
    #     for steps in range(len(bench_val[ep][0])):  # number of steps in episode
    #         for agent in range(len(bench_val[ep][0][steps])):  # number of agents in step
    #             collision += bench_val[ep][0][steps][agent]
    #     num_steps += 1
    #
    collision = collision / num_steps
    print("Average Collisions: ", collision)
    save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Collisions: %f" % collision + "\n\n")


def simple_adv(arglist):
    exp_name = 'PD5'
    arch_name = 'IC3'
    bench_name = 'bench3'
    bench_file = './exp_data/' + exp_name + '/' + arch_name + '/' + bench_name + '/' + arch_name + '-' + exp_name + '.pkl'
    bench_load = open(bench_file, 'rb')
    bench_val = pickle.load(bench_load)
    bench_load.close()
    arglist.scenario = "simple_adversary_4"
    adv_collision = 0
    good_collision = 0
    num_steps = arglist.benchmark_iters / arglist.max_episode_len
    # Metrics
    # Number of Collisions
    # print("Length of bench:", len(bench_val))
    # print(np.shape(bench_val))
    # print(np.shape(np.asarray(bench_val)))
    # print(np.shape(bench_val[0]))
    # print(np.shape(bench_val[0][0]))
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
                total_ep += 1
            agnt_adv_success.append(success_adv)
            agnt_gd_success.append(success_gd)
    # for ep in range(len(bench_val)):  # number of episodes in benchmark
    #     for steps in range(len(bench_val[ep][0])):  # number of steps in episode
    #         for agent in range(len(bench_val[ep][0][steps])):  # number of agents in step
    #             if agent < arglist.num_adversaries:
    #                 adv_collision += bench_val[ep][0][steps][agent]
    #             else:
    #                 good_collision += bench_val[ep][0][steps][agent]
    #             num_steps += 1



    success_gd_ep = np.mean(agnt_gd_success)
    success_adv_ep = np.mean(agnt_adv_success)
    good_collision = good_collision / num_steps
    adv_collision = adv_collision / num_steps
    print("Average Adv Collisions: ", adv_collision)
    print("Average Good Collisions: ", good_collision)
    print("Average Adv Success: ", success_adv_ep)
    print("Average Good Success: ", success_gd_ep)
    save_info_dir = './exp_data/' + '/' + exp_name + '/' + arch_name + '/' + bench_name + '/' + arch_name + '-' + exp_name + '.txt'
    # save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Adv Collisions: %f" % adv_collision + "\n\n")
        bench_file.write("Average Good Collisions: %f" % good_collision + "\n\n")
        bench_file.write("Average Adv Success: %f" % success_adv_ep + "\n\n")
        bench_file.write("Average Good Success: %f" % success_gd_ep + "\n\n")


def traffic_junction(arglist, bench_val):
    success_rate = 0
    # This is just the episode success rate, given as a single value
    # for each episode, divide it by the episode len, and you get the %
    # bench_val = bench_val / arglist.max_episode_len
    # print(np.shape(np.asarray(bench_val)))
    # for env_bench in bench_val:
    #     for agnt_idx, agent_bench in enumerate(env_bench):
    #         for ep_bench in agent_bench:
    #             for stps_bench in ep_bench:
    #                 # print(np.shape(stps_bench))
    #                 if agnt_idx < arglist.num_adversaries:
    #                     # print(stps_bench['collisions'][0])
    #                     try:
    #                         print(np.shape(stps_bench))
    #                         success_rate += int(stps_bench[0])
    #                     except:
    #                         continue
    mean_success_rate = np.mean(bench_val)
    print("Average Success Rate: ", mean_success_rate)
    save_info_dir = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.txt'
    with open(save_info_dir, 'w') as bench_file:
        bench_file.write("Model Name: %s" % arglist + "\n\n")
        bench_file.write("Average Success Rate: %f" % mean_success_rate + "\n\n")


def benchmark(arglist):
    bench_file = './exp_data/' + arglist.exp_name + '/' + arglist.exp_itr + '/' + arglist.benchmark_dir + '/' + arglist.exp_name + '.pkl'
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