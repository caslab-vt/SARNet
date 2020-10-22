import os, pickle, time, queue
import numpy as np

from experiments.benchmark import benchmark
import sarnet_td3.common.tf_util as U

def write_bench(exp_name, exp_itr, args, agent_info):
    benchmark_dir = os.path.join('./exp_data', exp_name, exp_itr, args.benchmark_dir)
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)
    file_name = './exp_data/' + exp_name + '/' + exp_itr + '/' + args.benchmark_dir + '/' + exp_name + '.pkl'
    print('Finished benchmarking, now saving...')
    with open(file_name, 'wb') as fp:
        pickle.dump(agent_info[:-1], fp)
    benchmark(args)

    return

def write_runtime(data_file, num_steps, main_run_time):
    with open(data_file, "a+") as f:
        final_run_time = round(time.time() - main_run_time, 3)
        print('...Finished total of {} episodes.'.format(num_steps))
        f.write('...Finished total of {} episodes.'.format(num_steps) + '\n')
        f.write("Total Run {} Time.".format(final_run_time) + '\n')
    return