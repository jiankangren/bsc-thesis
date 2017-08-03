from lib import Task, TaskSet
import generation as gen
import analysis as ana
import numpy as np
import numpy.random as nprd
import math
import collections as col
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools as ft
from time import time
import pickle
import os


##############
# Parameters #
##############

utils = [0.2 + k * 0.05 for k in range(47)]
nsets = 2
data_folder_path = './data/'
task_sets_list_path = data_folder_path + 'task_sets_list'
os.makedirs(data_folder_path, exist_ok=True)
Mode = col.namedtuple('Mode', ('name', 'func', 'color', 'linestyle', 'marker'))
modes = [
    Mode('dSMC', ana.d_smc, 'blue', 'dashed', 'o'),
    Mode('dAMC', ana.d_amc, 'red', 'dashed', 's'),
    Mode('p_smc', ana.p_smc, 'green', 'dashed', '^'),
    Mode('p_amc_bb', ana.p_amc_bb, 'magenta', 'dashed', '*'),
    # Mode('p_smc_monte_carlo', 'cyan', 'dashed', 'v')
]


################
# Eval Scripts #
################

def gen_task_sets():
    start = time()
    task_sets_list = []
    for util in utils:
        m = math.ceil(util)
        util_adj = util / m
        # print(util)
        task_sets = []
        for i in range(nsets):
            ts = gen.mc_fairgen(set_id=i, u_lo=util_adj, m=m, implicit_deadlines=True)
            gen.synth_c_dist(ts)
            gen.set_priorities_dm(ts)
            task_sets.append(ts)
        task_sets_list.append(task_sets)
    # print(task_sets_list)
    pickle.dump(task_sets_list, open(task_sets_list_path, 'wb'))
    stop = time()
    print('Task Set Generation: %.3fs' % (stop - start))


def eval_scheme(mode):
    start = time()
    pool = mp.Pool()
    task_sets_list = pickle.load(open(task_sets_list_path, 'rb'))
    rates = []
    for task_sets in task_sets_list:
        # print(mode.name, task_sets[0].u_lo)
        rates.append(100 * np.average(pool.map(mode.func, task_sets)))
    # print(rates)
    pickle.dump(rates, open(data_folder_path + mode.name, 'wb'))
    stop = time()
    print('%s: %.3fs' % (mode.name, (stop - start)))


# def schedulability_rates(util):
#     m = math.ceil(util)
#     util /= m
#     print(util, m)
#     results = {mode.name: [] for mode in modes}
#     for i in range(nsets):
#         task_set = gen.mc_fairgen(i, u_lo=util, m=m, implicit_deadlines=True)
#         gen.synth_c_dist(task_set)
#         gen.set_priorities_rm(task_set)
#         results['d_smc'].append(ana.d_smc(task_set))
#         results['d_amc'].append(ana.d_amc(task_set))
#         results['p_smc'].append(ana.p_smc(task_set, thresh_lo=1e-5))
#         results['p_amc_bb'].append(ana.p_amc_bb(task_set, thresh_lo=1e-5))  # TODO
#         # results['p_smc_monte_carlo'].append(ana.p_smc_monte_carlo(task_set))
#     return {mode.name: np.average(results[mode.name]) * 100 for mode in modes}


def plot_schedulability_rates():
    task_sets_list = pickle.load(open(task_sets_list_path, 'rb'))
    rates = {}

    plt.figure(figsize=(16, 9), dpi=180)
    plt.xlabel('LO mode utilization')
    plt.ylabel('Percentage of task sets schedulable')
    for name, _, color, linestyle, marker in modes:
        rates[name] = pickle.load(open(data_folder_path + name, 'rb'))
        plt.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)

    plt.legend(loc='center left')
    plt.axhline(0.0, color='black', linestyle='--')
    plt.axhline(100.0, color='black', linestyle='--')
    plt.axvline(1.0, color='black', linestyle='--')
    # plt.savefig('out.png')
    plt.show()


# def plot_gen_util_error():
#     """Sanity check: Plots error of desired vs. generated LO task util of mc_fairgen_stoch."""
#     gran = 100  # Time granularity
#     max_errs, avg_errs, min_errs = [], [], []
#     pool = mp.Pool()
#     for util in utils:
#         print(util)
#         m = math.ceil(util)
#         # util_adj = util / m
#         util_adj = (util / m) + (1. / gran)
#         task_sets = pool.map(ft.partial(gen.mc_fairgen_stoch, u_lo=util_adj, m=m, time_granularity=gran), range(nsets))
#         errs = [task_set.u_lo - util for task_set in task_sets]
#         max_errs.append(max(errs))
#         avg_errs.append(np.average(errs))
#         min_errs.append(min(errs))
#     plt.figure(figsize=(20, 10), dpi=200)
#     plt.title("Deviation of generated LO-mode utilizations vs. input generation parameter u_lo. Time granularity:%d"
#               % gran)
#     plt.xlabel('Input parameter u_lo')
#     plt.ylabel('Desired u_lo minus actual LO-mode utilization')
#     plt.plot(utils, max_errs, 'r-v', label='Max')
#     plt.plot(utils, avg_errs, 'g-o', label='Avg')
#     plt.plot(utils, min_errs, 'b-^', label='Min')
#     plt.axhline(1./gran, color='black', linestyle='dotted')
#     plt.axhline(0., color='black', linestyle='dashed')
#     plt.axhline(-1. / gran, color='black', linestyle='dotted')
#     plt.legend(loc='upper left')
#     # plt.savefig('u_lo_deviation_adj.png')
#     plt.show()
#
#
# def plot_rta_for_mc_systems():
#     """Sanity check: Reproduce (part of) the plot seen in [1], p. 8."""
#
#     utils = [0.025 + k * 0.05 for k in range(20)]
#     modes = [
#         Mode('SMC', 'green', 'solid', 'o'),
#         Mode('AMC-rtb', 'blue', 'solid', '^'),
#     ]
#     ntasks = 20
#     nsets = 1000
#     cf = 2.0  # Fixed multiplier for high criticality execution time
#     cp = 0.5  # Probability for a task being HI-critical
#
#     rates = {mode.name: [] for mode in modes}
#     pool = mp.Pool()
#
#     for util in utils:
#         print(util)
#         results = {mode.name: [] for mode in modes}
#         task_sets = []
#         for i in range(nsets):
#             periods = 10 ** np.random.uniform(1, 3, ntasks)  # log-uniform distribution w/ factor 100 difference
#             criticalities = np.random.choice(a=['HI', 'LO'], size=ntasks, p=[cp, 1 - cp])
#             utils_lo = gen.uunifast(ntasks, util)
#             tasks = [Task(j, criticalities[j], periods[j], periods[j],
#                           utils_lo[j] * periods[j],
#                           utils_lo[j] * cf * periods[j] if criticalities[j] == 'HI' else None
#                           ) for j in range(ntasks)]
#             task_set = TaskSet(i, tasks, discrete=False)
#             task_set.set_priorities_rm()
#             task_sets.append(task_set)
#         results['SMC'] = pool.map(ana.d_smc, task_sets)
#         results['AMC-rtb'] = pool.map(ana.d_amc, task_sets)
#         for mode in modes:
#             rates[mode.name].append(np.average(results[mode.name]) * 100)
#
#     plt.figure(figsize=(20, 10), dpi=200)
#     plt.title("Deterministic schedulability, task sets generated by UUniFast.")
#     plt.xlabel('LO mode utilization')
#     plt.ylabel('Percentage of task sets schedulable')
#     for name, color, linestyle, marker in modes:
#         # rates[name] = [result[name] for result in results]
#         plt.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)
#
#     plt.legend(loc='center left')
#     plt.axhline(0.0, color='black', linestyle='--')
#     plt.axhline(100.0, color='black', linestyle='--')
#     plt.axvline(1.0, color='black', linestyle='--')
#     plt.show()


#################
# Main Function #
#################

def main():
    print('Time elapsed:')
    gen_task_sets()

    for mode in modes:
        eval_scheme(mode)

    plot_schedulability_rates()
if __name__ == '__main__':
    main()
    # plot_schedulability_rates()
    # plot_gen_util_error()
    # plot_rta_for_mc_systems()

"""
[1] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
"""

