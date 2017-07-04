from class_lib import Task, TaskSet
import generation as gen
import analysis as ana
import numpy as np
import numpy.random as nprd
import math
import collections as col
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools as ft
import time

utils = [0.2 + k * 0.05 for k in range(37)] #+ [1.0 + k * 0.05 for k in range(11)]
Mode = col.namedtuple('Mode', ('name', 'color', 'linestyle', 'marker'))
modes = [
    Mode('d_smc', 'blue', 'dashed', 'o'),
    Mode('d_amc', 'red', 'dashed', 's'),
    Mode('p_smc', 'green', 'dashed', '^'),
    Mode('p_amc_bb', 'magenta', 'dashed', '*'),
    Mode('p_smc_monte_carlo', 'cyan', 'dashed', 'v')
]
nsets = 10


def schedulability_rates(util):
    m = math.ceil(util)
    util /= m
    print(util, m)
    results = {mode.name: [] for mode in modes}
    for i in range(nsets):
        task_set = gen.mc_fairgen_stoch(i, u_lo=util, m=m, implicit_deadlines=True)
        task_set.set_priorities_rm()
        results['d_smc'].append(ana.d_smc(task_set))
        results['d_amc'].append(ana.d_amc_rtb(task_set))
        results['p_smc'].append(ana.p_smc(task_set))
        results['p_amc_bb'].append(ana.p_amc_black_box(task_set))
        results['p_smc_monte_carlo'].append(ana.p_smc_monte_carlo(task_set))
    return {mode.name: np.average(results[mode.name]) * 100 for mode in modes}


def plot_schedulability_rates():

    rates = {mode.name: [] for mode in modes}

    with mp.Pool() as pool:

        start = time.time()
        results = pool.map(schedulability_rates, utils)
        stop = time.time()
        print("Time elapsed:", stop - start)

    plt.figure(figsize=(20, 10), dpi=200)
    plt.xlabel('LO mode utilization')
    plt.ylabel('Percentage of task sets schedulable')
    for name, color, linestyle, marker in modes:
        rates[name] = [result[name] for result in results]
        plt.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)

    plt.legend(loc='center left')
    plt.axhline(0.0, color='black', linestyle='--')
    plt.axhline(100.0, color='black', linestyle='--')
    plt.axvline(1.0, color='black', linestyle='--')
    # plt.savefig('out.png')
    plt.show()


def plot_gen_util_error():
    """Sanity check: Plots error of desired vs. generated LO task util of mc_fairgen_stoch."""
    gran = 100  # Time granularity
    max_errs, avg_errs, min_errs = [], [], []
    pool = mp.Pool()
    for util in utils:
        print(util)
        m = math.ceil(util)
        # util_adj = util / m
        util_adj = (util / m) + (1. / gran)
        task_sets = pool.map(ft.partial(gen.mc_fairgen_stoch, u_lo=util_adj, m=m, time_granularity=gran), range(nsets))
        errs = [task_set.u_lo - util for task_set in task_sets]
        max_errs.append(max(errs))
        avg_errs.append(np.average(errs))
        min_errs.append(min(errs))
    plt.figure(figsize=(20, 10), dpi=200)
    plt.title("Deviation of generated LO-mode utilizations vs. input generation parameter u_lo. Time granularity:%d"
              % gran)
    plt.xlabel('Input parameter u_lo')
    plt.ylabel('Desired u_lo minus actual LO-mode utilization')
    plt.plot(utils, max_errs, 'r-v', label='Max')
    plt.plot(utils, avg_errs, 'g-o', label='Avg')
    plt.plot(utils, min_errs, 'b-^', label='Min')
    plt.axhline(1./gran, color='black', linestyle='dotted')
    plt.axhline(0., color='black', linestyle='dashed')
    plt.axhline(-1. / gran, color='black', linestyle='dotted')
    plt.legend(loc='upper left')
    # plt.savefig('u_lo_deviation_adj.png')
    plt.show()


def plot_rta_for_mc_systems():
    """Sanity check: Reproduce (part of) the plot seen in [1], p. 8."""

    utils = [0.025 + k * 0.05 for k in range(20)]
    modes = [
        Mode('SMC', 'green', 'solid', 'o'),
        Mode('AMC-rtb', 'blue', 'solid', '^'),
    ]
    ntasks = 20
    nsets = 1000
    cf = 2.0  # Fixed multiplier for high criticality execution time
    cp = 0.5  # Probability for a task being HI-critical

    rates = {mode.name: [] for mode in modes}
    pool = mp.Pool()

    for util in utils:
        print(util)
        results = {mode.name: [] for mode in modes}
        task_sets = []
        for i in range(nsets):
            periods = 10 ** np.random.uniform(1, 3, ntasks)  # log-uniform distribution w/ factor 100 difference
            criticalities = np.random.choice(a=['HI', 'LO'], size=ntasks, p=[cp, 1 - cp])
            utils_lo = gen.uunifast(ntasks, util)
            tasks = [Task(j, criticalities[j], periods[j], periods[j],
                          utils_lo[j] * periods[j],
                          utils_lo[j] * cf * periods[j] if criticalities[j] == 'HI' else None
                          ) for j in range(ntasks)]
            task_set = TaskSet(i, tasks, discrete=False)
            task_set.set_priorities_rm()
            task_sets.append(task_set)
        results['SMC'] = pool.map(ana.d_smc, task_sets)
        results['AMC-rtb'] = pool.map(ana.d_amc_rtb, task_sets)
        for mode in modes:
            rates[mode.name].append(np.average(results[mode.name]) * 100)

    plt.figure(figsize=(20, 10), dpi=200)
    plt.title("Deterministic schedulability, task sets generated by UUniFast.")
    plt.xlabel('LO mode utilization')
    plt.ylabel('Percentage of task sets schedulable')
    for name, color, linestyle, marker in modes:
        # rates[name] = [result[name] for result in results]
        plt.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)

    plt.legend(loc='center left')
    plt.axhline(0.0, color='black', linestyle='--')
    plt.axhline(100.0, color='black', linestyle='--')
    plt.axvline(1.0, color='black', linestyle='--')
    plt.show()

if __name__ == '__main__':
    plot_schedulability_rates()
    # plot_gen_util_error()
    # plot_rta_for_mc_systems()


"""
[1] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
"""

