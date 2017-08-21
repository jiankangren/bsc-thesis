import lib
import synthesis as synth
import analysis as ana
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools as ft
from time import time
import pickle
import os
import matplotlib.ticker as ticker


##############
# Parameters #
##############

utils = [0.2 + k * 0.05 for k in range(37)]

task_sets_path = './data/task_sets/'
os.makedirs(task_sets_path, exist_ok=True)

eval_simplegen_path = './data/eval_simplegen/'
os.makedirs(eval_simplegen_path, exist_ok=True)
eval_fairgen_path = './data/eval_fairgen/'
os.makedirs(eval_fairgen_path, exist_ok=True)

eval_monte_carlo_path = './data/eval_monte_carlo/'
os.makedirs(eval_monte_carlo_path, exist_ok=True)

os.makedirs('./figures/', exist_ok=True)


###########
# Scripts #
###########

# Task Set Synthesis
####################

def gen_task_sets_simplegen(nsets):
    """Generate nsets task sets with SimpleGen and save them to disk."""
    start = time()
    eval_task_sets = []
    for util in utils:
        task_sets = []
        for i in range(nsets):
            ts = synth.simplegen(i, util, implicit_deadlines=True)
            synth.synth_c_pmf(ts, distribution_cls=lib.ExpExceedDist)
            synth.set_fixed_priorities(ts)
            task_sets.append(ts)
        eval_task_sets.append(task_sets)
    stop = time()
    print('Task Set Generation SimpleGen: %.3fs' % (stop - start))
    pickle.dump(eval_task_sets, open(task_sets_path + 'task_sets_simplegen', 'wb'))


def gen_task_sets_fairgen(nsets):
    """Generate nsets task sets with MC-FairGen and save them to disk."""
    start = time()
    eval_task_sets = []
    for util in utils:
        task_sets = []
        for i in range(nsets):
            ts = synth.mc_fairgen(set_id=i, u_lo=util, implicit_deadlines=True)
            synth.synth_c_pmf(ts)
            synth.set_fixed_priorities(ts)
            task_sets.append(ts)
        eval_task_sets.append(task_sets)
    stop = time()
    print('Task Set Generation MC-Fairgen: %.3fs' % (stop - start))
    pickle.dump(eval_task_sets, open(task_sets_path + 'task_sets_fairgen', 'wb'))


# Schedulability Tests
######################

def eval_simplegen():
    """Perform schedulability analysis on SimpleGen task set, using provided analysis schemes."""
    print("Evaluation: SimpleGen")
    modes = [
        # name, function
        ('dSMC', ana.d_smc),
        ('dAMC', ana.d_amc),
        ('EDF-VD', ana.d_edf_vd),
        ('pSMC', ana.p_smc),
        ('pAMC-BB', ana.p_amc_bb),
        ('pAMC-BB+', ft.partial(ana.p_amc_bb, ignore_hi_mode=True))
    ]

    pool = mp.Pool()
    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_simplegen', 'rb'))
    for name, func in modes:
        start = time()
        rates = []
        for task_sets in task_sets_list:
            rates.append(100 * np.average(pool.map(func, task_sets)))
        pickle.dump(rates, open(eval_simplegen_path + name, 'wb'))
        stop = time()
        print('%s: %.3fs' % (name, (stop - start)))


def eval_fairgen():
    """Perform schedulability analysis on MC-FairGen task set, using provided analysis schemes."""
    print("Evaluation: Fairgen")
    modes = [
        # name, function
        ('dSMC', ana.d_smc),
        ('dAMC', ana.d_amc),
        ('EDF-VD', ana.d_edf_vd),
        ('pSMC', ana.p_smc),
        ('pAMC-BB', ana.p_amc_bb),
        ('pAMC-BB+', ft.partial(ana.p_amc_bb, ignore_hi_mode=True))
    ]

    pool = mp.Pool()
    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_fairgen', 'rb'))
    for name, func in modes:
        start = time()
        rates = []
        for task_sets in task_sets_list:
            rates.append(100 * np.average(pool.map(func, task_sets)))
        pickle.dump(rates, open(eval_fairgen_path + name, 'wb'))
        stop = time()
        print('%s: %.3fs' % (name, (stop - start)))


def eval_monte_carlo():
    """Perform schedulability analysis on MC-FairGen task set, using Monte-Carlo analysis schemes."""
    print("Evaluation: Monte Carlo Schemes")
    modes = [
        # name, function
        ('pSMC', ft.partial(ana.p_smc, thresh_lo=1e-3, thresh_hi=1e-4)),
        ('pSMC (Monte Carlo)', ana.p_smc_monte_carlo),
        ('pAMC-BB', ft.partial(ana.p_amc_bb, hi_mode_duration=1, thresh_lo=1e-3, thresh_hi=1e-4)),
        ('pAMC-BB (Monte Carlo)', ana.p_amc_bb_monte_carlo)
    ]

    pool = mp.Pool()
    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_fairgen', 'rb'))
    for name, func in modes:
        start = time()
        rates = []
        for task_sets in task_sets_list:
            rates.append(100 * np.average(pool.map(func, task_sets)))
        pickle.dump(rates, open(eval_monte_carlo_path + name, 'wb'))
        stop = time()
        print('%s: %.3fs' % (name, (stop - start)))


# Visualization
###############

def plot_schedulability_rates_simplegen():
    """Plot schedulability rates in a graph, using the results from eval_simplegen()."""
    modes = [
        # name, color, linestyle, marker
        ('dSMC', 'blue', 'dashed', 'o'),
        ('dAMC', 'red', 'dashed', 'd'),
        ('EDF-VD', 'green', 'dashed', 's'),
        ('pSMC', 'orange', 'dashed', '^'),
        ('pAMC-BB', 'magenta', 'dashed', 'D'),
        ('pAMC-BB+', 'purple', 'dashed', 'v')
    ]

    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_simplegen', 'rb'))
    rates = {}
    fig = plt.figure(figsize=(12, 6), dpi=300)
    fig.suptitle('Evaluation: SimpleGen (n=%d)' % len(task_sets_list[0]))
    ax1 = fig.add_subplot(111)
    for name, color, linestyle, marker in modes:
        rates[name] = pickle.load(open(eval_simplegen_path + name, 'rb'))
        ax1.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)
    ax1.set_xlabel('LO mode utilization')
    ax1.set_ylabel('Percentage of task sets schedulable')
    ax1.set_xlim(0.1, 2.1)
    ax1.set_xticks([k * 0.5 for k in range(5)])
    ax1.set_yticks([k * 20 for k in range(6)])
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(1))
    ax1.minorticks_on()
    ax1.grid(which='both', linestyle='dashed')

    # Plot average system util:
    avg_utils = []
    for i, _ in enumerate(utils):
        avg_utils.append(np.average([task_set.u_avg for task_set in task_sets_list[i]]))
    ax2 = ax1.twinx()
    ax2.plot(utils, avg_utils, label='Avg Sys Util (right scale)', color='black', linestyle='dashed',
             marker=None)
    ylim = ax1.get_ylim()
    ax2.set_ylim(ylim[0] / 100, ylim[1] / 100)
    ax2.set_ylabel('U(Avg)')

    plt.axvline(1.0, color='black', linewidth=0.8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.xlim(0.1, 2.1)
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center left')
    plt.savefig('./figures/schedulability_rates_simplegen.png')
    # plt.show()


def plot_schedulability_rates_fairgen():
    """Plot schedulability rates in a graph, using the results from eval_fairgen()."""
    modes = [
        # name, color, linestyle, marker
        ('dSMC', 'blue', 'dashed', 'o'),
        ('dAMC', 'red', 'dashed', 'd'),
        ('EDF-VD', 'green', 'dashed', 's'),
        ('pSMC', 'orange', 'dashed', '^'),
        ('pAMC-BB', 'magenta', 'dashed', 'D'),
        ('pAMC-BB+', 'purple', 'dashed', 'v')
    ]

    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_fairgen', 'rb'))
    rates = {}
    fig = plt.figure(figsize=(12, 6), dpi=300)
    fig.suptitle('Evaluation: MC-Fairgen (n=%d)' % len(task_sets_list[0]))
    ax1 = fig.add_subplot(111)
    for name, color, linestyle, marker in modes:
        rates[name] = pickle.load(open(eval_fairgen_path + name, 'rb'))
        ax1.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)
    ax1.set_xlabel('LO mode utilization')
    ax1.set_ylabel('Percentage of task sets schedulable')
    ax1.set_xticks([k * 0.5 for k in range(5)])
    ax1.set_yticks([k * 20 for k in range(6)])
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(1))
    ax1.minorticks_on()
    ax1.grid(which='both', linestyle='dashed')

    # Plot average system util:
    avg_utils = []
    for i, _ in enumerate(utils):
        avg_utils.append(np.average([task_set.u_avg for task_set in task_sets_list[i]]))
    ax2 = ax1.twinx()
    ax2.plot(utils, avg_utils, label='Avg Sys Util (right scale)', color='black', linestyle='dashed',
             marker=None)
    ylim = ax1.get_ylim()
    ax2.set_ylim(ylim[0] / 100, ylim[1] / 100)
    ax2.set_ylabel('U(Avg)')

    plt.axvline(1.0, color='black', linewidth=0.8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.xlim(0.1, 2.1)
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.savefig('./figures/schedulability_rates_fairgen.png')
    # plt.show()


def plot_schedulability_rates_monte_carlo():
    """Plot schedulability rates in a graph, using the results from eval_monte_carlo()."""
    modes = [
        # name, color, linestyle, marker
        ('pSMC (Monte Carlo)', 'cyan', 'solid', 'D'),
        ('pSMC', 'blue', 'dashed', 's'),
        ('pAMC-BB (Monte Carlo)', 'purple', 'solid', 'v'),
        ('pAMC-BB', 'red', 'dashed', '^')
    ]

    rates = {}
    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_fairgen', 'rb'))
    fig = plt.figure(figsize=(12, 6), dpi=300)
    fig.suptitle('Evaluation: Monte Carlo Schemes (n=%d)' % len(task_sets_list[0]))
    ax1 = fig.add_subplot(111)
    for name, color, linestyle, marker in modes:
        rates[name] = pickle.load(open(eval_monte_carlo_path + name, 'rb'))
        ax1.plot(utils, rates[name], label=name, color=color, linestyle=linestyle, marker=marker)
    ax1.set_xlabel('LO mode utilization')
    ax1.set_ylabel('Percentage of task sets schedulable')
    ax1.set_xticks([k * 0.5 for k in range(5)])
    ax1.set_yticks([k * 20 for k in range(6)])
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(1))
    ax1.minorticks_on()
    ax1.grid(which='both', linestyle='dashed')

    plt.axvline(1.0, color='black', linewidth=0.8)
    plt.legend(loc='center left')
    plt.xlim(0.1, 2.1)
    plt.ylim(-5, 105)
    plt.savefig('./figures/schedulability_rates_monte_carlo.png')
    # plt.show()


# Performance Measurements
##########################

def measure_mp_speedup():
    """Measure sequential and parallel times elapsed for schedulability analysis."""
    modes = [
        # name, function
        ('dSMC', ana.d_smc),
        ('dAMC', ana.d_amc),
        ('EDF-VD', ana.d_edf_vd),
        ('pSMC', ana.p_smc),
        ('pAMC-BB', ana.p_amc_bb),
        ('pAMC-BB+', ft.partial(ana.p_amc_bb, ignore_hi_mode=True))
    ]
    times_seq = {}
    task_sets_list = pickle.load(open(task_sets_path + 'task_sets_fairgen', 'rb'))
    start_total_seq = time()
    for name, func in modes:
        start_mode_seq = time()
        rates = []
        for task_sets in task_sets_list:
            results = []
            for task_set in task_sets:
                results.append(func(task_set))
            rates.append(100 * np.average(results))
        stop_mode_seq = time()
        times_seq[name] = stop_mode_seq - start_mode_seq
    stop_total_seq = time()
    times_seq['Overall'] = stop_total_seq - start_total_seq

    times_par = {}
    start_total_par = time()
    pool = mp.Pool()
    for name, func in modes:
        start_mode_par = time()
        rates = []
        for task_sets in task_sets_list:
            rates.append(100 * np.average(pool.map(func, task_sets)))
        stop_mode_par = time()
        times_par[name] = stop_mode_par - start_mode_par
    stop_total_par = time()
    times_par['Overall'] = stop_total_par - start_total_par

    speedups = {}
    for name, _ in modes:
        speedups[name] = times_seq[name] / times_par[name]
    speedups['Overall'] = times_seq['Overall'] / times_par['Overall']

    print("PERFORMANCE MEASUREMENTS")
    print("Number of cores: %d" % mp.cpu_count())
    print("Scheme: Sequential time / Parallel time / Speedup")
    for name, _ in modes:
        print("%s: %.3fs / %.3fs / %.3f" % (name, times_seq[name], times_par[name], speedups[name]))
    print("Overall: %.3fs / %.3fs / %.3f" % (times_seq['Overall'], times_par['Overall'], speedups['Overall']))


#################
# Main Function #
#################

if __name__ == '__main__':
    gen_task_sets_simplegen(1000)
    eval_simplegen()
    plot_schedulability_rates_simplegen()
    gen_task_sets_fairgen(1000)
    eval_fairgen()
    plot_schedulability_rates_fairgen()
    eval_monte_carlo()
    plot_schedulability_rates_monte_carlo()
    measure_mp_speedup()
