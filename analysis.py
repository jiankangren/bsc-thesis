"""
This module contains any methods and functions related to the analysis of a task set's schedulability, deadline miss
probability and so forth. It makes use of the classes defined in module class_lib.

-- Luca Stalder, 2017
"""

import copy
import itertools
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import generation as gen
# import sympy as sp
from lib import TaskSet, WeibullDist, convolve_rescale_pmf, shrink


class BacklogSim(object):
    """Backlog simulation object.

    Offers a container used for iterative backlog analysis of stochastic task sets.

    Attributes:
        task_set: The TaskSet object to be analyzed. This needs priorities and release times assigned. 
        p_level: The backlog's P-level, meaning only jobs of priority level P or higher are considered.
        k: Number of completed hyperperiods.
        t: Elapsed time in current hyperperiod.
        backlog: The current backlog distribution, stored as a list.
        timeline: A list of NamedTuple, each modelling a job release with its release time and the corresponding task.
        jobs_remaining: A list of jobs that are not yet reached in the current hyperperiod.
    """

    def __init__(self, task_set, p_level=0, initial_backlog=None):
        self.task_set = task_set
        self.p_level = p_level
        self.k = 0
        self.t = 0
        self.backlog = np.array([1.0]) if initial_backlog is None else initial_backlog

        # Build timeline of one full hyperperiod
        self.timeline = [rel for rel in task_set.jobs if rel.task.static_prio >= self.p_level]
        self.jobs_remaining = list(self.timeline)  # Store a copy

    def step(self, dt=None, mode='before'):
        """Advance the model.

        Advances the model by dt time units. Perform convolution and shrinking where necessary.

         Args:
             dt: Size of the step. If no dt is given, step to the next job release and/or end of the hyperperiod.
             mode: Determines behavior if step lands exactly on a job release.
                'after' performs convolution on any job releases at the reached time, while 'before' does not.
        """
        if dt is None:
            if not self.jobs_remaining:
                dt = self.task_set.hyperperiod - self.t  # Step to end of hyperperiod if no remaining jobs
            else:
                dt = self.jobs_remaining[0].t - self.t

        while True:
            if not self.jobs_remaining:  # No more new jobs in this hyperperiod
                time_remaining = self.task_set.hyperperiod - self.t
                if dt < time_remaining:  # Step does not reach end of hyperperiod
                    self.backlog = shrink(self.backlog, dt)
                    self.t += dt
                    return
                else:  # Step reaches / exceeds end of hyperperiod
                    self.backlog = shrink(self.backlog, time_remaining)
                    self.k += 1
                    self.t = 0
                    self.jobs_remaining = list(self.timeline)
                    dt -= time_remaining
                    continue
            elif dt < self.jobs_remaining[0].release - self.t \
                    or dt == self.jobs_remaining[0].release - self.t and mode == 'before':
                # Step does not reach or convolve next job release
                self.backlog = shrink(self.backlog, dt)
                self.t += dt
                return
            else:  # Step to next job release
                next_job = self.jobs_remaining.pop(0)
                self.backlog = shrink(self.backlog, next_job.release - self.t)
                self.backlog = convolve_rescale_pmf(self.backlog, next_job.task.c_pmf)
                dt -= next_job.release - self.t
                self.t = next_job.release


def stationary_backlog_iter(task_set: TaskSet, p_level=0, max_iter=100, epsilon=1e-14):
    """Iterative calculation of the stationary backlog for a task set.

    This method finds the stationary backlog at the beginning of a task set's hyperperiod by repeatedly applying 
    convolution and shrinking until this backlog converges. Convergence is detected by a quadratic error < epsilon.

    This method is further described in [4].

    Args:
        task_set: The task set in consideration.
        p_level: The desired priority level of the resulting backlog. Only tasks with priority of at least p_level are 
            taken into consideration.
        max_iter: This bounds the maximum number of hyperperiods scanned. If no convergence is found after max_iter 
            iterations, the stationary backlog is assumed to grow unbounded.
        epsilon: Stopping value for convergence.

    Returns:
        last: An array representing the probability distribution over the task set's stationary backlog at the beginning
            of its hyperperiod. Returns None if the average system utilization is >100% (backlog growth unbounded).
        iters: The number of iterations needed until convergence. Returns -1 if convergence was not reached within
            max_iter iterations.
    """

    def quadratic_difference(p: np.ndarray, q: np.ndarray):
        """L2-Norm of difference between arrays p and q."""
        if len(p) <= len(q):
            a = np.array(p)
            b = np.array(q)
        else:
            a = np.array(q)
            b = np.array(p)
        a = np.concatenate((a, np.zeros(b.size - a.size)))
        return np.linalg.norm(a - b, 2)

    if task_set.u_avg > 1.0:
        print("Average system utilization above 100%, stationary backlog undefined.")
        return None, None

    if any([t.static_prio is None for t in task_set.tasks]):
        gen.set_priorities_dm(task_set)

    sim = BacklogSim(task_set=task_set, p_level=p_level)
    last = sim.backlog
    dist = 1.
    i = 0
    while dist > epsilon and i <= max_iter:
        sim.step(dt=task_set.hyperperiod, mode='before')
        dist = quadratic_difference(last, sim.backlog)
        last = sim.backlog
        i += 1
    if i > max_iter:
        iters = -1
    else:
        iters = i

    return last, iters


# def stationary_backlog_analytic(task_set: TaskSet, p_level=0):  # TODO REMOVE
#     """Exact solution for finding a task set's stationary backlog.
#
#     This method finds the stationary backlog at the beginning of a task set's hyperperiod by exact calculation, which
#     involves finding its Markov chain matrix. For further details refer to [4].
#
#     Args:
#         task_set: The task set in consideration.
#         p_level: The desired priority level of the resulting backlog. Only tasks with priority of at least p_level are
#             taken into consideration.
#
#     Returns:
#         An array representing the probability distribution over the task set's stationary backlog at the beginning of
#         its hyperperiod.
#     """
#     w_min = 0  # Backlog after first hyperperiod assuming all jobs require minimal execution time.
#     timeline = task_set.job_releases
#     t = 0
#     while timeline:
#         next_job = timeline.pop(0)
#         w_min = max(w_min - (next_job.t - t), 0)  # 'Shrink'
#         w_min += next_job.task.c_min()  # 'Convolve'
#         t = next_job.t
#     w_min = max(w_min - (task_set.hyperperiod - t), 0)
#
#     r = task_set.hyperperiod + w_min - sum([task.c_min() * len(task.rel_times) for task in task_set.tasks
#                                             if task.priority >= p_level])  # max idle time in any hyperperiod
#     sim = BacklogSim(task_set, p_level, initial_backlog=np.array([0.0] * r + [1.0]))
#     sim.step(dt=task_set.hyperperiod, mode='before')
#     m_r = len(sim.backlog) - 1  # max possible backlog for initial backlog r
#
#     first_backlogs = []
#     print(r, m_r)
#     for i in range(r + 1):  # TODO Add multi-threading
#         sim.__init__(task_set, p_level, initial_backlog=np.array([0.0] * i + [1.0]))
#         sim.step(task_set.hyperperiod, mode='before')
#         first_backlogs.append(np.concatenate((sim.backlog, np.zeros(m_r - len(sim.backlog) + 1))))
#         print(i)
#
#     for i in range(1, m_r + 1):
#         first_backlogs.append(np.array([0.0] * i + first_backlogs[r][:-i].tolist()))
#     p_upper_left = np.array(first_backlogs).T  # (m_r + 1) x (r + 1) matrix in the upper left corner of P
#     p_upper_left -= np.eye(p_upper_left.shape[0], p_upper_left.shape[1])  # Equate everything to zero
#
#     print(p_upper_left.shape)
#
#     unknowns = sp.symbols('p0:%d' % (m_r + r + 1))  # Tuple of unknowns
#     equations = p_upper_left.dot(unknowns)  # First subset of (m_r + 1) equations  # TODO Too slow
#
#     print(type(equations), equations.shape)
#
#     a_matrix = np.zeros((m_r, m_r))
#     a_matrix[:-1, 1:] = np.eye(m_r - 1)
#     a_matrix[-1, :] = (p_upper_left[-1:0:-1, r] / -p_upper_left[0, r]).T
#
#     D, V = np.linalg.eig(a_matrix)
#     print(D, D.shape)
#     print(V, V.shape)


def rta_fp(task_set: TaskSet):
    """Response time analysis for fixed-priority scheduled task sets.

    This method calculates the response time distribution for every job during one hyperperiod, assuming steady state
    backlog. This distribution is attached directly to each job object.

    Deadline miss probabilities can then be found by 1 - sum(response time distribution), since only response
    times up to and including the job's deadline are calculated.

    Args:
        task_set: Mixed-Criticality task set with probabilistic task execution times and fixed priority scheduling.
    """
    sims = [None] * len(task_set.tasks)
    for task in task_set.tasks:
        if sims[task.static_prio] is None:
            backlog, _ = stationary_backlog_iter(task_set, task.static_prio)
            if backlog is not None:
                sims[task.static_prio] = BacklogSim(task_set, task.static_prio, backlog)
            else:  # No stationary backlog
                for job in task_set.jobs:
                    job.response = np.zeros(1)
                return
    # Job Response Time Analysis
    for job in task_set.jobs:
        sim = sims[job.task.static_prio]
        sim.step(job.release - sim.t, mode='before')
        part_response = convolve_rescale_pmf(sim.backlog, job.task.c_pmf)
        preempts = [preempt for preempt in task_set.jobs
                    if job.release <= preempt.release
                    and job.task.static_prio < preempt.task.static_prio
                    and preempt.release < job.release + job.task.deadline]
        while preempts:
            next_preempt = preempts.pop(0)
            head, tail, _ = np.split(part_response,
                                     [next_preempt.release - job.release + 1, job.task.deadline + 1])
            tail_lim = job.task.deadline - head.size + 1
            if tail.size:
                tail = sig.convolve(tail, next_preempt.task.c_pmf[:tail_lim])[:tail_lim]
            part_response = np.concatenate((head, tail))
        part_response.resize(job.task.deadline + 1)
        job.response = part_response

    # TODO Remove
    # for task in task_set.tasks:
    #     task_responses = [j.response for j in task_set.jobs if j.task == task]
    #     task.avg_response = np.sum(task_responses, axis=0) / len(task_responses)
    #     task.dmp = 1. - np.sum(task.avg_response)


def plot_backlogs(task_set, p_level=0, idx=None, add_stationary=False, scale='log', path=None):
    """Plots the first n_subplots backlogs of task_set, considering only tasks of at least p_level priority."""
    print(task_set.description)
    if idx is None:
        idx = [1, 2, 5, 10, 50, 100]
    fig = plt.figure(figsize=(5, 10), dpi=180)
    sim = BacklogSim(task_set, p_level)
    if add_stationary:
        stationary, iters = stationary_backlog_iter(task_set)
        print("Iterations:", iters)
        xlim = len(stationary)
    else:
        sim.step(task_set.hyperperiod * max(idx))
        xlim = len(sim.backlog)
    sim = BacklogSim(task_set, p_level)
    for i, k in enumerate(idx):
        print(k, k - sim.k)
        sim.step(task_set.hyperperiod * (k - sim.k))
        plt.subplot(len(idx) + 1, 1, i + 1)
        plt.bar(range(len(sim.backlog)), sim.backlog)
        plt.xlim(-1, xlim + 1)
        plt.ylim(1e-15, 10)
        plt.yscale(scale)
        plt.title("Hyperperiods: %d" % sim.k)
    if add_stationary:
        plt.subplot(len(idx) + 1, 1, len(idx) + 1)
        plt.bar(range(len(stationary)), stationary)
        plt.xlim(-1, xlim + 1)
        plt.ylim(1e-15, 10)
        plt.yscale(scale)
        plt.title("Stationary backlog")
    plt.subplots_adjust(hspace=0.8)
    fig.suptitle(task_set.description)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


#########################################
# Deterministic Schedulability Analysis #
#########################################

def d_smc(task_set: TaskSet):
    """Deterministic SMC Response Time Analysis, as described in [2]."""

    def min_crit(c1, c2):
        """Returns the lower of both criticality levels."""
        if c1 == 'HI' and c2 == 'HI':
            return 'HI'
        else:
            return 'LO'

    for task in task_set.tasks:
        # 1) Build set of tasks with higher priority:
        hp = [j for j in task_set.tasks if j.static_prio > task.static_prio]

        # 2) Iteration to fixed point for solving recurrence relation:
        res = 0
        res_next = -1
        while res != res_next:
            res = res_next
            if res > task.deadline:  # res growing monotonically
                return False
            sum_hp = task.c_lo if task.criticality == 'LO' else task.c_hi
            for j in hp:
                if min_crit(task.criticality, j.criticality) == 'HI':
                    sum_hp += math.ceil(res / j.period) * j.c_hi
                else:
                    sum_hp += math.ceil(res / j.period) * j.c_lo
            res_next = sum_hp

    # No deadline overruns happened:
    return True


def d_amc(task_set):
    """Deterministic AMC-rtb Response Time Analysis, as described in [2]."""
    for task in task_set.tasks:
        # 1.1) Build set of all tasks with higher priority:
        hp = [j for j in task_set.tasks if j.static_prio > task.static_prio]

        # 1.2) Build sets of all HI- and LO-critical tasks with higher priority:
        hp_hi = [t for t in hp if t.criticality == 'HI']
        hp_lo = [t for t in hp if t.criticality == 'LO']

        # 2) Iteration to fixed point for solving recurrence relation:
        # 2.1) R_LO:
        res_lo = 0
        res_next = -1
        while res_lo != res_next:
            res_lo = res_next
            if res_lo > task.deadline:  # res growing monotonically
                return False
            sum_hp = task.c_lo if task.criticality == 'LO' else task.c_hi
            for j in hp:
                sum_hp += math.ceil(res_lo / j.period) * j.c_lo
            res_next = sum_hp

        # 2.2 R_HI (only defined for HI-critical tasks):
        if task.criticality == 'HI':
            res_hi = 0
            res_next = -1
            while res_hi != res_next:
                res_hi = res_next
                if res_hi > task.deadline:  # res growing monotonically
                    return False
                sum_hp = task.c_hi
                for j in hp_hi:
                    sum_hp += math.ceil(res_hi / j.period) * j.c_hi
                res_next = sum_hp

        # 2.3 R_* (criticality change, only defined for HI-critical tasks):
        if task.criticality == 'HI':
            res_asterisk = 0
            res_next = -1
            while res_asterisk != res_next:
                res_asterisk = res_next
                if res_asterisk > task.deadline:
                    return False
                sum_hp = task.c_hi
                for j in hp_hi:
                    sum_hp += math.ceil(res_asterisk / j.period) * j.c_hi
                for k in hp_lo:
                    sum_hp += math.ceil(res_lo / k.period) * k.c_lo
                res_next = sum_hp

    # No deadline overruns happened:
    return True


def d_edf_vd(task_set: TaskSet):
    """Deterministic EDF-VD schedulability analysis. Calculated according to theorem 1 in [3]."""

    def total_k_utilization(task_subset, k):
        """Total utilization at scenario criticality level k of tasks contained in task_subset."""
        result = 0.0
        for j in task_subset:
            c = j.c_hi if k == 'HI' else j.c_lo
            result += float(c) / float(j.period)
        return result

    subset_lo = [t for t in task_set.tasks if t.criticality == 'LO']
    subset_hi = [t for t in task_set.tasks if t.criticality == 'HI']
    u_1_1 = total_k_utilization(subset_lo, 'LO')
    u_2_1 = total_k_utilization(subset_hi, 'LO')
    u_2_2 = total_k_utilization(subset_hi, 'HI')

    return u_1_1 + min(u_2_2, u_2_1 / (1 - u_2_2)) <= 1     # Note that this condition is sufficient for schedulability.


#########################################
# Probabilistic Schedulability Analysis #
#########################################

def p_smc(task_set: TaskSet, thresh_lo: float=1e-5, thresh_hi: float=1e-9):
    """Probabilistic static mixed criticality schedulability analysis.
    
    This method decides whether a set of probabilistic tasks can be considered schedulable or not. The analysis is based
    on a task's probability of having no failing job in one hyperperiod, which is compared against the given thresholds.
    
    The effects of a mode switch are not considered in this analysis. The only difference between LO- and HI-critical
    tasks is the choice of DMP threshold.
    
    Args:
        task_set: Mixed-Criticality task set with probabilistic task execution times and fixed priority scheduling.
        thresh_lo: LO-critical tasks have their DMP compared against this float to determine schedulability.
        thresh_hi: HI-critical tasks have their DMP compared against this float to determine schedulability.
        
    Returns:
        True, if P["At least one deadline missed in one hyperperiod"] for every task lies below the corresponding 
        threshold, meaning the whole task set is considered 'schedulable'; else False.
    """
    rta_fp(task_set)
    for task in task_set.tasks:
        thresh = thresh_hi if task.criticality == 'HI' else thresh_lo
        p_failure = 1. - np.prod([1. - job.dmp for job in task_set.jobs if job.task == task])
        if p_failure > thresh:
            return False
    return True


def p_amc_bb(task_set: TaskSet,
             hi_mode_duration=1,
             thresh_lo=1e-5, thresh_hi=1e-9,
             ):
    """Probabilistic adaptive mixed criticality schedulability analysis.
    
    This method decides whether a set of probabilistic tasks can be considered schedulable or not. The analysis is based
    on the tasks' average response times - and following from that, their per-hyperperiod deadline miss probability 
    (DMP) with the system's stationary backlog.
    
    The probability of a mode switch happening is also taken into consideration. In this model, if a mode switch occurs,
    the system goes in a "reset state", lasting a fixed number of hyperperiods. 
    The behaviour in HI-mode is a "black box" and not considered in the analysis, it is simply assumed that all LO-tasks
    are killed (100% DMP), and all HI-tasks succeed (0% DMP).
    
    Args:
        task_set: Mixed-Criticality task set with probabilistic task execution times and fixed priority scheduling.
        hi_mode_duration: Theoretical duration of HI-mode after a mode switch occurred.
        thresh_lo: LO-critical tasks have their DMP compared against this float to determine schedulability.
        thresh_hi: HI-critical tasks have their DMP compared against this float to determine schedulability.
    """
    task_set_adj = TaskSet(task_set.set_id, copy.deepcopy(task_set.tasks))

    # Probability of mode switch:
    p_no_overrun = []
    for task in task_set_adj.tasks:
        if task.criticality == 'HI':
            p_no_overrun.append(np.sum(task.c_pmf[:task.c_lo + 1]) ** (task_set_adj.hyperperiod // task.period))
    p_switch = 1. - np.prod(p_no_overrun)

    # Adjusted deadline miss probability:
    for task in task_set_adj.tasks:
        if task.criticality == 'HI':
            task.c_pmf = task.c_pmf[:task.c_lo + 1] / np.sum(task.c_pmf[:task.c_lo + 1])
    rta_fp(task_set_adj)

    expected_switch_time = 1. / p_switch  # Expected number of hyperperiods until mode switch
    total_time = expected_switch_time + hi_mode_duration
    for task in task_set_adj.tasks:
        if task.criticality == 'HI':
            thresh = thresh_hi
            p_failure_hi = 0.
        else:
            thresh = thresh_lo
            p_failure_hi = 1.
        p_failure_lo = 1. - np.prod([1. - job.dmp for job in task_set_adj.jobs if job.task == task])
        if (expected_switch_time / total_time) * p_failure_lo + (hi_mode_duration / total_time) * p_failure_hi > thresh:
            return False
    return True


#######################
# Monte Carlo Schemes #
#######################

def p_smc_monte_carlo(task_set, nhyperperiods=10000, thresh_lo=1e-6, thresh_hi=1e-9):

    start_total = time.time()
    elapsed = 0

    class JobInstance(object):
        def __init__(self, task, release, remaining):
            self.task = task
            self.release = release  # Tuple (hyperperiod of release, release time)
            self.remaining = remaining
            self.abs_deadline = (release[0] + (release[1] + self.task.deadline) // task_set.hyperperiod,
                                 (release[1] + self.task.deadline) % task_set.hyperperiod)

    def exec_backlog(dt):
        """
        
        
        dt will never exceed the time to the next release or the end of the hyperperiod.  
        """
        nonlocal t
        while dt > 0:
            if not backlog:
                t += dt
                break
            elif backlog[0].remaining > dt:
                t += dt
                backlog[0].remaining -= dt
                break
            else:
                done = backlog.pop(0)
                t += done.remaining
                dt -= done.remaining
                if done.abs_deadline < (i, t):
                    done.task.failed_hps.add(done.release[0])

    ###
    class CustomContainer(object):
        def __init__(self):
            self.items = [[] for i in range(len(task_set.tasks))]
            self.size = 0

        def __bool__(self):
            return self.size > 0

        def __iter__(self):
            return self.as_list.__iter__()

        def __getitem__(self, item):
            return self.as_list.__getitem__(item)

        def add(self, new):
            self.items[new.task.static_prio].append(new)
            self.size += 1

        def pop(self, i):
            for a in self.items[::-1]:
                if a:
                    self.size -= 1
                    return a.pop(i)

        @property
        def as_list(self):
            return list(itertools.chain.from_iterable(self.items[::-1]))
    ###

    # backlog = socon.SortedListWithKey([], key=lambda rel: (-rel.task.static_prio, rel.release))  # ~42%
    backlog = CustomContainer()  # ~35%
    for task in task_set.tasks:
        task.exec_times = np.random.choice(a=range(len(task.c_pdf)),
                                           size=nhyperperiods * (task_set.hyperperiod // task.period),
                                           p=task.c_pdf)
        task.failed_hps = set()  # All hyperperiods where an instance of this task missed its deadline
    for i in range(nhyperperiods):
        # print(i)
        # Reset timer, build list of all releases in new hyperperiod
        t = 0
        releases = list(task_set.jobs)

        while releases:
            next_rel = releases.pop(0)
            exec_backlog(next_rel.release - t)
            exec_time, next_rel.task.exec_times = next_rel.task.exec_times[-1], next_rel.task.exec_times[:-1]
            start = time.time()
            backlog.add(JobInstance(task=next_rel.task, release=(i, next_rel.release), remaining=exec_time))
            stop = time.time()
            elapsed += stop - start
        else:
            exec_backlog(task_set.hyperperiod - t)
    stop_total = time.time()
    # print("Elapsed here:", elapsed)
    # print("Elapsed total:", stop_total - start_total)
    # print(100 * elapsed / (stop_total - start_total), "%")

    for task in task_set.tasks:
        p_fail = len(task.failed_hps) / nhyperperiods
        thresh = thresh_hi if task.criticality == 'HI' else thresh_lo
        if p_fail > thresh:
            return False
    return True








# def total_variation_distance(p: np.ndarray, q: np.ndarray):
#     """Total variation distance between arrays p and q."""
#     if len(p) <= len(q):
#         a = np.array(p)
#         b = np.array(q)
#     else:
#         a = np.array(q)
#         b = np.array(p)
#     a.resize(len(b))
#     return max([abs(x - y) for x, y in zip(a, b)])

if __name__ == '__main__':
    # ts = gen.simple_gen(0, 0.8)
    # ts.set_priorities_dm()
    # gen.synth_c_dist(ts, distribution_cls=ExpExceedDist)
    # print(ts.description)
    # ts.draw(scale='log')

    ts = gen.mc_fairgen(0, 0.8)
    gen.set_priorities_dm(ts)
    gen.synth_c_dist(ts, distribution_cls=WeibullDist)
    print(ts.description)
    ts.draw(scale='linear')

    # print("draw")
    # ts = gen.mc_fairgen_det(set_id=0, u_lo=0.95)
    # ts.set_priorities_rm()
    # # print(stationary_backlog_iter(task_set=ts))
    # # stationary_backlog_analytic(ts, p_level=0)
    # plot_backlogs(ts, add_stationary=True, scale='log')
    # ts = gen.dummy_taskset()
    # ts.set_priorities_rm()
    # p_smc_monte_carlo(ts)
    # rta_fp(ts)
    # for task in ts.tasks:
    #     print("Task ID: %d, DMP: %f, Avg Response Time Dist: %s" % (task.task_id, task.dmp, task.avg_response))
    # print(d_smc(ts), d_amc(ts), p_smc(ts), p_amc_black_box(ts))

    # ts_mod, p_switch = dmp_analysis_fp_given_lo_mode(ts)
    # print("Probability of mode switch:", p_switch)
    # for task in ts_mod.tasks:
    #     print("Task ID: %d, DMP: %f, Avg Response Time Dist: %s" % (task.task_id, task.dmp, task.avg_response))
    # for i in range(100):
    #     ts = gen.mc_fairgen_stoch(set_id=10, u_lo=0.95, m=1, mode='max', implicit_deadlines=True)
    #     ts.set_priorities_rm()
    #     print(p_smc_monte_carlo(ts))
    pass




"""
Literature:
[1] Audsley, Burns, Richardson, Tindell, Wellings
    Applying new scheduling theory to static priority preemptive scheduling.

[2] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
    
[3] Baruah, Bonifaci, D'Angelo, Marchetti-Spaccamela, van der Ster, Stougie
    Mixed-Criticality Scheduling of Sporadic Task Systems
    
[4] Diaz, Garcia, Kim, Lee, Bello, Lopez, Min, Mirabella
    Stochastic Analysis of Periodic Real-Time Systems
"""