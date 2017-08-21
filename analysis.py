"""
This module contains any methods and functions related to the analysis of a task set's schedulability, deadline miss
probability, etc. It makes use of the classes defined in module class_lib.

-- Luca Stalder, 2017
"""

import copy
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import synthesis as synth
from lib import Task, TaskSet, ExpExceedDist, WeibullDist, \
    convolve_rescale_pmf, shrink, split_convolve_merge


################################
# Response Time Analysis Tools #
################################

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
        synth.set_fixed_priorities(task_set)

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

        # Step up to job release; yielding job-level backlog
        sim = sims[job.task.static_prio]
        sim.step(job.release - sim.t, mode='before')

        # Split-convolve-merge
        part_response = convolve_rescale_pmf(sim.backlog, job.task.c_pmf)
        preempts = [preempt for preempt in task_set.jobs
                    if job.release <= preempt.release
                    and job.task.static_prio < preempt.task.static_prio
                    and preempt.release < job.release + job.task.deadline]
        while preempts:
            next_preempt = preempts.pop(0)
            part_response = split_convolve_merge(
                part_response, next_preempt.task.c_pmf, next_preempt.release)
            # head, tail, _ = np.split(part_response,
            #                          [next_preempt.release - job.release + 1, job.task.deadline + 1])
            # tail_lim = job.task.deadline - head.size + 1
            # if tail.size:
            #     tail = sig.convolve(tail, next_preempt.task.c_pmf[:tail_lim])[:tail_lim]
            # part_response = np.concatenate((head, tail))

        part_response = part_response[:job.task.deadline + 1]
        job.response = part_response


def plot_backlogs(task_set, p_level=0, idx=None, add_stationary=False, scale='log', path=None):
    """Plots backlog distributions of task_set, considering only tasks of at least p_level priority."""
    if idx is None:
        idx = [1, 2, 5, 10, 50, 100]  # A plot is drawn for the end of each of these hyperperiods
    fig = plt.figure(figsize=(3, 10), dpi=180)
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
        ax = plt.subplot(len(idx) + 1, 1, i + 1)
        plt.bar(range(len(sim.backlog)), sim.backlog)
        plt.xlim(-1, xlim + 1)
        if scale == 'log':
            plt.ylim(1e-15, 10)
        plt.yscale(scale)
        ax.set_yticks([])
        plt.title("Hyperperiods: %d" % sim.k)
    if add_stationary:
        ax = plt.subplot(len(idx) + 1, 1, len(idx) + 1)
        plt.bar(range(len(stationary)), stationary)
        plt.xlim(-1, xlim + 1)
        if scale == 'log':
            plt.ylim(1e-15, 10)
        plt.yscale(scale)
        ax.set_yticks([])
        plt.title("Stationary backlog")
    plt.subplots_adjust(hspace=0.8)
    # fig.suptitle(task_set.description, fontsize=10)
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

    u_hi_hi = task_set.u_hi
    if u_hi_hi >= 1:
        return False

    u_lo_lo = sum([task.u_lo for task in task_set.tasks if task.criticality == 'LO'])
    u_hi_lo = sum([task.u_lo for task in task_set.tasks if task.criticality == 'HI'])

    return u_lo_lo + min(u_hi_hi, u_hi_lo / (1 - u_hi_hi)) <= 1  # sufficient condition


#########################################
# Probabilistic Schedulability Analysis #
#########################################

def p_smc(task_set: TaskSet, thresh_lo: float=1e-4, thresh_hi: float=1e-9):
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
             ignore_hi_mode=False,
             thresh_lo=1e-4, thresh_hi=1e-9,
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
        ignore_hi_mode: If True, LO-task DMP is also assumed to be 0% in HI-mode, leading to an alternative, more 
        generous analysis scheme.
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

    if p_switch == 0:
        expected_switch_time = 1
        total_time = 1
    else:
        expected_switch_time = 1. / p_switch  # Expected number of hyperperiods until mode switch
        total_time = expected_switch_time + hi_mode_duration
    for task in task_set_adj.tasks:
        if task.criticality == 'HI':
            thresh = thresh_hi
            p_failure_hi = 0.
        else:
            thresh = thresh_lo
            p_failure_hi = 1. if not ignore_hi_mode else 0.
        p_failure_lo = 1. - np.prod([1. - job.dmp for job in task_set_adj.jobs if job.task == task])
        if (expected_switch_time / total_time) * p_failure_lo + (hi_mode_duration / total_time) * p_failure_hi > thresh:
            return False
    return True


#######################
# Monte Carlo Schemes #
#######################

def p_smc_monte_carlo(task_set, nhyperperiods=10000, thresh_lo=1e-3, thresh_hi=1e-4):
    """Monte Carlo Simulation for pSMC.
    
    Decides on schedulability by just simulating the system for a large number of hyperperiods. Execution times for
    jobs are drawn independently from their task's PMF, and all deadline misses are counted for each task.

    Args:
        task_set: Mixed-Criticality task set with probabilistic task execution times and fixed priority scheduling.
        nhyperperiods: Sample set size for the simulation. Determines the resolution of the result; DMP probabilites 
            for thresholds smaller than 1 / nhyperperiods can not be registered in a reliable way (DMP is either 0 or
            at least 1 / nhyperperiods.
        thresh_lo: LO-critical tasks have their DMP compared against this float to determine schedulability.
        thresh_hi: HI-critical tasks have their DMP compared against this float to determine schedulability.
    """
    class JobInstance(object):
        def __init__(self, task, release, remaining):
            self.task = task
            self.release = release  # Tuple (hyperperiod of release, release time)
            self.remaining = remaining
            self.abs_deadline = (release[0] + (release[1] + self.task.deadline) // task_set.hyperperiod,
                                 (release[1] + self.task.deadline) % task_set.hyperperiod)

    class BacklogContainer(object):
        """Optimized for backlog push and pop."""
        def __init__(self):
            self.items = [[] for _ in range(len(task_set.tasks))]
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

    def exec_backlog(dt):
        nonlocal t, backlog
        while dt > 0:  # dt will never exceed the time to the next release or the end of the hyperperiod
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
                    if len(done.task.failed_hps) > max_failures[done.task.criticality]:
                        return False  # max allowed failed hyperperiods exceeded
        return True

    max_failures = {'LO': int(thresh_lo * nhyperperiods), 'HI': int(thresh_hi * nhyperperiods)}
    backlog = BacklogContainer()

    for task in task_set.tasks:
        task.exec_times = np.random.choice(a=range(len(task.c_pmf)),
                                           size=nhyperperiods * (task_set.hyperperiod // task.period),
                                           p=task.c_pmf)
        task.failed_hps = set()  # All hyperperiods where an instance of this task missed its deadline
    for i in range(nhyperperiods):
        # Reset timer, build list of all releases in new hyperperiod
        t = 0
        releases = list(task_set.jobs)

        while releases:
            next_rel = releases.pop(0)
            success = exec_backlog(next_rel.release - t)
            if not success:
                return False
            exec_time, next_rel.task.exec_times = next_rel.task.exec_times[-1], next_rel.task.exec_times[:-1]
            backlog.add(JobInstance(task=next_rel.task, release=(i, next_rel.release), remaining=exec_time))
        else:
            success = exec_backlog(task_set.hyperperiod - t)
            if not success:
                return False
    return True


def p_amc_bb_monte_carlo(task_set, hi_mode_duration=1, nhyperperiods=10000, thresh_lo=1e-3, thresh_hi=1e-4):
    """Monte Carlo Simulation for pAMC-BB.
    
    Decides on schedulability by just simulating the system for a large number of hyperperiods. Execution times for
    jobs are drawn independently from their task's PMF, and all deadline misses are counted for each task. Mode switches
    are also simulated; every time a switch happens, the simulation fast-forwards for hi_mode_duration and LO-critical 
    tasks get hi_mode_duration deadline misses added to their counter.

    Args:
        task_set: Mixed-Criticality task set with probabilistic task execution times and fixed priority scheduling.
        hi_mode_duration: Number of hyperperiods the "black-box" takes after a mode switch occurred. Only whole
            hyperperiods possible (integer values)! 
        nhyperperiods: Sample set size for the simulation. Determines the resolution of the result; DMP probabilites 
            for thresholds smaller than 1 / nhyperperiods can not be registered in a reliable way (DMP is either 0 or
            at least 1 / nhyperperiods.
        thresh_lo: LO-critical tasks have their DMP compared against this float to determine schedulability.
        thresh_hi: HI-critical tasks have their DMP compared against this float to determine schedulability.
    """

    class JobInstance(object):
        def __init__(self, task, release, remaining):
            self.task = task
            self.release = release  # Tuple (hyperperiod of release, release time)
            self.remaining = remaining
            self.abs_deadline = (release[0] + (release[1] + self.task.deadline) // task_set.hyperperiod,
                                 (release[1] + self.task.deadline) % task_set.hyperperiod)
            self.c_lo_budget = task.c_lo

    class BacklogContainer(object):
        """Optimized for backlog push and pop."""

        def __init__(self):
            self.reset()

        def __bool__(self):
            return self.size > 0

        def __iter__(self):
            return self.as_list.__iter__()

        def __getitem__(self, item):
            return self.as_list.__getitem__(item)

        def add(self, new):
            self.items[new.task.static_prio].append(new)
            self.size += 1

        def pop(self, k):
            for a in self.items[::-1]:
                if a:
                    self.size -= 1
                    return a.pop(k)

        def reset(self):
            self.items = [[] for _ in range(len(task_set.tasks))]
            self.size = 0

        @property
        def as_list(self):
            return list(itertools.chain.from_iterable(self.items[::-1]))

    def exec_backlog(dt):
        nonlocal t, backlog
        while dt > 0:  # dt will never exceed the time to the next release or the end of the hyperperiod
            if not backlog:
                t += dt
                break
            elif backlog[0].remaining > dt and backlog[0].c_lo_budget > dt:  # job not done yet, no budget overrun
                t += dt
                backlog[0].remaining -= dt
                backlog[0].c_lo_budget -= dt
                break
            elif backlog[0].remaining > dt >= backlog[0].c_lo_budget:  # job not done yet, switch to HI-mode
                t += backlog[0].c_lo_budget
                return -1
            else:  # backlog[0].remaining <= dt --> job complete
                done = backlog.pop(0)
                if done.remaining > done.c_lo_budget:  # budget overrun, switch to HI-mode
                    t += done.c_lo_budget
                    return -1
                t += done.remaining
                dt -= done.remaining
                if done.abs_deadline < (i, t):  # i denotes the i-th hyperperiod
                    done.task.failed_hps.add(done.release[0])
                    if len(done.task.failed_hps) > max_failures[done.task.criticality]:
                        return 1  # max allowed failed hyperperiods exceeded
            # else:  # job complete, switch to HI-mode
            #     pass
        return 0

    max_failures = {'LO': int(thresh_lo * nhyperperiods), 'HI': int(thresh_hi * nhyperperiods)}
    backlog = BacklogContainer()
    for task in task_set.tasks:
        task.exec_times = np.random.choice(a=range(len(task.c_pmf)),
                                           size=nhyperperiods * (task_set.hyperperiod // task.period),
                                           p=task.c_pmf)
        task.failed_hps = set()  # All hyperperiods where an instance of this task missed its deadline
    i = 0
    while i < nhyperperiods:
        # Reset timer, build list of all releases in new hyperperiod
        t = 0
        releases = list(task_set.jobs)

        while releases:
            next_rel = releases.pop(0)
            event = exec_backlog(next_rel.release - t)
            if not event:
                exec_time, next_rel.task.exec_times = next_rel.task.exec_times[-1], next_rel.task.exec_times[:-1]
                backlog.add(JobInstance(task=next_rel.task, release=(i, next_rel.release), remaining=exec_time))
            elif event == 1:  # Task DMP exceeded limit
                return False
            else:  # Job exceeded c_lo budget, switch to HI-mode
                for task in task_set.tasks:
                    if task.criticality == 'LO':
                        task.failed_hps |= set(range(i, i + hi_mode_duration))
                        if len(task.failed_hps) > max_failures['LO']:
                            return False
                # Jump hi_mode_duration ahead, reset backlog, forward to next job release:
                i += hi_mode_duration
                backlog.reset()
                t = next_rel.release
                exec_time, next_rel.task.exec_times = next_rel.task.exec_times[-1], next_rel.task.exec_times[:-1]
                backlog.add(JobInstance(task=next_rel.task, release=(i, next_rel.release), remaining=exec_time))

        event = exec_backlog(task_set.hyperperiod - t)
        if not event:
            pass
        elif event == 1:  # Task DMP exceeded limit
            return False
        else:  # Job exceeded c_lo budget, switch to HI-mode
            for task in task_set.tasks:
                if task.criticality == 'LO':
                    task.failed_hps |= set(range(i, i + hi_mode_duration))
                    if len(task.failed_hps) > max_failures['LO']:
                        return False
            # Jump hi_mode_duration ahead, reset backlog, forward to next hyperperiod:
            i += hi_mode_duration
            backlog.reset()
        i += 1
    return True


if __name__ == '__main__':
    # Plot examples of backlog convergence for different task sets
    t00 = Task(0, 'LO', 6, 6, 2)
    t00.c_pmf = np.array([0.0, 0.5, 0.5])
    t01 = Task(1, 'LO', 8, 8, 2)
    t01.c_pmf = np.array([0.0, 0.5, 0.5])
    t02 = Task(2, 'LO', 12, 12, 3)
    t02.c_pmf = np.array([0.0, (1. / 3), (1. / 3), (1. / 3)])
    ts0 = TaskSet(0, [t00, t01, t02])
    synth.set_fixed_priorities(ts0)
    print(ts0.description)
    plot_backlogs(ts0, add_stationary=True, path='./figures/ex_backlogs0.png')

    t10 = Task(0, 'LO', 6, 6, 3)
    t10.c_pmf = np.array([0.0, 0.0, 0.5, 0.5])
    t11 = Task(1, 'LO', 8, 8, 3)
    t11.c_pmf = np.array([0.0, 0.0, 0.5, 0.5])
    t12 = Task(2, 'LO', 12, 12, 4)
    t12.c_pmf = np.array([0.0, 0.0, (1. / 3), (1. / 3), (1. / 3)])
    ts1 = TaskSet(1, [t10, t11, t12])
    synth.set_fixed_priorities(ts1)
    print(ts1.description)
    plot_backlogs(ts1, add_stationary=True, path='./figures/ex_backlogs1.png')

    t20 = Task(0, 'LO', 6, 6, 4)
    t20.c_pmf = np.array([0.0, 0.0, (1. / 3), (1. / 3), (1. / 3)])
    t21 = Task(1, 'LO', 8, 8, 4)
    t21.c_pmf = np.array([0.0, 0.0, (1. / 3), (1. / 3), (1. / 3)])
    t22 = Task(2, 'LO', 12, 12, 4)
    t22.c_pmf = np.array([0.0, 0.0, (1. / 3), (1. / 3), (1. / 3)])
    ts2 = TaskSet(2, [t20, t21, t22])
    synth.set_fixed_priorities(ts2)
    print(ts2.description)
    plot_backlogs(ts2, add_stationary=False, path='./figures/ex_backlogs2.png')
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