"""
This module contains any methods and functions related to the analysis of a task set's schedulability, deadline miss
probability and so forth. It makes use of the classes defined in module class_lib.

-- Luca Stalder, 2017
"""

import collections
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import sympy as sp
from class_lib import TaskSet
import generation as gen


class BacklogSim(object):
    """Backlog simulation class.
    
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
        self.p = p_level
        self.k = 0
        self.t = 0
        self.backlog = np.array([1.0]) if initial_backlog is None else initial_backlog  # TODO Build custom container

        # Build timeline of one full hyperperiod
        self.timeline = [rel for rel in task_set.job_releases if rel.task.priority >= self.p]
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
            elif dt < self.jobs_remaining[0].t - self.t or dt == self.jobs_remaining[0].t - self.t and mode == 'before':
                # Step does not reach or convolve next job release
                self.backlog = shrink(self.backlog, dt)
                self.t += dt
                return
            else:  # Step to next job release
                next_job = self.jobs_remaining.pop(0)
                self.backlog = shrink(self.backlog, next_job.t - self.t)
                self.backlog = convolve_pdf(self.backlog, next_job.task.c_pdf)
                dt -= next_job.t - self.t
                self.t = next_job.t


def scheduling_analysis_smc(task_set: TaskSet):
    """SMC Response Time Analysis, as described in [2]."""

    def min_crit(c1, c2):
        """Returns the lower of both criticality levels."""
        if c1 == 'HI' and c2 == 'HI':
            return 'HI'
        else:
            return 'LO'

    for task in task_set.tasks:
        # 1) Build set of tasks with higher priority:
        hp = []
        for j in task_set.tasks:
            if j.priority > task.priority:
                hp.append(j)

        # 2) Iteration to fixed point for solving recurrence relation:
        task.r = 0
        r_next = -1
        while task.r != r_next:
            task.r = r_next
            if task.r > task.deadline:  # task.r growing monotonically
                return False
            sum_hp = task.c[task.criticality]
            for j in hp:
                sum_hp += math.ceil(task.r / j.period) * j.c[min_crit(task.criticality, j.criticality)]
            r_next = sum_hp

    # No deadline overruns happened:
    return True


def scheduling_analysis_amc(task_set):
    """AMC-rtb Response Time Analysis, as described in [2]."""
    for task in task_set.tasks:
        # 1.1) Build set of all tasks with higher priority:
        hp = []
        for j in task_set.tasks:
            if j.priority > task.priority:
                hp.append(j)
        # 1.2) Build sets of all HI- and LO-critical tasks with higher priority:
        hp_hi = [t for t in hp if t.criticality == 'HI']
        hp_lo = [t for t in hp if t.criticality == 'LO']

        # 2) Iteration to fixed point for solving recurrence relation:
        # 2.1) R_LO:
        task.r_lo = 0
        r_next = -1
        while task.r_lo != r_next:
            task.r_lo = r_next
            if task.r_lo > task.deadline:
                return False
            sum_hp = task.c[task.criticality]
            for j in hp:
                sum_hp += math.ceil(task.r_lo / j.period) * j.c['LO']
            r_next = sum_hp

        # 2.2 R_HI (only defined for HI-critical tasks):
        if task.criticality == 'HI':
            task.r_hi = 0
            r_next = -1
            while task.r_hi != r_next:
                task.r_hi = r_next
                if task.r_hi > task.deadline:
                    return False
                sum_hp = task.c['HI']
                for j in hp_hi:
                    sum_hp += math.ceil(task.r_hi / j.period) * j.c['HI']
                r_next = sum_hp

        # 2.3 R_* (criticality change, only defined for HI-critical tasks):
        if task.criticality == 'HI':
            task.r_asterisk = 0
            r_next = -1
            while task.r_asterisk != r_next:
                task.r_asterisk = r_next
                if task.r_asterisk > task.deadline:
                    return False
                sum_hp = task.c['HI']
                for j in hp_hi:
                    sum_hp += math.ceil(task.r_asterisk / j.period) * j.c['HI']
                for k in hp_lo:
                    sum_hp += math.ceil(task.r_lo / k.period) * k.c['LO']
                r_next = sum_hp

    # No deadline overruns happened:
    return True


def scheduling_analysis_edf_vd(task_set: TaskSet):
    """Calculated according to theorem 1 in [3]."""

    def total_k_utilization(task_subset, k):
        """Total utilization at scenario criticality level k of tasks contained in task_subset."""
        result = 0.0
        for j in task_subset:
            result += float(j.c[k]) / float(j.period)
        return result

    subset_lo = [t for t in task_set.tasks if t.criticality == 'LO']
    subset_hi = [t for t in task_set.tasks if t.criticality == 'HI']
    u_1_1 = total_k_utilization(subset_lo, 'LO')
    u_2_1 = total_k_utilization(subset_hi, 'LO')
    u_2_2 = total_k_utilization(subset_hi, 'HI')

    return u_1_1 + min(u_2_2, u_2_1 / (1 - u_2_2)) <= 1     # Note that this condition is sufficient for schedulability.


def stationary_backlog_iter(task_set: TaskSet, p_level=0, max_iter=200, epsilon=1e-14):
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
        An array representing the probability distribution over the task set's stationary backlog at the beginning of 
        its hyperperiod.
    """
    sim = BacklogSim(task_set=task_set, p_level=p_level)
    last = sim.backlog
    dist = 1.
    i = 0
    while dist > epsilon and i <= max_iter or i < 10:
        sim.step(dt=task_set.hyperperiod, mode='before')
        dist = quadratic_difference(last, sim.backlog)
        print(dist)
        last = sim.backlog
        i += 1
    if i > max_iter:
        print("Diverging backlog.")
        return
    else:
        print("Convergence after %d iterations." % i)
        return last


def stationary_backlog_analytic(task_set: TaskSet, p_level=0):  # TODO Incomplete
    """Exact solution for finding a task set's stationary backlog.
    
    This method finds the stationary backlog at the beginning of a task set's hyperperiod by exact calculation, which
    involves finding its Markov chain matrix. For further details refer to [4].
    
    Args:
        task_set: The task set in consideration.
        p_level: The desired priority level of the resulting backlog. Only tasks with priority of at least p_level are 
            taken into consideration.
            
    Returns:
        An array representing the probability distribution over the task set's stationary backlog at the beginning of 
        its hyperperiod.    
    """
    w_min = 0  # Backlog after first hyperperiod assuming all jobs require minimal execution time.
    timeline = task_set.job_releases
    t = 0
    while timeline:
        next_job = timeline.pop(0)
        w_min = max(w_min - (next_job.t - t), 0)  # 'Shrink'
        w_min += next_job.task.c_min()  # 'Convolve'
        t = next_job.t
    w_min = max(w_min - (task_set.hyperperiod - t), 0)

    r = task_set.hyperperiod + w_min - sum([task.c_min() * len(task.rel_times) for task in task_set.tasks
                                            if task.priority >= p_level])  # max idle time in any hyperperiod
    sim = BacklogSim(task_set, p_level, initial_backlog=np.array([0.0] * r + [1.0]))
    sim.step(dt=task_set.hyperperiod, mode='before')
    m_r = len(sim.backlog) - 1  # max possible backlog for initial backlog r

    first_backlogs = []
    print(r, m_r)
    for i in range(r + 1):  # TODO Add multi-threading
        sim.__init__(task_set, p_level, initial_backlog=np.array([0.0] * i + [1.0]))
        sim.step(task_set.hyperperiod, mode='before')
        first_backlogs.append(np.concatenate((sim.backlog, np.zeros(m_r - len(sim.backlog) + 1))))
        print(i)

    for i in range(1, m_r + 1):
        first_backlogs.append(np.array([0.0] * i + first_backlogs[r][:-i].tolist()))
    p_upper_left = np.array(first_backlogs).T  # (m_r + 1) x (r + 1) matrix in the upper left corner of P
    p_upper_left -= np.eye(p_upper_left.shape[0], p_upper_left.shape[1])  # Equate everything to zero

    print(p_upper_left.shape)

    unknowns = sp.symbols('p0:%d' % (m_r + r + 1))  # Tuple of unknowns
    equations = p_upper_left.dot(unknowns)  # First subset of (m_r + 1) equations  # TODO Too slow

    print(type(equations), equations.shape)

    a_matrix = np.zeros((m_r, m_r))
    a_matrix[:-1, 1:] = np.eye(m_r - 1)
    a_matrix[-1, :] = (p_upper_left[-1:0:-1, r] / -p_upper_left[0, r]).T

    D, V = np.linalg.eig(a_matrix)
    print(D, D.shape)
    print(V, V.shape)


def dmp_analysis_fp(task_set: TaskSet):
    """
    Deadline miss probability analysis for fixed priority scheduled task sets.
    
    This method calculates the response time distribution for every job during one hyperperiod, assuming steady state
    backlog. The average over all response time distributions is then taken over all jobs belonging to one task,
    resulting in the average response time distribution of this task.
    
    Deadline miss probability per task is then given by 1 - sum(average response time distribution), since only response
    times up to and including the task's deadline are kept.
     
    Note that the method does not return anything, and instead writes the results directly to the corresponding Task
    objects.
    
    Args:
        task_set: The task set under consideration. Note that this has to have set (fixed) task priorities.
    """
    sims = [None] * len(task_set.tasks)
    for task in task_set.tasks:
        if sims[task.priority] is None:
            backlog = stationary_backlog_iter(task_set, task.priority)
            sims[task.priority] = BacklogSim(task_set, task.priority, backlog)

    # Job Response Time Analysis
    Job = collections.namedtuple('Job', ['rel_time', 'task'])
    jobs = [Job(rel_time, task) for rel_time, task in task_set.job_releases]
    job_responses = []
    for job in jobs:
        sim = sims[job.task.priority]
        sim.step(job.rel_time - sim.t, mode='before')
        part_response = convolve_pdf(sim.backlog, job.task.c_pdf)
        preempts = [preempt for preempt in jobs
                    if job.rel_time <= preempt.rel_time
                    and job.task.priority < preempt.task.priority
                    and preempt.rel_time < job.rel_time + job.task.deadline]
        while preempts:
            next_preempt = preempts.pop(0)
            head, tail, _ = np.split(part_response,
                                     [next_preempt.rel_time - job.rel_time + 1, job.task.deadline + 1])
            tail_lim = job.task.deadline - head.size + 1
            if tail.size:
                tail = sig.convolve(tail, next_preempt.task.c_pdf[:tail_lim])[:tail_lim]
            part_response = np.concatenate((head, tail))
        part_response.resize(job.task.deadline + 1)
        job_responses.append((job, part_response))

    for task in task_set.tasks:
        task_responses = [res for j, res in job_responses if j.task == task]
        task.avg_response = np.sum(task_responses, axis=0) / len(task_responses)
        task.dmp = 1. - np.sum(task.avg_response)


def plot_backlogs(task_set, p_level=0, n_subplots = 9, add_stationary=False, scale='log'):
    """Plots the first n_subplots backlogs of task_set, considering only tasks of at least p_level priority."""
    fig = plt.figure()
    sim = BacklogSim(task_set, p_level)
    if add_stationary:
        stationary = stationary_backlog_iter(ts)
        xlim = len(stationary)
    else:
        sim.step(task_set.hyperperiod * n_subplots)
        xlim = len(sim.backlog)
    sim.__init__(task_set, p_level)
    for i in range(n_subplots):
        sim.step(task_set.hyperperiod)
        plt.subplot(n_subplots + 2 // 3, 3, i + 1)
        plt.bar(range(len(sim.backlog)), sim.backlog)
        plt.xlim(0, xlim)
        plt.yscale(scale)
        plt.title("Hyperperiods: %d" % sim.k)
    if add_stationary:
        plt.subplot(n_subplots + 2 // 3, 3, n_subplots + 1)
        plt.bar(range(len(stationary)), stationary)
        plt.xlim(0, xlim)
        plt.yscale(scale)
        plt.title("Stationary backlog")
    plt.subplots_adjust(hspace=0.8)
    plt.show()


def animate_backlog(ts):  # TODO Fix animation (copy into separate script)
    sim = BacklogSim(task_set=ts)
    dt = ts.hyperperiod // 300  # 30 fps
    dt = 1
    xrange = len(sim.backlog)
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, xrange), ylim=(10 ** -6, 1.))
    plt.yscale('log')
    line, = ax.plot([0], [1.0])
    k_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        k_text.set_text('')
        time_text.set_text('')
        return line, k_text, time_text

    def animate(i):
        global sim, dt, xrange, ax
        sim.step(dt, mode='before')
        xrange = max(xrange, len(sim.backlog))
        print(xrange)
        ax.set_xlim(0, xrange)
        x_val = range(len(sim.backlog))
        y_val = sim.backlog
        line.set_data(x_val, y_val)
        k_text.set_text('hyperperiod = %d' % sim.k)
        time_text.set_text('time = %d' % sim.t)
        return line, k_text, time_text

    ani = animation.FuncAnimation(fig, animate, frames=20 * ts.hyperperiod, interval=500, blit=False, init_func=init)
    plt.show()


def convolve_pdf(a, b, percentile=1 - 1e-14):
    """Convolution of two discrete probability distribution functions."""
    # Convolve
    conv = sig.convolve(a, b)

    # Crop
    i = 0
    psum = 0.0
    while psum < percentile and i < len(conv):
        psum += conv[i]
        i += 1
    conv = conv[:i]

    # Rescale
    return conv / sum(conv)


def shrink(pdf, t):
    """Shrinking of PDF c by t time units."""

    if t >= len(pdf):
        return np.array([sum(pdf)])
    else:
        result = pdf[t:]
        result[0] += sum(pdf[:t])
        return result


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


def total_variation_distance(p: np.ndarray, q: np.ndarray):
    """Total variation distance between arrays p and q."""
    if len(p) <= len(q):
        a = np.array(p)
        b = np.array(q)
    else:
        a = np.array(q)
        b = np.array(p)
    a.resize(len(b))
    return max([abs(x - y) for x, y in zip(a, b)])

if __name__ == '__main__':
    # ts = gen.mc_fairgen_stoch(set_id=0, u_lo=0.95, mode='avg', max_tasks=10)
    ts = gen.dummy_taskset()
    ts.set_priorities_rm()
    ts.set_rel_times_fp()
    # ts.draw()
    # print(stationary_backlog_iter(task_set=ts))
    # stationary_backlog_analytic(ts, p_level=0)
    # plot_backlogs(ts, add_stationary=True, scale='log')
    dmp_analysis_fp(ts)


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