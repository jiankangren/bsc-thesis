from taskset_class_lib import Task, TaskSet
from math import ceil
import numpy as np
import sympy as sp
import taskset_generation as gen
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time


class BacklogSim(object):
    """Backlog simulation class
    
    Can be used for iterative backlog analysis of stochastic task sets.
    
    Attributes:
        task_set: The TaskSet object to be analyzed. This needs priorities and release times assigned. 
        p: The backlog's P-level, meaning only jobs of priority level P or higher are considered.
        k: Number of completed hyperperiods.
        t: Elapsed time in current hyperperiod.
        backlog: The current backlog distribution, stored as a list.    
    """

    def __init__(self, task_set, p=0, initial_backlog=None):
        """Initializes new simulation object."""
        self.task_set = task_set
        self.p = p
        self.k = 0
        self.t = 0
        self.backlog = np.array([1.0]) if initial_backlog is None else initial_backlog  # TODO Build custom container

        # Build timeline of one full hyperperiod
        self.timeline = []
        for task in self.task_set.tasks:
            self.timeline.extend([{'t': t, 'task': task} for t in task.rel_times if task.priority >= self.p])
        self.timeline.sort(key=lambda x: x['t'])
        self.jobs_remaining = list(self.timeline)  # Store a copy

    def step(self, dt=None, mode='after'):
        """Advance the model.
        
        Advances the model by dt time units. Perform convolution and shrinking where necessary.
         
         Args:
             dt: Size of the step. If no dt is given, step to the next job release and/or end of the hyperperiod.
             mode: Determines behavior if step lands exactly on a job release.
                'after' performs convolution on any job releases at the reached time, while 'before' does not.
        """
        start = time()
        if dt is None:
            if not self.jobs_remaining:
                dt = self.task_set.hyperperiod - self.t  # Step to end of hyperperiod if no remaining jobs
            else:
                dt = self.jobs_remaining[0]['t'] - self.t

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
                    stop = time()
                    # print("step:", stop - start)
                    start = time()
                    continue
            elif dt < self.jobs_remaining[0]['t'] - self.t or dt == self.jobs_remaining[0]['t'] and mode == 'before':
                # Step does not reach or convolve next job release
                self.backlog = shrink(self.backlog, dt)
                self.t += dt
                return
            else:  # Step to next job release
                next_job = self.jobs_remaining.pop(0)
                self.backlog = shrink(self.backlog, next_job['t'] - self.t)
                self.backlog = convolve_and_crop(self.backlog, next_job['task'].c_pdf)
                dt -= next_job['t'] - self.t
                self.t = next_job['t']


def dummy_taskset():
    t1 = Task(task_id=0, criticality='HI', period=4, deadline=4, u_lo=0.25, c_lo=1, u_hi=0.5, c_hi=2, phase=0)
    t2 = Task(task_id=1, criticality='LO', period=6, deadline=6, u_lo=0.667, c_lo=4, u_hi=None, c_hi=None, phase=0)
    t1.c_pdf = np.array([0.0, 0.5, 0.5])
    t2.c_pdf = np.array([0.0, 0.0, 0.2, 0.3, 0.5])
    ts = TaskSet(0, [t1, t2])
    ts.assign_priorities_rm()
    ts.assign_rel_times()
    return ts


def initialize_task_set(path):
    """Reads task set from path."""
    # Sample task set:
    return gen.mc_fairgen_det(n_sets=1)


def scheduling_analysis_smc(task_set: TaskSet):
    """SMC Response Time Analysis, as mentioned in [2]."""

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
                sum_hp += ceil(task.r / j.period) * j.c[min_crit(task.criticality, j.criticality)]
            r_next = sum_hp

    # No deadline overruns happened:
    return True


def scheduling_analysis_amc(task_set):
    """AMC-rtb Response Time Analysis, as mentioned in [2]."""
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
                sum_hp += ceil(task.r_lo / j.period) * j.c['LO']
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
                    sum_hp += ceil(task.r_hi / j.period) * j.c['HI']
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
                    sum_hp += ceil(task.r_asterisk / j.period) * j.c['HI']
                for k in hp_lo:
                    sum_hp += ceil(task.r_lo / k.period) * k.c['LO']
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


def scheduling_analysis(task_set_path, scheme):
    """Wrapper function"""

    task_set = initialize_task_set(task_set_path)
    return scheme(task_set)


def convolve_and_crop(a, b, percentile=1-1e-14):
    """"""
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


def p_level_backlog(task_set: TaskSet, p: int=0, n: int=0, t: int=0) -> [float]:
    """
    Iterative calculation.
    :param task_set: Needs assigned job release times.
    :param p: Priority level.
    :param n: The n-th hyperperiod is considered.
    :param t: Time inside the considered hyperperiod.
    :return: Returns PDF of the P-level backlog at time t during hyperperiod k.
    """
    # Build timeline of one full hyperperiod
    timeline = []
    for task in task_set.tasks:
        timeline.extend([{'t': t, 'task': task} for t in task.rel_times if task.priority >= p])
    timeline.sort(key=lambda x: x['t'])

    backlog = [1.0]

    for k in range(n):
        # Start scanning through this hyperperiod.
        start = time()
        now = 0
        remaining = list(timeline)

        while len(remaining) > 0:
            next_rel = remaining.pop(0)
            if next_rel['t'] > now:

                if False: # and next_rel['t'] > t:
                    shrink(backlog, t - now)
                    return backlog
                else:
                    backlog = shrink(backlog, next_rel['t'] - now)
                    now = next_rel['t']
            backlog = convolve_and_crop(backlog, next_rel['task'].c_pdf)
        else:
            backlog = shrink(backlog, task_set.hyperperiod - now)
        stop = time()
        print(k, stop - start)
    return backlog


def steady_state_iter(task_set: TaskSet, max_iter=200, epsilon=1e-14):
    sim = BacklogSim(task_set=task_set, p=0)
    last = list(sim.backlog)
    dist = 1.
    i = 0
    while dist > epsilon and i <= max_iter or i < 10:
        sim.step(dt=task_set.hyperperiod, mode='before')
        dist = total_variation_distance(last, sim.backlog)
        print(i, dist)
        # print(sim.backlog)
        last = list(sim.backlog)
        i += 1
    if i > max_iter:
        print("Diverging backlog.")
        return
    else:
        print("Convergence after %d iterations." % i)
        return last


def stationary_backlog(task_set: TaskSet, p_level=0):  # TODO Docstring
    """"""
    w_min = 0  # Backlog after first hyperperiod assuming all jobs require minimal execution time.
    timeline = []
    for task in task_set.tasks:  # Build p-level timeline of one full hyperperiod
        timeline.extend([{'t': t, 'task': task} for t in task.rel_times if task.priority >= p_level])
    timeline.sort(key=lambda x: x['t'])
    t = 0
    while timeline:
        next_job = timeline.pop(0)
        w_min = max(w_min - (next_job['t'] - t), 0)  # 'Shrink'
        w_min += next_job['task'].c_min()  # 'Convolve'
        t = next_job['t']
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


    # a_matrix = np.eye(m_r - 1)
    # a_matrix = a_matrix.col_insert(0, sp.zeros(m_r - 1, 1))
    # a_matrix = a_matrix.row_insert(m_r, (p_upper_left[-1:0:-1, r] / -p_upper_left[0, r]).T)
    # P, D = a_matrix.diagonalize()
    # print(P)
    # print(D)



""""
ts = gen.generate_tasksets_stoch(n_sets=1)[0]
while ts.u_lo < 0.9:
    ts = gen.generate_tasksets_stoch(n_sets=1)[0]
ts.assign_priorities_rm()
ts.assign_rel_times()
ts.draw()
"""

# ts = gen.mc_fairgen_stoch(set_id=0, u_lo=1.0, mode='avg')
# ts = dummy_taskset()
# ts.assign_priorities_rm()
# ts.assign_rel_times()
# ts.draw()
# start = time()
# log = p_level_backlog(task_set=ts, n=10)
# stop = time()
# print("p_level_backlog:", stop - start)
# plt.plot(range(len(log)), log, 'o')
# start = time()
# sim = BacklogSim(task_set=ts)
# sim.step(ts.hyperperiod, mode='before')
# stop = time()
# print("BacklogSim:", stop - start)
# plt.plot(range(len(sim.backlog)), sim.backlog, 'o')
# start = time()
# sim = BacklogSim(task_set=ts)
# sim.step(ts.hyperperiod, mode='after')
# stop = time()
# print("BacklogSim:", stop - start)
# plt.plot(range(len(sim.backlog)), sim.backlog, 'o')
# plt.yscale('log')
# plt.show()


# print(scheduling_analysis("", 'smc'))
# print(scheduling_analysis("", 'amc'))
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

if __name__ == '__main__':
    # ts = gen.mc_fairgen_stoch(set_id=0, u_lo=0.9, mode='avg', max_tasks=10)
    ts = dummy_taskset()
    ts.assign_priorities_rm()
    ts.assign_rel_times()
    # ts.draw()
    print(steady_state_iter(task_set=ts))
    # stationary_backlog(ts, p_level=0)



"""
Literature:
[1] Audsley, Burns, Richardson, Tindell, Wellings
    Applying new scheduling theory to static priority preemptive scheduling.

[2] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
    
[3] Baruah, Bonifaci, D'Angelo, Marchetti-Spaccamela, van der Ster, Stougie
    Mixed-Criticality Scheduling of Sporadic Task Systems
"""