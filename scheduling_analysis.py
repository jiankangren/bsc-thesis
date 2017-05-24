from taskset_class_lib import Task, TaskSet
from math import ceil
import taskset_generation as gen
import scipy.signal as sig
import matplotlib.pyplot as plt
from operator import itemgetter

def dummy_taskset():
    t1 = Task(0, 'HI', 15, 10, 9, 0.6, 6, 0.4, 0)
    t2 = Task(1, 'LO', 12, 8, None, None, 7, 0.583, 0)
    t1.c_pdf = [0.0, 0.0, 0.0, 0.2, 0.4, 0.2, 0.1, 0.05, 0.03, 0.02]
    t2.c_pdf = [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
    ts = TaskSet(0, {0: t1, 1: t2})
    ts.assign_priorities_rm()
    ts.assign_rel_times()
    ts.draw()
    return ts

def initialize_task_set(path):
    """Reads task set from path."""
    # Sample task set:
    return gen.mc_fairgen_modified(n_sets=1)


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
    if scheme == 'smc':
        return scheduling_analysis_smc(task_set)
    elif scheme == 'amc':
        return scheduling_analysis_amc(task_set)
    elif scheme == 'edf-vd':
        return scheduling_analysis_edf_vd(task_set)
    # Add new scheduling schemes here.
    else:
        pass


def convolve_and_crop(a, b, percentile=0.99999):
    """"""
    # Convolve
    conv = sig.convolve(a, b)

    # Crop
    i = 0
    psum = 0.0
    while psum < percentile:
        psum += conv[i]
        i += 1
    conv = conv[:i]

    # Rescale
    return conv / sum(conv)


def shrink(pdf, t):
    """Shrinking of PDF c by t time units."""

    if t >= len(pdf):
        return [sum(pdf)]
    else:
        result = pdf[t:]
        result[0] += sum(pdf[:t])
        return result


def p_level_backlog(taskset: TaskSet, p: int=0, n: int=0, t: int=0) -> [float]:
    """
    Iterative calculation.
    :param taskset: Needs assigned job release times.
    :param p: Priority level.
    :param n: The n-th hyperperiod is considered.
    :param t: Time inside the considered hyperperiod.
    :return: Returns PDF of the P-level backlog at time t during hyperperiod k.
    """
    # Build timeline of one full hyperperiod.
    timeline = []
    for key, item in taskset.tasks.items():
        timeline.extend([{'t': t, 'key': key} for t in item.rel_times if taskset.tasks[key].priority >= p])
    timeline.sort(key=lambda x: x['t'])
    print("Timeline:", timeline)

    backlog = [1.0]

    for k in range(n):
        # Start scanning through this hyperperiod.
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
            backlog = convolve_and_crop(backlog, taskset.tasks[next_rel['key']].c_pdf)
        else:
            backlog = shrink(backlog, taskset.hyperperiod - now)
    return backlog

"""
ts = gen.generate_tasksets_stoch(n_sets=1)[0]
while ts.u_lo < 0.9:
    ts = gen.generate_tasksets_stoch(n_sets=1)[0]
ts.assign_priorities_rm()
ts.assign_rel_times()
ts.draw()
"""
ts = dummy_taskset()
log = p_level_backlog(taskset=ts, n=1)
print(log)
log = p_level_backlog(taskset=ts, n=20)
print(log)
plt.bar(range(len(log)), log)
plt.show()

# print(scheduling_analysis("", 'smc'))
# print(scheduling_analysis("", 'amc'))

"""
Literature:
[1] Audsley, Burns, Richardson, Tindell, Wellings
    Applying new scheduling theory to static priority preemptive scheduling.

[2] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
    
[3] Baruah, Bonifaci, D'Angelo, Marchetti-Spaccamela, van der Ster, Stougie
    Mixed-Criticality Scheduling of Sporadic Task Systems
"""