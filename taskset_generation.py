from taskset_class_lib import Task, TaskSet, WeibullDist
import matplotlib.pyplot as plt
from math import ceil
from math import floor
import numpy as np
import numpy.random as nprd
from StaffordRandFixedSum import StaffordRandFixedSum_modified as randfixedsum

periods = [1, 2, 3, 4, 5, 6, 12, 15]


def bounded_uniform(u_hi_lo, u_hi_hi, m, n_hi, u_min, u_hi):
    u_lo = []
    arr = sorted(u_hi, reverse=True)
    u_lo_rem = u_hi_lo * m
    u_hi_rem = u_hi_hi * m
    n_hi_rem = n_hi - 1
    for u in u_hi:
        u_hi_rem -= u
        curr = nprd.uniform(max(u_min, u_lo_rem - u_hi_rem), min((u_lo_rem - (n_hi_rem * u_min)), u))
        n_hi_rem -= 1
        u_lo_rem -= curr
        u_lo.append(curr)
    return u_lo


def mc_fairgen_modified(m=1, u_min=0.001, u_max=0.99, max_tasks=20, time_granularity=1000, implicit_deadlines=False) \
        -> {Task}:
    """Returns a list of tasks, containing at most m*max_tasks."""
    # utilizations of hi- and lo-criticality tasks:
    u_hi_hi = nprd.randint(1, 11) / 10.0
    u_hi_lo = nprd.choice(np.arange(0.05, u_hi_hi, 0.1))
    u_lo_lo = nprd.choice(np.arange(0.05, 1.1 - u_hi_lo, 0.1))

    # percentage of hi-criticality tasks:
    p_hi = nprd.randint(1, 10) / 10.0

    # minimum required total tasks:
    n_min_hi = int(ceil(u_hi_hi * m / u_max))
    n_min_lo = int(ceil(u_lo_lo * m / u_max))
    n_min = max(m + 1, int(ceil(n_min_hi / p_hi)), int(ceil(n_min_lo / (1 - p_hi))))

    # total numbers of tasks:
    n = nprd.randint(n_min, max_tasks * m + 1)
    #n = 10
    n_hi = max(int(p_hi * n), n_min_hi)
    n_lo = n - n_hi

    t = [time_granularity * i for i in nprd.choice(a=periods, size=n)]
    u_hi = randfixedsum(n=n_hi, u=u_hi_hi * m, nsets=1, a=u_min, b=u_max)[0]
    u_lo = bounded_uniform(u_hi_lo, u_hi_hi, m, n_hi, u_min, u_hi)
    u_lo.extend(randfixedsum(n=n_lo, u=u_lo_lo * m, nsets=1, a=u_min, b=u_max)[0])

    c_lo = []
    c_hi = []
    d = []

    for i in range(n):
        c_lo.append(floor(u_lo[i] * t[i]))

    for i in range(n_hi):
        c_hi.append(floor(u_hi[i] * t[i]))

    if implicit_deadlines:
        d = list(t)
    else:
        for i in range(n_hi):
            d.append(nprd.randint(c_hi[i], t[i]))
        for i in range(n_hi, n):
            d.append(nprd.randint(c_lo[i], t[i]))

    taskset = {}
    for i in range(n_hi):

        taskset[i] = Task(task_id=i, criticality='HI', period=t[i], c_hi=c_hi[i], c_lo=c_lo[i], deadline=d[i],
                          u_lo=u_lo_lo + u_hi_lo, u_hi=u_hi_hi)

    for i in range(n_hi, n):
        taskset[i] = Task(task_id=i, criticality='LO', period=t[i], c_lo=c_lo[i], deadline=d[i],
                          u_lo=u_lo_lo + u_hi_lo)
    return taskset


def generate_tasksets_det(n_sets=1000) -> [TaskSet]:
    """Generates deterministic task sets."""
    tasksets = []
    for k in range(n_sets):
        tasksets.append(TaskSet(k, mc_fairgen_modified()))
    return tasksets


def generate_tasksets_stoch(n_sets=1000, m=1, c_lo_percentile=0.999, c_hi_percentile=0.99999) -> [TaskSet]:
    """"""
    tasksets = []
    for k in range(n_sets):
        while True:
            ts = mc_fairgen_modified(m=m)
            u_hi = 0.0
            for task in ts.values():
                gamma = nprd.uniform(1.1, 3)
                weib = WeibullDist(gamma=gamma)
                weib.rescale_beta(task.c['LO'], c_lo_percentile)
                if task.criticality == 'HI':
                    task.c['HI'] = ceil(weib.percentile(c_hi_percentile))
                    u_hi += float(task.c['HI']) / task.period
                task.c_pdf = weib.discrete_pd(bound=ceil(weib.percentile(c_hi_percentile)))
                psum = sum(task.c_pdf)
                task.c_pdf = [p / psum for p in task.c_pdf]
            if u_hi <= m * 1.0 and not [t for t in ts.values() if t.criticality == 'HI' and t.c['HI'] > t.deadline]:
                tasksets.append(TaskSet(set_id=k, tasks=ts))
                break
    return tasksets


"""
ts = generate_tasksets_stoch(n_sets=100)
for x in ts:
    print(x.description, x.hyperperiod)
    x.assign_priorities_rm()
    x.assign_rel_times()
    for t in x.tasks.values():
        print(t.period, t.rel_times)
        if t.criticality == 'HI' and t.c['LO'] > t.c['HI'] \
                or t.c['LO'] == 0:
                print('!!! ' + t.criticality, t.period, t.c['LO'], t.c['HI'], t.deadline, t.u_lo, t.u_hi, sum(t.c_pdf))
ts[0].draw()
"""