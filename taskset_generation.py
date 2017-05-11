from taskset_class_lib import Task, TaskSet
from math import ceil
from math import floor
import numpy as np
import numpy.random as nprd
from StaffordRandFixedSum import StaffordRandFixedSum as randfixedsum


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


def mc_fairgen_modified(m=1, u_min=0.0001, u_max=0.99, n_sets=1000, max_tasks=20, time_granularity=100, implicit_deadlines=True) -> [TaskSet]:
    """Returns a list of n_sets task sets, each containing at most m*max_tasks."""
    for k in range(n_sets):
        # utilizations of hi- and lo-criticality tasks:
        u_hi_hi = nprd.randint(1, 11) / 10.0
        u_hi_lo = nprd.choice(np.arange(0.05, u_hi_hi, 0.1).tolist() + [u_hi_hi])
        u_lo_lo = nprd.choice(np.arange(0.05, 1 - u_hi_lo, 0.1).tolist() + [1 - u_hi_lo])

        # percentage of hi-criticality tasks:
        p_hi = nprd.randint(1, 10) / 10.0

        # minimum required total tasks:
        n_min_hi = int(ceil(u_hi_hi * m / u_max))
        n_min_lo = int(ceil(u_lo_lo * m / u_max))
        n_min = max(m + 1, int(ceil(n_min_hi / p_hi)), int(ceil(n_min_lo / (1 - p_hi))))

        # total numbers of tasks:
        n = nprd.randint(n_min, max_tasks * m + 1)
        n_hi = max(int(p_hi * n), n_min_hi)
        n_lo = n - n_hi

        #
        t = []

        for i in range(n):
            t.append((2 ** nprd.randint(2, 8)) * time_granularity)

        u_hi = randfixedsum(n_hi, u_hi_hi * m, 1)[0]
        u_lo = bounded_uniform(u_hi_lo, u_hi_hi, m, n_hi, u_min, u_hi)
        u_lo.extend(randfixedsum(n_lo, u_lo_lo * m, 1)[0])

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

        taskset = []
        for i in range(n_hi):
            taskset.append(Task(criticality='HI', period=t[i], c_hi=c_hi[i], c_lo=c_lo[i], deadline=d[i]))

        for i in range(n_hi, n):
            taskset.append(Task(criticality='LO', period=t[i], c_lo=c_lo[i], deadline=d[i]))
        nprd.shuffle(taskset)
        return taskset

ts = mc_fairgen_modified(n_sets=1, implicit_deadlines=False)
for t in ts:
    print(t.criticality, t.period, t.c['LO'], t.c['HI'], t.deadline)
