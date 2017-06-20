from taskset_class_lib import Task, TaskSet, WeibullDist
import matplotlib.pyplot as plt
from math import ceil
from math import floor
import numpy as np
import numpy.random as nprd
from time import time


def randfixedsum(n, u, nsets, a, b):
    """
    All courtesy to P. Emberson, R. Stafford, R. Davis 
    for the randfixedsum algorithm and their implementation in Python.
    Paper: http://retis.sssup.it/waters2010/waters2010.pdf#page=6
    Matlab: https://www.mathworks.com/matlabcentral/fileexchange/9700-random-vectors-with-fixed-sum
    Python: https://github.com/brandenburg/schedcat/blob/master/schedcat/generator/generator_emstada.py
    """

    # Check the arguments.
    if (n != round(n)) or (nsets != round(nsets)) or (nsets < 0) or (n < 1):
        print('n must be a whole number and m a non-negative integer.')
    elif (u < n * a) or (u > n * b) or (a >= b):
        print('Inequalities n * a <= u <= nsets * b and a < b must hold.', n, u, nsets, a, b)

    # deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = np.floor(u)
    # s = u  # Staffords Version
    s = (u - n * a) / (b - a)  # Modified to include parameters a and b. (L. Stalder, 2017)
    step = 1 if k < (k-n+1) else -1
    s1 = s - np.arange(k, (k - n + 1) + step, step)
    step = 1 if (k+n) < (k-n+1) else -1
    s2 = np.arange((k + n), (k + 1) + step, step) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, (n+1)):
        tmp1 = w[i - 2, np.arange(1, (i + 1))] * s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * s2[np.arange((n - i), n)] / float(i)
        w[i - 1, np.arange(1, (i + 1))] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, (i + 1))] + tiny
        tmp4 = np.array((s2[np.arange((n - i), n)] > s1[np.arange(0, i)]))
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    m = nsets
    x = np.zeros((n, m))
    if m == 0:
        return
    rt = np.random.uniform(size=(n - 1, m))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, m))  # rand position in simplex
    s = np.repeat(s, m)
    j = np.repeat(int(k + 1), m)
    sm = np.repeat(0, m)
    pr = np.repeat(1, m)

    for i in np.arange(n-1, 0, -1):  # iterate through dimensions
        e = (rt[(n-i)-1, ...] <= t[i-1, j-1]) # decide which direction to move in this dimension (1 or 0)
        sx = rs[(n-i)-1, ...] ** (1/float(i))  # next simplex coord
        sm = sm + (1-sx) * pr * s/float(i+1)
        pr = sx * pr
        x[(n-i)-1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n-1, ...] = sm + pr * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    for i in range(0, m):                            # Adjusted xrange to range for Python 3 (L. Stalder, 2017)
        x[..., i] = x[np.random.permutation(n), i]

    x = (b - a)*x + a  # Rescale and return: (L. Stalder, 2017)
    return np.transpose(x)


def bounded_uniform(u_hi_lo, u_hi_hi, m, n_hi, u_min, u_hi):
    u_lo = []
    arr = sorted(u_hi, reverse=True)
    u_lo_rem = u_hi_lo * m
    u_hi_rem = u_hi_hi * m
    n_hi_rem = n_hi - 1
    for u in arr:
        u_hi_rem -= u
        curr = nprd.uniform(max(u_min, u_lo_rem - u_hi_rem), min((u_lo_rem - (n_hi_rem * u_min)), u))
        n_hi_rem -= 1
        u_lo_rem -= curr
        u_lo.append(curr)
    return u_lo


def mc_fairgen_det(
        set_id,  # Identifier for newly generated task set
        u_lo=None,  # Normalized (per-core) system utilization in LO-mode
        u_hi=None,  # Normalized (per-core) system utilization in HI-mode
        m=1,  # No. of cores
        u_min=0.01,  # Minimum per-task utilization
        u_max=0.99,  # Maximum per-task utilization
        max_tasks=10,
        periods=None,  # List of possible period values (a default is assigned if this is None)
        time_granularity=100,  # Multiplier for smaller discrete time units.
        implicit_deadlines=False
) -> TaskSet:
    """
    Returns a set of tasks, containing at most m*max_tasks.
    
    Can either be used to generate task sets with specific system utilizations or with random values.
    """

    if u_hi is None:
        u_hi_hi = nprd.uniform(low=max_tasks*u_min, high=1.0)
    else:
        u_hi_hi = u_hi

    p_hi = nprd.randint(1, 10) / 10.0  # percentage of hi-criticality tasks
    p_lo = 1. - p_hi

    if u_lo is None:
        u_hi_lo = nprd.uniform(low=max_tasks*p_hi*u_min, high=u_hi_hi)
        u_lo_lo = nprd.uniform(low=max_tasks*p_lo*u_min, high=1 - u_hi_lo)
    else:
        u_hi_lo = nprd.uniform(low=max_tasks*p_hi*u_min, high=min(u_hi_hi, u_lo) - max_tasks*p_lo*u_min)
        u_lo_lo = u_lo - u_hi_lo


    n_min_hi = int(ceil(u_hi_hi * m / u_max))  # minimum required total tasks
    n_min_lo = int(ceil(u_lo_lo * m / u_max))
    n_min = max(m + 1, int(ceil(n_min_hi / p_hi)), int(ceil(n_min_lo / (1 - p_hi))))
    if n_min > max_tasks:
        n_min = max_tasks
    n = nprd.randint(n_min, max_tasks * m + 1)  # total numbers of tasks
    n_hi = max(int(p_hi * n), n_min_hi)
    n_lo = n - n_hi


    # print("u_hi_hi:%f\nu_hi_lo:%f\nu_lo_lo:%f\np_hi:%f\nn_min_hi:%d\nn_min_lo:%d\nn_min:%d\nn:%d\n"
    #       "n_hi:%d\nn_lo:%d" % (u_hi_hi, u_hi_lo, u_lo_lo, p_hi, n_min_hi, n_min_lo, n_min, n, n_hi, n_lo))

    if periods is None:
        periods = [1, 2, 3, 4, 5, 6, 15]  # Small hyperperiods with these period values
    t = [time_granularity * i for i in nprd.choice(a=periods, size=n)]
    utils_hi = randfixedsum(n=n_hi, u=u_hi_hi * m, nsets=1, a=u_min, b=u_max)[0]
    utils_lo = bounded_uniform(u_hi_lo=u_hi_lo, u_hi_hi=u_hi_hi, m=m, n_hi=n_hi, u_min=u_min, u_hi=utils_hi)
    utils_lo.extend(randfixedsum(n=n_lo, u=u_lo_lo * m, nsets=1, a=u_min, b=u_max)[0])

    c_lo = []
    c_hi = []
    d = []

    for i in range(n):
        c_lo.append(floor(utils_lo[i] * t[i]))

    for i in range(n_hi):
        c_hi.append(floor(utils_hi[i] * t[i]))

    if implicit_deadlines:
        d = list(t)
    else:
        for i in range(n_hi):
            d.append(nprd.randint(c_hi[i], t[i]))
        for i in range(n_hi, n):
            d.append(nprd.randint(c_lo[i], t[i]))

    tasks = []
    for i in range(n_hi):
        tasks.append(Task(task_id=i, criticality='HI', period=t[i], u_lo=utils_lo[i], c_lo=c_lo[i],
                          u_hi=utils_hi, c_hi=c_hi[i], deadline=d[i]))

    for i in range(n_hi, n):
        tasks.append(Task(task_id=i, criticality='LO', period=t[i], u_lo=utils_lo[i], c_lo=c_lo[i],
                          deadline=d[i]))
    return TaskSet(set_id, tasks)


def mc_fairgen_stoch(
        set_id,  # Identifier for newly generated task set
        u_lo=None,  # Normalized (per-core) system utilization in LO-mode
        u_hi=None,  # Normalized (per-core) system utilization in HI-mode
        mode='max',  # Setting whether u_lo is meant to be the maximum (c_lo_percent) or average system utilization
        m=1,  # No. of cores
        u_min=0.01,  # Minimum per-task utilization
        u_max=0.99,  # Maximum per-task utilization
        max_tasks=10,
        periods=None,  # List of possible period values (a default is assigned if this is None)
        time_granularity=100,  # Multiplier for smaller discrete time units.
        implicit_deadlines=False,
        distribution_cls=WeibullDist,
        c_lo_percent=0.999, c_hi_percent=0.99999) -> [TaskSet]:  # TODO Docstring
    """
    
    """
    start = time()
    while True:
        ts = mc_fairgen_det(set_id=set_id,
                            u_lo=u_lo,
                            u_hi=u_hi,
                            m=m,
                            u_min=u_min,
                            u_max=u_max,
                            max_tasks=max_tasks,
                            periods=periods,
                            time_granularity=time_granularity,
                            implicit_deadlines=implicit_deadlines
                            )
        for t in ts.tasks:
            if mode == 'max':
                distribution = distribution_cls.from_percentile(x=t.c_lo, percentile=c_lo_percent)
            elif mode == 'avg':
                distribution = distribution_cls.from_ev(ev=t.c_lo)
            t.c_lo = ceil(distribution.percentile(p=c_lo_percent))
            if t.c_lo > t.deadline:
                break
            if t.criticality == 'HI':
                t.c_hi = ceil(distribution.percentile(p=c_hi_percent))
                if t.c_hi > t.deadline:
                    break
            cutoff = c_hi_percent if t.criticality == 'HI' else c_lo_percent
            t.c_pdf = distribution.discrete_pd(cutoff=cutoff)
        else:
            stop = time()
            return ts


if __name__ == '__main__':
    elapsed = 0
    for k in range(10):
        start = time()
        taskset = mc_fairgen_stoch(set_id=0, u_lo=0.9, mode='max', implicit_deadlines=False,
                                   u_min=0.01, max_tasks=10, time_granularity=100)
        stop = time()
        elapsed += stop - start
    print(elapsed)
    taskset.draw(scale='log')

"""
ts = 
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