"""
This module offers various methods and functions for the synthesi of fair deterministic and stochastic task sets.
It makes use of the classes defined in module class_lib.

-- Luca Stalder, 2017
"""

import math

import numpy as np
import numpy.random as nprd
import decimal as dec

from lib import Task, TaskSet, ExpExceedDist, WeibullDist


def uunifast(n, util):
    """Returns an array of n values in (0, 1) which sum up to util. See [3]."""
    sum_u = util
    vect_u = np.empty(n)
    for i in range(n - 1):
        next_sum_u = sum_u * nprd.random() ** (1.0 / (float(n - i)))
        vect_u[i] = sum_u - next_sum_u
        sum_u = next_sum_u
    vect_u[-1] = sum_u
    return vect_u


def randfixedsum(n, u, nsets, a, b):
    """Randomly and uniformly generates vectors with a specified sum and values in a specified interval.
    
    All courtesy to P. Emberson, R. Stafford, R. Davis for the randfixedsum algorithm and their implementation in 
    Python. Only minor adjustments have been made and are annotated in the corresponding lines.
    
    Paper: [1]
    Matlab: https://www.mathworks.com/matlabcentral/fileexchange/9700-random-vectors-with-fixed-sum
    Python: https://github.com/brandenburg/schedcat/blob/master/schedcat/generator/generator_emstada.py
    
    Args:
        n: Size of each returned vector.
        u: Float value every vector sums up to.
        nsets: Number of vectors that are returned.
        a: Lower bound for every element in all vectors.
        b: Upper bound for every element in all vectors.
    
    Returns:
        An array of nsets vectors, each containing n float values that lie between a and b and sum up to u. 
    """

    # Check the arguments.
    if (n != round(n)) or (nsets != round(nsets)) or (nsets < 0) or (n < 1):
        print('n must be a whole number and nsets a non-negative integer.')
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
        tmp1 = (w[i - 2, np.arange(1, (i + 1))] / float(i)) * s1[np.arange(0, i)]
        tmp2 = (w[i - 2, np.arange(0, i)] / float(i)) * s2[np.arange((n - i), n)]
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
    """Generates a list u_lo of uniformly random values, each corresponding to a value in u_hi, with following 
    conditions:
    
    (1) sum(u_lo) is equal to m * u_hi_lo.
    (2) No value in u_lo is larger than its corresponding value in u_hi.
    
    This algorithm is taken from [2] and used in mc_fairgen.
    
    Args:
        u_hi_lo: Desired total utilization of HI-criticality tasks in LO-mode.
        u_hi_hi: Given total utilization of HI-criticality tasks in HI-mode.
        m: Number of processors in the system.
        n_hi: Number of HI-criticality tasks.
        u_min: Minimum utilization per task.
        u_hi: List of HI-mode task utilizations.
        
    Returns:
        A list u_lo of task utilizations for every HI-criticality task.
    """
    u_lo = []
    # arr = sorted(u_hi, reverse=True)
    u_hi[::-1].sort()

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


def set_fixed_priorities(task_set, mode='dm'):
    """Fixed priority scheduling. Assigns rate- or deadline-monotonic priorities (shorter period -> higher priority)."""
    if mode == 'rm':
        tasks = sorted(task_set.tasks, key=lambda t: t.period, reverse=True)
    elif mode == 'dm':
        tasks = sorted(task_set.tasks, key=lambda t: t.deadline, reverse=True)
    for idx, task in enumerate(tasks):
        task.static_prio = idx


################################
# Task Set Parameter Synthesis #
################################

def dummy_taskset():
    """Returns a small dummy task set with only two tasks. These can be used for testing."""
    t1 = Task(task_id=0, criticality='HI', period=4, deadline=4, c_lo=1, c_hi=2, phase=0)
    t2 = Task(task_id=1, criticality='LO', period=6, deadline=6, c_lo=4, phase=0)
    t1.c_pmf = np.array([0.0, 0.5, 0.5])
    t2.c_pmf = np.array([0.0, 0.0, 0.2, 0.3, 0.5])
    return TaskSet(0, [t1, t2])


def simplegen(
        set_id,
        u_lo=None,
        cf=1.5,
        cp=0.5,
        n_tasks=10,
        implicit_deadlines=False
) -> TaskSet:
    """Generates mixed-criticality task sets based on the UUniFast algorithm.
    
    Args:
        set_id: Identifier for newly generated task set.
        u_lo: Desired normalized system utilization in LO-mode. Uniformly picked at random if None.
        cf: Criticality factor. C(HI) values are cf times higher than the corresponding C(LO) value.
        cp: Criticality probability. Chance for a task to be of HI criticality.
        n_tasks: Number of tasks in the generated task set.
        implicit_deadlines: If true, deadline == period; if false, deadlines are picked uniformly at random.  
    
    Returns:
        A TaskSet object (defined in module class_lib) with the desired parameters. Note that this task set's
        parameters are non-integer values and thus unsuited for practical, hyperperiod-based response time analysis.
    """

    # Constants:
    time_granularity = 10  # Multiplier for smaller discrete time units.

    if u_lo is None:
        u_lo = nprd.uniform(0.05, 1.)

    # make sure u_lo is between 0.0 and 1.0; adjust if necessary:
    m = int(math.ceil(u_lo))
    u_lo_adj = u_lo / m

    utils_lo = []
    for i in range(m):
        utils_lo.extend(uunifast(n_tasks, u_lo_adj))
    tasks = []
    for i in range(m * n_tasks):
        periods = [5, 10, 20, 25, 50, 100]
        period = nprd.choice(periods) * time_granularity
        crit = nprd.choice(['HI', 'LO'], p=[cp, 1. - cp])
        c_lo = int(math.ceil(utils_lo[i] * period))
        c_hi = int(math.ceil(cf * c_lo))
        if implicit_deadlines:
            deadline = period
        else:
            deadline = nprd.randint(c_hi, period) if c_hi < period else period
        if crit == 'LO':
            c_hi = None
        tasks.append(Task(task_id=i, criticality=crit, period=period, deadline=deadline,
                          c_lo=c_lo, c_hi=c_hi))
    return TaskSet(set_id, tasks)


def mc_fairgen(
        set_id,
        u_lo=None,
        u_hi=None,
        periods=None,
        implicit_deadlines=False,
) -> TaskSet:
    """Returns a fair set of deterministic tasks.
    
    Can either be used to generate task sets with specific system utilizations or with random values.
    
    Fair in this context means introducing as little bias towards specific scheduling policies, etc. as possible.
    
    This method is based on [2].
    
    Args:
        set_id: Identifier for newly generated task set.
        u_lo: Desired normalized (per-core) system utilization in LO-mode. Fairly picked at random if None.
        u_hi: Desired normalized (per-core) system utilization in HI-mode. Fairly picked at random if None.
        periods: List of possible period values. A default list resulting in a small hyperperiod is assigned if None.
        implicit_deadlines: If true, deadline == period; if false, deadlines are picked uniformly at random.
            
    Returns:
        A TaskSet object (defined in module class_lib) with the desired parameters. Note that this task set does not 
        have any priorities or job release times assigned yet. 
    """

    # Constants:
    m = 1  # Number of cores in the system.
    u_min = 0.01  # Minimum per-task utilization.
    u_max = 0.99  # Maximum per-task utilization.
    max_tasks = 10  # Maximum number of tasks in the generated task set.
    time_granularity = 10  # Multiplier to introduce smaller discrete time units.

    if u_hi is None:
        u_hi_hi = nprd.uniform(low=max_tasks * u_min, high=1.0)
    else:
        u_hi_hi = u_hi

    p_hi = nprd.randint(1, 10) / 10.  # fraction of hi-criticality tasks
    p_lo = 1. - p_hi

    if u_lo is None:
        u_hi_lo = nprd.uniform(low=max_tasks * p_hi * u_min, high=u_hi_hi)
        u_lo_lo = nprd.uniform(low=max_tasks * p_lo * u_min, high=1 - u_hi_lo)
    else:
        # make sure u_lo is between 0.0 and 1.0; adjust if necessary:
        factor = int(math.ceil(u_lo))
        m *= factor
        u_lo /= factor

        u_hi_lo = nprd.uniform(low=max_tasks * p_hi * u_min, high=min(u_hi_hi, u_lo) - max_tasks * p_lo * u_min)
        u_lo_lo = u_lo - u_hi_lo

    n_min_hi = int(math.ceil(u_hi_hi * m / u_max))  # minimum required total tasks
    n_min_lo = int(math.ceil(u_lo_lo * m / u_max))
    n_min = max(m + 1, int(math.ceil(n_min_hi / p_hi)), int(math.ceil(n_min_lo / (1 - p_hi))))
    if n_min < max_tasks * m:  # randint only working if low < high
        n = nprd.randint(n_min, max_tasks * m + 1)
    else:
        n = max_tasks * m
    n_hi = max(int(p_hi * n), n_min_hi)
    n_lo = n - n_hi

    if periods is None:
        periods = [5, 10, 20, 25, 50, 100]  # This will yield a manageably small hyperperiod.
    t = [time_granularity * i for i in nprd.choice(a=periods, size=n)]
    utils_hi = randfixedsum(n=n_hi, u=u_hi_hi * m, nsets=1, a=u_min, b=u_max)[0]
    utils_lo = bounded_uniform(u_hi_lo=u_hi_lo, u_hi_hi=u_hi_hi, m=m, n_hi=n_hi, u_min=u_min, u_hi=utils_hi)
    utils_lo.extend(randfixedsum(n=n_lo, u=u_lo_lo * m, nsets=1, a=u_min, b=u_max)[0])

    c_lo, c_hi, d = [], [], []

    for i in range(n):
        c_lo.append(int(dec.Decimal(utils_lo[i] * t[i]).to_integral_value(dec.ROUND_HALF_UP)))

    for i in range(n_hi):
        c_hi.append(int(dec.Decimal(utils_hi[i] * t[i]).to_integral_value(dec.ROUND_HALF_UP)))

        # Note: The use of Decimal here is to avoid rounding c_lo = 0.5 down to 0 (as done with mathematical rounding).

    if implicit_deadlines:
        d = list(t)
    else:
        for i in range(n_hi):
            d.append(nprd.randint(c_hi[i], t[i]))
        for i in range(n_hi, n):
            d.append(nprd.randint(c_lo[i], t[i]))

    tasks = []
    for i in range(n_hi):
        tasks.append(Task(task_id=i, criticality='HI', period=t[i], c_lo=c_lo[i], c_hi=c_hi[i], deadline=d[i]))

    for i in range(n_hi, n):
        tasks.append(Task(task_id=i, criticality='LO', period=t[i], c_lo=c_lo[i], deadline=d[i]))
    return TaskSet(set_id, tasks)


################################
# Execution Time PMF Synthesis #
################################

def synth_c_pmf(task_set,
                distribution_cls=WeibullDist,
                c_lo_percent=1 - 1e-5, c_hi_percent=1 - 1e-9):
    """Adds a computation time PMF to every task in the set.

    Distributions are generated based on the task's deterministic parameters. C(LO) can either set to be the maximum,
    or average computation time. New distributions other than the default one can be implemented as well, but need to
    offer the entire interface given in the default one to be fully usable.

    Args:
        task_set: Set of tasks to which PMFs are added.
        distribution_cls: The class of distribution used for generating task sets. As standard, WeibullDist introduced
            in module class_lib is used, but in principle, any custom distribution class can be implemented, as long as
            they contain EVERY method seen in class_lib.WeibullDist.
        c_lo_percent: LO-criticality tasks' distribution is chosen such that in discretizing them, their PDF gets 
            cropped at the c_lo_percent-th percentile.
        c_hi_percent: HI-criticality tasks' distribution is chosen such that in discretizing them, their PDF gets 
            cropped at the c_hi_percent-th percentile.    
    """
    for t in task_set.tasks:
        distribution = distribution_cls.from_percentile(t.c_lo, c_lo_percent, t.c_hi, c_hi_percent)
        cutoff = c_hi_percent if t.criticality == 'HI' else c_lo_percent
        t.c_pmf = distribution.discrete_pmf(cutoff=cutoff)


if __name__ == '__main__':
    # Plot example task sets for both synthesis methods
    ex_simplegen = simplegen(0, 0.8)
    synth_c_pmf(ex_simplegen, distribution_cls=ExpExceedDist)
    ex_simplegen.draw(scale='log', path='./figures/ex_simplegen.png')

    ex_mc_fairgen = mc_fairgen(0, 0.8)
    synth_c_pmf(ex_mc_fairgen)
    ex_mc_fairgen.draw(scale='linear', path='./figures/ex_mc_fairgen.png')


"""
Literature:
[1] Emberson, Stafford, Davis
    Techniques For The Synthesis Of Multiprocessor Tasksets
[2] Ramanathan, Easwaran
    Evaluation of Mixed-Criticality Scheduling Algorithms using a Fair Taskset Generator
[3] Bini, Buttazzo
    Measuring the Performance of Schedulability Tests
[4] Maxim, Davis, Cucu-Grosjean, Easwaran
    Probabilistic Analysis for Mixed Criticality Scheduling with SMC and AMC
"""