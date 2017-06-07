from math import exp, log, gcd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


class Task(object):
    """Input for the analysis."""
    def __init__(self,
                 task_id,
                 criticality,
                 period,
                 deadline,
                 u_lo=None, c_lo=None,
                 u_hi=None, c_hi=None,
                 phase=None,
                 priority=None):
        self.task_id = task_id
        self.criticality = criticality
        self.period = period
        self.phase = phase
        self.deadline = deadline
        self.c = {'HI': c_hi, 'LO': c_lo}
        self.u_hi = u_hi
        self.u_lo = u_lo
        self.priority = priority
        self.c_pdf = []
        self.rel_times = []

    def plot_c_pdf(self):
        plt.bar(range(len(self.c_pdf)), self.c_pdf)
        plt.axvline(self.c['LO'], color="orange", linestyle="--")
        plt.axvline(self.c['HI'], color="red", linestyle="--")
        plt.show()


class TaskSet(object):
    """Set of sporadic tasks."""
    def __init__(self, set_id, tasks: [Task]):
        self.id = set_id
        self.tasks = tasks
        self.n_lo = len([t for t in self.tasks.values() if t.criticality == 'LO'])
        self.n_hi = len(tasks) - self.n_lo
        self.u_lo = sum([float(t.c['LO']) / t.period for t in self.tasks.values()])
        self.u_hi = sum([float(t.c['HI']) / t.period for t in self.tasks.values() if t.criticality == 'HI'])
        self.description = "Task Set {0}: {1} LO task(s) @ {3} util, {2} HI task(s) @ {4} util."\
            .format(self.id, self.n_lo, self.n_hi, self.u_lo, self.u_hi)
        self.hyperperiod = lcm([t.period for t in self.tasks.values()])  # TODO: Add lazy evaluation

    def draw(self):
        fig = plt.figure()
        for i in range(len(self.tasks)):
            plt.subplot(7, 3, i+1)
            t = self.tasks[i]
            plt.title('Task: {0}, Criticality: {1}, Period: {2}, C_LO: {3}, C_HI: {4}'
                      .format(t.task_id, t.criticality, t.period, t.c['LO'], t.c['HI']))
            plt.bar(range(len(t.c_pdf)), t.c_pdf)
            plt.axvline(t.c['LO'], color='orange', linestyle='--')
            if t.criticality == 'HI':
                plt.axvline(t.c['HI'], color='red', linestyle='--')

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(self.description)
        plt.show()

    def assign_priorities_rm(self):
        tasks = sorted(self.tasks.values(), key=lambda t: t.period, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def assign_priorities_dm(self):
        tasks = sorted(self.tasks.values(), key=lambda t: t.deadline, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def assign_rel_times(self):
        for t in self.tasks.values():
            t.rel_times = [k * t.period for k in range(self.hyperperiod // t.period)]


class WeibullDist(object):
    """Probability distribution with shape parameter gamma and scale parameter beta."""
    def __init__(self, gamma=None, beta=None):
        self.gamma = gamma
        self.beta = beta

    def pdf(self, x):
        gamma = self.gamma
        beta = self.beta
        return (gamma / beta) * (x / beta) ** (gamma - 1) * exp(-(x / beta) ** gamma)

    def percentile(self, p):
        return self.beta * (log(1.0/(1.0 - p)))**(1.0/self.gamma)

    def cdf(self, x):
        return 1 - exp(-(x / self.beta)**self.gamma)

    def rescale_beta(self, x, p):
        """Adjusts beta such that percentile p lies at x."""
        self.beta = x / pow(-log(1-p), 1.0/self.gamma)

    def discrete_pd(self, bound) -> [float]:
        """Returns probabilities for x in [0, bound] as a vector. 
        Values above bound are cropped and probabilities are rescaled."""
        partial_sums = [self.cdf(x + 1) - self.cdf(x) for x in range(0, bound)]
        partial_sums.insert(0, 0.0)
        factor = sum(partial_sums)
        return [x * factor for x in partial_sums]


def lcm(numbers: [int]):
    """Least common multiple."""
    def lcm2(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm2, numbers)
