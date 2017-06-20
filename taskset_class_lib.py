from math import exp, log, gcd, gamma, ceil
import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt
from functools import reduce


class Task(object):  # TODO Docstring
    """Input for the analysis."""
    def __init__(self,
                 task_id,
                 criticality,
                 period,
                 deadline,
                 u_lo=None, c_lo=None,
                 u_hi=None, c_hi=None,
                 phase=0,
                 priority=None):
        self.task_id = task_id
        self.criticality = criticality
        self.period = period
        self.phase = phase
        self.deadline = deadline
        self.u_lo = u_lo
        self.c_lo = c_lo
        self.u_hi = u_hi
        self.c_hi = c_hi
        self.priority = priority
        self.c_pdf = []
        self.rel_times = []

    def plot_c_pdf(self):
        plt.bar(range(len(self.c_pdf)), self.c_pdf)
        plt.axvline(self.c_lo, color="orange", linestyle="--")
        plt.axvline(self.c_hi, color="red", linestyle="--")
        plt.show()

    def c_max(self):
        """Return the maximum execution time."""
        return self.c_lo if self.criticality == 'LO' else self.c_hi

    def c_min(self, epsilon=1e-14):
        """Return the minimum execution time."""
        if self.c_pdf is None:
            return self.c_lo
        else:
            return [t for t in range(self.c_max()) if self.c_pdf[t] > epsilon][0]


class TaskSet(object):  # TODO Docstring
    """Set of sporadic tasks."""
    def __init__(self, set_id, tasks: [Task]):
        self.id = set_id
        self.tasks = tasks
        self.n_lo = len([t for t in self.tasks if t.criticality == 'LO'])
        self.n_hi = len(tasks) - self.n_lo
        self.u_lo = sum([float(t.c_lo) / t.period for t in self.tasks])
        self.u_hi = sum([float(t.c_hi) / t.period for t in self.tasks if t.criticality == 'HI'])
        self.description = "Task Set {0}: {1} LO task(s) @ {3} util, {2} HI task(s) @ {4} util."\
            .format(self.id, self.n_lo, self.n_hi, self.u_lo, self.u_hi)
        self.hyperperiod = lcm([t.period for t in self.tasks])  # TODO: Add lazy evaluation

    def draw(self, scale='linear'):
        fig = plt.figure()
        for i in range(len(self.tasks)):
            plt.subplot(7, 3, i+1)
            t = self.tasks[i]
            plt.title('Task: {0}, Criticality: {1}, Period: {2}, C_LO: {3}, C_HI: {4}'
                      .format(t.task_id, t.criticality, t.period, t.c_lo, t.c_hi))
            plt.bar(range(len(t.c_pdf)), t.c_pdf)
            plt.axvline(t.c_lo, color='orange', linestyle='--')
            if t.criticality == 'HI':
                plt.axvline(t.c_hi, color='red', linestyle='--')
            plt.yscale(scale)
            # plt.axvline(t.deadline, color='black', linestyle='--') TODO: Remove this

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(self.description)
        plt.show()

    def assign_priorities_rm(self):
        tasks = sorted(self.tasks, key=lambda t: t.period, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def assign_priorities_dm(self):
        tasks = sorted(self.tasks, key=lambda t: t.deadline, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def assign_rel_times(self):
        for t in self.tasks:
            t.rel_times = [k * t.period + t.phase for k in range(self.hyperperiod // t.period)]


class WeibullDist(object):  # TODO Docstring
    """Probability distribution with shape parameter k and scale parameter beta."""
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    @classmethod
    def from_ev(cls, ev):
        k = nprd.uniform(1.1, 3)
        beta = ev / gamma(1. + (1. / k))
        return cls(k=k, beta=beta)

    @classmethod
    def from_percentile(cls, x, percentile):
        k = nprd.uniform(1.1, 3)
        beta = x / (-log(1 - percentile)) ** (1. / k)
        return cls(k=k, beta=beta)

    def pdf(self, x):
        k = self.k
        beta = self.beta
        return (k / beta) * (x / beta) ** (k - 1.) * exp(-(x / beta) ** k)

    def percentile(self, p: float):
        return self.beta * (log(1./(1. - p)))**(1./self.k)

    def cdf(self, x: float):
        return 1 - exp(-(x / self.beta)**self.k)

    def discrete_pd(self, cutoff):
        """Returns probabilities up to the cutoff-percentile as a vector. 
        Values above cutoff are cropped and probabilities are rescaled."""
        bound = int(ceil(self.percentile(cutoff)))
        probabilities = np.zeros(bound + 1)
        last = 0.0
        for x in range(1, bound + 1):
            curr = self.cdf(x)
            probabilities[x] = curr - last
            last = curr
        factor = 1. / sum(probabilities)
        return probabilities * factor


def lcm(numbers: [int]):
    """Least common multiple."""
    def lcm2(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm2, numbers)


if __name__ == '__main__':
    weib = WeibullDist(beta=10, k=5)
    vec = weib.discrete_pd(0.9999)
