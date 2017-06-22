"""
This module offers custom classes needed for the generation and analysis for mixed-criticality task sets and other
operations related to them.

-- Luca Stalder, 2017
"""

import math
import functools
import collections
import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt


class Task(object):
    """General-purpose task.
    
    Task can be used for periodic, both uni- and mixed-criticality tasks with either deterministic or stochastic 
    execution times.
    
    Attributes:
        task_id: Integer value that uniquely identifies the task inside its belonging TaskSet.
        criticality: String of either 'LO' or 'HI'.
        period: Time that passes between two job releases for this task.
        deadline: Relative deadline, task execution has to be completed before this amount of time since release has 
            passed.
        u_lo: Task utilization in LO-mode, i.e. either (average exec time / period) or (c_lo / period).
        c_lo: Upper bound on exec time in LO-mode.
        u_hi: Task utilization in HI-mode, i.e. c_hi / period (if defined).
        c_hi: Upper bound on exec time in HI-mode, only defined for HI-critical tasks.
        phase: Shift of release times in positive direction of time, relative to the TaskSet's hyperperiod.
        priority: Integer value for task's priority in a TaskSet with fixed priority scheduling (0 means 'Lowest prio').
        c_pdf: Array of floats. task.c_pdf[i] returns P["Task takes i time units for execution."]
        rel_times: List of release times scheduled in the corresponding TaskSet's hyperperiod.
        avg_response: An array modelling the distribution of average task response time. Only including values up to
            its deadline, as the others are not considered for deadline miss probability analysis.
        dmp: Average deadline miss probability.
    """
    def __init__(self,
                 task_id,
                 criticality='LO',
                 period=None,
                 deadline=None,
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
        self.avg_response = None
        self.dmp = None

    def plot_c_pdf(self, scale='linear'):
        """Simple method to display a plot of the task's execution time probability distribution.
        
        c_lo and c_hi values are displayed as orange and red dotted lines, respectively.
        
        Args:
            scale: Either 'linear' or 'log'.
        """
        plt.bar(range(len(self.c_pdf)), self.c_pdf)
        plt.axvline(self.c_lo, color="orange", linestyle="--")
        plt.axvline(self.c_hi, color="red", linestyle="--")
        plt.yscale(scale)
        plt.show()

    def c_max(self):
        """Returns the task's maximum execution time."""
        return self.c_lo if self.criticality == 'LO' else self.c_hi

    def c_min(self, epsilon=1e-14):
        """Returns the task's minimum execution time. Values smaller than epsilon are considered zero."""
        if self.c_pdf is None:
            return self.c_lo
        else:
            return [t for t in range(self.c_max()) if self.c_pdf[t] > epsilon][0]


class TaskSet(object):
    """Set of periodic tasks.
    
    TaskSet acts as a container class for the Task class introduced above.
    
    Attributes:
        set_id: Integer value that uniquely identifies the instance of TaskSet.
        tasks: List of tasks contained in this task set.
        n_lo: Number of LO-criticality tasks.
        n_hi: Number of HI-criticality tasks.
        u_lo: Total (system) utilization in LO-mode. Either based on c_lo, or on average execution times.
        u_hi: Total (system) utilization in HI-mode. Based on c_hi values.
        description: A short descriptive string about this task set. Can be used for plotting.
    """
    def __init__(self, set_id, tasks: [Task]):
        self.id = set_id
        self.tasks = tasks
        self.n_lo = len([t for t in self.tasks if t.criticality == 'LO'])
        self.n_hi = len(tasks) - self.n_lo
        self.u_lo = sum([float(t.c_lo) / t.period for t in self.tasks])
        self.u_hi = sum([float(t.c_hi) / t.period for t in self.tasks if t.criticality == 'HI'])
        self.description = "Task Set {0}: {1} LO task(s) @ {3} util, {2} HI task(s) @ {4} util."\
            .format(self.id, self.n_lo, self.n_hi, self.u_lo, self.u_hi)

    @property
    def hyperperiod(self):
        """This TaskSet's hyperperiod, defined as the least common multiple over all its tasks' periods."""
        return lcm([t.period for t in self.tasks])  # TODO: Add lazy evaluation

    @property
    def job_releases(self):
        """A list of tuples, each representing the release of a new job instance, ordered by release time."""
        timeline = []
        JobRelease = collections.namedtuple('JobRelease', ['t', 'task'])
        for task in self.tasks:
            timeline.extend([JobRelease(t, task) for t in task.rel_times])
        timeline.sort(key=lambda x: x.t)
        return timeline

    def draw(self, scale='linear'):
        """Method for displaying a graphic representation of the task set.
        
        Every task is drawn as a subplot with a short description. Orange and red dotted lines correspond to the task's
        c_lo and c_hi values, respectively.
        
        Args:
         scale: Either 'linear' or 'log'.
        """
        # TODO Add possibility of saving into an image file.
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

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(self.description)
        plt.show()

    def set_priorities_rm(self):
        """Fixed priority scheduling. Assigns rate-monotonic priorities (shorter period -> higher priority."""
        tasks = sorted(self.tasks, key=lambda t: t.period, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def set_priorities_dm(self):
        """Fixed priority scheduling. Assigns deadline-monotonic priorities (shorter deadline -> higher priority."""
        tasks = sorted(self.tasks, key=lambda t: t.deadline, reverse=True)
        for i in range(len(tasks)):
            tasks[i].priority = i

    def set_rel_times(self):
        """Builds the list of release times for every task.
        
        Release times are always relative to one hyperperiod.
        """
        for t in self.tasks:
            t.rel_times = [k * t.period + t.phase for k in range(self.hyperperiod // t.period)]


class WeibullDist(object):
    """Example distribution, based on a Weibull Distribution with shape parameter k and scale parameter beta.
    
    Every method defined here MUST be implemented, if the user chooses to use a custom distribution for task generation.
    
    Attributes:
        k: Shape parameter, a reasonable choice for task execution time distributions lies in the interval (1, 5).
        beta: Scale parameter.
    """
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    @classmethod
    def from_ev(cls, ev):
        """Constructor method for a given fixed expected value.
        
        Args:
            ev: Desired expected value of the distribution.
            
        Returns:
            A distribution object that can be used to model a task's execution time distribution.
            The distribution's expected value is ev.
        """
        k = nprd.uniform(1.1, 3)
        beta = ev / math.gamma(1. + (1. / k))
        return cls(k=k, beta=beta)

    @classmethod
    def from_percentile(cls, x, p):
        """Constructor method for a given fixed percentile.
        
        Args:
            x: Point of the p-th percentile.
            p: Desired percentile, in interval (0, 1).
            
        Returns:
            A distribution object that can be used to model a task's execution time distribution.
            The distribution's p-th percentile lies at point x.
        """
        k = nprd.uniform(1.1, 3)
        beta = x / (-math.log(1 - p)) ** (1. / k)
        return cls(k=k, beta=beta)

    def pdf(self, x):
        """Exact probability density function."""
        k = self.k
        beta = self.beta
        return (k / beta) * (x / beta) ** (k - 1.) * math.exp(-(x / beta) ** k)

    def cdf(self, x: float):
        """Exact cumulative distribution function."""
        return 1 - math.exp(-(x / self.beta)**self.k)

    def percentile(self, p: float):
        """Exact percentile function."""
        return self.beta * (math.log(1./(1. - p)))**(1./self.k)

    def discrete_pd(self, cutoff):
        """Method to find a discrete, cropped approximation of the distribution's PDF. 
        
        Args:
            cutoff: Percentile value, at which the approximation should stop. Avoids returning infinitely long arrays.
            
        Returns:
            An array object containing the discrete distribution. For example:
            
            [0.0, 0.6, 0.4]
            
            stands for a discrete distribution with P[x=0] = 0.0, P[x=1] = 0.6, P[x=2] = 0.4.
        """
        bound = int(math.ceil(self.percentile(cutoff)))
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
        return (a * b) // math.gcd(a, b)
    return functools.reduce(lcm2, numbers)


if __name__ == '__main__':
    weib = WeibullDist(beta=10, k=5)
    vec = weib.discrete_pd(0.9999)
