"""
This module offers custom classes needed for the generation and analysis for mixed-criticality task sets and other
operations related to them.

-- Luca Stalder, 2017
"""

import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprd
import scipy.signal as sig


#####################
# Container Classes #
#####################

class Task(object):
    """General-purpose task.
    
    Task can be used for periodic, both uni- and mixed-criticality tasks with either worst-case execution times (WCET)
    or with an execution time probability mass function (PMF).
    
    Attributes:
        task_id: Integer value that uniquely identifies the task inside its belonging TaskSet.
        criticality: String of either 'LO' or 'HI'.
        period: Time that passes between two job releases for this task.
        deadline: Relative deadline, task execution has to be completed before this amount of time since release has 
            passed.
        c_lo: Upper bound on exec time in LO-mode.
        c_hi: Upper bound on exec time in HI-mode, only defined for HI-critical tasks.
        c_pmf: Array of floats. task.c_pmf[i] returns P["Task takes i time units for execution."]
        phase: Shift of release times in positive direction of time, relative to the TaskSet's hyperperiod.
        static_prio: Integer value for task's priority in a TaskSet with fixed priority scheduling 
            (0 means 'Lowest prio').
        ### avg_response: An array modelling the distribution of average task response time. Only including values up to
            its deadline, as the others are not considered for deadline miss probability analysis.
        ### dmp: Average deadline miss probability.
    """
    def __init__(self,
                 task_id,
                 criticality,
                 period,
                 deadline,
                 c_lo,
                 c_hi=None,
                 phase=0,
                 static_prio=None):
        self.task_id = task_id
        self.criticality = criticality
        self.period = period
        self.deadline = period if deadline is None else deadline
        self.c_lo = c_lo
        self.c_hi = c_hi
        self.c_pmf = None
        self.phase = phase
        self.static_prio = static_prio
        # self.avg_response = None
        # self.dmp = None

    @property
    def u_lo(self):
        """Task utilization in LO-mode."""
        return self.c_lo / self.period

    @property
    def u_hi(self):
        """Task utilization in HI-mode."""
        return None if self.c_hi is None else self.c_hi / self.period

    @property
    def u_avg(self):
        """Average utilization over whole c_pmf"""
        if self.c_pmf is not None:
            return np.average(a=range(self.c_pmf.size), weights=self.c_pmf) / self.period

    def plot_c_pmf(self, scale='linear', show=False):
        """Simple method to display a plot of the task's execution time probability mass function.
        
        c_lo and c_hi values are displayed as orange and red dotted lines, respectively.
        
        Args:
            scale: Either 'linear' or 'log'.
            show: Show plot immediately if true. 
        """
        plt.bar(range(len(self.c_pmf)), self.c_pmf)
        plt.axvline(self.c_lo, color="orange", linestyle="--")
        plt.axvline(self.c_hi, color="red", linestyle="--")
        plt.yscale(scale)
        if show:
            plt.show()

    def c_max(self):
        """Returns the task's maximum execution time."""
        return self.c_lo if self.criticality == 'LO' else self.c_hi

    def c_min(self, epsilon=1e-14):
        """Returns the task's minimum execution time. Values smaller than epsilon are considered zero."""
        if self.c_pmf is None:
            return self.c_lo
        else:
            return [t for t in range(self.c_max()) if self.c_pmf[t] > epsilon][0]


class TaskSet(object):
    """Set of periodic tasks.
    
    TaskSet acts as a container class for the Task class introduced above. Note that the list of tasks should be treated
    as read-only, and not be changed after initialization.
    
    Attributes:
        set_id: Integer value that uniquely identifies the instance of TaskSet.
        tasks: List of tasks contained in this task set.
        hyperperiod: Duration of one hyperperiod, defined as the least common multiple over all its tasks' periods.
        jobs: A list of namedtuples over one hyperperiod, each representing the release of a new job instance, ordered 
            by release time.
    """
    def __init__(self, set_id, tasks: [Task], build_job_list=True):
        self.set_id = set_id
        self.tasks = tasks
        if build_job_list:

            def lcm(numbers: [int]):
                """Least common multiple."""

                def lcm2(a, b):
                    return (a * b) // math.gcd(a, b)

                return functools.reduce(lcm2, numbers)

            self.hyperperiod = lcm([t.period for t in self.tasks])
            self.jobs = []
            for task in self.tasks:
                releases = [k * task.period + task.phase for k in range(self.hyperperiod // task.period)]
                self.jobs.extend([Job(rel, task) for rel in releases])
            self.jobs.sort(key=lambda x: x.release)

    @property
    def n_lo(self):
        """Number of LO-criticality tasks."""
        return len([t for t in self.tasks if t.criticality == 'LO'])

    @property
    def n_hi(self):
        """Number of HI-criticality tasks."""
        return len([t for t in self.tasks if t.criticality == 'HI'])

    @property
    def u_lo(self):
        """Maximum system utilization in LO-mode."""
        return sum([task.u_lo for task in self.tasks])

    @property
    def u_hi(self):
        """Maximum system utilization in HI-mode."""
        return sum([task.u_hi for task in self.tasks if task.criticality == 'HI'])

    @property
    def u_avg(self):
        """Average system utilization, if computation time PFs are defined."""
        if all(task.c_pmf is not None for task in self.tasks):
            return sum([task.u_avg for task in self.tasks])
        else:
            return None

    @property
    def description(self):
        """A short descriptive string about this task set. Can be used for plotting."""
        u_avg = round(self.u_avg, 3) if self.u_avg is not None else None
        return "Task Set {0}: #Tasks LO/HI: ({1}/{2})  Utils LO/HI/Avg: ({3}/{4}/{5})".format(
            self.set_id, self.n_lo, self.n_hi, round(self.u_lo, 3), round(self.u_hi, 3), u_avg)

    def draw(self, scale='linear', path=None):
        """Method for displaying a graphic representation of the task set.
        
        Every task is drawn as a subplot with a short description. Orange and red dotted lines correspond to the task's
        c_lo and c_hi values, respectively.
        
        Args:
         scale: Either 'linear' or 'log'.
         path: Path to save figure to image file. If None, plot will be shown immediately instead.
        """
        fig = plt.figure(figsize=(10, 10), dpi=180)
        for i in range(len(self.tasks)):
            # print(i)
            plt.subplot(math.ceil(len(self.tasks) / 2), 2, i+1)
            t = self.tasks[i]
            plt.title('Task: {0}, {1}, Period: {2}, C(LO/HI): {3}/{4}'
                      .format(t.task_id, t.criticality, t.period, t.c_lo, t.c_hi))
            plt.bar(range(len(t.c_pmf)), t.c_pmf)
            plt.axvline(t.c_lo, color='orange', linestyle='--')
            if t.criticality == 'HI':
                plt.axvline(t.c_hi, color='red', linestyle='--')
            plt.axvline(t.deadline, color='black', linestyle='--')
            plt.yscale(scale)

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(self.description)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


class Job(object):
    """Lightweight container for a task instance.
     
    Attributes:
        release = Release time inside the task set's hyperperiod
        task = Reference to the generating task.
        response = Array of the job-level response time PF.    
    """
    def __init__(self, release, task: Task, response=None):
        self.release = release
        self.task = task
        self.response = response

    @property
    def dmp(self):
        """Deadline miss probability; 1 - sum of the response time PMF up to and including the task's deadline."""
        return None if self.response is None else 1. - np.sum(self.response[:self.task.deadline + 1])


####################
# Basic Operations #
####################

def convolve_rescale_pmf(a, b, percentile=1 - 1e-14):
    """Convolution of two discrete probability mass functions."""
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
    """Shrinking of PMF c by t time units."""

    if t >= len(pdf):
        return np.array([sum(pdf)])
    else:
        result = pdf[t:]
        result[0] += sum(pdf[:t])
        return result


####################################
# Probability Distribution Classes #
####################################

class ExpExceedDist(object):
    """
    Probability distribution based on the exceedance function (1 - CDF) being a straight line on a log-plot.
    
    Maxim et al. [1] introduced this method of generating synthetic probability distributions for tasks. The exceedance
    function 1 - CDF is assumed to be a straight line on a graph with probabilities on a log scale, i.e. of the form
    1-CDF = a * exp(b * x). The parameters a and b are then found by extrapolating through the points (x_lo, 1 - p_lo)
    and (x_hi, 1 - p_hi).
    """
    def __init__(self, a, b, x_min):
        self.a = a
        self.b = b
        self.x_min = x_min

    @classmethod
    def from_percentile(cls, x_lo, p_lo, x_hi, p_hi):
        """Constructor method for a given fixed percentile.

        Args:
            x_lo, x_hi: Points of the p-th percentiles.
            p_lo, p_hi: Desired percentile, in interval (0, 1).

        Returns:
            A distribution object that can be used to model a task's execution time distribution.
        """
        if x_hi is None:
            x_hi = x_lo * 1.5
        ex1, ex2 = 1. - p_lo, 1. - p_hi  # Exceedance probabilities
        b = (math.log(ex2) - math.log(ex1)) / (x_hi - x_lo)
        a = ex1 / (math.exp(b * x_lo))
        x_min = - math.log(a) / b
        return cls(a=a, b=b, x_min=x_min)

    def pdf(self, x):
        """Exact probability density function."""
        return -self.a * self.b * math.exp(self.b * x) if x >= self.x_min else 0

    def cdf(self, x: float):
        """Exact cumulative distribution function."""
        return 1 - self.a * math.exp(self.b * x) if x >= self.x_min else 0

    def percentile(self, p: float):
        """Exact percentile function."""
        return (1. / self.b) * math.log((1 - p) / self.a)

    def discrete_pmf(self, cutoff):
        """Method to find a discrete, cropped approximation of the distribution's PDF. 

        Args:
            cutoff: Percentile value, at which the approximation should stop. Avoids returning infinitely long arrays.

        Returns:
            An array object containing the discrete distribution. For example:

            [0.0, 0.6, 0.4]

            stands for a discrete distribution with P[x=0] = 0.0, P[x=1] = 0.6, P[x=2] = 0.4.
        """
        x_min = self.x_min
        x_max = int(math.ceil(self.percentile(cutoff)))
        probabilities = np.zeros(x_max + 1)
        last = 0.0
        for x in range(int(math.ceil(x_min)), x_max + 1):
            curr = self.cdf(x)
            probabilities[x] = curr - last
            last = curr
        factor = 1. / sum(probabilities)
        return probabilities * factor


class WeibullDist(object):
    """Example distribution, based on a Weibull Distribution with shape parameter k and scale parameter beta.
    
    Attributes:
        k: Shape parameter; a reasonable choice for task execution time distributions lies in the interval (1, 5).
        beta: Scale parameter.
    """
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    @classmethod
    def from_percentile(cls, x_lo, p_lo, x_hi, p_hi):
        """Constructor method for a given fixed percentile.
        
        Args:
            x_lo, x_hi: Points of the p-th percentiles.
            p_lo, p_hi: Desired percentile, in interval (0, 1).
            
        Returns:
            A distribution object that can be used to model a task's execution time distribution.
        """
        k = nprd.uniform(1.5, 3)
        beta = x_lo / (-math.log(1 - p_lo)) ** (1. / k)
        return cls(k=k, beta=beta)

    @classmethod
    def from_ev(cls, ev, x_hi, p_hi):
        """Constructor method for a given fixed expected value.
        
        Args:
            ev: Desired expected value of the distribution.
            x_hi: Point of p_hi-th percentile.
            p_hi: p_hi-th percentile.
            
        Returns:
            A distribution object that can be used to model a task's execution time distribution.
            The distribution's expected value is ev.
        """
        k = nprd.uniform(1.5, 3)
        beta = ev / math.gamma(1. + (1. / k))
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

    def discrete_pmf(self, cutoff):
        """Method to find a discrete, cropped approximation of the distribution's PDF. 
        
        Args:
            cutoff: Percentile value, at which the approximation should stop. Avoids returning infinitely long arrays.
            
        Returns:
            An array object containing the discrete probability mass function. For example:
            
            [0.0, 0.6, 0.4]
            
            stands for a discrete distribution with P[x=0] = 0.0, P[x=1] = 0.6, P[x=2] = 0.4.
        """
        c_max = int(math.ceil(self.percentile(cutoff)))
        probabilities = np.zeros(c_max + 1)
        last = 0.0
        for x in range(1, c_max + 1):
            curr = self.cdf(x)
            probabilities[x] = curr - last
            last = curr
        if sum(probabilities) == 0:  # TODO
            print(self.k, self.beta, cutoff, c_max, probabilities)
        factor = 1. / sum(probabilities)
        return probabilities * factor

if __name__ == '__main__':
    pass

"""
Literature:
[1] Maxim, Davis, Cucu-Grosjean, Easwaran
    Probabilistic Analysis for Mixed Criticality Scheduling with SMC and AMC
"""