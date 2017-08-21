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

    @property
    def description(self):
        return ('Task: {0}, {1}, T: {2}, C(LO/HI): {3}/{4}, D: {5}'
                .format(self.task_id, self.criticality, self.period, round(self.c_lo), self.c_hi, self.deadline))

    def draw(self, scale='linear', labels=False, path=None):
        """Simple method to display a plot of the task's execution time probability mass function.
        
        c_lo and c_hi values are displayed as orange and red dotted lines, respectively.
        
        Args:
            scale: Either 'linear' or 'log'.
            labels: Shows number values over each bar.
            path: Path for saving figure. If None, the figure will be displayed immediately instead.
        """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        bars = ax.bar(range(len(self.c_pmf)), self.c_pmf)
        ax.axvline(self.c_lo, color="orange", linestyle="--", linewidth=1)
        ax.text(self.c_lo*1.01, 0.8, "C(LO)", color='orange', verticalalignment='bottom', rotation='vertical')
        if self.criticality == 'HI':
            ax.axvline(self.c_hi, color="red", linestyle="--", linewidth=1)
            ax.text(self.c_hi * 1.01, 0.8, "C(HI)", color='red', verticalalignment='bottom',
                    rotation='vertical')
        if scale == 'linear':
            ax.set_ylim(0, 1.05)
        ax.set_yscale(scale)
        ax.set_xlabel('Execution Time')
        ax.set_ylabel('Probability')
        ax.set_title(self.description)
        if labels:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., 1.0 * height,
                        '%g' % height,
                        fontsize=11,
                        ha='center', va='bottom')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)

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
        fig = plt.figure(figsize=(8, 8), dpi=300)

        for i, task in enumerate(self.tasks):
            ax = plt.subplot(math.ceil(len(self.tasks) / 2), 2, i+1)
            ax.set_title(task.description, fontsize=10)
            ax.bar(range(len(task.c_pmf)), task.c_pmf)
            ax.axvline(task.c_lo, color='orange', linestyle='--')
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.text(task.c_lo, -0.02*y_range, "C(LO)",
                    color='orange', horizontalalignment='center', verticalalignment='top', fontsize=6)
            if task.criticality == 'HI':
                ax.axvline(task.c_hi, color='red', linestyle='--')
                ax.text(task.c_hi, -0.02 * y_range, "C(HI)",
                        color='red', horizontalalignment='center', verticalalignment='top', fontsize=6)            # plt.axvline(task.deadline, color='black', linestyle='--')
            ax.set_yscale(scale)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(self.description, fontsize=12)
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

def convolve_rescale_pmf(a, b, percentile=1 - 1e-14, rescale=True):
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
    if rescale:
        conv /= sum(conv)

    return conv


def shrink(pmf, t):
    """Shrinking of PMF c by t time units."""

    if t >= len(pmf):
        return np.array([sum(pmf)])
    else:
        result = pmf[t:]
        result[0] += sum(pmf[:t])
        return result


def split_convolve_merge(response_pmf, preempt_pmf, release):
    """Split response_pmf at release, convolve tail with preempt_pmf, merge back together."""
    head, tail = np.split(response_pmf, [release + 1])
    if len(tail):
        tail = convolve_rescale_pmf(tail, preempt_pmf, rescale=False)
    return np.concatenate((head, tail))


####################################
# Probability Distribution Classes #
####################################

class ExpExceedDist(object):
    """
    Probability distribution based on the exceedance function (1 - CDF) being a straight line on a log-plot.
    
    Maxim et al. [1] introduced this method of generating synthetic probability distributions for tasks. The exceedance
    function 1 - CDF is assumed to be a straight line on a graph with probabilities on a log scale, i.e. of the form
    1-CDF = a * exp(b * x). The parameters a and b are then found by extrapolating through the points (c_lo, 1 - p_lo)
    and (c_hi, 1 - p_hi).
    """
    def __init__(self, a, b, c_min):
        self.a = a
        self.b = b
        self.c_min = c_min

    @classmethod
    def from_percentile(cls, c_lo, p_lo, c_hi, p_hi):
        """Constructor method for a given fixed percentile.

        Args:
            c_lo, c_hi: Points of the p-th percentiles.
            p_lo, p_hi: Desired percentile, in interval (0, 1).

        Returns:
            A distribution object that can be used to model a task's execution time distribution.
        """
        cp = 1.5  # Criticality factor parameter
        if c_hi is None:
            c_hi = c_lo * cp
        ex_lo, ex_hi = 1. - p_lo, 1. - p_hi  # Exceedance probabilities
        b = (math.log(ex_hi) - math.log(ex_lo)) / (c_hi - c_lo)
        a = ex_lo / (math.exp(b * c_lo))
        c_min = - math.log(a) / b
        return cls(a, b, c_min)

    def pdf(self, x):
        """Exact probability density function."""
        return -self.a * self.b * math.exp(self.b * x) if x >= self.c_min else 0

    def cdf(self, x: float):
        """Exact cumulative distribution function."""
        return 1 - self.a * math.exp(self.b * x) if x >= self.c_min else 0

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
        c_min = self.c_min
        c_max = int(math.floor(self.percentile(cutoff)))
        probabilities = np.zeros(c_max + 1)
        last = 0.0
        for x in range(int(math.ceil(c_min)), c_max + 1):
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
        factor = 1. / sum(probabilities)
        return probabilities * factor

if __name__ == '__main__':
    # Plot examples for stochastic MC-tasks
    ex_task_lo = Task(task_id=0, criticality='LO', period=8, deadline=7, c_lo=4)
    ex_task_lo.c_pmf = [0.0, 0.25, 0.25, 0.25, 0.25]
    ex_task_lo.draw(labels=True, path='./figures/ex_task_lo.png')
    ex_task_hi = Task(task_id=1, criticality='HI', period=10, deadline=6, c_lo=3, c_hi=5)
    ex_task_hi.c_pmf = [0.0, 0.2, 0.4, 0.3, 0.08, 0.02]
    ex_task_hi.draw(labels=True, path='./figures/ex_task_hi.png')


    def draw_pmf(pmf, xlim, ylim=(0, 1), figsize=(3, 3), barfont=11):
        """Helper function for following plots."""
        fig = plt.figure(figsize=figsize, dpi=180)
        ax = plt.axes()
        bars = ax.bar(range(len(pmf)), pmf)
        ax.set_xlim(xlim)
        ax.set_xticks(range(int(ax.get_xlim()[1])))
        ax.set_ylim(ylim)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     '%g' % height,
                     fontsize=barfont,
                     ha='center', va='bottom', rotation='vertical')
        return fig

    # Plot example for convolution and shrinking
    ex_backlog = [0.0, 0.5, 0.5]
    fig1 = draw_pmf(ex_backlog, (-0.5, 6))
    plt.savefig('./figures/ex_backlog.png')

    ex_job_pmf = [0.0, 0.2, 0.5, 0.3]
    fig2 = draw_pmf(ex_job_pmf, (-0.5, 6))
    plt.savefig('./figures/ex_job_pmf.png')

    ex_convolve = convolve_rescale_pmf(ex_backlog, ex_job_pmf)
    fig3 = draw_pmf(ex_convolve, (-0.5, 6))
    plt.savefig('./figures/ex_convolve.png')
    plt.axvline(3, color='black', linestyle='dashed', linewidth=1)
    plt.savefig('./figures/ex_convolve_line.png')

    ex_shrink = shrink(ex_convolve, 3)
    fig4 = draw_pmf(ex_shrink, (-0.5, 6))
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.savefig('./figures/ex_shrink.png')

    # Plot example for split-convolve-merge
    ex_job_response = np.array([0.0, 0.0, 0.2, 0.3, 0.3, 0.1, 0.1])  # t = 0
    fig5_1a = draw_pmf(ex_job_response, (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.axvline(3, color='black', linestyle='dashed', linewidth=0.8)
    plt.text(3, 0.76, r'$\lambda_k^\prime$', horizontalalignment='center', verticalalignment='bottom')
    plt.savefig('./figures/ex_split_1a.png')

    ex_job_response_head, ex_job_response_tail = ex_job_response[:3+1], ex_job_response[3+1:]
    fig5_1b = draw_pmf(ex_job_response_head, (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.savefig('./figures/ex_split_1b.png')
    plt.savefig('./figures/ex_split_3a.png')

    fig5_1c = draw_pmf(np.concatenate((np.zeros(4), ex_job_response_tail)),
                       (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.savefig('./figures/ex_split_1c.png')
    plt.savefig('./figures/ex_split_2a.png')

    ex_job_preempt = np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.4])  # Release at t = 3
    fig5_2b = draw_pmf(ex_job_preempt, (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.savefig('./figures/ex_split_2b.png')

    ex_tail_convolve = convolve_rescale_pmf(ex_job_response_tail, ex_job_preempt, rescale=False)
    fig5_2b = draw_pmf(np.concatenate((np.zeros(4), ex_tail_convolve)),
                       (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.savefig('./figures/ex_split_2c.png')
    plt.savefig('./figures/ex_split_3b.png')

    ex_job_response_merge = np.concatenate((ex_job_response_head, ex_tail_convolve))
    fig5_3c = draw_pmf(ex_job_response_merge, (-0.5, 12), ylim=(0, 0.75), figsize=(3, 2), barfont=10)
    plt.savefig('./figures/ex_split_3c.png')

"""
Literature:
[1] Maxim, Davis, Cucu-Grosjean, Easwaran
    Probabilistic Analysis for Mixed Criticality Scheduling with SMC and AMC
"""