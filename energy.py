import math
import numpy as np
import collections as col
import matplotlib.pyplot as plt

from analysis import convolve_rescale_pmf


def transient_system(harvest_dist, burst, c_job, p_job):
    """Response time analysis for a transient system executing a single job.
    
    A transient system collects a random amount of energy per time unit and releases it in a burst after a
    certain threshold is exceeded. This method will plot the distribution of harvested energy, the distribution
    of burst inter-arrival time, and the resulting response time distribution for the executed job.
    
    Burst size is assumed to be orders of magnitude smaller than job power dissipation, i.e. job response time
    mainly depends on energy consumption.
    
    Args:
        harvest_dist: Array describing the probability mass function for energy collected per time unit.
        burst: Capacity of the energy buffer and thus also size of released energy bursts.
        c_job: Deterministic job execution time.
        p_job: Deterministic job power dissipation per time unit.    
    """
    Job = col.namedtuple('Job', ('c', 'p', 'cp'))  # execution time, power dissipation
    job = Job(c_job, p_job, c_job * p_job)
    epsilon = 1e-14
    burst_time = []

    def burst_time_cdf(y):
        result = np.array([1.])
        for _ in range(y):
            result = convolve_rescale_pmf(result, harvest_dist)
        return np.sum(result[burst:])

    def burst_time_pmf(y):
        return burst_time_cdf(y) - burst_time_cdf(y - 1)

    def response_time():
        result = np.array([1.])
        psum = 0.
        y = 0
        while psum < 1. - epsilon:
            burst_time.append(burst_time_pmf(y))
            psum += burst_time[-1]
            y += 1
        for i in range(int(math.ceil(job.cp / burst))):
            result = convolve_rescale_pmf(result, burst_time)
        l = burst / job.p if job.cp % burst == 0 else (job.cp % burst) / job.p
        delta = np.array([0.] * int(l) + [1.0])
        result = convolve_rescale_pmf(result, delta)
        return result

    response = response_time()

    fig = plt.figure(figsize=(4, 6), dpi=200)

    plt.subplot(311)
    plt.title("Energy harvest distribution")
    plt.bar(range(len(harvest_dist)), harvest_dist)

    plt.subplot(312)
    plt.title("Burst inter-arrival time PMF, B = %d" % burst)
    plt.bar(range(len(burst_time)), burst_time)

    plt.subplot(313)
    plt.title("Response time PMF for Job: C = %d, P = %d" % (job.c, job.p))
    plt.plot(range(len(response)), response)

    plt.tight_layout()
    plt.savefig('./figures/energy.png')
    plt.show()


if __name__ == '__main__':
    def uniform_dist(a, b):
        return np.concatenate((np.zeros(a), np.full(b - a + 1, 1. / (b - a + 1))))

    transient_system(harvest_dist=uniform_dist(2, 5), burst=5, c_job=5, p_job=300)
    transient_system(harvest_dist=uniform_dist(2, 5), burst=50, c_job=5, p_job=300)
    transient_system(harvest_dist=uniform_dist(2, 40), burst=50, c_job=5, p_job=300)
    transient_system(harvest_dist=uniform_dist(15, 27), burst=50, c_job=5, p_job=300)

