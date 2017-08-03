import math
import numpy as np
import collections as col
import matplotlib.pyplot as plt

from analysis import convolve_rescale_pmf


def uniform_dist(a, b):
    """"""
    return np.concatenate((np.zeros(a), np.full(b - a + 1, 1. / (b - a + 1))))


def transient_system(harvest_dist, burst, c_job, p_job):
    """
     
    """
    Job = col.namedtuple('Job', ('c', 'p', 'cp'))  # execution time, power dissipation
    job = Job(c_job, p_job, c_job * p_job)
    epsilon = 1e-14
    burst_time = []

    def burst_time_cdf(y):
        result = np.array([1.])
        for i in range(y):
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
    # response = []
    # psum = 0.
    # y = 0
    # while psum < 1. - epsilon:
    #     response.append(response_time(y))
    #     psum += response[-1]
    #     y += 1

    print(harvest_dist)
    print(burst_time)
    print(response)

    fig = plt.figure(figsize=(20, 10), dpi=200)

    plt.subplot(311)
    plt.title("Energy harvest distribution")
    plt.bar(range(len(harvest_dist)), harvest_dist)

    plt.subplot(312)
    plt.title("Burst inter-arrival time distribution, burst size = %d" % burst)
    plt.bar(range(len(burst_time)), burst_time)

    plt.subplot(313)
    plt.title("Response time distribution for Job: Cj = %d, Pj = %d" % (job.c, job.p))
    plt.bar(range(len(response)), response)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

transient_system(uniform_dist(2, 5), 5, 30, 3)
