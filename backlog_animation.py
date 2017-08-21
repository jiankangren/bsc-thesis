"""
This module offers a simple script to show an animation of a random stochastic task set. This serves to visualize the
concepts of convolution and shrinking.
"""

import synthesis as synth
from analysis import BacklogSim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

if __name__ == '__main__':

    ####################
    # Example Task Set #
    ####################

    ts = synth.mc_fairgen(0, 2.5)
    synth.synth_c_pmf(ts)
    synth.set_fixed_priorities(ts)

    #############
    # Animation #
    #############

    sim = BacklogSim(task_set=ts)
    dt = 1
    fig = plt.figure(figsize=(16, 9), dpi=180)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, ts.hyperperiod), ylim=(1e-14, 1))
    ax.set_title(ts.description)
    plt.xlabel('Backlog size')
    plt.ylabel('Probability')
    plt.yscale('log')
    line, = ax.plot([0], [1.0], linestyle='', marker='s')
    k_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=18, ha='right')
    time_text = ax.text(0.98, 0.9, '', transform=ax.transAxes, fontsize=18, ha='right')


    def init():
        line.set_data([], [])
        k_text.set_text('')
        time_text.set_text('')
        return line, k_text, time_text


    def animate(i):
        global sim, dt, ax
        sim.step(dt, mode='before')
        x_val = range(len(sim.backlog))
        y_val = sim.backlog
        line.set_data(x_val, y_val)
        k_text.set_text('hyperperiod = %d' % sim.k)
        time_text.set_text('time = %d' % sim.t)
        return line, k_text, time_text

    t0 = time()
    animate(0)
    t1 = time()
    interval = 20 * dt - (t1 - t0)
    ani = animation.FuncAnimation(fig, animate, frames=ts.hyperperiod*20, interval=interval, blit=True, init_func=init)
    plt.show()
