"""
This module offers a simple script to show an animation of a random stochastic task set. This serves to visualize the
concepts of convolution and shrinking.
"""

import generation as gen
from analysis import BacklogSim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ts = gen.mc_fairgen_stoch(0, u_lo=0.95, mode='avg')
ts.set_priorities_rm()
ts.set_rel_times()

sim = BacklogSim(task_set=ts)
dt = ts.hyperperiod // 2000  # 30 fps
xrange = len(sim.backlog)
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, ts.hyperperiod), ylim=(10 ** -6, 1.))
plt.yscale('log')
line, = ax.plot([0], [1.0])
k_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    k_text.set_text('')
    time_text.set_text('')
    return line, k_text, time_text


def animate(i):
    global sim, dt, xrange, ax
    sim.step(dt, mode='before')
    # xrange = max(xrange, len(sim.backlog))
    # ax.set_xlim(0, xrange)
    x_val = range(len(sim.backlog))
    y_val = sim.backlog
    line.set_data(x_val, y_val)
    k_text.set_text('hyperperiod = %d' % sim.k)
    time_text.set_text('time = %d' % sim.t)
    return line, k_text, time_text

ani = animation.FuncAnimation(fig, animate, frames=20 * ts.hyperperiod, interval=30, blit=True, init_func=init)
plt.show()
