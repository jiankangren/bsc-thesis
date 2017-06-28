import class_lib
import generation as gen
import analysis as ana
import numpy as np
import math
import pickle
import collections as col
import matplotlib.pyplot as plt

def schedulability_rates(util, nsets=100):
    m = math.ceil(util)
    util /= m
    results = {'d_smc': [], 'd_amc': [], 'p_smc': [], 'p_amc_bb': []}
    print(util, m)
    for i in range(nsets):
        print(i)
        task_set = gen.mc_fairgen_stoch(i, u_lo=util, m=m, implicit_deadlines=True)
        task_set.set_priorities_rm()
        # task_set.draw()
        results['d_smc'].append(ana.d_smc(task_set))
        results['d_amc'].append(ana.d_amc(task_set))
        results['p_smc'].append(ana.p_smc(task_set))
        results['p_amc_bb'].append(ana.p_amc_black_box(task_set))
    return (sum(results['d_smc']) / nsets, sum(results['d_amc']) / nsets,
            sum(results['p_smc']) / nsets, sum(results['p_amc_bb']) / nsets)


# nsets = 100
# utils = [0.95 + k * 0.05 for k in range(4)]
# task_sets = []
#
#
# for idx, util in enumerate(utils):
#     task_sets.append([])
#     for i in range(nsets):
#         task_set = gen.mc_fairgen_stoch(idx*nsets + i, u_lo=util, implicit_deadlines=True)
#         task_set.set_priorities_rm()
#         task_sets[idx].append(task_set)
#         # print(idx, i)
#         pass
# # pickle.dump(task_sets, open("task_sets", 'wb'))
# # task_sets = pickle.load(open("task_sets", 'rb'))
#
# success_rates = []
#
# for idx, util in enumerate(utils):
#     results = {'d_smc': [], 'd_amc': [], 'p_smc': [], 'p_amc_bb': []}
#     for task_set in task_sets[idx]:
#         print(idx, task_set.id)
#         results['d_smc'].append(ana.d_smc(task_set))
#         results['d_amc'].append(ana.d_amc(task_set))
#         results['p_smc'].append(ana.p_smc(task_set))
#         results['p_amc_bb'].append(ana.p_amc_black_box(task_set))
#     success_rates.append((util, np.sum(results['d_smc']), np.sum(results['d_amc']),
#           np.sum(results['p_smc']), np.sum(results['p_amc_bb'])))
#     print(success_rates[idx])

# pickle.dump(success_rates, open("success_rates", 'wb'))

utils = [0.8 + k * 0.01 for k in range(20)] + [1.0 + k * 0.05 for k in range(11)]
rates_d_smc, rates_d_amc, rates_p_smc, rates_p_amc_bb = zip(*[schedulability_rates(util, nsets=100) for util in utils])
ax_d_smc, ax_d_amc, ax_p_smc, ax_p_amc_bb = plt.plot(utils, rates_d_smc, 'b-o', utils, rates_d_amc, 'r-s',
                                                     utils, rates_p_smc, 'g-^', utils, rates_p_amc_bb, 'm-*')
plt.legend((ax_d_smc, ax_d_amc, ax_p_smc, ax_p_amc_bb), ('dSMC', 'dAMC', 'pSMC', 'pAMC-BB'), loc='center left')
plt.axvline(1.0, color='black', linestyle='--')
plt.show()
