from math import ceil


class Task(object):
    """Input for the analysis."""
    def __init__(self, criticality, period, phase, deadline, c_hi, c_lo, priority):
        self.criticality = criticality
        self.period = period
        self.phase = phase
        self.deadline = deadline
        self.c = {'HI': c_hi, 'LO': c_lo}
        self.priority = priority


def initialize_task_set(path):
    """Reads task set from path."""
    # Sample task set:
    t1 = Task('LO', 2, 0, 2, 1, 1, 1)
    t2 = Task('HI', 10, 0, 10, 5, 1, 2)
    t3 = Task('HI', 100, 0, 100, 20, 20, 3)
    return[t1, t2, t3]


def scheduling_analysis_smc(task_set):
    """SMC Response Time Analysis, as mentioned in [2]."""

    def min_crit(c1, c2):
        """Returns the lower of both criticality levels."""
        if c1 == 'HI' and c2 == 'HI':
            return 'HI'
        else:
            return 'LO'

    for task in task_set:
        # 1) Build set of tasks with higher priority:
        hp = []
        for j in task_set:
            if j.priority < task.priority:
                hp.append(j)

        # 2) Iteration to fixed point for solving recurrence relation:
        task.r = 0
        r_next = -1
        while task.r != r_next:
            task.r = r_next
            if task.r > task.deadline:  # task.r growing monotonically
                return False
            sum_hp = task.c[task.criticality]
            for j in hp:
                sum_hp += ceil(task.r / j.period) * j.c[min_crit(task.criticality, j.criticality)]
            r_next = sum_hp

    # No deadline overruns happened:
    return True


def scheduling_analysis_amc(task_set):
    """AMC-rtb Response Time Analysis, as mentioned in [2]."""
    for task in task_set:
        # 1.1) Build set of all tasks with higher priority:
        hp = []
        for j in task_set:
            if j.priority < task.priority:
                hp.append(j)
        # 1.2) Build sets of all HI- and LO-critical tasks with higher priority:
        hp_hi = [t for t in hp if t.criticality == 'HI']
        hp_lo = [t for t in hp if t.criticality == 'LO']

        # 2) Iteration to fixed point for solving recurrence relation:
        # 2.1) R_LO:
        task.r_lo = 0
        r_next = -1
        while task.r_lo != r_next:
            task.r_lo = r_next
            if task.r_lo > task.deadline:
                return False
            sum_hp = task.c[task.criticality]
            for j in hp:
                sum_hp += ceil(task.r_lo / j.period) * j.c['LO']
            r_next = sum_hp

        # 2.2 R_HI (only defined for HI-critical tasks):
        if task.criticality == 'HI':
            task.r_hi = 0
            r_next = -1
            while task.r_hi != r_next:
                task.r_hi = r_next
                if task.r_hi > task.deadline:
                    return False
                sum_hp = task.c['HI']
                for j in hp_hi:
                    sum_hp += ceil(task.r_hi / j.period) * j.c['HI']
                r_next = sum_hp

        # 2.3 R_* (criticality change, only defined for HI-critical tasks):
        if task.criticality == 'HI':
            task.r_asterisk = 0
            r_next = -1
            while task.r_asterisk != r_next:
                task.r_asterisk = r_next
                if task.r_asterisk > task.deadline:
                    return False
                sum_hp = task.c['HI']
                for j in hp_hi:
                    sum_hp += ceil(task.r_asterisk / j.period) * j.c['HI']
                for k in hp_lo:
                    sum_hp += ceil(task.r_lo / k.period) * k.c['LO']
                r_next = sum_hp

    # No deadline overruns happened:
    return True


def scheduling_analysis_edf_vd(task_set):
    """Calculated according to theorem 1 in [3]."""

    def total_k_utilization(task_subset, k):
        """Total utilization at scenario criticality level k of tasks contained in task_subset."""
        result = 0.0
        for j in task_subset:
            result += float(j.c[k]) / float(j.period)
        return result

    subset_lo = [t for t in task_set if t.criticality == 'LO']
    subset_hi = [t for t in task_set if t.criticality == 'HI']
    u_1_1 = total_k_utilization(subset_lo, 'LO')
    u_2_1 = total_k_utilization(subset_hi, 'LO')
    u_2_2 = total_k_utilization(subset_hi, 'HI')

    return u_1_1 + min(u_2_2, u_2_1 / (1 - u_2_2)) <= 1     # Note that this condition is sufficient for schedulability.


def scheduling_analysis(task_set_path, scheme):
    """Wrapper function"""

    task_set = initialize_task_set(task_set_path)
    if scheme == 'smc':
        return scheduling_analysis_smc(task_set)
    elif scheme == 'amc':
        return scheduling_analysis_amc(task_set)
    elif scheme == 'edf-vd':
        return scheduling_analysis_edf_vd(task_set)
    # Add new scheduling schemes here.
    else:
        pass

print(scheduling_analysis("", 'smc'))
print(scheduling_analysis("", 'amc'))
print(scheduling_analysis("", 'edf-vd'))

"""
Literature:
[1] Audsley, Burns, Richardson, Tindell, Wellings
    Applying new scheduling theory to static priority preemptive scheduling.

[2] Baruah, Burns, Davis
    Response-Time Analysis for Mixed Criticality Systems
    
[3] Baruah, Bonifaci, D'Angelo, Marchetti-Spaccamela, van der Ster, Stougie
    Mixed-Criticality Scheduling of Sporadic Task Systems
"""