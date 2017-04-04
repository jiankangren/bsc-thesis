from math import ceil


class Task(object):
    """Input for the analysis."""
    def __init__(self, criticality, priority, period, phase, deadline, c_hi, c_lo):
        self.criticality = criticality
        self.priority = priority
        self.period = period
        self.phase = phase
        self.deadline = deadline
        self.c = {'HI': c_hi, 'LO': c_lo}


def min_crit(c1, c2):
    """Returns the lower of both criticality levels."""
    if c1 == 'HI' and c2 == 'HI':
        return 'HI'
    else:
        return 'LO'


def solve_response_time(task, task_set):
    """Iteratively solve recurrence relation, see Audsley et al."""
    l_i = task.criticality
    c_i = task.c[l_i]
    hp = []
    for j in task_set:
        if j.priority < task.priority:
            hp.append(j)
    r_i = 0
    r_i_next = -1

    while r_i != r_i_next:
        r_i = r_i_next
        sigma = c_i
        for j in hp:
            sigma += ceil(r_i / j.period) * j.c[min_crit(l_i, j.criticality)]
        r_i_next = sigma

    return r_i


def response_time_analysis(task_set):
    """SMC RTA, as mentioned by Baruah et al."""
    for tau in task_set:
        if solve_response_time(tau, task_set) > tau.deadline:
            return False
    return True

# Sample task set:
t1 = Task('LO', 1, 2, 0, 2, 1, 1)
t2 = Task('HI', 2, 10, 0, 10, 2, 1)
t3 = Task('HI', 3, 100, 0, 100, 20, 20)
sample_set = [t1, t2, t3]

print(response_time_analysis(sample_set))
