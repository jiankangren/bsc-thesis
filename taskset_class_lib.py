class Task(object):
    """Input for the analysis."""
    def __init__(self, criticality, period, deadline, c_hi=None, c_lo=None, phase=None, priority=None):
        self.criticality = criticality
        self.period = period
        self.phase = phase
        self.deadline = deadline
        self.c = {'HI': c_hi, 'LO': c_lo}
        self.priority = priority


class TaskSet(object):
    """Set of sporadic tasks."""
    def __init__(self, set_id, tasks: [Task] = None, scheduled: bool = False):
        self.id = set_id
        self.tasks = tasks
        self.scheduled = scheduled
