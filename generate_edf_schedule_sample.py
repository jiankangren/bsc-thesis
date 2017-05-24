from taskset import *
from fractions import gcd
import sys

def earliest_deadline(ts, time):
    taskToExecute=-1;    
    current_deadline = sys.maxsize
    for i,t in enumerate(ts):
        if t.next_arrival <= time and (t.next_arrival+t.deadline) < current_deadline:
            current_deadline=t.next_arrival + t.deadline
            taskToExecute=i
    return taskToExecute


def lcm(nums): #assumes i to be a list of numbers
    x=1
    for n in nums:
        x=x*n/gcd(x,n)
    
    return x


def edf_schedule(ts, hyperperiod=-1, overhead=0): # idle time is signalled by taskid -1
    if hyperperiod==-1:
        hyperperiod=lcm([t.period for t in ts])
    
    schedule=[]
    arrivals=[]
    
    for t in ts:
        arrivals.extend([t.period*i for i in range(hyperperiod/t.period +1)])
    arrivals=sorted(list(set(arrivals)))
    for t in ts:
        t.next_arrival=0
        t.done_comps=0

    start=0
    preemptionOverhead=0
    while start < hyperperiod:
        i=earliest_deadline(ts, start)
        nextArrivalBoundry = min([x for x in arrivals if x>start])
        if i>=0:
            remainingComps=ts[i].c_current-ts[i].done_comps
            if (start+remainingComps+preemptionOverhead)<nextArrivalBoundry: # task will finish
                schedule.append([start, start + remainingComps + preemptionOverhead, i]) # start, end, task
                ts[i].next_arrival+= ts[i].period
                ts[i].done_comps=0
                start=start+remainingComps+ preemptionOverhead
                preemptionOverhead=0
            else: # task may not run to completion
                schedule.append([start, nextArrivalBoundry, i])
                ts[i].done_comps=ts[i].done_comps+max(nextArrivalBoundry-start-preemptionOverhead, 0)
                start=nextArrivalBoundry
                preemptionOverhead = overhead if earliest_deadline(ts, start) != i else 0                    
        else:
            preemptionOverhead=0
            schedule.append([start, nextArrivalBoundry, -1])
            start=nextArrivalBoundry
            
    consolidatedSchedule=[schedule[0]]
    for i, s in enumerate(schedule[:-1]):
        if consolidatedSchedule[-1][1]==schedule[i+1][0] and consolidatedSchedule[-1][2]== schedule[i+1][2]: 
            consolidatedSchedule[-1][1]=schedule[i+1][1]
        else:
            consolidatedSchedule.append(schedule[i+1])
    
    status=1 if sum([1 if t.next_arrival == hyperperiod else 0 for t in ts])==len(ts) else 0
        
    return consolidatedSchedule, status



