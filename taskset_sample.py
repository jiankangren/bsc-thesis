import numpy as np


class task:
    def __init__(self, c_low=None, c_high=None, period=None, crit=False,deadline=None, taskclass=None):
        self.c_low=c_low
        self.c_high=c_high
        self.period=period
        self.deadline=deadline
        if deadline==None:
            self.deadline=period
        self.crit=crit
        self.next_arrival=0.0
        self.done_comps=0.00
        self.c_current=self.c_low
        self.core=-1
        self.taskclass=taskclass
    def displayTask(self):
        print ("Computation Lo : ", self.c_low,  ", Computation Hi: ", self.c_high, ", Period: ", self.period, ", Deadline: ",self.deadline, "Criticality: ", self.crit, "Task Class", self.taskclass)
    def highUtil(self):
        return (self.c_high/self.period if self.crit==True else 0.00)
    def lowUtil(self):
        return (self.c_low/self.period)

def compareUtil(a):
    return a.lowUtil()

def uunifast(util, n):
    sumU = util
    vectU=[]
    for i in range(int(n)-1):
        nextSumU = sumU*np.random.random()**(1.0/(float(n-i)))
        vectU.append(sumU - nextSumU)
        sumU = nextSumU
    vectU.append(sumU)
    return vectU

def uunifastDiscard(util, n,maxutil=1.0):
    utils=list(np.ones(n)*maxutil)
    while(max(utils)>=maxutil):
        utils=uunifast(util,n)    
    return utils

def generateTaskset(util, n,pLow, pHigh,maxutil=1.0, periodMode=1): #pLow and pHigh are given in muS
    taskset=[]
    utilization=uunifastDiscard(util, n,maxutil)
    periods=list(np.random.randint(pLow, pHigh,n)) if periodMode==0 \
            else [pLow*(2**p) for p in np.random.randint(1,int(np.log2(pHigh/pLow))+1, n)] if periodMode==1\
            else list(np.ones(n))*pLow    
    for i,u in enumerate(utilization):
        taskset.append(task(c_low=periods[i]*u, c_high=periods[i]*u, \
                        period= periods[i], deadline=periods[i], taskclass=0))        
    return taskset

taskset = generateTaskset(0.5, 100, 0.5, 0.8, maxutil=0.8, periodMode=1)
for t in taskset:
    t.displayTask()

