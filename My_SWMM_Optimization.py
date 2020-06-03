# -*- coding: utf-8 -*-

from pyswmm import Simulation,Nodes,Links
import numpy as np
import scipy.special as sc_special
import time
import datetime

# 运行时有两处需要更改名字

My_timestep=5 #单位：min
SIZE = 288  # 总的优化步长数目，时间=SIZE*My_timestep
HorizonLength=4 # 预测时域长度
my_Start_time=datetime.datetime(2018,7,8,0,0) # 模型开始报告时间

def cuckoo_search(n, m, fit_func, lower_boundary, upper_boundary,My_Current_step,CONTROL_RULES,Current_States, iter_num=50,pa=0.01, beta = 1.5, step_size = 0.1):
    """
    Cuckoo search function
    ---------------------------------------------------
    Input parameters:
        n: Number of nests
        m: Number of dimensions
        fit_func: User defined fitness evaluative function
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        iter_num: Number of iterations (default: 100) 
        pa: Possibility that hosts find cuckoos' eggs (default: 0.25)
        beta: Power law index (note: 1 < beta < 2) (default: 1.5)
        step_size:  Step size scaling factor related to the problem's scale (default: 0.1)
    Output:
        The best solution and its value and the best solution for each iteration
    """
    # print(fit_func)
    # get initial nests' locations 
    nests = generate_nests(n, m, lower_boundary, upper_boundary)
    fitness = calc_fitness(fit_func, nests,My_Current_step,CONTROL_RULES,Current_States)

    # get the best nest and record it
    best_nest_index = np.argmin(fitness)
    best_fitness = fitness[best_nest_index]
    best_nest = nests[best_nest_index].copy()
    nestpara = nests[best_nest_index,:]
    nestparalist = nestpara.tolist()
    nestparalist.append(best_fitness)
    best_nests = nestparalist
    best_nests = np.array([best_nests])
    for _ in range(iter_num):
        nests = update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_size,My_Current_step,CONTROL_RULES,Current_States)
        nests = abandon_nests(nests, lower_boundary, upper_boundary, pa)
        fitness = calc_fitness(fit_func, nests,My_Current_step,CONTROL_RULES,Current_States)
        
        min_nest_index = np.argmin(fitness)
        min_fitness = fitness[min_nest_index]
        min_nest = nests[min_nest_index]
        nestpara = nests[min_nest_index,:]
        nestparalist = nestpara.tolist()
        nestparalist.append(min_fitness)
        nestparalis1 = np.array([nestparalist])
        best_nests = np.append(best_nests,nestparalis1,axis=0)
        if (min_fitness < best_fitness):
            best_nest = min_nest.copy()
            best_fitness = min_fitness

    return best_nest # 新增best_nests储存最优解

def generate_nests(n, m, lower_boundary, upper_boundary):
    """
    Generate the nests' locations
    ---------------------------------------------------
    Input parameters:
        n: Number of nests
        m: Number of dimensions
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
    Output:
        generated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    nests = np.empty((n, m))

    for each_nest in range(n):
        nests[each_nest] = lower_boundary + np.array([np.random.rand() for _ in range(m)]) * (upper_boundary - lower_boundary)

    return nests

def update_nests(fit_func, lower_boundary, upper_boundary, nests, best_nest, fitness, step_coefficient,My_Current_step,CONTROL_RULES,Current_States):
    """
    This function is to get new nests' locations and use new better one to replace the old nest
    ---------------------------------------------------
    Input parameters:
        fit_func: User defined fitness evaluative function
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        nests: Old nests' locations 
        best_nest: Nest with best fitness
        fitness: Every nest's fitness
        step_coefficient:  Step size scaling factor related to the problem's scale (default: 0.1)
    Output:
        Updated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    #print(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape
    # generate steps using levy flight
    steps = levy_flight(n, m, 1.5)
    new_nests = nests.copy()

    for each_nest in range(n):
        # coefficient 0.01 is to avoid levy flights becoming too aggresive
        # and (nest[each_nest] - best_nest) could let the best nest be remained
        step_size = step_coefficient * steps[each_nest] * (nests[each_nest] - best_nest)
        step_direction = np.random.rand(m)
        new_nests[each_nest] += step_size * step_direction
        # apply boundary condtions
        new_nests[each_nest][new_nests[each_nest] < lower_boundary] = lower_boundary[new_nests[each_nest] < lower_boundary]
        new_nests[each_nest][new_nests[each_nest] > upper_boundary] = upper_boundary[new_nests[each_nest] > upper_boundary]

    new_fitness = calc_fitness(fit_func, new_nests,My_Current_step,CONTROL_RULES,Current_States)
    nests[new_fitness < fitness] = new_nests[new_fitness < fitness]
    
    return nests

def abandon_nests(nests, lower_boundary, upper_boundary, pa):
    """
    Some cuckoos' eggs are found by hosts, and are abandoned.So cuckoos need to find new nests.
    ---------------------------------------------------
    Input parameters:
        nests: Current nests' locations
        lower_boundary: Lower bounary (example: lower_boundary = (-2, -2, -2))
        upper_boundary: Upper boundary (example: upper_boundary = (2, 2, 2))
        pa: Possibility that hosts find cuckoos' eggs
    Output:
        Updated nests' locations
    """
    lower_boundary = np.array(lower_boundary)
    upper_boundary = np.array(upper_boundary)
    n, m = nests.shape
    for each_nest in range(n):
        if (np.random.rand() < pa):
            step_size = np.random.rand() * (nests[np.random.randint(0, n)] - nests[np.random.randint(0, n)])
            nests[each_nest] += step_size
            # apply boundary condtions
            nests[each_nest][nests[each_nest] < lower_boundary] = lower_boundary[nests[each_nest] < lower_boundary]
            nests[each_nest][nests[each_nest] > upper_boundary] = upper_boundary[nests[each_nest] > upper_boundary]
    
    return nests

def levy_flight(n, m, beta):
    """
    This function implements Levy's flight.
    ---------------------------------------------------
    Input parameters:
        n: Number of steps 
        m: Number of dimensions
        beta: Power law index (note: 1 < beta < 2)
    Output:
        'n' levy steps in 'm' dimension
    """
    sigma_u = (sc_special.gamma(1+beta)*np.sin(np.pi*beta/2)/(sc_special.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    sigma_v = 1

    u =  np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, sigma_v, (n, m))

    steps = u/((np.abs(v))**(1/beta))
    #print(n)
    return steps

def calc_fitness(fit_func, nests,My_Current_step,CONTROL_RULES,Current_States):
    """
    calculate each nest's fitness
    ---------------------------------------------------
    Input parameters:
        fit_func: User defined fitness evaluative function
        nests:  Nests' locations
    Output:
        Every nest's fitness
    """
    n, m = nests.shape
    fitness = np.empty(n)

    for each_nest in range(n):
        fitness[each_nest] = fit_func(nests[each_nest],My_Current_step,CONTROL_RULES,Current_States)

    return fitness
def mytrape(mynumpy,steps):
    return (sum(mynumpy)*2-mynumpy[0]-mynumpy[-1])/2*steps

def Initial_State():
    sim= Simulation(r"./test.inp")
    links=Links(sim)
    nodes = Nodes(sim)
    Linkname=["C1","C2","O1","O2"]
    Nodename=["St1","St2","J1","J2"]
    T=len(Linkname)+len(Nodename)
    Current_States = np.zeros([T,1])
    t = 0
    for link in Linkname:
       a2 = links[link]
       Current_States[t][0] = a2.initial_flow
       t =t+1
    for node in Nodename:
        a1 = nodes[node]
        Current_States[t][0] = a1.initial_depth
        t =t+1
    sim.close()
    return Current_States

def State_Update(CONTROL_RULES,Current_States,My_Current_step):
    sim= Simulation(r"./test.inp")
    Linkname=["C1","C2","O1","O2"]
    Nodename=["St1","St2","J1","J2"]
    links=Links(sim)
    nodes = Nodes(sim) 
    o1=Links(sim)["O1"]
    o2=Links(sim)["O2"]
    t = 0
    sim.start_time = my_Start_time + datetime.timedelta(minutes=My_timestep*(My_Current_step))
    sim.end_time=my_Start_time + datetime.timedelta(minutes=My_timestep*(My_Current_step+HorizonLength+1))
    sim.step_advance(My_timestep*60)
    for link in Linkname:
        a2 = links[link]
        a2.initial_flow = Current_States[t][0]
        t =t+1          
    for node in Nodename:
        a1 = nodes[node]
        a1.initial_depth = Current_States[t][0] 
        t =t+1 
    i = 0
    for step in sim:
         o1.target_setting= CONTROL_RULES[i+My_Current_step][0]
         o2.target_setting= CONTROL_RULES[i+My_Current_step][1]
         i=i+1
         
    t = 0
    for link in Linkname:
        a2 = links[link]
        Current_States[t][0] = a2.flow
        t =t+1          
    for node in Nodename:
        a1 = nodes[node]
        Current_States[t][0]  = a1.depth
        t =t+1  
    return Current_States

if __name__ == '__main__':
    n=100
    m=HorizonLength*2
    pa_main=0.25
    lower_boundary1=[0,0]
    upper_boundary1=[1,1]
    lower_boundary=[]
    upper_boundary=[]
    v=np.zeros((SIZE,3),dtype=float)
    My_Current_step=0    
    Current_States = Initial_State()
    for i in range(HorizonLength):
        lower_boundary.extend(lower_boundary1.copy())
        upper_boundary.extend(upper_boundary1.copy())
    CONTROL_RULES=np.zeros((SIZE+HorizonLength,2),dtype=float)    
    def fit_func(nest,My_Current_step,CONTROL_RULES,Current_States):
        sim= Simulation(r"./test.inp")
        Linkname=["C1","C2","O1","O2"]
        Nodename=["St1","St2","J1","J2"]
        links=Links(sim)
        nodes = Nodes(sim) 
        t = 0
        for link in Linkname:
            a2 = links[link]
            a2.initial_flow = Current_States[t][0]
            t =t+1          
        for node in Nodename:
            a1 = nodes[node]
            a1.initial_depth = Current_States[t][0] 
            t =t+1        
        flood1=np.array([])
        flood2=np.array([])
        flood3=np.array([])
        flood4=np.array([])
        o1=Links(sim)["O1"]
        o2=Links(sim)["O2"]
        j1 = Nodes(sim)["J1"]
        j2 = Nodes(sim)["J2"]
        j3 = Nodes(sim)["St1"]
        j4 = Nodes(sim)["St2"]
        sim.start_time = my_Start_time + datetime.timedelta(minutes=My_timestep*(My_Current_step))
        sim.end_time=my_Start_time + datetime.timedelta(minutes=My_timestep*(My_Current_step+HorizonLength+1))
        sim.step_advance(My_timestep*60)
        CONTROL_RULES[My_Current_step:My_Current_step+HorizonLength]=nest.reshape([4,-1])
        i=0
        for step in sim:
            o1.target_setting= CONTROL_RULES[i+My_Current_step][0]
            o2.target_setting= CONTROL_RULES[i+My_Current_step][1]
            i=i+1
            flood1 = np.append(flood1,[j1.flooding])
            flood2 = np.append(flood2,[j2.flooding])
            flood3 = np.append(flood3,[j3.flooding])
            flood4 = np.append(flood4,[j4.flooding])
        sim.close()
        return mytrape(flood1,60*My_timestep)+ mytrape(flood2,60*My_timestep)+ mytrape(flood3,60*My_timestep)+ mytrape(flood4,60*My_timestep)
    for i in range(SIZE):
        time_start=time.time()        
        print('第%d轮优化开始'%(i+1))
        best_nest=cuckoo_search(n, m, fit_func, lower_boundary, upper_boundary,i,CONTROL_RULES,Current_States) 
        CONTROL_RULES[i:i+HorizonLength]=best_nest.reshape([4,-1])
        Current_States = State_Update(CONTROL_RULES,Current_States,i)
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
 



