import numpy as np
import matplotlib.pyplot as plt
import random

class Problem:
  def __init__(self, N, f1, f2, cons, s1, s2): # N: number of variables, f1,f2: objectives, c: list of constraints (size=N), s1,s2: solution
    self.N=N
    self.f1=f1 # objectives
    self.f2=f2
    self.cons=np.array(cons) # constraints
    self.s1=s1 # solutions
    self.s2=s2


class ABC:
  def __init__(self, prob, colony_size, L=100, acceleration=1.0, CL_m=1, EA_size_limit=100):
    self.N = prob.N # number of viables (dimension of problem = coordinate dimension)
    self.f1=prob.f1
    self.f2=prob.f2
    self.cons=np.array(prob.cons)
    
    # MOABC params
    self.colony_size=colony_size # == size of employed bees = size of onlooker bees
    self.num_employed=colony_size//2
    self.num_onlooker=colony_size//2
    self.L=L                            # Abandonment Limited param
    self.acceleration=acceleration
    self.CL_m=CL_m                      # Comprehensive learning's m
    self.EA_size_limit=EA_size_limit    # External Archive size
    # self.limited_cd=limited_cd          # limited crowding distance, default = num_objectives/EA_size_limit, 
    # #                                       (objectives already normalized in these two MOP)

    # below will be re-calculated every iterations
    self.food_pos=None
    self.food_counter=None
    # self.food_objectives=None
    # self.food_fitness=None
    self.EA=None

    self.pareto_pos=None

  def init_ABC(self):
    # init food pos
    r=np.random.random_sample((self.num_employed,self.N))
    self.food_pos=(r*(self.cons[:,1]-self.cons[:,0])) + self.cons[:,0]
    self.food_counter=np.zeros(self.num_employed)
    # self.food_fitness=self.getFitness(self.getObjectives(self.food_pos))
    #init EA
    self.EA=np.array([[]])
    self.updateEA(self.food_pos)

  def train(self, iters):
    for _ in range(iters):
      # employed bees
      self.beesRoutine(target=self.food_pos)

      # onlooker bees
        # prepare Comprehensive Learning
      target_m=self.EA[np.random.randint(len(self.EA), size=len(self.food_pos))]
      target_notm=np.array([self.EA[np.random.choice(np.arange(0, len(self.EA)), size=self.N),np.arange(self.N)] for _ in range(len(self.food_pos))])
      CL_mask=[True for _ in range(self.CL_m)]+[False for _ in range(self.CL_m, self.N)]
      CL_masks=np.array([random.sample(CL_mask,len(CL_mask)) for _ in range(len(self.food_pos))])
      target = target_notm
      target[CL_masks]=target_m[CL_masks]
        #
      self.beesRoutine(target=target)

      # scout bees
        # abandon food pos
      indices=self.food_counter>self.L
      r=np.random.random_sample((self.num_employed,self.N))
      rand_food_pos=(r*(self.cons[:,1]-self.cons[:,0])) + self.cons[:,0]
      self.food_pos[indices]=rand_food_pos[indices]
      self.updateEA(self.food_pos[indices])
      self.food_counter[indices]=0.
    return self.getObjectives(self.food_pos)

  def getObjectives(self, Ps):
    f1s = self.f1(Ps)
    f2s = self.f2(Ps)
    return np.array([f1s, f2s])
  
  def getSigma(self, objectives):
    o1s2=np.power(objectives[0],2)
    o2s2=np.power(objectives[1],2)
    return (o1s2-o2s2)/(o1s2+o2s2+1e-6)

  # def getFitness(self, objectives):
  #   return -1*(objectives[0]+objectives[1]) # my own definition # the more the better 

  def reposToConstraints(self, Ps):
    for i in range(len(Ps)):
      indices=Ps[i]<self.cons[:,0]
      # print(indices)
      Ps[i][indices]=self.cons[:,0][indices]
      indices=Ps[i]>self.cons[:,1]
      Ps[i][indices]=self.cons[:,1][indices]
    return Ps

  def beesRoutine(self, target=None):
    assert(not target is None)
    r1=np.random.random_sample((len(self.food_pos),1))
    new_pos=self.food_pos+r1*self.acceleration*(target-self.food_pos)
    new_pos = self.reposToConstraints(new_pos)
    selected_idx = self.greedySelect(self.food_pos, new_pos)
    self.food_pos[selected_idx]=new_pos[selected_idx]
    self.updateEA(self.food_pos[selected_idx])
    self.food_counter=self.food_counter+np.logical_not(selected_idx)


  def greedySelect(self, old_pos, new_pos):
    old_objs=self.getObjectives(old_pos).T
    new_objs=self.getObjectives(new_pos).T
    dominant_idx=np.logical_and(new_objs[:,0]>old_objs[:,0], new_objs[:,1]>old_objs[:,1])
    nondominant_idx=np.logical_or(new_objs[:,0]>old_objs[:,0], new_objs[:,1]>old_objs[:,1])
    half_selection=np.random.choice(a=[False, True], size=len(new_pos))
    nondominant_idx=np.logical_and(nondominant_idx,half_selection)
    selected_idx = np.logical_or(dominant_idx, nondominant_idx)
    return selected_idx

  def updateEA(self, Ps):
    # Evaluate new food position(s) and add if is non-dominated
    if self.EA.size==0:
      self.EA=self.getNonDominant(Ps)
    else:
      self.EA=self.getNonDominant(np.concatenate([self.EA,Ps]))
    self.EAPopControl()
    
  def getNonDominant(self, Ps):
    # calculate ranking and sort by ranking
    Ps=np.unique(Ps, axis=0)
    objs=np.concatenate((self.getObjectives(Ps).T,Ps), axis=1)
    objs=objs[objs.argsort(axis=0)[:,0]]
    firstRanks=[objs[0]]
    for entry in objs[1:]:
      if entry[1]> firstRanks[-1][1]:
        continue
      firstRanks.append(entry)
    firstRanks=np.array(firstRanks)
    return firstRanks[:,2:2+self.N]


  def EAPopControl(self):
    # Use crowd distancing to reduce population of EA if it is exceeding EA_size_limit
    if len(self.EA) <= self.EA_size_limit:
      return 0
    exceeding=len(self.EA)-self.EA_size_limit
    dist=np.zeros(len(self.EA))
    objs=np.concatenate((self.getObjectives(self.EA).T,np.array([np.arange(len(self.EA))]).T), axis=1)
    
    # print(objs.shape, len(self.EA), len(dist))
    objs=objs[objs[:,1].argsort(axis=0)]
    for i in range(1,len(self.EA)-1):
      cur_idx=int(objs[i][-1])
      # print(cur_idx, i)
      dist[cur_idx]+=objs[i,1]-objs[i-1,1]
      dist[cur_idx]+=objs[i+1,1]-objs[i,1]
    dist[int(objs[0,-1])]=1000 # infinite crowd distance for boundary solutions
    dist[int(objs[-1,-1])]=1000
    
    objs=objs[objs[:,0].argsort(axis=0)]
    for i in range(1,len(self.EA)-1):
      cur_idx=int(objs[i,-1])
      dist[cur_idx]+=objs[i,0]-objs[i-1,0]
      dist[cur_idx]+=objs[i+1,0]-objs[i,0]
    dist[int(objs[0,-1])]=1000
    dist[int(objs[-1,-1])]=1000
    dist=dist/2
    self.EA=self.EA[dist.argsort(axis=0)][::-1][:self.EA_size_limit]

def MOP_f1(particles):
  return particles[:,0] # x1

def MOP_f2(particles):
  num_particle, N = particles.shape
  g=1+10*(N-1)+np.sum(np.power(particles[:,1:],2)-10*np.cos(2*np.pi*particles[:,1:]), axis=1)
  h=np.zeros(num_particle)
  indices=g>particles[:,0]
  # h[g>particles[:,0]]=1-np.power(particles[:,0]/g,0.5)
  h[indices]=1-np.power(particles[:,0][indices]/g[indices],0.5)
  # h[h<0.0]=0.0
  f2=g*h
  return f2

def Concave_f1(particles):
  return particles[:,0] # x1

def Concave_f2(particles, alpha):
  num_particle, N = particles.shape
  g=1+10*(np.sum(particles[:,1:], axis=1)/(N-1))
  h=np.zeros(num_particle)
  indices=g>particles[:,0]
  h[indices]=1-np.power(particles[:,0][indices]/g[indices],alpha)
  # h[h<0.0]=0.0
  f2=g*h
  return f2


if __name__=='__main__':
  np.random.seed(0)

  ######################################################
  # Problem setting
  n=3
  cons1=[[0,1]]+[[-30,30] for _ in range(n-1)]
  prob1=Problem(n, MOP_f1, MOP_f2, cons1, s1=lambda x: x, s2=lambda x: 1-x**0.5) # for abc1

  alpha=2
  cons2=[[0,1]]+[[0,1] for _ in range(n-1)]
  prob2=Problem(n, Concave_f1, lambda x: Concave_f2(x,alpha), cons2, s1=lambda x: x, s2=lambda x: 1-x**alpha) # for abc2,3

  # ABC # Solution
    # prob1
  # solx=np.linspace(0,1,50)
  # soly=prob1.s2(solx)
  # abc = ABC(prob1, 10000, acceleration=1.5)
    # prob2
  solx=np.linspace(0,1,50)
  soly=prob2.s2(solx)
  abc = ABC(prob2, 10000, acceleration=1.5) 
  
  abc.init_ABC()
  obj=abc.train(100)
  #####################################################

  # benchmark
  EA_obj=abc.getObjectives(abc.EA)
  sigmas=abc.getSigma(obj)
  EA_sigma=abc.getSigma(EA_obj)
  mx=max(max(sigmas),max(EA_sigma))
  mn=min(min(sigmas),min(EA_sigma))
  print("Min, Max sigmas = ", mn, mx)
  print("Sigmas range = ", mx-mn)

  # Visualization
    # plot solution
  plt.plot(solx,soly, c='black', linewidth=0.5, ls='--', label='solution')
    # plot particles
  plt.scatter(x=obj[0],y=obj[1], alpha=0.7, s=3, label='bees')
    # plot guided particle (global best particle of each region in sigma method)
  plt.scatter(x=EA_obj[0],y=EA_obj[1], alpha=0.7, edgecolors='red', marker='^', facecolors='none', label='EA')

  plt.xlabel('f1')
  plt.ylabel('f2')
  plt.legend(loc="upper right")
  plt.ylim(0,1)
  plt.xlim(0,1)
  plt.show()

## the paper didn't mention about how to define the personal best.
