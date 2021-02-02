import numpy as np
import matplotlib.pyplot as plt

class Problem:
  def __init__(self, N, f1, f2, cons, s1, s2): # N: number of variables, f1,f2: objectives, c: list of constraints (size=N), s1,s2: solution
    self.N=N
    self.f1=f1 # objectives
    self.f2=f2
    self.cons=np.array(cons) # constraints
    self.s1=s1 # solutions
    self.s2=s2


# could have defined 'Particle' as a class separately but numpy fast matrix operation cannot be used.
class PSO:
  def __init__(self, prob, num_particle, w=0.4, C1=2, C2=2,sigma_bins=[-1.0,-0.930, -0.766, -0.174, 0.5, 0.940, 1.0]):
    self.N = prob.N # number of viables (dimension of each particle)
    self.f1=prob.f1
    self.f2=prob.f2
    self.cons=np.array(prob.cons)
    
    # PSO params
    self.num_particle=num_particle
    self.w=w
    self.C1=C1
    self.C2=C2

    # below will be re-calculated every iterations
    self.paricles=None
    self.velocities=None
    self.fitnesses=None
    self.sigmas=None
    self.personal_best_particles=None
    self.personal_best_fitnesses=None
    # self.global_best=None
    self.sigma_bins=np.array(sigma_bins)
    self.guided_particles=None # size = sigma_bins
    self.guided_fitnesses=None

    self.pareto_particles=None

  def init_PSO(self):
    r=np.random.random_sample((self.num_particle,self.N))
    # each particles
    self.particles=(r*(self.cons[:,1]-self.cons[:,0])) + self.cons[:,0]
    self.velocities=np.zeros(self.particles.shape)
    objectives=self.getObjectives(self.particles)
    self.fitnesses=self.getFitness(objectives)
    self.sigmas=self.getSigma(objectives)

    # particles' personal best
    self.personal_best_particles=self.particles.copy()
    self.personal_best_fitnesses=self.getFitness(objectives)

    # Guided Particles for sigma method (<-> global best)
    # self.guided_particles=np.array([None for _ in range(self.sigma_bins.shape[0]+1)])
    self.guided_particles=np.zeros((self.sigma_bins.shape[0]+1, self.N))
    self.guided_fitnesses=np.zeros(self.guided_particles.shape[0])-1e8
    self.updateGuided(self.particles, self.fitnesses, self.sigmas)


  def train(self, iters):
    for _ in range(iters):
      objectives=self.getObjectives(self.particles)
      self.fitnesses=self.getFitness(objectives)
      self.sigmas=self.getSigma(objectives)
      # print(self.sigmas)
      self.updateGuided(self.particles, self.fitnesses, self.sigmas)
      self.updatePersonalbest()

      r1=np.random.random_sample((self.particles.shape[0],1)) # each particle have different ratio of acceleration
      r2=np.random.random_sample((self.particles.shape[0],1))
      assigned_guided_particles=self.getGuided(self.particles, self.fitnesses, self.sigmas)
      # print(objectives)
      self.velocities=self.w*self.velocities \
            +self.C1*r1*(self.personal_best_particles-self.particles)\
            +self.C2*r2*(assigned_guided_particles-self.particles)
      self.particles=self.particles+self.velocities
      self.reposToConstraints() # Push particles that are out of bound back to the constraint boundaries
    
    objectives=self.getObjectives(self.particles)
    self.fitnesses=self.getFitness(objectives)
    self.sigmas=self.getSigma(objectives)
    self.updateGuided(self.particles, self.fitnesses, self.sigmas)
    return objectives

  def getObjectives(self, Ps):
    f1s = self.f1(Ps)
    f2s = self.f2(Ps)
    return np.array([f1s, f2s])

  def getFitness(self, objectives):
    return -1*(objectives[0]+objectives[1]) # my own definition # the more the better 

  def getSigma(self, objectives):
    o1s2=np.power(objectives[0],2)
    o2s2=np.power(objectives[1],2)
    return (o1s2-o2s2)/(o1s2+o2s2+1e-12)

  def reposToConstraints(self):
    for i in range(self.num_particle):
      indices=self.particles[i]<self.cons[:,0]
      # print(indices)
      self.particles[i][indices]=self.cons[:,0][indices]
      indices=self.particles[i]>self.cons[:,1]
      self.particles[i][indices]=self.cons[:,1][indices]
      

  def updatePersonalbest(self): ###
    indices = self.fitnesses>self.personal_best_fitnesses
    self.personal_best_particles[indices]=self.particles[indices]
    self.personal_best_fitnesses[indices]=self.fitnesses[indices]

  def updateGuided(self, Ps, Fs, Ss): # O(num_particles * log bin_size)
    # Ps, Fs, Ss : particles, fitnesses, sigmas
    for p,f,s in zip(Ps, Fs, Ss):
      idx = np.searchsorted(self.sigma_bins, s)
      if f>self.guided_fitnesses[idx]:
        self.guided_fitnesses[idx]=f
        self.guided_particles[idx]=p

  def getGuided(self, Ps, Fs, Ss):
    valid_bin=np.unique(np.searchsorted(self.sigma_bins, Ss))
    guided_idx = valid_bin[np.random.randint(valid_bin.shape[0], size=Ps.shape[0])] # randomly assigned
    for i in range(Ps.shape[0]):
      idx = np.searchsorted(self.sigma_bins, Ss[i])
      # if the particle is already guiding particle itself, don't assign to other sigma bins' guiding particle
      if Fs[i]==self.guided_fitnesses[idx]: 
        guided_idx[i]=idx 
    return self.guided_particles[guided_idx]



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
  prob1=Problem(n, MOP_f1, MOP_f2, cons1, s1=lambda x: x, s2=lambda x: 1-x**0.5) # for pso1

  alpha=2
  cons2=[[0,1]]+[[0,1] for _ in range(n-1)]
  prob2=Problem(n, Concave_f1, lambda x: Concave_f2(x,alpha), cons2, s1=lambda x: x, s2=lambda x: 1-x**alpha)

  # PSO # Solution
    # prob1
  # solx=np.linspace(0,1,50)
  # soly=prob1.s2(solx)
  # # pso = PSO(prob1, 10000) # defailt 6 bins
  # pso = PSO(prob1, 10000, sigma_bins=[-1.0,-0.99, -0.98, -0.96, -0.930, -0.766, -0.174, 0.5, 0.940, 0.96, 0.98, 0.99, 1.0])
    # prob2
  solx=np.linspace(0,1,50)
  soly=prob2.s2(solx)
  # pso = PSO(prob2, 10000) # default 6 bins
  pso = PSO(prob2, 10000, sigma_bins=[-1.0,-0.99, -0.98, -0.96, -0.930, -0.766, -0.174, 0.5, 0.940, 0.96, 0.98, 0.99, 1.0])
  
  pso.init_PSO()
  obj=pso.train(100)
  #####################################################

  # benchmark
  guided_obj=pso.getObjectives(pso.guided_particles[pso.guided_fitnesses>-1e3])
  sigmas=pso.getSigma(obj)
  guided_sigma=pso.getSigma(guided_obj)
  # print(guided_sigma)
  mx=max(max(sigmas),max(guided_sigma))
  mn=min(min(sigmas),min(guided_sigma))
  print("Min, Max sigmas = ", mn, mx)
  print("Sigmas range = ", mx-mn)

  # Visualization
    # plot solution
  plt.plot(solx,soly, c='black', linewidth=0.5, ls='--', label='solution')
    # plot particles
  plt.scatter(x=obj[0],y=obj[1], alpha=0.7, s=3, label='particle')
    # plot guided particle (global best particle of each region in sigma method)
  plt.scatter(x=guided_obj[0],y=guided_obj[1], alpha=0.7, edgecolors='red', marker='^', facecolors='none', label='guided particle')

  plt.xlabel('f1')
  plt.ylabel('f2')
  plt.legend(loc="upper right")
  plt.ylim(0,1)
  plt.xlim(0,1)
  plt.show()

## the paper didn't mention about how to define the personal best.