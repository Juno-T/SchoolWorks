import cv2
import numpy as np
import matplotlib.pyplot as plt

class evo:
  def __init__(self, img, numTri=10, generationSize=10, crossoverProb=0.5, mutationProb=0.05):
    self.numTri=numTri
    self.bandSize=255//(self.numTri+1)
    self.TriAlpha=1/(self.numTri)
    self.imgShape=img.shape

    ## Evolutionary algorithm setup
    self.goal=self._calcGoal(img)
    self.generationSize=generationSize
    self.crossoverProb=crossoverProb
    self.mutationProb=mutationProb
    self.generation=None
    self.generationNumber=0
    self.probDist = []
    accumProb=0.0
    for i in range(self.generationSize-1):
      self.probDist.append((1-crossoverProb)*(1-accumProb))
      accumProb+=self.probDist[i]
    self.probDist.append(1-accumProb)
    self.probDist=sorted(self.probDist)[::-1]
    # self.unit=np.floor(np.random.rand(numTri,3,2)*[[np.array(self.imgShape)]])

  def _calcGoal(self, img):
    tmp=np.array(img)
    return tmp/255.0
  
  def populate(self, n):
    return np.array([np.floor(np.random.rand(self.numTri,4,2)*[[np.array(self.imgShape)]]) for _ in range(n)])
    # return self.generation

  def evaluation(self):
    # sort the generation by error and return the errs
    rendered = np.array([self.getImg(unit) for unit in self.generation])
    errs=np.sum(np.sum(np.power(rendered-[self.goal], 2),axis=1),axis=1)
    indices=np.argsort(errs)
    # print(errs)
    self.generation=self.generation[indices]
    return errs[indices]
    

  def selectAndPopulate(self):
    # pool=self.generation[:self.selectionSize]
    # pool=pool.reshape(E.selectionSize*E.numTri,4,2)
    # pool=np.unique(pool, axis=0)
    # children=np.array([pool[np.random.choice(pool.shape[0],self.numTri)] for _ in range(self.generationSize)])
    children = [self.generation[np.random.choice(np.arange(0, self.generationSize), p=self.probDist, size=self.numTri),np.arange(0,self.numTri)] for _ in range(self.generationSize)]
    # newUnit=self.populate(self.newUnitPerGeneration)
    # self.generation=np.append(children,newUnit,axis=0)
    self.generation=np.array(children)
    self.generationNumber+=1
    return self.generation

  def mutation(self):
    for i in range(self.generationSize):
      if np.random.rand()>self.mutationProb:
        continue
      for _ in range(1):
        # triangle=np.random.randint(self.numTri)
        # self.generation[i][triangle]=np.random.rand(4,2)*[np.array(self.imgShape)]
        if(np.random.rand()<0.5):
          triangle=np.random.randint(self.numTri)
          # self.generation[i][triangle]=np.random.rand(4,2)*[np.array(self.imgShape)]
          vertex=np.random.randint(3)
          oldval=self.generation[i][triangle][vertex]
          self.generation[i][triangle][vertex]=np.mod(oldval+np.random.randint(-200,200,2),self.imgShape)
        else:
          triangle=np.random.randint(self.numTri)
          col=self.generation[i][triangle][3][0]
          self.generation[i][triangle][3]=col+(np.random.rand()-0.5)*col

  def getImg(self, unit):
    out = np.zeros(self.imgShape)
    for triangle in unit:
      out += self._drawTriangle(triangle)
    out=out/np.max(out)
    return 1-out
  
  def _drawTriangle(self, triangle):
    out = np.zeros(self.imgShape)
    col=triangle[3][0]/self.imgShape[0]
    vertices = np.array(triangle[:3], np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=1., thickness=1)
    cv2.fillPoly(out,[pts],1.)
    out=np.array(out)*col
    return out
  
def evol(maxGeneration, E):
  E.generation=E.populate(E.generationSize)
  errors=[]
  while maxGeneration>0:
    maxGeneration-=1
    score=E.evaluation()
    # best=E.generation[0] #
    E.selectAndPopulate()
    # E.generation[0]=best #
    E.mutation()
    cv2.imshow("evolved",E.getImg(E.generation[0]))
    cv2.waitKey(1)
    errors.append(score[0])
    print(E.generationNumber,'/',E.generationNumber+maxGeneration, score[0])
  score=E.evaluation()
  return errors

if __name__ == '__main__' :
  np.random.seed(0)
  img=cv2.imread("goldhill.png",0)
  E = evo(img, numTri=100, generationSize=40, crossoverProb=0.8, mutationProb=0.5)
  cv2.imshow("original", img)
  # cv2.imshow("gen",255-E.overlapImg*E.TriAlpha)
  # cv2.imshow("tri",E.getImg(E.unit))
  cv2.imshow("goal",E.goal)
  errors=evol(100,E)
  cv2.imshow("evolved",E.getImg(E.generation[0]))
  plt.plot(errors)
  plt.xlabel('generation')
  plt.ylabel('errors')
  plt.show()
  print("Done!")
  # print(E.generation[0].shape)
  cv2.waitKey()
  cv2.destroyAllWindows()

### Record
# Mix top 3 and Pick 50
  # with color, mutate whole 1 triangle
  # evo1 -> evo(img, numTri=50, generationSize=20, selectionSize=3, newUnitPerGeneration=0, mutationProb=0.5) 100 gens
  # '', mutate by small change 1 triangle
  # evo2 -> evo(img, numTri=50, generationSize=20, selectionSize=3, newUnitPerGeneration=0, mutationProb=0.5) 100 gens
  # '', mutate by big pos change 1 triangle, big color change 1 triangle
  # evo3 -> evo(img, numTri=50, generationSize=20, selectionSize=3, newUnitPerGeneration=0, mutationProb=0.5) 100 gens
  # evo4 -> evo3 2000 gens
  # evo5 -> '', evo3 but either one of the mutation per unit

# Crossover respectively to the triangle number.
  ## evo_co1 -> evo(img, numTri=50, generationSize=20, crossoverProb=0.5, mutationProb=0.5) 100 gens slowest
  # evo_co2 -> evo_co1 w/ mutationProb=0.8 100 gens
  ## evo_co3 -> evo_co1 w/ crossoverProb=0.8 100 gens fastest but a bit unstable
  ## evo_co4 -> evo_co1 w/ crossoverProb=0.3 100 gens faster
  # evo_co5 -> evo_co1 w/ mutationProb=0.3 100 gens
  # evo_co6 -> co 0.8, mut 0.8
  # evo_co7 -> evoco1 with genSize 40
  # evo_co8 -> genSize 40 mut 0.8 So good -> Could be because that it is big enough to find good one among the weird combinations
  #         -> 1000 gen 1.2s/gen
  # evo_co11 -> evo_co8 with gensize 100, 2.9s/gen

  ### Mutate with a whole new triangle
  # evo_co9 -> evo_co1 new mut
  # evo_co10 -> evo_co9 genSize40
  #         -> co 0.8 1000 gen
  # evo_co12 -> evo(img, numTri=100, generationSize=40, crossoverProb=0.8, mutationProb=0.5)
