import pandas as pd
import numpy as np
from igraph import *
from scipy import spatial
import sys
import argparse

class CommunityDetection:
  def __init__(self, graph, alpha):
    self.graph = graph
    self.v = graph.vcount()
    self.alpha = alpha

  def phase1(self):
    maxIteration = 0
    improvement = 1
    C = list(range(self.v))

    while (improvement == 1 and maxIteration < 15):
      improvement = 0
      for vi in range(self.v):
        maxV = -1
        maxDeltaQ = 0.0
        clusters = list(set(C))
        
        for vj in clusters:
          if (C[vi] != vj):
            dQ = self.__deltaQ(C, vi, vj)
            if (dQ > maxDeltaQ):
              maxDeltaQ = dQ
              maxV = vj
        
        if (maxDeltaQ > 0.0 and maxV != -1):
          improvement = 1
          C[vi] = maxV
      
      maxIteration += 1
    
    return C

  def phase2 (self, C):
    newC = self.sequential_clusters(C)
    temp = list(Clustering(newC))
    L = len(set(newC))
    simMatrix = np.zeros((L,L))
    
    for i in range(L):
      for j in range(L):
        similarity = 0.0
        for k in temp[i]:
          for l in temp[j]:
            similarity = similarity + simMatrix2[k][l]
        simMatrix[i][j] = similarity
    
    self.graph.contract_vertices(newC)
    self.graph.simplify(combine_edges=sum)
    return self.graph

  def sequential_clusters(self, C):
    mapping = {}
    newC = []
    c = 0
    for i in C:
      if i in mapping:
        newC.append(mapping[i])
      else:
        newC.append(c)
        mapping[i] = c
        c = c + 1
    return newC

  def composite_modularity(self, C):
    return self.graph.modularity(C, weights='weight') + self.__Q_attr(C)

  def __Q_attr(self, C):
    clusters = list(Clustering(C))
    S = 0.0
    for c in clusters:
      T = 0.0
      for v1 in c:
        for v2 in C:
          if (v1 != v2):
            T = T + simMatrix[v1][v2]
      T = T/len(c)
      S = S + T
    return S/(len(set(C)))

  def __deltaQ(self, C, v1, v2):
    d1 = self.__deltaQ_new(C, v1, v2)
    d2 = self.__deltaQ_attr(C, v1, v2)
    return (self.alpha*d1) + ((1-self.alpha)*d2)

  def __deltaQ_new(self, C, v1, v2):
    Q1 = self.graph.modularity(C, weights='weight')
    temp = C[v1]
    C[v1] = v2
    Q2 = self.graph.modularity(C, weights='weight')
    C[v1] = temp
    return (Q2-Q1);

  def __deltaQ_attr(self, C, v1, v2):
    S = 0.0;
    indices = [i for i, x in enumerate(C) if x == v2]
    for v in indices:
      S = S + simMatrix[v1][v]
    return S/(len(indices)*len(set(C)))


class LoadData:
  def __init__(self):
    pass

  def graph(self):
    self.g = Graph()
    self.__add_vertices_with_attributes()
    self.__add_edges_with_weight()
    return self.g

  def __add_vertices_with_attributes(self):
    attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')
    V = len(attributes)
    self.g.add_vertices(V)

    for col in attributes.keys():
      self.g.vs[col] = attributes[col]

  def __add_edges_with_weight(self):
    with open('data/fb_caltech_small_edgelist.txt') as f:
      edges = f.readlines()
    
    edges = [tuple([int(x) for x in line.strip().split(" ")]) for line in edges]
    self.g.add_edges(edges)
    self.g.es['weight'] = [1]*len(edges)

    

class SAC1:
  def __init__(self, alpha):
    self.alpha = alpha
    loader = LoadData()
    self.graph = loader.graph()
    self.vertices = self.graph.vcount()

  def run(self):
    self.similarity_between_nodes()

    comDec = CommunityDetection(self.graph, self.alpha)
    clusterP1 = comDec.phase1()
    clusterP1 = comDec.sequential_clusters(clusterP1)
    modularityP1 = comDec.composite_modularity(clusterP1)
    self.graph = comDec.phase2(clusterP1)
    self.vertices = self.graph.vcount()

    comDec = CommunityDetection(self.graph, self.alpha)
    clusterP2 = comDec.phase1()
    clusterP2new = comDec.sequential_clusters(clusterP2)
    clustersPhase2 = list(Clustering(clusterP2new))
    modularityP2 = comDec.composite_modularity(clusterP1)

    C1new = comDec.sequential_clusters(clusterP1)
    clustersPhase1 = list(Clustering(C1new))

    finalC = []
    for c in clustersPhase2:
      t = []
      for v in c:
        t.extend(clustersPhase1[v])
      finalC.append(t)

    if (modularityP1 > modularityP2):
      self.__write_to_file(clustersPhase1)
    else:
      self.__write_to_file(clustersPhase2)


  def similarity_between_nodes(self):
    global simMatrix
    global simMatrix2
    
    simMatrix = np.zeros((self.vertices, self.vertices))
    
    for v1 in range(self.vertices):
      for v2 in range(self.vertices):
        simMatrix[v1][v2] = self.__cosine_similarity(v1, v2)

    simMatrix2 = np.array(simMatrix)

  def __cosine_similarity(self, v1, v2):
    v1_attributes = list(self.graph.vs[v1].attributes().values())
    v2_attributes = list(self.graph.vs[v2].attributes().values())
    return 1 - spatial.distance.cosine(v1_attributes, v2_attributes)

  def __write_to_file(self, clusters):
    if self.alpha < 1:
      self.alpha *= 10

    file = open("communities_new_"+ str(int(self.alpha)) +".txt", 'w+')
    for c in clusters:
      for i in range(len(c)-1):
        file.write("%s," % c[i])
      file.write(str(c[-1]))
      file.write('\n')
    file.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=str, help='enter alpha value')
    
    return parser.parse_args()
  

if __name__ == "__main__":
  args = get_args()
  algo = SAC1(float(args.alpha))
  algo.run()
