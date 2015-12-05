# Benchmark of the functions

import FEM
import numpy as np 
import scipy.sparse as sparse

# 1.- 
# a)
#Define the triangle with its nodes coordinates

p=np.array([[1.,1.],
	        [1.,2.],
	        [2.,1.]])

# c) Lets define the source function by a lambda funcion

f = lambda x1,x2: x1*x2


# 2 Assembling the elements

# We gonna use the functions generated in the last homework to generate
# the meshes to study as benchmark

import meshes as msh

a=1
h0=np.sqrt(2)/14

mesh=msh.grid_square(a,h0)

p=mesh[0]
t=mesh[1]
be=mesh[2]

K=0
nodes=np.array([p[i-1] for i in t[K]])
(T(t,K).transpose().dot(elemStiffness(nodes))).dot(T(t,K).toarray())