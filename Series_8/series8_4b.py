# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series8_4b                                  ###
###            " Solution problem with                        ###
###                   zero average  "                         ###
###                                                           ###
###-----------------------------------------------------------###

# We firs import the module FEM and meshes (to generate the mesh)

import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# Lets define a solver function that gives the solution (coefficient
# vector) of the problem in the exercise 4, with homogeneous boundary
# conditions

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2
# f : source function
# n : the order of the numerical quadrature in the load term

def homogen(h0,f,n):
	# First lets generate the uniform mesh for this mesh width
	mesh=msh.grid_square(1,h0)
	# Matrix of nodes
	p=mesh[0]
	# Matrix of triangles index
	t=mesh[1]
	# Now lets compute the stiffness matrix
	A=fem.stiffness(p,t)
	# Lets compute the load vector
	Load=fem.load(p,t,n,f)
	# Lets compute the vector m that analogous to the load vector
	# with f(x.y)=1
	m=fem.load(p,t,n,lambda x,y:1)
	# Lets create the matrix and vectors of the modified bigger system to
	# solve with lagrangian multipliers of size (size(A)+(1,1))
	size=A.shape[0]
	B=sparse.lil_matrix(np.zeros([size+1,size+1]))
	B[0:size,0:size]=A
	B[0:size,size]=m
	B[size,0:size]=m.transpose()
	# We add a zero at the end of f 
	Load=np.concatenate((Load,np.array([[0]])))
	# Now lets get the solution of the linear system using spsolve function
	U=spla.spsolve(B,Load)
	#We extract the solution u and the multiplicer l
	u=U[0:size]
	l=U[size]
	# We return [p,t,u,l]
	return p,t,u,l



# To get a solution u(x,y)=cos(pi x)*cos(pi y) the source function must 
# be f(x,y)=(2 pi^2)cos(pi x)*cos(pi y)

#lets define the source function
f= lambda x1,x2: (2*np.pi**2)*np.cos(np.pi*x1)*np.cos(np.pi*x2)
# The order of the quadrature will be 3
n=3
# The closest value of h0 to 0.1 to be able to generate a regular 
h0=np.sqrt(2)/14

# Lets get the solution
Sol=homogen(h0,f,n)
p=Sol[0]
t=Sol[1]
u=Sol[2]
l=Sol[3]

#Change the t to put the value 1 to zero to be able to plot with the 
# function in the FEM module
t=np.array([list(ti-1) for ti in t])

#Now lets define the exact solution as a function 
def uexact(x1,x2):
	return np.cos(np.pi*x1)*np.cos(np.pi*x2)

#Lets get a np.array of the exact solution evaluated in each 

Uexact=np.array([uexact(pi[0],pi[1]) for pi in p])

# Lets generate the plot of both
fem.plot(p,t,u,"fem_homogen.png","FEM homogeneous solution")
fem.plot(p,t,Uexact,"exact_homogen.png","Exact homogeneous solution")

# Lets get the discretization error and plot it 
error=Uexact-u
fem.plot(p,t,error,"error_homogen.png","Discretization homogeneous error")

# Is also noticable that l=1.5075455299440885e-05 that is approximately
# zero which is the exact value of lambda with g=0


