# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series7_4c                                  ###
###            " Solving Dirichlet                            ###
###              homogeneous Problem  "                       ###
###                                                           ###
###-----------------------------------------------------------###

# We firs import the module FEM and meshes (to generate the mesh)

import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# Lets define a function that gives the solution (coefficient vector) 
# of the homogeneous Dirichlet problem (like g=0) 

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2
# f : source function
# n : the order of the numerical quadrature in the load term

def dirichlet_homogeneous(h0,f,n):
	# First lets generate the uniform mesh for this mesh width
	mesh=msh.grid_square(1,h0)
	# Matrix of nodes
	p=mesh[0]
	# Matrix of triangles index
	t=mesh[1]
	# Matrix of border element index
	be=mesh[2]
	#Lets get the interior nodes in the mesh
	innod=fem.interiorNodes(p,t,be)
	# Rewrite the innod to match with indices beginning at 0
	innod=[i-1 for i in innod]
	# Now lets compute the stiffness matrix
	Stiff=fem.stiffness(p,t)
	# Now lets compute the mass matrix
	Mass=fem.mass(p,t)
	# Lets compute the load vector
	Load=fem.load(p,t,n,f)
	# The complete matrix for the bilinear form is given by the sum of 
	#the stiffness and the mass
	B=Stiff+Mass
	# Now lets initialize the array in zeros to respect the homogeneous 
	# Dirichlet boundary conditions 
	U=np.zeros(len(p))
	# Lets take just the interior points in the matrix B and the Load vector
	Bint=B[innod][:,innod]
	Loadint=Load[innod]
	# Now lets get the solution of the linear system of the interior points
	# using spsolve function
	Uint=spla.spsolve(Bint,Loadint)
	# Finally we put them in the correspondence interior points of the complete
	# solution
	U[innod]=Uint
	# We return [p,t,U]
	return p,t,U


#Now lets get the solution u=sin(pi*x)sin(pi*y) that gives a source
# function f(x,y)=(2pi^2+1)sin(pi*x)sin(pi*y)

#Lets define first the function f
f= lambda x1,x2: (2*(np.pi**2)+1)*np.sin(np.pi*x1)*np.sin(np.pi*x2)
# The closest value of h0 to 1 to be able to generate a regular 
h0=np.sqrt(2)/14
n=3

# Lets get the solution
Sol=dirichlet_homogeneous(h0,f,n)
p=Sol[0]
t=Sol[1]
u=Sol[2]
#Change the t to put the value 1 to zero to be able to plot with the 
# function in the FEM module
t=np.array([list(ti-1) for ti in t])

#Now lets define the exact solution as a function 
def uexact(x1,x2):
	return np.sin(np.pi*x1)*np.sin(np.pi*x2)

#Lets get a np.array of the exact solution evaluated in each 

Uexact=np.array([uexact(pi[0],pi[1]) for pi in p])

# Lets generate the plot of both
fem.plot(p,t,u,"fem_dirichhom.png","FEM dirichlet homogeneous solution")
fem.plot(p,t,Uexact,"exact_dirichhom.png","Exact dirichlet homogeneous solution")

# Lets get the discretization error and plot it 
error=Uexact-u
fem.plot(p,t,error,"error_dirichhom.png","Discretization homogeneous dirichleterror")
