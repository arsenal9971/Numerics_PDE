# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series7_4e                                  ###
###            " Solving Dirichlet                            ###
###             non homogeneous Problem "                     ###
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

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2
# f : source function
# g : the function in the boundary
# n : the order of the numerical quadrature in the load term

# For solve the nonhomogeneous dirichlet problem we gonna expand the solution
# u=u_0+u_g, where u_g is the Dirichlet lift that is g in the boundary
# and 0 everywhere else, and u_0 is the solution of a dirichlet problem
# with new load vector given by the load original substracted by
# B u_g_n, where u_g_n corresponds to a vector of u_g evaluated in the nodes
# and B=Stiff+Mass 

def dirichlet_nonhomogeneous(h0,f,g,n):
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
	# Lets define a vector with of the values of u_g in each node
	# that is zero in the innernodes and g everywhere else
	ug=np.array([g(p[i][0],p[i][1]) for i in range(len(p))])
	ug[innod]=np.zeros(len(innod))
	#Calculate the new load vector with the Dirichleft taken in account
	Load=Load-B.dot(ug).reshape(len(p),1)
	#Now lets get the homogeneous problem solution as we did before
	# First lets initialize the array in zeros to respect the homogeneous 
	# Dirichlet boundary conditions 
	U=np.zeros(len(p))
	# Lets take just the interior points in the matrix B and the Load vector
	Bint=B[innod][:,innod]
	Loadint=Load[innod]
	# Now lets get the solution of the linear system of the interior points
	# using spsolve function
	Uint=spla.spsolve(Bint,Loadint)
	# We put them in the correspondence interior points of the complete
	# solution
	U[innod]=Uint
	# Finally we sum up the two solutions to get the final solution
	u=U+ug
	# We return [p,t,U]
	return p,t,u

#Now lets get the solution u=sin(pi*x)sin(pi*y) that gives a source
# function f(x,y)=(2pi^2+1)sin(pi*x)sin(pi*y)

#Lets define first the function f
f= lambda x1,x2: 0
#Lets define the function 
g= lambda x1,x2: x1+x2
# The closest value of h0 to 1 to be able to generate a regular 
h0=np.sqrt(2)/14
n=3

# Lets get the solution
Sol=dirichlet_nonhomogeneous(h0,f,g,n)
p=Sol[0]
t=Sol[1]
u=Sol[2]
#Change the t to put the value 1 to zero to be able to plot with the 
# function in the FEM module
t=np.array([list(ti-1) for ti in t])

# Lets finally generate the plot
fem.plot(p,t,u,"fem_dirichnonhom.png","FEM dirichlet nonhomogeneous solution")
