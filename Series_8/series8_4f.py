# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series8_4f                                   ###
###            " Discretization error in                      ###
###                energy norm "                              ###
###                                                           ###
###-----------------------------------------------------------###

# We firs import the module FEM and meshes (to generate the mesh)

import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os

# Lets define a function that generates a mesh of the unit circle
# in dependence of the maximal width h0

# input:

# h0 : maximal width

def mesh_gen(h0):
	line=''
	line=line+'// Creating a mesh for the unit circle\n ' 
	line=line+'// Maximal width in the mesh h0=0.1\n '
	line=line+'h0= '+str(h0)+';\n '
	line=line+'radius = 1.0;\n'
	line=line+'// Creating the points\n'
	line=line+'Point(1) = {0, 0, 0, h0};\n'
	line=line+'Point(2) = {-radius, 0, 0, h0};\n'
	line=line+'Point(3) = {0, radius, 0, h0};\n'
	line=line+'Point(4) = {radius, 0, 0, h0};\n'
	line=line+'Point(5) = {0, -radius, 0, h0};\n'
	line=line+'Circle(6) = {2, 1, 3};\n'
	line=line+'Circle(7) = {3, 1, 4};\n'
	line=line+'Circle(8) = {4, 1, 5};\n'
	line=line+'Circle(9) = {5, 1, 2};\n'
	line=line+'// Define a surface by a Line Loop\n'
	line=line+'Line Loop(10) = {6, 7, 8, 9};\n'
	line=line+'Plane Surface(11) = {10};\n'
	line=line+'// Define a surface by a Line Loop\n'
	line=line+'Physical Line(101) = {6, 7, 8, 9};\n'
	line=line+'Physical Surface(201) = {11};'
	#Write to the file
	f=open('circle.geo','w')
	f.write(line)
	f.close()
	# Generate the msh file
	os.system('gmsh circle.geo -2')


# Lets define a solver function that gives the solution (coefficient
# vector) of the problem in the exercise 4, with non homogeneous boundary
# conditions 

# input:

# f : source function
# h0 : maximal 
# n : the order of the numerical quadrature in the load term
# g: the Neumann boundary data

def no_homogen(h0,f,n,g):
	# First lets generate the mesh for the unite circle
	mesh_gen(h0)
	mesh=msh.read_gmsh('circle.msh')
	# Matrix of nodes
	p=mesh[0]
	# Matrix of triangles index
	t=mesh[1]
	# Matrix of boundary edges 
	be=mesh[2]
	# Now lets compute the stiffness matrix
	A=fem.stiffness(p,t)
	# Lets compute the load vector
	Load=fem.load(p,t,n,f)
	# Lets compute the load neumann vector
	Loadneumann=fem.loadNeumann(p,be,n,g)
	#Finally the Loadvector will be the sum of the one with the source f
	# and the one with f
	Load=Load-Loadneumann
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




# To get a solution u(x1,x2)=x1*sin(pi*r)
# We define the source function f and the Neumann Data g as follows

def f(x1,x2):
	r=np.sqrt(x1**2+x2**2)
	return -3*np.pi*(x1/r)*np.cos(np.pi*r)+(np.pi**2)*x1*np.sin(np.pi*r)

def g(x1,x2):
	r=np.sqrt(x1**2+x2**2)
	return np.pi*x1*np.cos(np.pi*r)+(x1/r)*np.sin(np.pi*r)

#We gonna use maximal width h0=0.1 and order of quadrature n=3
h0=0.1
n=3

# Lets get the solution
Sol=no_homogen(h0,f,n,g)
p=Sol[0]
t=Sol[1]
u=Sol[2]
l=Sol[3]

#Change the t to put the value 1 to zero to be able to plot with the 
# function in the FEM module
t=np.array([list(ti-1) for ti in t])

#Lets define the exact solution
def uexact(x1,x2):
	r=np.sqrt(x1**2+x2**2)
	return x1*np.sin(np.pi*r)

#Lets get a np.array of the exact solution evaluated in each 
Uexact=np.array([uexact(pi[0],pi[1]) for pi in p])

# Lets generate the plot of both
fem.plot(p,t,u,"fem_non_homogen.png","FEM nonhomogeneous solution")
fem.plot(p,t,Uexact,"exact_non_homogen.png","Exact nonhomogeneous solution")

# Lets get the discretization error and plot it 
error=Uexact-u
fem.plot(p,t,error,"error_non_homogen.png","Discretization non homogeneous error")
