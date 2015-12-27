
# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###              Module Neumann Solver                        ###
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
# of the Neumann problem 

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2
# f : source function
# n : the order of the numerical quadrature in the load term

def neumann(h0,f,n):
	# First lets generate the uniform mesh for this mesh width
	mesh=msh.grid_square(1,h0)
	# Matrix of nodes
	p=mesh[0]
	# Matrix of triangles index
	t=mesh[1]
	# Now lets compute the stiffness matrix
	Stiff=fem.stiffness(p,t)
	# Now lets compute the mass matrix
	Mass=fem.mass(p,t)
	# Lets compute the load vector
	Load=fem.load(p,t,n,f)
	# The complete matrix for the bilinear form is given by the sum of 
	#the stiffness and the mass
	B=Stiff+Mass
	# Now lets get the solution of the linear system using spsolve function
	U=spla.spsolve(B,Load)
	# We return [p,t,U]
	return p,t,U

