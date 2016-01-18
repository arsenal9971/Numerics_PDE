# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series9_3g                                  ###
###            " Neumann Homogeneous                          ###
###             u(x, y)=cos(2πx)cos(2πy)                      ###
###                quadratic"                                 ###
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

