# coding=utf-8
# Héctor Andrade Loarca
#  
# gaussTriangle(n)
# 
# returns abscissas and weights for "Gauss integration" in the triangle with 
# vertices (-1,-1), (1,-1), (-1,1)
#
# input:
# n - order of the numerical integration (1 <= n <= 5)
#
# output:
# x - 1xp-array of abscissas, that are 1x2-arrays (p denotes the number of 
#     abscissas/weights)
# w - 1xp-array of weights (p denotes the number of abscissas/weights)
#

import numpy as np
import scipy.sparse as sparse

def gaussTriangle(n):

  if n == 1:
      x = [[-1/3., -1/3.]];
      w = [2.];
  elif n == 2:
      x = [[-2/3., -2/3.],
           [-2/3.,  1/3.],
           [ 1/3., -2/3.]];
      w = [2/3.,
           2/3.,
           2/3.];
  elif n == 3:
      x = [[-1/3., -1/3.],
           [-0.6, -0.6],
           [-0.6,  0.2],
           [ 0.2, -0.6]];
      w = [-1.125,
            1.041666666666667,
            1.041666666666667,
            1.041666666666667];
  elif n == 4:
      x = [[-0.108103018168070, -0.108103018168070],
           [-0.108103018168070, -0.783793963663860],
           [-0.783793963663860, -0.108103018168070],
           [-0.816847572980458, -0.816847572980458],
           [-0.816847572980458,  0.633695145960918],
           [ 0.633695145960918, -0.816847572980458]];
      w = [0.446763179356022,
           0.446763179356022,
           0.446763179356022,
           0.219903487310644,
           0.219903487310644,
           0.219903487310644];
  elif n == 5:
      x = [[-0.333333333333333, -0.333333333333333],
           [-0.059715871789770, -0.059715871789770],
           [-0.059715871789770, -0.880568256420460],
           [-0.880568256420460, -0.059715871789770],
           [-0.797426985353088, -0.797426985353088],
           [-0.797426985353088,  0.594853970706174],
           [ 0.594853970706174, -0.797426985353088]];
      w = [0.450000000000000,
           0.264788305577012,
           0.264788305577012,
           0.264788305577012,
           0.251878361089654,
           0.251878361089654,
           0.251878361089654];
  else:
      print 'numerical integration of order ' + str(n) + 'not available';
      
  return x, w


#
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
#
# I changed it a little to generate the plot as an image
# in the work directory

def plot(p,t,u,file,title):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  plt.title(title)
  plt.savefig(file)
  plt.close()

# Function that computes the elemStiffness in a triangle

# input
# p - 3x2-matrix of the coordinates of the triangle nodes

#First lets define a function that calculates the area of a triangle
# with input the point of the vertices

def area(p):
  return 1./2.*(np.abs(p[1,0]*p[2,1]-p[2,0]*p[1,1]+
                       p[0,0]*p[1,1]-p[1,0]*p[0,1]+
                       p[2,0]*p[0,1]-p[0,0]*p[2,1]))

def elemStiffness(p):
  #We compute the area of the triangle using shoelace formula
  Area=area(p)
  # We compute the coordinate difference matrix DK in the triangle K
  DK=np.array([[p[1,1]-p[2,1],p[2,1]-p[0,1],p[0,1]-p[1,1]],
               [p[2,0]-p[1,0],p[0,0]-p[2,0],p[1,0]-p[0,0]]])
  # Finally we compute the element stiffness matrix
  return (1/(4*Area))*np.transpose(DK).dot(DK)


#Function that computes the elemMass of element mass matrix
# for a constant coeficcient cK=1

# input
# p - 3x2-matrix of the coordinates of the triangle nodes

def elemMass(p):
  #We compute the area of the triangle using shoelace formula
  Area=area(p)
  #We compute the element mass matrix which have 1/6 in the diagonal and 
  # and 1/12 otherwise
  return (1./12*np.ones((3,3))+1./12*np.eye(3))*Area

# Function that computes the element Load vector
 
# Input:
# p - 3x2 matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

# First lets define the function that transform the coordinates from the
# the triangle with points coordinates in p to the reference triangle 

#Input: p - 3x2 matrix of the coordinates of the triangle nodes
#       (x1,x2)- coordinates of evaluation in the reference triangle

def transform(p,x1,x2):
  return p[0]+x1*(p[1]-p[0])+x2*(p[2]-p[0])

# Lets define the shape functions in the reference triangle 
#Input:
#       j - the index of the function
#       (x1,x2)- coordinates of evaluation in the reference triangle

def Nshape(j,x1,x2):
  return [1-x1-x2,x1,x2][j]

# Lets define the function that computes the element load vector

# input:
  # p - 3x2 matrix of the coordinates of the triangle nodes
  # n - order of the numerical quadrature (1 <= n <= 5)
  # f - source term function


def elemLoad(p,n,f):
  #Lets get the vectors and weights in the quadrature
  quad=gaussTriangle(n)
  xquad=quad[0]
  wquad=quad[1]
  #Transformed each element of the quadrature points to np.array
  xquad=map(lambda x: np.array(x),xquad)
  #Lets generate a list of the vector transformed to the reference triangle
  #to integrate in the triangle where the quadrature is defined
  xref=map(lambda x: (x+1)/2,xquad)
  #Lets transformed to the original triangle the new coordinates
  xtrans=map(lambda x:transform(p,x[0],x[1]), xref)
  #The determinant of the Jacobian
  detJ=2*area(p)
  #Finally we obtain the element load vector
  Load=map(lambda i:sum(map(lambda j:wquad[j]*f(xtrans[j][0],xtrans[j][1])
                 *Nshape(i,xref[j][0],xref[j][1])*np.abs(detJ),range(0,n))),range(0,3))
  return 1./2*np.array(Load).reshape(3,1)


# Now we gonna assemble the whole elements to obtain the total stiffenss
# matrix


# First lets define a function that gives you the T matrix of a triangle K

# input:
  # t : Mx3 matrix with the triangle-node numbering
  # K : the number of the triangle in the t matrix

def T(t,K):
  # Start the T matrix with just zeros
  n=t.max()
  Tm=np.zeros((3,n))
  # extraxt nodes index in the K triangle
  index=t[K]
  # Put the ones
  Tm[0,index[0]-1]=1
  Tm[1,index[1]-1]=1
  Tm[2,index[2]-1]=1
  #Returning the matrix T in lil sparse format
  return sparse.lil_matrix(Tm)

# Now lets get the global stiffnes matrix

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles

def stiffness(p,t):
  # Sum up the matrix element stiffness weighted with the T matrices (the conecction)
  # in lil_matrix format
  stiff=sum(map(lambda K:(T(t,K).transpose()
    .dot(elemStiffness(np.array([p[i-1] for i in t[K]]))))
  .dot(T(t,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(stiff)

# Now lets get the global mass matrix in the same way that the stiffness

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles


def mass(p,t):
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  # in lil_matrix format
  massm=sum(map(lambda K:(T(t,K).transpose()
    .dot(elemMass(np.array([p[i-1] for i in t[K]]))))
  .dot(T(t,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(massm)

# Now lets get the global load vector 

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

def load(p,t,n,f):
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  return sum(map(lambda K:(T(t,K).transpose()
    .dot(elemLoad(np.array([p[i-1] for i in t[K]]),n,f))),range(0,len(t))))
  
# Lets define the function to get the interior nodes as indices into p

# input:
# p  - Nx2 array with coordinates of the nodes
# t  - Mx3 array with indices of nodes of the triangles
# be - Bx2 array with indices of nodes on boundary edges

def interiorNodes(p, t, be):
     # First get a list of the indices of the nodes in the boundary
     bound=sum(be.tolist(),[])
     # We take out the duplicates
     bound=set(bound)
     # Generate a set of the whole indices of the vertices in the mesh
     indices=set([i for i in range(1,len(p)+1)])
     # Finally we return the substraction of the sets to get the 
     # interior points indices
     return list(indices-bound)

