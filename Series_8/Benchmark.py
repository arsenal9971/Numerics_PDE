import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# To get a solution u(x,y)=cos(pi x)*cos(pi y) the source function must 
# be f(x,y)=(2 pi^2)cos(pi x)*cos(pi y)

#lets define the source function
f= lambda x1,x2: (2*np.pi**2)*np.cos(np.pi*x1)*np.cos(np.pi*x2)
# The order of the quadrature will be 3
n=3
# The closest value of h0 to 0.1 to be able to generate a regular 
h0=np.sqrt(2)/10

# We create the mesh
mesh=msh.grid_square(1,h0)
# Matrix of nodes
pi=mesh[0]
# Matrix of triangles index
t=mesh[1]
# Matrix of boudary elements vertices
be=mesh[2]

index=be[39]

p=pi[index-1]

g= lambda x1,x2: np.cos(np.pi*x1)*np.cos(np.pi*x2)

sum(map(lambda K:(TB(be,p,K).transpose()
    .dot(elemLoadNeumann(np.array([p[i-1] for i in be[K]]),n,g))),range(0,len(be))))