import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import itertools as it



##physical parameters and constants
width = 6
height = 8

bounds = 0 #known edge potentials

subDivHor = 6
subDivVer = 8

e0 = 8.854e-12 

rho = 2*e0

rounds = 50

##digitize space
hy = width / subDivVer
hx = height / subDivHor
a = hy * hx
A = a / 2
elementTotal = subDivHor * subDivVer * 2

nodeHor = ( subDivHor + 1 ) 
nodeVer = ( subDivVer + 1 )

nodeTotal = nodeHor * nodeVer

freeNodeTotal = (subDivHor - 1 ) * ( subDivVer - 1 )

print(
      '\n\ntotal elements', elementTotal,
      '\n\ntotal nodes', nodeTotal,
      '\n\ntotal unknown nodes', freeNodeTotal,
      '\n\nhorizontal nodes', nodeHor,
      '\n\nvertical nodes', nodeVer
      )

# nodes of geometry     
geom = (subDivVer+1,subDivHor+1)
v = np.zeros((1,nodeTotal))
vg = np.reshape(v,geom)
prev = 0

vi = v.copy()
vig = np.reshape(vi,geom).view()

freeNode = vg[1:-1,1:-1].view()
freeNodei = vig[1:-1,1:-1].view()
freeNode[:] = 0

rhorho = vg.copy()
rhorho[1:-1,1:-1] = rho
Rho = np.reshape(rhorho,(1,nodeTotal))

##build coefficient matrices
Ce = 1/(4*A) * np.array(    # assuming uniform mesh
                         [[ hy**2+hx**2,-hy**2, -hx**2 ], 
                          [-hy**2,       hy**2,      0 ],
                          [-hx**2,           0,  hx**2 ]] 
                        )


cGlobal = np.zeros((1,nodeTotal))
cGlobal[0,0] = 2 * (Ce[0,0] + Ce[1,1] + Ce[2,2]) 
cGlobal[0,1] = 2 * Ce[0,1]
cGlobal[0, nodeHor] = 2 * Ce[0,2]
cGlobal = la.toeplitz(cGlobal)



plt.close('all') 
plt.matshow(cGlobal)

t = A/12 * np.array( # for triangle elements
                     [[ 2, 1, 1 ],
                      [ 1, 2, 1 ],
                      [ 1, 1, 2 ]]
                    )

T = np.zeros((1,nodeTotal))
T[0,0] = 6 * (t[0,0]) 
T[0,1] = 2 * t[0,1]
T[0, nodeHor] = 2 * t[0,2]
T[0, nodeHor+1] = 4 * t[0,2]

T = la.toeplitz(T)
corner= np.arange(nodeHor-1, nodeTotal-1, nodeHor)
T[corner,corner + 1] = 0
T[corner + 1,corner] = 0

cornert= np.arange(nodeHor-1, nodeTotal-nodeHor -2, nodeHor)
T[cornert,cornert + nodeHor +1] = 0
T[cornert + nodeHor +1 ,cornert] = 0


plt.matshow(T)
print('\n\n\nBegining iterations for ',rounds,' rounds')
## FEM iteration method

for round in range(0,rounds):

    for k,i in it.product(range(0,nodeTotal),range(0,nodeTotal)):

        if i != k:
            vi[0,k] += -1 / (cGlobal[k,k]) * v[0,i] * cGlobal[k,i]
        
        vi[0,k] += 1 / (e0*cGlobal[k,k]) * T[k,i] * Rho[0,k]

    conv = vg - prev
    
    freeNode[:] = freeNodei
    vi[:] = 0

    prev = vg
    

    

print(vg)
plt.figure()    
plt.imshow(vg)
plt.colorbar()