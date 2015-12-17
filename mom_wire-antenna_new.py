import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import linalg
from scipy import pi
from scipy import integrate
from mpl_toolkits.mplot3d.axes3d import Axes3D

# def integrand(z, k, rad, z, zp):
#     return 

# of segments on wire (odd numbers only)
seg = 11

##physical parameters and constants
lamRat = 1
lineDia = .01

voltage = 1

e0 = 8.854e-12
eta = 120*pi
k = 2 * pi

##digitize physical space
segLen = lamRat / seg
zDel = segLen / 2

z = np.empty( (seg + 1, seg + 1) )

# z[:,0:-1]
zCore = z[0:-1,0:-1].view()
zi = np.linspace(zDel/2, lamRat-zDel/2, seg) - lamRat / 2

zCore[:] = linalg.toeplitz(zi)

zCore[:] = integrate.fixed_quad(integrand, aLim, bLim, )

plt.matshow(z)
plt.colorbar()