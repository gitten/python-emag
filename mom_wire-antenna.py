import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import linalg
from scipy import pi
from mpl_toolkits.mplot3d.axes3d import Axes3D

# of segments on wire
seg = 10

##physical parameters and constants
lineLen = 1
lineDia = .01

gap = .1
condLen = lineLen - gap

voltage = 1

e0 = 8.854e-12


##digitize physical space
dl = condLen / seg
gapRat = gap - dl

opointz = np.linspace(0, condLen, seg + 1)

print('condLen = ', condLen, '\ndl = ', dl, '\nlineLen-dl = ', lineLen-dl, '\ngap = ',gap, '\ngapRat = ', gapRat, '\nopointz = ', opointz)

opointz[seg/2:] += gapRat
opointz[seg/2] -= gapRat / 2

print('\nwith gap  ', opointz)