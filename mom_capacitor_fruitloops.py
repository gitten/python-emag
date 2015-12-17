import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import linalg
from scipy import pi
from mpl_toolkits.mplot3d.axes3d import Axes3D
#segmentation size
dl = .25
dx = dl
dy = dl

##physical parameters and constants
plate_len = 2
plate_width = plate_len

gap = .05

gap_init = .01
gap_points = 10

voltage = 2

ofx = 0
ofy = 0

e0 = 8.854e-11
col = 1.602176565e-19

c0 = e0 * plate_len * plate_width / gap


##digitize physical space
opoint_plate = int(plate_len * plate_width / dl ** 2) # of observation points on one plate
opoint_tot = 2 * opoint_plate #number of total observation points(both plates)
opoint_hvector = np.arange(0, plate_len, dl)
opoint_vvector = np.arange(0, plate_width, dl)

gap_vector = np.linspace(gap_init, plate_len * plate_width, num=10)

##prepare distance vectors for coefficient matrix
force = .282 * dl

k = dl ** 2  / (4 * pi)

block = (len(opoint_vvector), len(opoint_vvector))
x = np.kron(np.ones(block),linalg.toeplitz(opoint_hvector))
y = np.kron(linalg.toeplitz(opoint_hvector),np.ones(block))

quad2 = k / np.sqrt(x**2 + y**2)
quad2[np.diag_indices(opoint_plate)] = force

quad1 = k / np.sqrt( (x + ofx)**2 + (y + ofy)**2 + gap**2)
x


##build coefficient matrix
coeff_mat = np.vstack( ( np.hstack((quad2,quad1)), np.hstack((quad1,quad2)) ) )

cv = np.empty([opoint_tot,1])
cv[:] = voltage / 2
cv[opoint_tot/2:] = cv[opoint_tot/2:] * -1

## plot coefficient matrix
plt.close('all')
plt.matshow(coeff_mat) #np.hstack((coeff_mat , cv)))
plt.colorbar()

## solve
rho = linalg.solve(coeff_mat, cv) # e0 constant initially factored out of coeff matrix
rho = np.reshape(rho[:opoint_tot/2], [len(opoint_hvector),len(opoint_vvector)])

c = -e0 * np.sum(rho) * dl * dl / voltage
print(c)


plt.matshow(rho)
plt.colorbar()
##pretty plot
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-plate_len/2, plate_len / 2, dl)
Y = np.arange(-plate_width /2 , plate_width / 2, dl)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, rho, rstride=1, cstride=1,
        linewidth=0, antialiased=True)

plt.show() 

## Normalized gap vs. cap plots
c_vector = np.empty(np.size(gap_vector))
count = 0

for gaped in gap_vector:
    coi = k / np.sqrt( (x + ofx)**2 + (y + ofy)**2 + gaped**2 )
    coeff_mat[0:opoint_plate,opoint_plate::] = coi
    coeff_mat[opoint_plate::,0:opoint_plate] = coi
    rho = linalg.solve(coeff_mat, cv)
    c_vector[count] = -e0 * np.sum(rho[opoint_plate::]) * dl * dl / voltage
    count += 1

plt.figure()
plt.plot(gap_vector / (plate_len * plate_width), c_vector / c0)


