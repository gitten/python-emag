import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import linalg
from scipy import pi
from scipy.linalg import kron

# of segments
seg = 20

##physical parameters and constants
line_len = 1
line_dia = .01

bend_ang = 45
bend_loc = seg/2

voltage = 1

e0 = 8.854e-12
col = 1.602176565e-19

##digitize physical space
dl = line_len / seg

cl = np.cos(pi * bend_ang / 180)
sl = np.sin(pi * bend_ang / 180)

x_quad2 = linalg.toeplitz(np.arange(0, bend_loc) * dl)
xy_quad4 = linalg.toeplitz(np.arange(0, seg - bend_loc))
quad13 = kron(np.ones((bend_loc,1)), xy_quad4[0,:])
quad13 = quad13.reshape((bend_loc,seg - bend_loc))
y_quad13 = quad13 * sl*dl + sl*dl
x_quad13 = quad13 * cl*dl + cl*dl+ x_quad2[:,-1:]

x =  np.vstack( (np.hstack( (x_quad2,x_quad13) ), np.hstack((x_quad13.T,xy_quad4 * cl*dl) ) ) )

y =  np.vstack( (np.hstack( (np.zeros(np.shape(x_quad2)),y_quad13) ),np.hstack( (y_quad13.T,xy_quad4 * sl*dl) ) ))



##build coefficient matrix
coeff_mat = np.sqrt(x**2 + y**2)
coeff_mat =  2 * line_dia * dl / coeff_mat
coeff_mat[np.diag_indices(seg)] = 4 * pi * line_dia * np.log(dl / line_dia)
# coeff_mat[0,:] = 2 * line_dia * dl / opoint
# coeff_mat[0,0] = 4 * pi * line_dia * np.log(dl / line_dia)
# 
cv = np.empty([seg,1])
cv[:] = 4 * pi * e0 * voltage

## plot coefficient matrix
plt.close('all')
plt.matshow(coeff_mat)
plt.colorbar()

plt.figure
plt.matshow(np.hstack((x,y)))
# plt.figure
# plt.matshow(debug2)

## solve
q = linalg.solve(coeff_mat, cv)

##pretty plot
fig = plt.figure()

# plt.bar(np.arange(1,seg + 1), q, alpha=0.6)
# # plt.yticks(y_pos, people)
# # plt.xlabel('Performance')
# plt.title('Moooo')
# ax = plt.gca()
# ax.relim()
# ax.autoscale_view(True,True,True)
# plt.show()
plt.plot(q)