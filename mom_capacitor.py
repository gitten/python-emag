import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy import linalg
from scipy import pi

#segmentation size
dl = .5
dx = dl
dy = dl

##physical parameters and constants
plate_len = 2
plate_width = plate_len

gap = .5

voltage = 1

ofx = 0
ofy = 0

e0 = 8.854e-12
col = 1.602176565e-19

##digitize physical space
opoint_plate = int(plate_len * plate_width / dl ** 2) # of observation points on one plate
opoint_tot = 2 * opoint_plate #number of total observation points(both plates)
opoint_hvector = np.arange(0, plate_len, dl)
opoint_vvector = np.arange(0, plate_width, dl)
#gap_vector = np.arange(0, gap / dl)

##prepare distance vectors for coefficient matrix
opoint_sref = opoint_hvector

for x in opoint_hvector[1:]:
    row = np.sqrt(opoint_hvector ** 2 + x ** 2)
    opoint_sref = np.hstack((opoint_sref, row))

opoint_gapref = np.sqrt( opoint_hvector ** 2 + gap ** 2)

for x in opoint_hvector[1:]:
    row = np.sqrt(opoint_hvector ** 2 + x ** 2)
    opoint_gapref = np.hstack((opoint_gapref, row))

##build coefficient matrix
row = len(opoint_sref)
col = len(opoint_gapref)

coeff_mat = np.empty([opoint_tot,opoint_tot])  
plt.close('all')
plt.matshow(coeff_mat-coeff_mat)

opoint_sref[0] = 0

coeff_mat[:row,col:] = 1#linalg.toeplitz(opoint_sref)#quad 1  (standard xy quadrant numbering)
coeff_mat[:row, :col] = 2#linalg.toeplitz(opoint_gapref) #quad 2
coeff_mat[row:,:col] = coeff_mat[:row, col:] #quad 3 = quad 1
coeff_mat[row:,col:] = coeff_mat[:row,:col] #quad 4 = quad 2

# cv = np.empty([opoint_tot,1])
# cv[:] = 4 * pi * e0 * voltage

## plot coefficient matrix
# plt.close('all')
plt.matshow(coeff_mat) #np.hstack((coeff_mat , cv)))
plt.colorbar()

# plt.figure
# plt.matshow(debug2)

# ## solve
# q = linalg.solve(coeff_mat, cv)
# 
# ##pretty plot
# # plt.figure()
# # plt.plot(q)
# fig = plt.figure()
# 
# 
# plt.bar(opoint, q, dl, alpha=0.6)
# # plt.yticks(y_pos, people)
# # plt.xlabel('Performance')
# plt.title('How fast do you want to go today?')
# ax = plt.gca()
# ax.relim()
# ax.autoscale_view(True,True,True)
# pylab.show()