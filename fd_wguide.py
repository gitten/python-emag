import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

#mesh arm length
h = .25

#physical parameters and constants

a_width = 2 #cm
b_height = 3 #cm
a_notch = 1
b_notch = 1

e0 = 8.854e-12
c = 3e10

##digitize physical space ( pulled from fd_line.py )
unk_rowlen = int(a_width / h - 1) #int(Tline_width / h - 1) #number of unknown nodes along horzontal axis
unk_collen = int(b_height / h - 1) #int(Tline_thickness / h - 1) #number of unknown nodes along vertical axis
unk_total  = int(unk_rowlen * unk_collen)

#notch
notch_len = int(a_notch / h - 1)
notch_hi = int(a_notch / h )

#debug
test = np.arange(unk_total)
test = np.reshape(test, (unk_collen, unk_rowlen))


##determine 4-arm star coeficient diagonals and conductor and dielectric indicies ( pulled from fd_line.py )

upper_diag = (np.arange(unk_total - unk_rowlen),np.arange(unk_rowlen,unk_total))
lower_diag = (np.arange(unk_rowlen,unk_total),np.arange(unk_total - unk_rowlen))

left_diag  = (np.arange(1,unk_total),np.arange(unk_total - 1))
right_diag = (np.arange(unk_total-1),np.arange(1, unk_total))

# # cond_iindex = unk_total - 1 - (unk_rowlen - cond_len) // 2 - unk_rowlen * off_len #index of right node on lower conductor. int div = maybreak
# # bottom_cond = np.arange(cond_iindex - cond_len, cond_iindex) + 1
# # top_cond    = bottom_cond - dualcond_len * unk_rowlen

# # #dielectric indicies
# # D12_rowstart = (unk_collen - int(D1_D2 / h)) * unk_rowlen
# # D23_rowstart = (unk_collen - int(D2_D3 / h)) * unk_rowlen
# # 
# # D12_index = D12_rowstart + np.arange(0, unk_rowlen)
# # D23_index = D23_rowstart + np.arange(0, unk_rowlen)
# # 
# # D12_index = D12_index[-np.in1d(D12_index,np.hstack((top_cond,bottom_cond)))]
# # D23_index = D23_index[-np.in1d(D23_index,np.hstack((top_cond,bottom_cond)))]



##build coefficient array ( pulled from fd_line.py )
# memory problems for very large arrays. look into "HDF5 for python"
coeff_mat = np.eye((unk_total)) * -4

coeff_mat[upper_diag] = 1
coeff_mat[lower_diag] = 1
coeff_mat[left_diag]  = 1
coeff_mat[right_diag] = 1

#corner indicies
corner= np.arange(unk_rowlen - 1, unk_total - 1, unk_rowlen)

coeff_mat[corner,corner + 1] = 0
coeff_mat[corner + 1,corner] = 0

# # coeff_mat[top_cond[0] - 1, top_cond[0]] = 0
# # coeff_mat[top_cond[-1:] + 1, top_cond[-1:]] = 0
# # coeff_mat[bottom_cond[0] - 1, bottom_cond[0]] = 0
# # coeff_mat[bottom_cond[-1:] + 1, bottom_cond[-1:]] = 0

# # coeff_mat[bottom_cond] = 0 
# # coeff_mat[bottom_cond,bottom_cond] = -1
# # coeff_mat[bottom_cond - unk_rowlen, bottom_cond] = 0
# # coeff_mat[bottom_cond + unk_rowlen, bottom_cond] = 0

# # coeff_mat[top_cond] = 0 
# # coeff_mat[top_cond,top_cond] = -1
# # coeff_mat[top_cond - unk_rowlen, top_cond] = 0
# # coeff_mat[top_cond + unk_rowlen, top_cond] = 0

# # cv = np.zeros((unk_total, 1))
# # 
# # cv[np.hstack((bottom_cond + unk_rowlen,bottom_cond - unk_rowlen,bottom_cond[0] - 1, bottom_cond, bottom_cond[-1] + 1))] -= voltage
# # cv[np.hstack((top_cond - unk_rowlen,top_cond + unk_rowlen,top_cond[0] - 1, top_cond, top_cond[-1] + 1))] -= voltage

##Build coefficient array (Waveguide)
#account for boundary symetry

#vertical and horizontal boundary indicies
bounds_v = np.arange(0, unk_total, unk_rowlen)
bounds_h = np.arange(0, unk_rowlen)

#left boundary (acts on center right diagonal)
coeff_mat[bounds_v, bounds_v +1] = 2

#right boundary (acts on center left diagonal)
coeff_mat[bounds_v + unk_rowlen - 1, bounds_v  + unk_rowlen - 2] = 2

#top boundary (acts on bottom diagonal)
coeff_mat[unk_total - unk_rowlen + bounds_h, unk_total - 2 * unk_rowlen + bounds_h] = 2

#bottom boundary (acts on top diagonal)
coeff_mat[bounds_h, bounds_h + unk_rowlen] = 2

## plot coefficient matrix
plt.close('all')
plt.matshow(coeff_mat) #np.hstack((coeff_mat , cv)))
plt.colorbar()


## solve eigenvalues
plt.figure()
(m,v) = linalg.eig(coeff_mat)

m = np.sort(-m * c / (h * 2 * np.pi))

plt.plot(m[0:20],'r*')