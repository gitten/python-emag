import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def distribute(h):
    #physical parameters
    Tline_width = 6 #meters
    Tline_thickness = 8 #meters
    
    
    
    
    #mesh arm length
    #h = 1#meters
    
    
    
    #digitize physical space
    unk_rowlen = int(Tline_width / h - 1) #number of unknown nodes along horzontal axis
    unk_collen = int(Tline_thickness / h - 1) #number of unknown nodes along vertical axis
    unk_total  = int(unk_rowlen * unk_collen)
    
    
    #debug
    test = np.arange(unk_total)
    test = np.reshape(test, (unk_collen, unk_rowlen))
    
    #determine 4-arm star coeficient diagonals and conductor and dielectric indicies 
    
    upper_diag = (np.arange(unk_total - unk_rowlen),np.arange(unk_rowlen,unk_total))
    lower_diag = (np.arange(unk_rowlen,unk_total),np.arange(unk_total - unk_rowlen))
    
    left_diag  = (np.arange(1,unk_total),np.arange(unk_total - 1))
    right_diag = (np.arange(unk_total-1),np.arange(1, unk_total))
    
    
    #build coefficient array
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
    
    
    cv = np.ones(unk_total) * -2 * h**2
    
    V = linalg.solve(coeff_mat, cv)
    Vair = np.reshape(V, (unk_collen, unk_rowlen))
    
    return Vair,coeff_mat


## Processsssss

h1,c1 = distribute(1)
h1 = h1[0:4,0:3]
print('\n\nfor h = 1 meter\n\n',h1)

hp5,cp5 = distribute(.5)
hp5i = hp5[1:8:2,1:6:2]
print('\n\nfor h = 0.5 meter\n\n',hp5i)

h25,c25 = distribute(.25)
h25i = h25[3:16:4,3:12:4]
print('\n\nfor h = 0.25 meter\n\n',h25i)

h125,c125 = distribute(.125)
h125i = h125[7:16*2:4*2,7:12*2:4*2]
print('\n\nfor h = 0.125 meter\n\n',h125i)


BookValue = np.array([[2.04,3.05,3.35],
                      [3.12,4.79,5.32],
                      [3.66,5.69,6.34],
                      [3.82,5.96,6.65]])
 
print('\n\nBook Values\n\n', BookValue)              
plt.close('all')
plt.figure()
plt.imshow(hp5i)
plt.colorbar()             


plt.matshow(c1)
plt.colorbar()              
