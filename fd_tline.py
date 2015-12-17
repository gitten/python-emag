import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

voltage = 1

#physical parameters
Tline_width = 5 #mm
Tline_thickness = 5 #mm

center_conductor_width   = 2 #mm
center_conductor_offset = 2 #mm
dual_conductor_height = 1 #mm

#dielectric boundaries (distance from base)
D1_D2= 3 #mm
D2_D3 = 2 #mm

#permativity coefficients
e0 = 8.854 * 10 ** -12 
air = 1

er1 = air
er2 = 2.35
er3 = air

er12 = er1 + er2
er23 = er2 + er3

e1 = er1 * e0
e2 = er2 * e0

#mesh arm length
h = .25#mm


#digitize physical space
unk_rowlen = int(Tline_width / h - 1) #number of unknown nodes along horzontal axis
unk_collen = int(Tline_thickness / h - 1) #number of unknown nodes along vertical axis
unk_total  = int(unk_rowlen * unk_collen)

cond_len = int(center_conductor_width / h + 1)# number of nodes on conductor
off_len = int(center_conductor_offset / h - 1) #number of unknown nodes on offset length before conductor row
dualcond_len = int(dual_conductor_height / h)#number of unknown nodes between center conductors



#debug
test = np.arange(unk_total)
test = np.reshape(test, (unk_collen, unk_rowlen))

##determine 4-arm star coeficient diagonals and conductor and dielectric indicies 

upper_diag = (np.arange(unk_total - unk_rowlen),np.arange(unk_rowlen,unk_total))
lower_diag = (np.arange(unk_rowlen,unk_total),np.arange(unk_total - unk_rowlen))

left_diag  = (np.arange(1,unk_total),np.arange(unk_total - 1))
right_diag = (np.arange(unk_total-1),np.arange(1, unk_total))

cond_iindex = unk_total - 1 - (unk_rowlen - cond_len) // 2 - unk_rowlen * off_len #index of right node on lower conductor. int div = maybreak
bottom_cond = np.arange(cond_iindex - cond_len, cond_iindex) + 1
top_cond    = bottom_cond - dualcond_len * unk_rowlen

#dielectric indicies
D12_rowstart = (unk_collen - int(D1_D2 / h)) * unk_rowlen
D23_rowstart = (unk_collen - int(D2_D3 / h)) * unk_rowlen

D12_index = D12_rowstart + np.arange(0, unk_rowlen)
D23_index = D23_rowstart + np.arange(0, unk_rowlen)

D12_index = D12_index[-np.in1d(D12_index,np.hstack((top_cond,bottom_cond)))]
D23_index = D23_index[-np.in1d(D23_index,np.hstack((top_cond,bottom_cond)))]
##build coefficient array
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
coeff_mat[top_cond[0] - 1, top_cond[0]] = 0
coeff_mat[top_cond[-1:] + 1, top_cond[-1:]] = 0
coeff_mat[bottom_cond[0] - 1, bottom_cond[0]] = 0
coeff_mat[bottom_cond[-1:] + 1, bottom_cond[-1:]] = 0

coeff_mat[bottom_cond] = 0 
coeff_mat[bottom_cond,bottom_cond] = -1
coeff_mat[bottom_cond - unk_rowlen, bottom_cond] = 0
coeff_mat[bottom_cond + unk_rowlen, bottom_cond] = 0

coeff_mat[top_cond] = 0 
coeff_mat[top_cond,top_cond] = -1
coeff_mat[top_cond - unk_rowlen, top_cond] = 0
coeff_mat[top_cond + unk_rowlen, top_cond] = 0

cv = np.zeros((unk_total, 1))

cv[np.hstack((bottom_cond + unk_rowlen,bottom_cond - unk_rowlen,bottom_cond[0] - 1, bottom_cond, bottom_cond[-1] + 1))] -= voltage
cv[np.hstack((top_cond - unk_rowlen,top_cond + unk_rowlen,top_cond[0] - 1, top_cond, top_cond[-1] + 1))] -= voltage

plt.close('all')
plt.matshow(np.hstack((coeff_mat , cv)))
plt.colorbar()

## solve air filled
V = linalg.solve(coeff_mat, cv)
Vair = np.reshape(V, (unk_collen, unk_rowlen))
plt.figure()
plt.imshow(Vair)
plt.colorbar()

##dielectric analasys
dia = coeff_mat
#12 boundry
if not er12 == 2:
    dia[D12_index,D12_index] = -4 * er12
    dia[D12_index,D12_index - 1] = er12
    dia[D12_index,D12_index + 1] = er12
    dia[D12_index,D12_index - unk_rowlen] = 2 * er1
    dia[D12_index,D12_index + unk_rowlen] = 2 * er2
    #rezero knowns ( place holders for now )
    dia[top_cond[0] - 1, top_cond[0]] = 0
    dia[top_cond[-1:] + 1, top_cond[-1:]] = 0

#23 boundry
if not er23 == 2:
    dia[D23_index,D23_index] = -4 * er12
    dia[D23_index,D23_index - 1] = er12
    dia[D23_index,D23_index + 1] = er12
    dia[D23_index,D23_index - unk_rowlen] = 2 * er1
    dia[D23_index,D23_index + unk_rowlen] = 2 * er2
    #rezero knowns ( place holders for now )
    dia[bottom_cond[0] - 1, bottom_cond[0]] = 0
    dia[bottom_cond[-1:] + 1, bottom_cond[-1:]] = 0
    
if not (er12 or er23) == 2:
    dia[corner,corner + 1] = 0
    dia[corner + 1,corner] = 0
    
plt.matshow(np.hstack((dia , cv)))
plt.colorbar()

## solve dielectric filled
V = linalg.solve(coeff_mat, cv)
V = np.reshape(V, (unk_collen, unk_rowlen))
plt.figure()
plt.imshow(V)
plt.colorbar()

##contours for/and capacitance (applied on the Vair and V array elements)
#defaults (middle of the road contours)
vernode_len = off_len // 2
hornode_len = (unk_rowlen - cond_len) // 4

#defaults (middle of the road contours)
#vertical components
vernode_vair_left = Vair[vernode_len: unk_collen - vernode_len , hornode_len - 1:hornode_len + 2]
vernode_vair_right = Vair[vernode_len: unk_collen - vernode_len , unk_rowlen - hornode_len - 2:unk_rowlen - hornode_len + 1]
vernode_v_left = V[vernode_len: unk_collen - vernode_len , hornode_len - 1:hornode_len + 2]
vernode_v_right = V[vernode_len: unk_collen - vernode_len , unk_rowlen - hornode_len - 2:unk_rowlen - hornode_len + 1]
#horizontal components
hornode_vair_up = Vair[vernode_len - 1:vernode_len + 2,hornode_len:unk_rowlen - hornode_len]
hornode_vair_dwn = Vair[unk_collen - vernode_len - 2:unk_collen - vernode_len + 1,hornode_len:unk_rowlen - hornode_len]
hornode_v_up = V[vernode_len - 1:vernode_len + 2,hornode_len:unk_rowlen - hornode_len]
hornode_v_dwn = V[unk_collen - vernode_len - 2:unk_collen - vernode_len + 1,hornode_len:unk_rowlen - hornode_len]
#out from in
q0 = np.sum(vernode_vair_left[:,2] - vernode_vair_left[:,0])
q0 = q0 + np.sum(vernode_vair_right[:,0] - vernode_vair_right[:,2])
q0 = q0 + np.sum(hornode_vair_up[2,:] - hornode_vair_up[0,:])
q0 = q0 + np.sum(hornode_vair_dwn[0,:] - hornode_vair_dwn[2,:])

q = np.sum(vernode_v_left[:,2] - vernode_v_left[:,0])
q = q + np.sum(vernode_v_right[:,0] - vernode_v_right[:,2])
q = q + np.sum(hornode_v_up[2,:] - hornode_v_up[0,:])
q = q + np.sum(hornode_v_dwn[0,:] - hornode_v_dwn[2,:])

C0 = (e0 / 2) * q0 / voltage
C = (e0 / 2) * q / voltage # wrong: did not acount for dielectric in sum

z = 1 / (3 * 10 ** 8 * np.sqrt( (C0 * C)))

print(z)
# plt.figure()
plt.matshow(Vair)

##contour markers (careful with adding wrong values to the cumsum process)
# vernode_vair_left[:,1] += 5
# vernode_vair_right[:,1] += 5
# hornode_vair_up[1,:] += 5
# hornode_vair_dwn[1,:] += 5


