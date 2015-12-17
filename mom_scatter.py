import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import pi
from scipy import integrate
from scipy import special


nodeTot = 72

inRad = .25
outRad = .30

er = 4

ei = 1
c = 3e8

## Digitize physical space
dt = 2 * pi / nodeTot

rd = (inRad + outRad) / 2

# rt = np.sqrt( (rd * dt * (outRad - inRad )) / pi)

rti, err = integrate.quad(lambda ang: (outRad**2 - inRad**2)/2, 0, dt)
rt = np.sqrt(rti/pi)
print(dt*180/pi,rti,rt)
    
theta = np.linspace(0,2*pi - dt, nodeTot)

r = rd * np.sqrt( 
                ( np.cos(theta[0]) - np.cos(theta) )**2 +
                ( np.sin(theta[0]) - np.sin(theta) )**2
                )

## build coefficient matrix
r = linalg.toeplitz(r)

z = 1j  * pi**2 * rt * (er - 1) * (
                                    special.jv(1, 2*pi*rt) * 
                                    special.hankel2(0, 2 * pi * r)
                                  ) 

#diagonal
di = np.diag_indices(nodeTot)

z[di] = 1 + (er-1) * .5j * ( 2 * pi**2 * rt * special.hankel2(1, 2*pi*rt) - 2j )

cv = ei * np.exp(-2j * pi * rd * np.cos(theta))


# plt.matshow(np.imag(z))

#solve for field
e = linalg.solve(z, cv)

##echo
esi = np.empty((nodeTot))
for i in np.arange(0,nodeTot):
    esi[i] = np.sum( 
                    (er -1) * e * rt *
                    special.jv(1, 2 * pi) *
                    np.exp( 2j*pi*rd * (
                                        np.cos(theta)*np.cos(theta[i]) +
                                        np.sin(theta)*np.cos(theta[i])
                                       ) )
                  )

es = 2*pi**4 * np.abs(esi)**2

## pretty plot
plt.close('all')

x = theta[:nodeTot//2]*180/pi

plt.subplot(2, 1, 1)
plt.plot(x, abs(e[:nodeTot//2]), 'ko-')
plt.title('A tale of 2 subplots')
plt.ylabel('Electric Field Distribution')

plt.subplot(2, 1, 2)
plt.plot(x, es[:nodeTot//2], 'r.-')
plt.xlabel('angle(degrees)')
plt.ylabel('Echo Width')

plt.show()