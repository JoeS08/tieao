import numpy as np
import LightPipes as lp
import cv2
import matplotlib.pyplot as plt


nGrid = 256
size = 20  # size of the field [mm]
x = np.arange(-nGrid/2, nGrid/2, 1)
y = np.arange(-nGrid/2, nGrid/2, 1)[::-1]
wv = 500 * 1e-6  # lambda [mm]
A = wv / (2 * np.pi)
# print(A)

r = np.floor(nGrid * 0.2 / 2)
obj = np.zeros((nGrid, nGrid))

for i in range(nGrid):
    for j in range(nGrid):
        if x[j]**2 + y[i]**2 < r**2:
            obj[i, j] = 1

ft_obj = np.fft.fftshift(np.fft.fft2(obj))
#amp_obj = np.abs(ft_obj)**2
amp_obj = np.abs(ft_obj)
phase_obj = np.angle(ft_obj)

noll = 8
f = lp.Begin(size, wv, nGrid)
(nz, mz) = lp.noll_to_zern(noll)  # coma
f = lp.Zernike(f, nz, mz, size/2, A)
f = lp.CircAperture(f, size/2, 0, 0)
phi = lp.Phase(f)

ft_obj_ = ft_obj * np.exp(1j * phi)

cv2.imwrite('object.bmp', obj*255)
cv2.imwrite('amp.bmp', amp_obj/np.max(amp_obj) * 255)
cv2.imwrite('phase.bmp', phase_obj/np.max(phase_obj) * 255)
abeName = lp.ZernikeName(noll)
cv2.imwrite(abeName + '.bmp', phi/np.max(phi) * 255)


abe_obj = np.fft.ifft2(ft_obj_)
abe_obj_ = np.abs(abe_obj)**2
abe_int = abe_obj_/np.max(abe_obj_)*255
cv2.imwrite('Aberated_object_' + abeName +
            '.bmp', abe_int)

abe_int = abe_int / 255
_abe_int = np.fft.fftshift(np.fft.fft2(abe_int))
ftphi = np.angle(_abe_int)
# print(ftphi)
ftao = ft_obj_ * np.exp(-1j * ftphi)
ftao = np.fft.ifftshift(np.fft.ifft2(ftao))
ftao_int = np.abs(ftao)**2

plt.figure()
plt.imshow(ftao_int/np.max(ftao_int)*255)
plt.show()
