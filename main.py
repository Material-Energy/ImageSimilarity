import numpy as np
from numpy.linalg import eig
import imageio.v3 as iio

def svd(m, dim):
    m_t = np.transpose(m) 
    sigma, vec2 = eig(m_t @ m)
    _, vec1 = eig(m @ m_t)

    out = np.zeros(m.shape)
    for i in range(dim):
        out += (np.transpose(vec1[i]) @ vec2[i] * (sigma[i] ** 0.5)).astype(int)
    return out

    return vec1[:dim], sigma[:dim], vec2[:dim]

def rgbToImg(r, g, b):
    shape = r.shape + (3,)
    img = np.zeros(shape)
    print(img.shape)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            img[i][j][0] = r[i][j]
            img[i][j][1] = g[i][j]
            img[i][j][2] = b[i][j]
    return img


test = np.matrix([[35, 23], [58, 92], [28, 64]])
print(svd(test, 1))
u, e, v_t = np.linalg.svd(test, full_matrices=False)
sig = np.zeros((3, 2))
for i in range(2):
    sig[i, i] = e[i]
print(u @ sig @ v_t)

img = iio.imread("orange1.png")
r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]

print(r)

r = svd(r, 4)
g = svd(g, 4)
b = svd(b, 4)

# print(rgbToImg(r, g, b))
# iio.imwrite("out.png", img)