import numpy as np
from numpy.linalg import eig

def svd(m, dim):
    m_t = np.transpose(m) 
    sigma, vec2 = eig(m_t @ m)
    _, vec1 = eig(m @ m_t)

    out = np.zeros(m.shape)
    for i in range(dim):
        out += np.transpose(vec1[i]) @ vec2[i] * (sigma[i] ** 0.5)
    return out

    return vec1[:dim], sigma[:dim], vec2[:dim]


test = np.matrix([[35, 23], [58, 92], [28, 64]])
print(svd(test, 1))
print(np.linalg.svd(test))