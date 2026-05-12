import numpy as np
from numpy.linalg import eig

def svd(m, dim):
    m_t = np.transpose(m) 
    sigma, vec2 = eig(m_t @ m)
    _, vec1 = eig(m @ m_t)

    out = np.zeros(m.shape)
    for i in dim:
        out += vec1[i] * sigma[i] * np.transpose(vec2[i])
    return out

    return vec1[:dim], sigma[:dim], vec2[:dim]


