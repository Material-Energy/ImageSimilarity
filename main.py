import math
import numpy as np
from numpy.linalg import eigh
import imageio.v3 as iio

def svd(m, dim):
    u, e, v_t = np.linalg.svd(m, full_matrices=False)

    out = np.zeros(m.shape)
    for i in range(min(dim, min(m.shape))):
        out += np.outer(u[:,i], v_t[i]) * e[i]
    
    return out

def rgbToImg(r, g, b):
    shape = r.shape + (3,)
    img = np.zeros(shape)
    print(img.shape)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            img[i][j][0] = min(max(r[i][j], 0), 255)
            img[i][j][1] = min(max(g[i][j], 0), 255)
            img[i][j][2] = min(max(b[i][j], 0), 255)
    return img

def svdColored(img, dim):
    r = svd(img[:,:,0], dim)
    g = svd(img[:,:,1], dim)
    b = svd(img[:,:,2], dim)

    return rgbToImg(r, g, b).astype(np.uint8)

def get_norm(svd):
    size = svd.shape[0] * svd.shape[1]
    norm_r = np.linalg.matrix_norm(svd[:,:,0]) / size
    norm_g = np.linalg.matrix_norm(svd[:,:,1]) / size
    norm_b = np.linalg.matrix_norm(svd[:,:,2]) / size

    return math.sqrt(norm_r ** 2 + norm_g ** 2 + norm_b ** 2)


# print(u @ np.diag(e) @ v_t)

img_og = iio.imread("orange_og.png")
img_og_svd = svdColored(img_og, 2)

iio.imwrite("out2.png", img_og_svd)

img_1 = iio.imread("orange1.png")
img_1_svd = svdColored(img_1, 2)

img_apple = iio.imread("apple.png")
img_apple_svd = svdColored(img_apple, 2)

iio.imwrite("out_apple.png", img_apple_svd)

norm_og = get_norm(img_og_svd)
norm_img_1 = get_norm(img_1_svd)
norm_img_apple = get_norm(img_apple)

print(norm_img_1 - norm_og)
print(norm_img_apple - norm_og)