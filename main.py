import math
import numpy as np
from numpy.linalg import eigh
import imageio.v3 as iio
from PIL import Image, ImageOps

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

    return (norm_r + norm_g + norm_b) / 3

def get_norm_r(svd):
    size = svd.shape[0] * svd.shape[1]
    norm_r = np.linalg.matrix_norm(svd[:,:,0]) / size
    return norm_r

def get_norm_g(svd):
    size = svd.shape[0] * svd.shape[1]
    norm_g = np.linalg.matrix_norm(svd[:,:,1]) / size
    return norm_g

def get_norm_b(svd):
    size = svd.shape[0] * svd.shape[1]
    norm_b = np.linalg.matrix_norm(svd[:,:,2]) / size
    return norm_b

# experimentation with colored images

# img_og = iio.imread("orange_og.png")
# img_og_svd = svdColored(img_og, 2)

# iio.imwrite("out_og.png", img_og_svd)

# img_1 = iio.imread("orange1.png")
# img_1_svd = svdColored(img_1, 2)

# iio.imwrite("out_1.png", img_1_svd)

# img_apple = iio.imread("apple.png")
# img_apple_svd = svdColored(img_apple, 2)

# iio.imwrite("out_apple.png", img_apple_svd)

# norm_og = get_norm(img_og_svd)
# norm_img_1 = get_norm(img_1_svd)
# norm_img_apple = get_norm(img_apple)

# print(norm_img_1 - norm_og)
# print(norm_img_apple - norm_og)

def img_to_vector(img):
    matrix = np.array(img)
    out = []
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            out.append(matrix[x][y])
    return np.array(out)


def import_img(src):
    img = Image.open(src)
    return ImageOps.grayscale(img)

def resize(img):
    return img.resize((512, 512), Image.BILINEAR)

def cosine(vec_1, vec_2):
    return (np.dot(vec_1, vec_2)) / (np.sqrt(vec_1.dot(vec_1)) * np.sqrt(vec_2.dot(vec_2)))


orange_og = import_img("orange_og.png")
orange_og = resize(orange_og)
iio.imwrite("out_og.png", np.array(orange_og).astype(np.uint8))
vec_og = img_to_vector(orange_og)

orange1 = import_img("orange1.png")
orange1 = resize(orange1)
iio.imwrite("out_1.png", np.array(orange1).astype(np.uint8))
vec_orange = img_to_vector(orange1)

apple = import_img("apple.png")
apple = resize(apple)
iio.imwrite("out_apple.png", np.array(apple).astype(np.uint8))
vec_apple = img_to_vector(apple)

print("Cosine of orange and tangerine: ")
print(cosine(vec_og, vec_orange))

print("Cosine of orange and apple: ")
print(cosine(vec_og, vec_apple))


ex_1 = import_img("example_1.png")

r_row = np.array( # 3 x 4 
    [[0.83, 0.17, 0, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0.17, 0.83]]
)

r_col = np.array( # 3 x 6
    [[0.5, 0.5, 0, 0, 0, 0],
    [0, 0, 0.5, 0.5, 0, 0],
    [0, 0, 0, 0, 0.5, 0.5]]
)

out = r_row @ ex_1 @ r_col.T

iio.imwrite("ex_out.png", out.astype(np.uint8))