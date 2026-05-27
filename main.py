import math
import numpy as np
from numpy.linalg import eigh
import imageio.v3 as iio
from PIL import Image, ImageOps

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
    return img.resize((64, 64), Image.BILINEAR)

def cosine(vec_1, vec_2):
    return (np.dot(vec_1, vec_2)) / (np.sqrt(vec_1.dot(vec_1)) * np.sqrt(vec_2.dot(vec_2)))


orange = import_img("photos/orange.png")
orange = resize(orange)
iio.imwrite("out_orange.png", np.array(orange).astype(np.uint8))
vec_orange = img_to_vector(orange)

tangerine = import_img("photos/tangerine.png")
tangerine = resize(tangerine)
iio.imwrite("out_tangerine.png", np.array(tangerine).astype(np.uint8))
vec_tangerine = img_to_vector(tangerine)

apple = import_img("photos/apple.png")
apple = resize(apple)
iio.imwrite("out_apple.png", np.array(apple).astype(np.uint8))
vec_apple = img_to_vector(apple)

print("Cosine of orange and tangerine: ")
print(cosine(vec_orange, vec_tangerine))

print("Cosine of orange and apple: ")
print(cosine(vec_orange, vec_apple))


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

lemon = import_img("photos/lemon.png")
lemon = resize(lemon)
iio.imwrite("out_lemon.png", np.array(lemon).astype(np.uint8))
vec_lemon = img_to_vector(lemon)

lime = import_img("photos/lime.jpg")
lime = resize(lime)
iio.imwrite("out_lime.png", np.array(lime).astype(np.uint8))
vec_lime = img_to_vector(lime)

grape = import_img("photos/grape.jpg")
grape = resize(grape)
iio.imwrite("out_grape.png", np.array(grape).astype(np.uint8))
vec_grape = img_to_vector(grape)

def normalize(vec):
    return vec / np.linalg.norm(vec)

images = np.array([
    normalize(vec_tangerine),
    normalize(vec_orange),
    normalize(vec_apple),
    normalize(vec_lemon),
    normalize(vec_lime),
    normalize(vec_grape)
])



test_image = import_img("photos/test.jpg")
test_image = resize(test_image)
iio.imwrite("out_test.png", np.array(test_image).astype(np.uint8))
vec_test = normalize(img_to_vector(test_image))

output = (images @ vec_test)
print("Similarity of Test Image: ")
print(output)

ranked = np.argsort(output)[::-1]
labels = ['Tangerine', 'Orange', 'Apple', 'Lemon', 'Lime', 'Grape']
for i in ranked:
    print(f"  {labels[i]}: {output[i]:.3f}")


## SVD time

