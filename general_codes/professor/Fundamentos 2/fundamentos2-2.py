import cv2 as cv
import os
from matplotlib import pyplot as plt

std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
# file_name = 'lena_gray_512.tif'
file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)


cv.imshow('teste',img)


img_sub = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_AREA)
cv.imshow('image2',img_sub)


plt.imshow(img_sub, cmap = 'gray', interpolation = 'nearest')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure()

plt.imshow(img_sub, cmap = 'gray', interpolation = 'lanczos')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()