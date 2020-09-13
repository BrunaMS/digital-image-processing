import cv2 as cv
import os
from matplotlib import pyplot as plt

plt.close('all')

std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
# file_name = 'lena_gray_512.tif'
file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure()

plt.imshow(img[-128:-1,0:127], cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.figure()

plt.imshow(img[-128:-1,0:127], cmap = 'gray', interpolation = 'bicubic', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
