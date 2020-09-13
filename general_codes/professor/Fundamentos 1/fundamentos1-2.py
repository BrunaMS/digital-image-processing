import cv2 as cv
import os


std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
file_name = 'lena_gray_512.tif'
# file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)

print(img.shape)
# h, w = img.shape

print(img.size)
# npix = img.size

print(img.dtype)

x=400
y=115

print(img[x,y])

print(img[...].min())
print(img[...].max())