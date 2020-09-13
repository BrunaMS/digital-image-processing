import cv2 as cv
import os
import numpy as np

std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
# file_name = 'lena_gray_512.tif'
file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)


cv.imshow('teste',img)


img_sub = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_AREA)
cv.imshow('image sub',img_sub)

img2 = cv.resize(img_sub,None,fx=2, fy=2, interpolation = cv.INTER_NEAREST)
cv.imshow('image2',img2)

img3 = cv.resize(img_sub,None,fx=2, fy=2, interpolation = cv.INTER_LINEAR)
cv.imshow('image3',img3)

img4 = cv.resize(img_sub,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
cv.imshow('image4',img4)

img5 = cv.resize(img_sub,None,fx=2, fy=2, interpolation = cv.INTER_LANCZOS4)
cv.imshow('image5',img5)

dif2 = np.uint8(np.int32(img2) - np.int32(img) + 128)
cv.imshow('dif2',dif2)

dif5 = np.uint8(np.int32(img5) - np.int32(img) + 128)
cv.imshow('dif5',dif5)