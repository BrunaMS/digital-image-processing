import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

plt.close('all')

img = np.ones([512,512],dtype=np.uint8)*128

cv.imshow('teste',img)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


plt.imshow(img, cmap = 'gray', interpolation = 'bicubic', vmin=0, vmax=255)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

img[0:127,0:127]=118

img[-128:-1,-128:-1]=138

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
