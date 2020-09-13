import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
# file_name = 'img_gray_512.tif'
file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)

xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

# create the figure
# =============================================================================
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap=plt.cm.jet,
#                 linewidth=0)
# 
# # show it
# plt.show()
# =============================================================================


from mayavi import mlab

mlab.figure(bgcolor=(1,1,1))

# We'll use "surf" to display a 2D grid...
# warp_scale='auto' guesses a vertical exaggeration for better display.
# Feel free to remove the "warp_scale" argument for "true" display.
mlab.surf(xx, yy, img, warp_scale='auto')

mlab.show()
