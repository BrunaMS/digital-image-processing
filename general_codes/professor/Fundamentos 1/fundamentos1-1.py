import cv2 as cv
import os

std_images_path = os.path.abspath('../../Material_complementar/standard_test_images/gray/')
# file_name = 'lena_gray_512.tif'
file_name = 'cameraman.tif'

file = os.path.join(std_images_path, file_name)

img = cv.imread(file,0)

# =============================================================================
# cv.namedWindow('teste', cv.WINDOW_NORMAL)
# cv.waitKey(0)
# =============================================================================

cv.imshow('teste',img)

# =============================================================================
# cv.waitKey(0)
# cv.destroyAllWindows()
# 
# cv.imwrite('teste.jpg',img)
# =============================================================================
