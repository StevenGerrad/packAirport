

###############################################################################################
# 
#       2019.8.3
#       机场行李分割
#       ---------
#       水平集
# 
###############################################################################################

'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import string
import pylab

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

def claneH(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def sharpenImg(src):
    kernel_sharpen_1 = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]])
    src = cv.filter2D(src,-1,kernel_sharpen_1)
    return src

img = cv.imread('packAirport/image/'+'001.jpg')
img = sharpenImg(img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = claneH(gray)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plt.subplot(2,2,1),plt.title('thresh'),plt.imshow(thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
plt.subplot(2,2,2),plt.title('opening'),plt.imshow(opening)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
plt.subplot(2,2,3),plt.title('unknown'),plt.imshow(unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.subplot(2,2,4),plt.title('img'),plt.imshow(img)

plt.show()
'''


###############################################################################################
# 
#       2019.8.3
#       机场行李分割
#       ---------
#       canny+finContours+grabCut
# 
###############################################################################################


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import string
import pylab

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

def fillHole(im_in):
	im_floodfill = im_in.copy()
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	# Floodfill from point (0, 0)
	cv.floodFill(im_floodfill, mask, (0,0), 255)
	# Invert floodfilled image
	im_floodfill_inv = cv.bitwise_not(im_floodfill)
	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv
	return im_out

def claneH(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def sharpenImg(src):
    kernel_sharpen_1 = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]])
    kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0
    # src = cv.filter2D(src,-1,kernel_sharpen_1)
    src = cv.filter2D(src,-1,kernel_sharpen_3)
    return src

img = cv.imread('packAirport/image/pCut2/video0/'+'0-001.jpg')
img = sharpenImg(img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = claneH(gray)

edges = cv.Canny(gray,100,200,apertureSize = 3)

kernel = np.ones((3,3),np.uint8)
edges = cv.dilate(edges,kernel,iterations=2)
opening = cv.morphologyEx(edges,cv.MORPH_OPEN,kernel, iterations = 2)

thresh, contours, hierarchy = cv.findContours(opening, 2, 1)
cv.drawContours(img, contours, -1, (0,255,0), 3)

plt.subplot(1,2,1), plt.imshow(img)
plt.subplot(1,2,2), plt.imshow(edges)
plt.show()


'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/pCut2/video0/'+'0-001.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
thresh, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0,255,0), 3)

plt.subplot(1,2,1), plt.imshow(thresh)
plt.subplot(1,2,2), plt.imshow(img)
plt.show()
'''