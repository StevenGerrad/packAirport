
################################################################################################
# 
#       2019.7.22
#       Feature Detection and Description
#       ------------
#       https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html
# 
################################################################################################


########## ---------- ########## Harris Corner Detection ########## ---------- ##########

# Harris Corner Detector in OpenCV 
# 
# img - Input image, it should be grayscale and float32 type.
# blockSize - It is the size of neighbourhood considered for corner detection
# ksize - Aperture parameter of Sobel derivative used.
# k - Harris detector free parameter in the equation.

'''

import cv2
import numpy as np
#加载图像
image = cv2.imread('packAirport/image\IMG/test img-B/training/0.png')
#自定义卷积核
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
        [1,1,1],
        [1,-7,1],
        [1,1,1]])
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0
#卷积
output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
#显示锐化效果
cv2.imshow('Original Image',image)
cv2.imshow('sharpen_1 Image',output_1)
cv2.imshow('sharpen_2 Image',output_2)
cv2.imshow('sharpen_3 Image',output_3)
#停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()

'''

'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

filename = 'packAirport/image/bai1.png'
filename1 = 'packAirport/image/bai2.png'
filename2 = 'packAirport/image/bai3.png'

img = cv.imread(filename)
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
img1 = cv.filter2D(img,-1,kernel_sharpen_1)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
gray1 = np.float32(gray1)

dst = cv.cornerHarris(gray,2,3,0.04)
dst1 = cv.cornerHarris(gray1,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
dst1 = cv.dilate(dst1,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
img1[dst1>0.01*dst1.max()]=[0,0,255]

print ('==========')

plt.subplot(1,2,1),plt.title('img'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img1'),plt.imshow(img1)
plt.show()

'''

# cv.namedWindow("dst",cv.WINDOW_NORMAL)
# cv.imshow('dst',img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()

# Corner with SubPixel Accuracy 亚像素(类似动画图片那种的)、暂时不研究

# Shi-Tomasi Corner Detector & Good Features to Track

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/bai1.png')
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img,-1,kernel_sharpen_1)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
corners1 = cv.goodFeaturesToTrack(gray1,25,0.01,10)
corners1 = np.int0(corners1)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

for i in corners1:
    x,y = i.ravel()
    cv.circle(img1,(x,y),3,255,-1)

plt.subplot(1,2,1),plt.title('img'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img1'),plt.imshow(img1)
plt.show()
'''

# Introduction to SIFT (Scale-Invariant Feature Transform)

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/bai1.png')

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img,-1,kernel_sharpen_1)

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
# kp, des = sift.detectAndCompute(gray,None)
img=cv.drawKeypoints(gray,kp,img)

gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
kp1 = sift.detect(gray1,None)
# kp, des = sift.detectAndCompute(gray,None)
img1=cv.drawKeypoints(gray1,kp1,img1)

# cv.imwrite('sift_keypoints.jpg',img)
# cv.namedWindow("sift_keypoints",cv.WINDOW_NORMAL)
# cv.imshow('sift_keypoints',img)
# cv.waitKey(0)

plt.subplot(1,2,1),plt.title('img'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img1'),plt.imshow(img1)
plt.show()
'''


# Introduction to SURF (Speeded-Up Robust Features)
# SURF is good at handling images with blurring and rotation, 
# but not good at handling viewpoint change and illumination change.

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('D:/auxiliaryPlane/project/Python/packAirport/image\IMG/test img-B/training/0.png',0)
img = cv.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg',0)
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img = cv.filter2D(img,-1,kernel_sharpen_1)

surf = cv.xfeatures2d.SURF_create(400)
kp, des = surf.detectAndCompute(img,None)
print ('kp',len(kp))
print ('getHessianThreshold', surf.getHessianThreshold() )
# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(img,None)
print ('kp',len(kp) )

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

print( surf.getUpright() )
surf.setUpright(True)
# Recompute the feature points and draw it
kp = surf.detect(img,None)
img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

print( surf.descriptorSize() )

surf.getExtended()
# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(img,None)
print( surf.descriptorSize() )
# print( des.shape )
'''
# 这个不知道是参数不对还是咋的，没啥效果啊

'''
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('packAirport/image/bai1.png')

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv2.filter2D(img,-1,kernel_sharpen_1)

#参数为hessian矩阵的阈值
surf = cv2.xfeatures2d.SURF_create(400)
#找到关键点和描述符
key_query,desc_query = surf.detectAndCompute(img,None)
#把特征点标记到图片上
img=cv2.drawKeypoints(img,key_query,img)

key_query1,desc_query1 = surf.detectAndCompute(img1,None)
img1=cv2.drawKeypoints(img1,key_query1,img1)

plt.subplot(1,2,1),plt.title('img'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img1'),plt.imshow(img1)
plt.show()
'''

# FAST Algorithm for Corner Detection

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('packAirport/image/bai1.png',0)
# img = cv.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg',0)
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
# img = cv.filter2D(img,-1,kernel_sharpen_1)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# cv.imwrite('fast_true.png',img2)

# cv.namedWindow("fast_true",cv.WINDOW_NORMAL),cv.imshow('fast_true',img2),cv.waitKey(0)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
# cv.imwrite('fast_false.png',img3)

# cv.namedWindow("fast_false",cv.WINDOW_NORMAL),cv.imshow('fast_false',img3),cv.waitKey(0)

plt.subplot(1,2,1),plt.title('fast_true'),plt.imshow(img2)
plt.subplot(1,2,2),plt.title('fast_false'),plt.imshow(img3)
plt.show()

'''
# BRIEF (Binary Robust Independent Elementary Features)

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/bai1.png',0)
# img = cv.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg',0)
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img,-1,kernel_sharpen_1)
# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()
# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

kp1 = star.detect(img1,None)
kp1, des1 = brief.compute(img1, kp1)

img = cv.drawKeypoints(img, kp, None, color=(255,0,0))
img2 = cv.drawKeypoints(img1, kp, None, color=(255,0,0))

print( brief.descriptorSize() )
print( des.shape )

plt.subplot(1,2,1),plt.title('img1'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img2'),plt.imshow(img2)
plt.show()
'''

# ORB (Oriented FAST and Rotated BRIEF)

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/bai1.png',0)

# img = cv.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg',0)
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img,-1,kernel_sharpen_1)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

kp1 = orb.detect(img1,None)
kp1, des1 = orb.compute(img1, kp1)
img1 = cv.drawKeypoints(img1, kp, None, color=(0,255,0), flags=0)

plt.subplot(1,2,1),plt.title('img'),plt.imshow(img)
plt.subplot(1,2,2),plt.title('img1'),plt.imshow(img1)
plt.show()

'''

########## ========== ########## Feature Matching ########## ========== ##########

# Brute-Force Matching with ORB Descriptors

'''
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('packAirport/image/pCut1/video0/0-001.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('packAirport/image/pCut1/video1/1-001.jpg',cv.IMREAD_GRAYSCALE) # trainImage

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img1,-1,kernel_sharpen_1)
img2 = cv.filter2D(img2,-1,kernel_sharpen_1)

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
'''


# Brute-Force Matching with SIFT Descriptors and Ratio Test

'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('packAirport/image/pCut1/video0/0-001.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('packAirport/image/pCut1/video1/1-001.jpg',cv.IMREAD_GRAYSCALE) # trainImage

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img1,-1,kernel_sharpen_1)
img2 = cv.filter2D(img2,-1,kernel_sharpen_1)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
'''


# FLANN based Matcher
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('packAirport/image/pCut1/video0/0-001.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('packAirport/image/pCut1/video1/1-001.jpg',cv.IMREAD_GRAYSCALE) # trainImage

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img1,-1,kernel_sharpen_1)
img2 = cv.filter2D(img2,-1,kernel_sharpen_1)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
'''


########## ########## Feature Matching + Homography to find Objects ########## ##########



import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('packAirport/image/pCut1/video0/0-001.jpg',0)          # queryImage
img2 = cv.imread('packAirport/image/pCut1/video1/1-001.jpg',0)          # trainImage

kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img1 = cv.filter2D(img1,-1,kernel_sharpen_1)
img2 = cv.filter2D(img2,-1,kernel_sharpen_1)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

