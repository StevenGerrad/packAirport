
################################################################################
# 
#       --------------------- 
#       作者：章子雎Kevin 
#       来源：CSDN 
#       原文：https://blog.csdn.net/zhangziju/article/details/79754652 
#       版权声明：本文为博主原创文章，转载请附上博文链接！
# 
################################################################################

########## ========== ########## 基于BFmatcher的SIFT实现 ########## ========== ##########
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg'
imgname2 = 'D:/auxiliaryPlane/project/Python/packAirport/image/002.jpg'

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

mR = max(img1.shape[0], img2.shape[0])
mC = max(img1.shape[1], img2.shape[1])

img1 = cv2.resize(img1, (mC, mR), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (mC, mR), interpolation=cv2.INTER_CUBIC)


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)      #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)        #des是描述子

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)      #灰度处理图像
kp2, des2 = sift.detectAndCompute(img2,None)        #des是描述子

print (gray1.shape, gray2.shape)

hmerge = np.hstack((gray1, gray2))      #水平拼接

# cv2.namedWindow("gray",cv2.WINDOW_NORMAL)
# cv2.imshow("gray", hmerge)
# cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
hmerge = np.hstack((img3, img4))        #水平拼接

cv2.namedWindow("point",cv2.WINDOW_NORMAL)
cv2.imshow("point", hmerge)     #拼接显示为gray
cv2.waitKey(0)

# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.namedWindow("BFmatch",cv2.WINDOW_NORMAL)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
# img5.save('../test.jpg')
cv2.destroyAllWindows()
'''

########## ========== ########## 基于FlannBasedMatcher的SIFT实现 ########## ========== ##########

'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'packAirport/image/001.jpg'
imgname2 = 'packAirport/image/002.jpg'

sift = cv2.xfeatures2d.SIFT_create()

# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

mR = max(img1.shape[0], img2.shape[0])
mC = max(img1.shape[1], img2.shape[1])

img1 = cv2.resize(img1, (mC, mR), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (mC, mR), interpolation=cv2.INTER_CUBIC)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)#des是描述子

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2,None)

hmerge = np.hstack((gray1, gray2))      #水平拼接

# cv2.imshow("gray", hmerge)          #拼接显示为gray
# cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

hmerge = np.hstack((img3, img4))        #水平拼接
cv2.imshow("point", hmerge)         #拼接显示为gray
cv2.waitKey(0)

matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
# img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
print("===================================================")
cv2.imshow("FLANN", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

########## ========== ########## 基于FlannBasedMatcher的SURF实现 ########## ========== ##########

'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'packAirport/image/001.jpg'
imgname2 = 'packAirport/image/002.jpg'

surf = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

mR = max(img1.shape[0], img2.shape[0])
mC = max(img1.shape[1], img2.shape[1])

img1 = cv2.resize(img1, (mC, mR), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (mC, mR), interpolation=cv2.INTER_CUBIC)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = surf.detectAndCompute(img1,None)#des是描述子

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = surf.detectAndCompute(img2,None)

hmerge = np.hstack((gray1, gray2)) #水平拼接
cv2.imshow("gray", hmerge) #拼接显示为gray
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
cv2.waitKey(0)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("SURF", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''

########## ========== ########## 基于BFMatcher的ORB实现 ########## ========== ##########


import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'packAirport/image/001.jpg'
imgname2 = 'packAirport/image/002.jpg'

orb = cv2.ORB_create()

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

mR = max(img1.shape[0], img2.shape[0])
mC = max(img1.shape[1], img2.shape[1])

img1 = cv2.resize(img1, (mC, mR), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2, (mC, mR), interpolation=cv2.INTER_CUBIC)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)      #灰度处理图像
kp1, des1 = orb.detectAndCompute(img1,None)     #des是描述子

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = orb.detectAndCompute(img2,None)

hmerge = np.hstack((gray1, gray2))              #水平拼接
cv2.imshow("gray", hmerge)              #拼接显示为gray
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
cv2.waitKey(0)

# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("ORB", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
