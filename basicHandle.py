
#############################################################################################
#
#       2019.7.31
#       机场分拣行李匹配--优化图像，进行基本变换研究
#       -----------
#       https://docs.opencv.org/master/de/db2/tutorial_py_table_of_contents_histograms.html
#       
#############################################################################################


# High Dynamic Range (HDR) 高动态范围(明暗变换)
# 这个似乎是用于摄像机的，需要曝光度不同的图片


########## ########## Histograms - 1 : Find, Plot, Analyze !!! ########## ##########

# https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html

'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

img = cv.imread('packAirport/image/'+'001.jpg')
plt.subplot(2,3,1), plt.title('img'), plt.imshow(img)
plt.subplot(2,3,2), plt.hist(img.ravel(),256,[0,256])
plt.subplot(2,3,3)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

gray = img

#伽马变换
# https://blog.csdn.net/yawdd/article/details/80180848/

gamma=copy.deepcopy(gray)
rows=img.shape[0]
cols=img.shape[1]
for i in range(rows):
    for j in range(cols):
        gamma[i][j]=3*pow(gamma[i][j],0.8)

plt.subplot(2,3,4), plt.title('gamma'), plt.imshow(gamma)
plt.subplot(2,3,5), plt.hist(gamma.ravel(),256,[0,256])
plt.subplot(2,3,6)
for i,col in enumerate(color):
    histr = cv.calcHist([gamma],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()
'''


# Using OpenCV: Application of Mask
# 使用这个mask看看可不可以处理一下白大佬抠图的效果
# tmd这个直方图怎么这么个b型

'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

img = cv.imread('packAirport/image/'+img_path1,0)
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
# mask[100:300, 100:400] = 255
mask[img != 255] = 255
masked_img = cv.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
'''



########## ########## Histograms - 2: Histogram Equalization ########## ##########

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

img = cv.imread('packAirport/image/'+'001.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()             # 按照所给定的轴参数返回元素的梯形累计和
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.subplot(2,2,1), plt.title('original'), plt.imshow(img, cmap='gray')
plt.subplot(2,2,2)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')


cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

plt.subplot(2,2,3), plt.title('after masked_equal'), plt.imshow(img2, cmap='gray')
plt.subplot(2,2,4)
plt.plot(cdf, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()
'''


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# 好的，下面出场的这个我希望是个大杀器

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

img = cv.imread('packAirport/image/'+'001.jpg', 0)
hist_original = cv.calcHist([img],[0],None,[256],[0,256])
plt.subplot(2,2,1), plt.title('img'), plt.imshow(img, cmap='gray')
plt.subplot(2,2,2), plt.plot(hist_original)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
hist_handle = cv.calcHist([cl1],[0],None,[256],[0,256])
plt.subplot(2,2,3), plt.title('cl1'), plt.imshow(cl1, cmap='gray')
plt.subplot(2,2,4), plt.plot(hist_handle)

plt.show()

'''


########## ########## Histograms - 3 : 2D Histograms ########## ##########
# find color histograms where two features are Hue & Saturation values

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

img = cv.imread('packAirport/image/'+'001.jpg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

plt.subplot(1,2,1),plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(hist,interpolation = 'nearest')

plt.show()
'''


########## ########## Histogram - 4 : Histogram Backprojection ########## ##########

# Algorithm in Numpy

# Backprojection in OpenCV
# 这个有点不太明白
# https://segmentfault.com/a/1190000015676940

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

roi = cv.imread('packAirport/image/'+'002.jpg')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target  = cv.imread('packAirport/image/'+'bai1.png')
# plt.imshow(target), plt.show()

hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)

# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256])

# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))       # 多通道合并
res = cv.bitwise_and(target,thresh)

# res = np.vstack((target,thresh,res))
# cv.imwrite('res.jpg',res)
plt.subplot(1,3,1), plt.title('target'), plt.imshow(target)
plt.subplot(1,3,2), plt.title('thresh'), plt.imshow(thresh)
plt.subplot(1,3,3), plt.title('res'), plt.imshow(res)

plt.show()
'''

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# roi是我们需要找到的对象或区域
# roi = cv2.imread('img_roi.png')
roi = cv2.imread('packAirport/image/'+'002.jpg')


hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# target是我们搜索的图像
# target = cv2.imread('img.jpg')
target  = cv2.imread('packAirport/image/'+'bai1.png')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# 计算对象的直方图
roihist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

# 标准化直方图，并应用投影
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256], 1)

# 与磁盘内核进行卷积
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
cv2.filter2D(dst, -1, disc, dst)

# 阈值、二进制按位和操作
ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)

plt.subplot(1,3,1), plt.title('target'), plt.imshow(target)
plt.subplot(1,3,2), plt.title('thresh'), plt.imshow(thresh)
plt.subplot(1,3,3), plt.title('res'), plt.imshow(res)

plt.show()

res = np.vstack((target, thresh, res))
# cv2.imshow('res', res), cv2.waitKey()

'''

#############################################################################################
#
#       2019.7.31
#       交互式的调整亮度和对比度
#       -----------
#       https://www.cnblogs.com/lfri/p/10753019.html
#       
#############################################################################################

'''

import cv2
import numpy as np

alpha = 0.3
beta = 80
img_path = "packAirport/image/002.jpg"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)

def updateAlpha(x):
    global alpha,img,img2
    alpha = cv2.getTrackbarPos('Alpha','image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))

def updateBeta(x):
    global beta,img,img2
    beta = cv2.getTrackbarPos('Beta','image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
  

# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha','image',0,300,updateAlpha)
cv2.createTrackbar('Beta','image',0,255,updateBeta)
cv2.setTrackbarPos('Alpha','image',100)
cv2.setTrackbarPos('Beta','image',10)
# 设置鼠标事件回调
#cv2.setMouseCallback('image',update)  


while(True):
    cv2.imshow('image',img) 
    if cv2.waitKey(1) == ord('q'):  
        break

cv2.destroyAllWindows()

'''


###############################################################################################
# 
#       2019.8.2
#       ---------
#       关于多种方法，图像亮度调整，锐化滤波等的顺序问题
#       或许应该找一个方法测评一下
# 
###############################################################################################


import numpy as np
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

def claneH(img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 加入来自openCV的histogram的有限对比度直方图均衡
        hist_original = cv.calcHist([img],[0],None,[256],[0,256])
        # plt.subplot(2,2,1), plt.title('img'), plt.imshow(img, cmap='gray')
        # plt.subplot(2,2,2), plt.plot(hist_original)

        # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        hist_handle = cv.calcHist([cl1],[0],None,[256],[0,256])
        return cl1
        # plt.subplot(2,2,3), plt.title('cl1'), plt.imshow(cl1, cmap='gray')
        # plt.subplot(2,2,4), plt.plot(hist_handle)
        # plt.show()

def sharpenImg(src):
        kernel_sharpen_1 = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])
        src = cv.filter2D(src,-1,kernel_sharpen_1)
        return src

img = cv.imread('packAirport/image/'+'001.jpg')

print (len(img.shape), img.ndim)

img1 = sharpenImg(img)
img1 = claneH(img1)
print (len(img1.shape), img1.ndim)
img2 = claneH(img)
img2 = sharpenImg(img2)
# 好吧，看来第二个明显有点崩了

plt.subplot(2,2,1),plt.imshow(img1)
plt.subplot(2,2,2),plt.imshow(img2)
plt.show()
