
################################################################################
# 
#       2019.7.11
#       分拣图像直方图匹配....这个不是匹配是整合
#       -----------
#       https://www.jianshu.com/p/3f6abf3eeba2
#       
################################################################################

'''
# 这tm应该叫做直方图整合，不应该叫直方图匹配    

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib


#将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray,nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist

#计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray,h_s,nums):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp

    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    des = np.zeros((w,h),dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            des[i][j] = int((nums - 1)* h_acc[grayArray[i][j] ] +0.5)
    return des

#传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist,name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1#x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    #plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))#绘制直方图
    #plt.show()

#直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des
'''
#  =========================================== 我是分割线 =========================================== #
'''
imdir = 'D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg'
imdir_match = 'D:/auxiliaryPlane/project/Python/packAirport/image/101.jpg'

#直方图匹配
#打开文件并灰度化
im_s = Image.open(imdir).convert("L")
im_s = np.array(im_s)
print(np.shape(im_s))
#打开文件并灰度化
im_match = Image.open(imdir_match).convert("L")
im_match = np.array(im_match)
print(np.shape(im_match))
#开始绘图
plt.figure()

#原始图和直方图
plt.subplot(2,3,1)
plt.title("original")
plt.imshow(im_s,cmap='gray')

plt.subplot(2,3,4)
hist_s = arrayToHist(im_s,256)
drawHist(hist_s,"original")

#match图和其直方图
plt.subplot(2,3,2)
plt.title("match Photo")
plt.imshow(im_match,cmap='gray')

plt.subplot(2,3,5)
hist_m = arrayToHist(im_match,256)
drawHist(hist_m,"match Photo")

#match后的图片及其直方图
im_d = histMatch(im_s,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
plt.subplot(2,3,3)
plt.title("matched")
plt.imshow(im_d,cmap='gray')

plt.subplot(2,3,6)
hist_d = arrayToHist(im_d,256)
drawHist(hist_d,"matched")

plt.show()

'''


################################################################################
# 
#           2019.7.25
#           图像直方图研究
#       -----------------------
#       https://blog.csdn.net/m0_38007695/article/details/82718107
#       灰度直方图
#       Matplotlib本身提供了计算直方图的函数hist
#       线性变换
#       分段线性变换(*)
#       直方图正规化
#       正规化函数normalize
#       伽马变换
#       全局直方图均衡化(*)
#       限制对比度的自适应直方图均衡化(*)
#       
################################################################################


################# 灰度直方图 #################

# 灰度直方图是图像灰度级的函数，用来描述每个灰度级在图像矩阵中的像素个数或者占有率

'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist
 
img = cv.imread("packAirport/image/001.jpg", 0)
grayHist = calcGrayHist(img)
x = np.arange(256)
# 绘制灰度直方图
plt.title('image gray histogram')
plt.plot(x, grayHist, 'r', linewidth=2, c='black')
plt.xlabel("gray Label")
plt.ylabel("number of pixels")
plt.show()

# cv.imshow("img", img)
# cv.waitKey()

'''


################# Matplotlib本身也提供了计算直方图的函数hist #################

'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("packAirport/image/001.jpg", 0)
h, w = img.shape[:2]
pixelSequence = img.reshape([h * w, ])
numberBins = 256
histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                  facecolor='black', histtype='bar')
plt.xlabel("gray label")
plt.ylabel("number of pixels")
plt.axis([0, 255, 0, np.max(histogram)])
plt.show()
cv.imshow("img", img)
cv.waitKey()
'''


########### ========== ########### 线性变换 ########### ========== ###########

# 不应该用线性变换，本身就有一部分是比较亮的图，这样图像反而不清晰

'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
# 绘制直方图函数
def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
 
img = cv.imread("packAirport/image/001.jpg", 0)
out = 2.0 * img
# 进行数据截断，大于255的值截断为255
out[out > 255] = 255
# 数据类型转换
out = np.around(out)
out = out.astype(np.uint8)
# 分别绘制处理前后的直方图
# grayHist(img)
# grayHist(out)

plt.subplot(1,2,1), plt.title('img'), plt.imshow(img, cmap='gray')
plt.subplot(1,2,2), plt.title('out'), plt.imshow(out, cmap='gray')
plt.show()
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
'''


########### ========== ########### 分段线性变换 ########### ========== ###########


########### ========== ########### 直方图正规化 ########### ========== ###########

'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("packAirport/image/001.jpg", 0)
# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
Imin, Imax = cv.minMaxLoc(img)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)

hist_img = cv.calcHist([img],[0],None,[256],[0,256])
hist_out = cv.calcHist([out],[0],None,[256],[0,256])

plt.subplot(2,2,1), plt.title('img'), plt.imshow(img, cmap='gray')
plt.subplot(2,2,2), plt.plot(hist_img)
plt.subplot(2,2,3), plt.title('out'), plt.imshow(out, cmap='gray')
plt.subplot(2,2,4), plt.plot(hist_out)
plt.show()
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
'''

########### ========== ########### 正规化函数normalize ########### ========== ###########
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
    
img = cv.imread("packAirport/image/001.jpg", 0)
out = np.zeros(img.shape, np.uint8)
cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
cv.imshow("img", img)
cv.imshow("out", out)
grayHist(img)
grayHist(out)
cv.waitKey()

'''

########### ========== ########### 伽马变换,增加/降低图像对比度 ########### ========== ###########

# 这个变换在直方图上观察不是很显著啊

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()

img = cv.imread("packAirport/image/001.jpg", 0)
# 图像归一化
fi = img / 255.0
# 伽马变换
gamma = 1.2
out = np.power(fi, gamma)
cv.imshow("img", img)
cv.imshow("out", out)
grayHist(out)
cv.waitKey()


########### ========== ########### 全局直方图均衡化 ########### ========== ###########

########### ========== 限制对比度的自适应直方图均衡化 ========== ###########