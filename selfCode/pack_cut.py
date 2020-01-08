
###################################################################################
#
#       2019.7.30
#       机场分拣行李分割
#       行李箱整体分割
#       行李箱三个(两个面分割)
#       -----------
#       https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
#       分水岭
#       
###################################################################################

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

img = cv.imread('packAirport/image/'+img_path1)

# 加上图像锐化
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
img = cv.filter2D(img,-1,kernel_sharpen_1)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
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

###################################################################################
# 
#       2019.8.1
#       图像分割尝试
#       ----------
#       基于图的图像分割Effective graph-based image segmentation
#       https://blog.csdn.net/u014796085/article/details/83449972
# 
###################################################################################

# 得，这老外整的程序我根本就跑都跑不起来



###################################################################################
# 
#       2019.8.1
#       
#       ----------
#       教程 | OpenCV Grabcut对象分割
#       https://cloud.tencent.com/developer/article/1400019
#       图像亮度优化 https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
#       灰度图转化为rgb图像 https://blog.csdn.net/llh_1178/article/details/77833447
# 
###################################################################################

'''
import numpy as np
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

def gray2rgb(src, src_gray):
        # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
        B = src[:,:,0]
        G = src[:,:,1]
        R = src[:,:,2]
        # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
        g = src_gray[:]
        p = 0.2989; q = 0.5870; t = 0.1140
        B_new = (g-p*R-q*G)/t
        B_new = np.uint8(B_new)
        src_new = np.zeros((src.shape)).astype("uint8")
        src_new[:,:,0] = B_new
        src_new[:,:,1] = G
        src_new[:,:,2] = R
        # 显示图像
        return  src_new
        # cv2.imshow("input", src)
        # cv2.imshow("output", src_gray)
        # cv2.imshow("result", src_new)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

src = cv.imread('packAirport/image/'+'001.jpg')

cl1 = claneH(src)
# src = cl1
# src = gray2rgb(src, cl1)                # 这个效果，tm就像衣服掉色了一样

# cl1 = sharpenImg(cl1)         # 这个效果，图像似乎更粗糙了啊
src = cv.cvtColor(cl1, cv.COLOR_GRAY2BGR)

cv.imshow("input", src)

mask = np.zeros(src.shape[:2], dtype=np.uint8)
rect = (53,12,356,622)
bgdmodel = np.zeros((1,65),np.float64)
fgdmodel = np.zeros((1,65),np.float64)

cv.grabCut(src,mask,rect,bgdmodel,fgdmodel,5,mode=cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
print(mask2.shape)
result = cv.bitwise_and(src,src,mask=mask2)

cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()

'''

###############################################################################################
# 
#       2019.8.2
#       Wiener filter for out-of-focus image in Python
#       ---------
#       openCV 官网的 https://docs.opencv.org/master/de/d3c/tutorial_out_of_focus_deblur_filter.html
#       没有python版本
#       out-of-focus image
#       https://stackoverflow.com/questions/53782069/wiener-filter-for-out-of-focus-image-in-python
# 
###############################################################################################

# 老外的代码还是有点看不懂

'''
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import cv2
from skimage import restoration

kernel_size = 5
restoration_parameter = 1

# Read in images: out-of-focus, in-focus, and experimental point spread function
# img = cv2.imread('pictures/out_of_focus.jpg')
img = cv2.imread('packAirport/image/'+'001.jpg')

blur = img[:,:,0]
# img2 = cv2.imread('pictures/in_focus.jpg')
# clean = img2[:,:,0]
# img = cv2.imread('pictures/PSF.jpg')
# psf1 = img[:,:,0]

psf2 = np.ones((kernel_size, kernel_size)) / kernel_size**2  # A square kernal
psf = psf2  # set psf to be either psf1 (experimental point spread function) or psf2 (square kernal)
deconvolved_img = restoration.wiener(blur, psf, restoration_parameter, clip=False)

fig = plt.figure()
ax = plt.subplot(111)
new_image = ax.imshow(deconvolved_img)
plt.gray()
plt.show()

def update(kernel_size, restoration_parameter):
    psf2 = np.ones((kernel_size, kernel_size)) / kernel_size**2
    psf = psf2  # set psf to be either psf1
    deconvolved_img = restoration.wiener(blur, psf, restoration_parameter, clip=False)
    new_image.set_data(deconvolved_img)
    ax.set_title(r'kernel size = %2.0f, restoration parameter =%2.5f' % (kernel_size, restoration_parameter))
    return

widgets.interact(update, restoration_parameter=widgets.FloatSlider(min=0,max=100,step=0.1,value=epsilon,description=r'Res. Par.'),
                kernel_size=widgets.IntSlider(min=0,max=40,step=1,value=kernel_size,description=r'kernel size'))

# 后面这个交互式的好像也是跑不起来啊orz

'''


###############################################################################################
# 
#       2019.8.2
#       ---------
#       OpenCV卡尔曼滤波介绍与代码演示
#       https://blog.csdn.net/mangobar/article/details/86690578
# 
###############################################################################################

# 妈的这东西是图像处理吗


###############################################################################################
# 
#       2019.8.3
#       grabCut 再探索
#       ---------
#       Interactive Foreground Extraction using GrabCut Algorithm
#       https://www.cnblogs.com/zyly/p/9392881.html
# 
###############################################################################################

# -*- coding: utf-8 -*-

'''
基于图论的分割方法-GraphCut
【图像处理】图像分割之（一~四）GraphCut，GrabCut函数使用和源码解读（OpenCV）
https://blog.csdn.net/kyjl888/article/details/78253829
'''
'''
import numpy as np
import cv2
     
#鼠标事件的回调函数
def on_mouse(event,x,y,flag,param):        
    global rect
    global leftButtonDowm
    global leftButtonUp
    
    #鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDowm = True
        leftButtonUp = False
        
    #移动鼠标事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDowm and  not leftButtonUp:
            rect[2] = x
            rect[3] = y        
  
    #鼠标左键松开
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDowm and  not leftButtonUp:
            x_min = min(rect[0],rect[2])
            y_min = min(rect[1],rect[3])
            
            x_max = max(rect[0],rect[2])
            y_max = max(rect[1],rect[3])
            
            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDowm = False      
            leftButtonUp = True

#读入图片
img = cv2.imread('packAirport/image/'+'001.jpg')
#掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割的时候，也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函数；在处理结束之后，mask中会保存结果
mask = np.zeros(img.shape[:2],np.uint8)

#背景模型，如果为None，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
bgdModel = np.zeros((1,65),np.float64)
#fgdModel——前景模型，如果为None，函数内部会自动创建一个fgdModel；fgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
fgdModel = np.zeros((1,65),np.float64)

#用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
rect = [0,0,0,0]  
    
#鼠标左键按下
leftButtonDowm = False
#鼠标左键松开
leftButtonUp = True
    
#指定窗口名来创建窗口
cv2.namedWindow('img') 
#设置鼠标事件回调函数 来获取鼠标输入
cv2.setMouseCallback('img',on_mouse)

#显示图片
cv2.imshow('img',img)


while cv2.waitKey(2) == -1:
    #左键按下，画矩阵
    if leftButtonDowm and not leftButtonUp:  
        img_copy = img.copy()
        #在img图像上，绘制矩形  线条颜色为green 线宽为2
        cv2.rectangle(img_copy,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)  
        #显示图片
        cv2.imshow('img',img_copy)
        
    #左键松开，矩形画好 
    elif not leftButtonDowm and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        #转换为宽度高度
        rect[2] = rect[2]-rect[0]
        rect[3] = rect[3]-rect[1]
        rect_copy = tuple(rect.copy())   
        rect = [0,0,0,0]
        #物体分割
        cv2.grabCut(img,mask,rect_copy,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img_show = img*mask2[:,:,np.newaxis]
        #显示图片分割后结果
        cv2.imshow('grabcut',img_show)
        #显示原图
        cv2.imshow('img',img)    

cv2.waitKey(0)
cv2.destroyAllWindows()

'''



###############################################################################################
# 
#       2019.8.3
#       grabCut 再探索
#       ---------
#       运用GrabCut轻松玩转抠图（python实现）
#       https://www.jianshu.com/p/11b5dc8f0242
# 
###############################################################################################

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('packAirport/image/'+'001.jpg')
OLD_IMG = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
plt.subplot(2,2,1), plt.imshow(mask)

SIZE = (1, 65)
bgdModle = np.zeros(SIZE, np.float64)
fgdModle = np.zeros(SIZE, np.float64)
rect = (1, 1, img.shape[1], img.shape[0])
cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img *= mask2[:, :, np.newaxis]

plt.subplot(2,2,3), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])

plt.show()

'''


###############################################################################################
# 
#       2019.8.4
#       grabCut
#       ---------
#       Python 使用Opencv的GrabCut 算法实现前景检测以及分水岭算法实现图像分割
#       https://blog.csdn.net/HuangZhang_123/article/details/80535269
# 
###############################################################################################

'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('packAirport/image/pCut2/video0/'+'0-001.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])

# zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (1, 1, img.shape[1], img.shape[0])
# 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
# img - 输入图像
# mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
# rect - 包含前景的矩形，格式为 (x,y,w,h)
# bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
# iterCount - 算法的迭代次数
# mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# np.where 函数是三元表达式 x if condition else y的矢量化版本
# result = np.where(cond,xarr,yarr)
# 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
# | 是逻辑运算符or的另一种表现形式
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# mask2[:, :, np.newaxis] 增加维度
img = img * mask2[:, :, np.newaxis]

# 显示图片
plt.subplot(121), plt.imshow(img)
plt.title("grabcut"), plt.xticks([]), plt.yticks([])

plt.show()

'''

###############################################################################################
# 
#       2019.8.4
#       grabCut
#       ---------
#       opencv python交互式grabCut
#       https://blog.csdn.net/qq_41244435/article/details/86677495
# 
###############################################################################################

# 这个程序跑不起来噻

