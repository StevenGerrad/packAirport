
################################################################################################
# 
#       2019.7.25
#       图像锐化
#       ------------
#       https://blog.csdn.net/Miracle0_0/article/details/82051497
# 
################################################################################################

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

#加载图像
image = cv2.imread('packAirport/image/001.jpg')
#图像锐化卷积核
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
kernel_sharpen_4 = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]])
#卷积
output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
output_4 = cv2.filter2D(image,-1,kernel_sharpen_4)

# plt.subplot(2,3,1); plt.title('oriange'); plt.imshow(image)
plt.subplot(2,2,1); plt.title('sharpen_1'); plt.imshow(output_1)
plt.subplot(2,2,2); plt.title('sharpen_2'); plt.imshow(output_2)
plt.subplot(2,2,3); plt.title('sharpen_3'); plt.imshow(output_3)
plt.subplot(2,2,4); plt.title('sharpen_4'); plt.imshow(output_4)

plt.show()

'''

################################################################################################
# 
#       2019.7.25
#       图像边缘检测
#       ------------
#       https://blog.csdn.net/zouxy09/article/details/49080029
# 
################################################################################################

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

#加载图像
image = cv2.imread('packAirport/image/001.jpg')
#图像锐化卷积核
kernel_edge_1 = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [-1,-1,2,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],])
kernel_edge_2 = np.array([
        [0,0,-1,0,0],
        [0,0,-1,0,0],
        [0,0,4,0,0],
        [0,0,-1,0,0],
        [0,0,-1,0,0],])
kernel_edge_3 = np.array([
        [-1,0,0,0,0],
        [0,-2,0,0,0],
        [0,0,6,0,0],
        [0,0,0,-2,0],
        [0,0,0,0,-1],])
kernel_edge_4 = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]])
#卷积
output_1 = cv2.filter2D(image,-1,kernel_edge_1)
output_2 = cv2.filter2D(image,-1,kernel_edge_2)
output_3 = cv2.filter2D(image,-1,kernel_edge_3)
output_4 = cv2.filter2D(image,-1,kernel_edge_4)

# plt.subplot(2,3,1); plt.title('oriange'); plt.imshow(image)
plt.subplot(2,2,1); plt.title('edge_dect_1'); plt.imshow(output_1)
plt.subplot(2,2,2); plt.title('edge_dect_2'); plt.imshow(output_2)
plt.subplot(2,2,3); plt.title('edge_dect_3'); plt.imshow(output_3)
plt.subplot(2,2,4); plt.title('edge_dect_4'); plt.imshow(output_4)

plt.show()
'''


################################################################################################
# 
#       2019.7.26
#       图像边缘检测
#       ------------
#       skimage包——scikit-image SciKit (toolkit for SciPy) :: God::skimage图像处理速度太慢了
#       https://www.cnblogs.com/denny402/p/5125253.html
#       sobel算子、roberts算子、scharr算子、prewitt算子、canny算子、gabor滤波
# 
################################################################################################


# sobel算子、roberts算子、scharr算子、prewitt算子、canny算子
# sobel算子可用来检测边缘
'''

from skimage import data,filters,io,feature
import cv2
import matplotlib.pyplot as plt

img = io.imread('packAirport/image/001.jpg')
# img = data.camera()
grayImg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges1 = filters.sobel(grayImg)
edges2 = filters.roberts(grayImg)
edges3 = filters.scharr(grayImg)
edges4 = filters.prewitt(grayImg)

edges5 = feature.canny(grayImg)   #sigma=1
edges6 = feature.canny(grayImg,sigma=3)   #sigma=3

# plt.subplot(2,4,1), plt.title('oriange'),plt.imshow(img, plt.cm.gray)
# plt.subplot(2,4,2), plt.title('grayImg'),plt.imshow(grayImg, plt.cm.gray)
plt.subplot(2,2,1), plt.title('sobel'),plt.imshow(edges1, plt.cm.gray)
plt.subplot(2,2,2), plt.title('roberts'),plt.imshow(edges2, plt.cm.gray)
plt.subplot(2,2,3), plt.title('scharr'),plt.imshow(edges3, plt.cm.gray)
plt.subplot(2,2,4), plt.title('prewitt'),plt.imshow(edges4, plt.cm.gray)

# plt.subplot(2,4,7), plt.title('canny'),plt.imshow(edges5, plt.cm.gray)
# plt.subplot(2,4,8), plt.title('canny3'),plt.imshow(edges6, plt.cm.gray)

plt.show()

# gabor滤波，这个效果真的不怎么样
filt_real1, filt_imag1 = filters.gabor(grayImg,frequency=0.6)
filt_real2, filt_imag2 = filters.gabor(grayImg,frequency=0.1)

plt.subplot(2,2,1),plt.title('filt_real f:0.6'),plt.imshow(filt_real1,plt.cm.gray)
plt.subplot(2,2,2),plt.title('filt-imag f:0.6'),plt.imshow(filt_imag1,plt.cm.gray)
plt.subplot(2,2,3),plt.title('filt_real f:0.1'),plt.imshow(filt_real2,plt.cm.gray)
plt.subplot(2,2,4),plt.title('filt-imag f:0.1'),plt.imshow(filt_imag2,plt.cm.gray)

plt.show()
'''


################################################################################################
# 
#       2019.7.26
#       图像轮廓检测
#       ------------
#       skimage包——scikit-image SciKit (toolkit for SciPy)
#       https://www.cnblogs.com/denny402/p/5160955.html
# 
################################################################################################

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure,draw,io

# 生成二值测试图像
img = io.imread('packAirport/image/001.jpg')
grayImg= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 检测所有图形的轮廓
contours = measure.find_contours(grayImg, 0.5)

# 绘制轮廓
fig, (ax0,ax1) = plt.subplots(1,2,figsize=(8,8))
ax0.imshow(grayImg,plt.cm.gray)
grayImg[:,:] = 0
ax1.imshow(grayImg,plt.cm.gray)
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=10)
ax1.axis('image')
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()

'''

################################################################################################
# 
#       2019.7.26
#       图像边缘检测
#       ------------
#       https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
#       https://segmentfault.com/a/1190000015662096
# 
################################################################################################

# Canny Edge Detection

'''
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/001.jpg',0)
edges = cv.Canny(img,100,200)
edges2 = cv.Canny(img,50,250)
edges3 = cv.Canny(img,0,255)

plt.subplot(221),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image2'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(edges3,cmap = 'gray')
plt.title('Edge Image3'), plt.xticks([]), plt.yticks([])
plt.show()
'''

########## ++++++++++ ########## Image Pyramids ########## ++++++++++ ##########

# 观察一下行李图像缩小后的效果
'''
import cv2 as cv
import numpy as np,sys
from matplotlib import pyplot as plt

img = cv.imread('packAirport/image/packStudy/001_01.jpg',0)

G = img.copy()
gpA = [G]

G2 = cv.pyrDown(G)
G3 = cv.pyrDown(G2)
G4 = cv.pyrDown(G3)
G5 = cv.pyrDown(G4)
G6 = cv.pyrDown(G5)

plt.subplot(2,3,1),plt.title('oriange'),plt.imshow(G)
plt.subplot(2,3,2),plt.title('Pyramids2'),plt.imshow(G2)
plt.subplot(2,3,3),plt.title('Pyramids3'),plt.imshow(G3)
plt.subplot(2,3,4),plt.title('Pyramids4'),plt.imshow(G4)
plt.subplot(2,3,5),plt.title('Pyramids5'),plt.imshow(G5)
plt.subplot(2,3,6),plt.title('Pyramids6'),plt.imshow(G6)

plt.show()
'''


################################################################################################
# 
#       2019.7.26
#       Image Gradients
#       ------------
#       https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
# 
################################################################################################

'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('packAirport/image/packStudy/001_01.jpg',0)
img = cv.imread('packAirport/image/001.jpg',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

'''