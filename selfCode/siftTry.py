
################################################################################
# 
#       --------------------- 
#       作者：Rachel-Zhang 
#       来源：CSDN 
#       原文：https://blog.csdn.net/abcjennifer/article/details/7639681 
#       版权声明：本文为博主原创文章，转载请附上博文链接！
# 
################################################################################

# https://www.jianshu.com/p/14b92d3fd6f8
# Harris：该算法用于检测角点；
# SIFT：该算法用于检测斑点；
# SURF：该算法用于检测角点；
# FAST：该算法用于检测角点；
# BRIEF：该算法用于检测斑点；
# ORB：该算法代表带方向的FAST算法与具有旋转不变性的BRIEF算法；

'''
import cv2
import numpy as np
#import pdb
#pdb.set_trace()#turn on the pdb prompt
 
#read image
img = cv2.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('origin',img)
 
#SIFT
detector = cv2.SIFT()
keypoints = detector.detect(gray,None)
img = cv2.drawKeypoints(gray,keypoints)
#img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

################################################################################
# 
#       --------------------- 
#       作者：OnePunch-Man 
#       来源：CSDN 
#       原文：https://blog.csdn.net/u013837209/article/details/77411564 
#       版权声明：本文为博主原创文章，转载请附上博文链接！
# 
################################################################################

'''

import cv2
img = cv2.imread('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()

# error 信息，该算法（SIFT）已经获得专利，在此配置中不包括，
# 设置 OPENCV_ENABLE_NONFREE CMake 选项并在函数 ' cv :: xfeatures2d :: SIFT :: create ' 中重建库
# 查到这里我只想说cnmd,nmlgb,nmlglb

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)

'''


################################################################################
# 
#       2019.8.1
#       图像相似度比较-pHash算法（图像感知算法）
#       -----------
#       链接：https://zhuanlan.zhihu.com/p/63180171
#       来源：知乎
# 
################################################################################

'''
import cv2
import numpy as np
#均值哈希算法
def aHash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str
#差值感知算法
def dHash(img):
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

img1=cv2.imread('packAirport/image/001.jpg')
img2=cv2.imread('packAirport/image/002.jpg')
hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print ('均值哈希算法相似度：'+ str(n))

hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：'+ str(n))

'''
