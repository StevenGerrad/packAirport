# sift
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os.path import splitext, join
from math import sqrt
import shutil

# 颜色映射
def bgr_mapping(img_val):
    # 将bgr颜色分成8个区间做映射
    return int(img_val/16)
    # return int(img_val/32)
    # if img_val >= 0 and img_val <= 31: return 0
    # if img_val >= 32 and img_val <= 63: return 1
    # if img_val >= 64 and img_val <= 95: return 2
    # if img_val >= 96 and img_val <= 127: return 3
    # if img_val >= 128 and img_val <= 159: return 4
    # if img_val >= 160 and img_val <= 191: return 5
    # if img_val >= 192 and img_val <= 223: return 6
    # if img_val >= 224: return 7

# 颜色直方图的数值计算
def calc_bgr_hist(image):
    if not image.size: return False
    hist = {}
    # 缩放尺寸减小计算量
    # image = cv2.resize(image, (32, 32))
    image = cv2.resize(image, (64, 64))
    for bgr_list in image:
        for bgr in bgr_list:
            # 颜色按照顺序映射
            maped_b = bgr_mapping(bgr[0])
            maped_g = bgr_mapping(bgr[1])
            maped_r = bgr_mapping(bgr[2])
            # 计算像素值
            index   = maped_b * 16 * 16 + maped_g * 16 + maped_r
            hist[index] = hist.get(index, 0) + 1
    
    return hist

# 计算两张图片的相似度
def compare_similar_hist(h1, h2):
    if not h1 or not h2: return False
    sum1, sum2, sum_mixd = 0, 0, 0
    # 像素值key的最大数不超过512，直接循环到512，遍历取出每个像素值
    for i in range(4096):
    # for i in range(512):
        # 计算出现相同像素值次数的平方和
        sum1 = sum1 + (h1.get(i, 0) * h1.get(i, 0))
        sum2 = sum2 + (h2.get(i, 0) * h2.get(i, 0))
        # 计算两个图片次数乘积的和
        sum_mixd = sum_mixd + (h1.get(i, 0) * h2.get(i, 0))
    # 按照余弦相似性定理计算相似度
    return sum_mixd / (sqrt(sum1) * sqrt(sum2))

# 读取图片内容

imgname1="packAirport/packMatch-bai/test img-B/test/1.png"
path='packAirport/packMatch-bai/test img-B/training/'
# os.chdir(path)
filelist=os.listdir(path)
length=0
list_len=len(filelist)
for i in filelist:
	orb = cv2.ORB_create()
	imgname2=path+i 
	length+=1;
	print(i,end=": ")
	img1 = cv2.imread(imgname1)

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
	kp1, des1 = orb.detectAndCompute(img1,None)#des是描述子

	img2 = cv2.imread(imgname2)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	kp2, des2 = orb.detectAndCompute(img2,None)

	hmerge = np.hstack((gray1, gray2)) #水平拼接

	img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
	img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

	hmerge = np.hstack((img3, img4)) #水平拼接

	# BFMatcher解决匹配
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	# 调整ratio
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append([m])

	img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
	ax=plt.subplot(int(list_len+1)/2,2,length);
	ans1=compare_similar_hist(calc_bgr_hist(img1), calc_bgr_hist(img2))
	ans2=len(good)/len(matches)
	ax.set_title(i+": "+format(0.0*ans1+1*ans2,'.4f'));
	plt.imshow(img5);
plt.show();
plt.close('all') 