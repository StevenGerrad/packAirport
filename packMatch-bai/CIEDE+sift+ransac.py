
import cv2
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import splitext, join
import shutil
from operator import itemgetter
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_diff import delta_e_cmc
from PIL import Image

#区域划分
def bgr_mapping(img_val):
    # 将bgr颜色分成256/64个区间做映射
    return int(img_val/64)

# RGb的加权数值计算
def calc_bgr_hist(im):
	 # 缩放尺寸减小计算量
    scale=0.2
    width = int(im.size[0]*scale)
    height =int(im.size[1]*scale)
    im = im.resize((width, height), Image.ANTIALIAS)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    hist1={}; hist2={}; hist3={};

    width = im.size[0]
    height = im.size[1]
    for x in range(width):
        for y in range(height):
            if(len(pix[x,y])==3):
                maped_r, maped_g, maped_b= pix[x, y]
            else:
                maped_r, maped_g, maped_b ,appra= pix[x, y]
                if(appra==0):
                    continue
            # 计算像素值
            maped_b = bgr_mapping(maped_b)
            maped_g = bgr_mapping(maped_g)
            maped_r = bgr_mapping(maped_r)
            hist1[maped_b] = hist1.get(maped_b, 0) + 1
            hist2[maped_g] = hist2.get(maped_g, 0) + 1
            hist3[maped_r] = hist3.get(maped_r, 0) + 1
    list1=sorted(hist1.items(),key=itemgetter(1),reverse=True);
    list2=sorted(hist2.items(),key=itemgetter(1),reverse=True);
    list3=sorted(hist3.items(),key=itemgetter(1),reverse=True);

    x1,v1=list1[0];
    x2,v2=list1[1];
    x3,v3=list1[2];
    B_mean = (x1*v1+x2*v2+x3*v3)/(1.0*v1+v2+v3)

    x1,v1=list2[0];
    x2,v2=list2[1];
    x3,v3=list2[2];
    G_mean = (x1*v1+x2*v2+x3*v3)/(1.0*v1+v2+v3)

    x1,v1=list3[0];
    x2,v2=list3[1];
    x3,v3=list3[2];
    R_mean = (x1*v1+x2*v2+x3*v3)/(1.0*v1+v2+v3)
    return [B_mean*64,G_mean*64,R_mean*64]

# region 辅助函数
# RGB2XYZ空间的系数矩阵
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


# im_channel取值范围：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion


# region RGB 转 Lab
# 像素值RGB转XYZ空间，pixel格式:(B,G,R)
# 返回XYZ空间下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
    XYZ空间转Lab空间
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab


# endregion

# region Lab 转 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb
# endregion

#CIEDE
def compare_similar_hist(rgb_1, rgb_2):
    # print("rgb1",rgb_1)
    # print("rgb2",rgb_2)

    lab_l,lab_a,lab_b=RGB2Lab(rgb_1)
    color1 = LabColor(lab_l, lab_a, lab_b)

    lab_l,lab_a,lab_b=RGB2Lab(rgb_2)
    color2 = LabColor(lab_l, lab_a, lab_b)
    delta_e = delta_e_cie2000(color1, color2)
    # delta_e = delta_e_cmc(color1, color2)
    return delta_e

def claneH(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist_original = cv2.calcHist([img],[0],None,[256],[0,256])
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    # hist_handle = cv2.calcHist([cl1],[0],None,[256],[0,256])
    return cl1
    # plt.imshow(cl1, cmap='gray'),plt.show()

def sharpenImg(src):
    kernel_sharpen_1 = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1]])
    src = cv2.filter2D(src,-1,kernel_sharpen_1)
    return src

# 读取图片内容
path='packAirport/packMatch-bai/test img/training/'
filelist=os.listdir(path)
length=0
list_len = len(filelist)
img1_path="packAirport/packMatch-bai/test img/test/1.png"
img1 = cv2.imread(img1_path)
img1 = sharpenImg(img1)         # wjychanged
im1 = Image.open(img1_path)

for i in filelist:
    length+=1
    img2_path=path+i
    im2 = Image.open(img2_path)
    # 色差评价函数
    ans1=compare_similar_hist(calc_bgr_hist(im1), calc_bgr_hist(im2))
    
    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
    gray1 = claneH(gray1)       # wjychanged
    kp1, des1 = sift.detectAndCompute(img1,None)   #des是描述子

    img2 = cv2.imread(img2_path)
    img2 = sharpenImg(img2)     # wjychanged
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
    gray2 = claneH(gray2)       # wjychanged
    kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子

    img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
    hmerge = np.hstack((img3, img4)) #水平拼接

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # 调整ratio
    good = []
    tmp = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
            tmp.append([m])
    # ???????????????????
    if len(good) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
        ans=[]
        for j in range(len(status)):
            if status[j]!=0:
                ans.append(tmp[j])
        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,ans,None,flags=2)       # 画出特征匹配点连线
        ans2=len(ans)/len(matches)
    else:
        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,tmp,None,flags=2)
        ans2=len(tmp)/len(matches)
    #准确度
    end_ans=100-2*(ans1*0.7+ans2*0.3)       # 评价机制
    ax=plt.subplot(int(list_len+1)/2,2,length);
    ax.set_title(i+": "+"%.3f%%" % (end_ans));
    plt.axis("off");
    plt.imshow(img5);

plt.show();
plt.close('all') 
