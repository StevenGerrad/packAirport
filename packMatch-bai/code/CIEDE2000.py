
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


def bgr_mapping(img_val):
    # 将bgr颜色分成16个区间做映射
    return int(img_val/64)
# 颜色直方图的数值计算
def calc_bgr_hist(im):
    scale=0.2
    width = int(im.size[0]*scale)
    height =int(im.size[1]*scale)
    im = im.resize((width, height), Image.ANTIALIAS)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]
    hist1={}; hist2={}; hist3={};
    # 缩放尺寸减小计算量
    width = im.size[0]
    height = im.size[1]
    for x in range(width):
        for y in range(height):
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

# def calc_bgr_hist(im):
#     scale=0.2
#     width = int(im.size[0]*scale)
#     height =int(im.size[1]*scale)
#     im = im.resize((width, height), Image.ANTIALIAS)
#     pix = im.load()
#     width = im.size[0]
#     height = im.size[1]
#     per_image_Rmean = []
#     per_image_Gmean = []
#     per_image_Bmean = []
#     length=0
#     for x in range(width):
#         for y in range(height):
#             maped_r, maped_g, maped_b ,appra= pix[x, y]
#             if(appra==0):
#                 length+=1;
#                 continue
#             # 计算像素值
#             maped_b = bgr_mapping(maped_b)
#             maped_g = bgr_mapping(maped_g)
#             maped_r = bgr_mapping(maped_r)
#             # 计算像素值
#             per_image_Rmean.append(maped_r)
#             per_image_Gmean.append(maped_g)
#             per_image_Bmean.append(maped_b)
#     R_mean = np.mean(per_image_Rmean)
#     G_mean = np.mean(per_image_Gmean)
#     B_mean = np.mean(per_image_Bmean)
#     # return [B_mean, G_mean, R_mean]

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


def compare_similar_hist(rgb_1, rgb_2):
    # print("rgb1",rgb_1)
    # print("rgb2",rgb_2)

    lab_l,lab_a,lab_b=RGB2Lab(rgb_1)
    color1 = LabColor(lab_l, lab_a, lab_b)

    # color1 = LabColor(lab_l=0.9, lab_a=16.3, lab_b=-2.22)
    print("lab",lab_l,lab_a,lab_b)
    lab_l,lab_a,lab_b=RGB2Lab(rgb_2)
    color2 = LabColor(lab_l, lab_a, lab_b)
    print("lab",lab_l,lab_a,lab_b)
    # delta_e = delta_e_cie2000(color1, color2)
    delta_e = delta_e_cmc(color1, color2)
    return delta_e

# 读取图片内容


path='packAirport/packMatch-bai/test img-B/training/'
filelist=os.listdir(path)
length=0
list_len=len(filelist)
img1_path="packAirport/packMatch-bai/test img-B/test/1.png"
img1 = cv2.imread(img1_path)
im1 = Image.open(img1_path)
for i in filelist:
    length+=1
    img2_path=path+i 
    img2 = cv2.imread(img2_path)
    im2 = Image.open(img2_path)
    ans=compare_similar_hist(calc_bgr_hist(im1), calc_bgr_hist(im2))
    ax=plt.subplot(int(1+list_len)/2,2,length);
    ax.set_title(i+": "+format(ans,'.4f'));
    img5 = np.hstack((img1, img2)) #水平拼接
    plt.imshow(img5);

plt.show();
plt.close('all') 
