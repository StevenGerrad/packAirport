################################################################################
# 
#       
#       --------------------- 
#       作者：JerrySing 
#       来源：CSDN 
#       原文：https://blog.csdn.net/u010977034/article/details/82733137 
# 
################################################################################
# SSIM（结构相似性度量）
'''
from skimage.measure import compare_ssim
from scipy.misc import imread
import numpy as np
 
img1 = imread("D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg")
img2 = imread("D:/auxiliaryPlane/project/Python/packAirport/image/002.jpg")
 
img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
 
print("------------------",img2.shape)
print("------------------",img1.shape)
ssim = compare_ssim(img1, img2, multichannel=True)
 
print("==================",ssim)

'''

# cosin相似度（余弦相似度）

'''
from PIL import Image
from numpy import average, linalg, dot


def get_thumbnail(image, size=(1200, 750), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res


image1 = Image.open('D:/auxiliaryPlane/project/Python/packAirport/image/001.jpg')
image2 = Image.open('D:/auxiliaryPlane/project/Python/packAirport/image/002.jpg')

cosin0 = image_similarity_vectors_via_numpy(image1, image2)
print("=========0=========",cosin0)

image3 = Image.open('D:/auxiliaryPlane/project/Python/packAirport/image/101.jpg')
cosin1 = image_similarity_vectors_via_numpy(image1, image3)
print("=========1=========",cosin1)

'''


#######################################################################################
# 
#       2019.8.4
#       --------------------- 
#       LineSegmentDetector in Opencv 3 with Python
#       https://www.e-learn.cn/content/wangluowenzhang/961239
# 
#######################################################################################

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read gray image
img1 = cv2.imread('packAirport/image/pCut2/video0/'+'0-001.jpg',0)
img2 = cv2.imread('packAirport/image/pCut2/video1/'+'1-001.jpg',0)

#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image
lines1 = lsd.detect(img1)[0] #Position 0 of the returned tuple are the detected lines
lines2 = lsd.detect(img2)[0]

#Draw detected lines in the image
drawn_img1 = lsd.drawSegments(img1,lines1)
drawn_img2 = lsd.drawSegments(img2,lines2)

#Show image
plt.subplot(1,2,1), plt.imshow(drawn_img1)
plt.subplot(1,2,2), plt.imshow(drawn_img2)
plt.show()

'''

#######################################################################################
# 
#       2019.8.4
#       --------------------- 
# 
#######################################################################################

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('packAirport/image/pCut2/video0/'+'0-001.jpg')
img2 = cv2.imread('packAirport/image/pCut2/video1/'+'1-001.jpg')

img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

LineSegmentDetector = cv2.createLineSegmentDetector()
lines1, _, prec1, _ = LineSegmentDetector.detect(img_gray1)
lines2, _, prec1, _ = LineSegmentDetector.detect(img_gray2)

bf = cv2.BFMatcher()
# matches = bf.knnMatch(prec1,prec1, k=2)

#Draw detected lines in the image
drawn_img1 = LineSegmentDetector.drawSegments(img1,lines1)
drawn_img2 = LineSegmentDetector.drawSegments(img2,lines2)

#Show image
plt.subplot(1,2,1), plt.imshow(drawn_img1)
plt.subplot(1,2,2), plt.imshow(drawn_img2)
plt.show()

'''

#######################################################################################
# 
#       2019.8.4
#       (*)尝试使用line-feature-match
#       --------------------- 
#       https://answers.opencv.org/question/159732/binary-descriptor-and-keyline/
#       https://docs.opencv.org/master/d6/d83/classcv_1_1line__descriptor_1_1BinaryDescriptor.html#ad896d4dfe36452cf01bd578f78d3062a
#       https://docs.opencv.org/master/d1/dbd/classcv_1_1line__descriptor_1_1LSDDetector.html#a8de4c87977c2d3fa54b765df532cbf96
#       似乎是openCV还没有对python的此接口，额。。。。
# 
#######################################################################################







