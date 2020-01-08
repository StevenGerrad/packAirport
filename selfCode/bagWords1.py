
###################################################################################
# 
#       2019.9.5
#       --------------
#       https://github.com/bikz05/bag-of-words
# 
###################################################################################

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
# scipy.cluster是scipy下的一个做聚类的package, 共包含了两类聚类方法: 
# 1. 矢量量化(scipy.cluster.vq):支持vector quantization 和 k-means 聚类方法 
# 2. 层次聚类(scipy.cluster.hierarchy):支持hierarchical clustering 和 agglomerative clustering(凝聚聚类)
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


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

train_path = "D:/auxiliaryPlane/project/Python/packAirport/image/img_test1"
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []
kps_list = []

for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    im = cv2.imread(image_path)
    im = sharpenImg(im)
    im = claneH(im)
    kpts, des = sift.detectAndCompute(im,None)        #des是描述子
    print (kpts[0].pt[0])
    kps_list.append(kpts)
    des_list.append((image_path, des))

# 后续再考虑对特征点查找进行优化
print ( len(des_list) )

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    print (image_path)
    descriptors = np.vstack((descriptors, descriptor))
    # 按垂直方向（行顺序）堆叠数组构成一个新的数组
print ("descriptors", descriptors.shape)

# 2019.9.6
# https://www.cnblogs.com/bincoding/p/8876002.html

# 3.kmeans(obs, k_or_guess, iter=20, thresh=1e-05, check_finite=True) 
# 输入obs是数据矩阵,行代表数据数目,列代表特征维度; k_or_guess表示聚类数目;iter表示循环次数,最终返回损失最小的那一次的聚类中心; 
# 输出有两个,第一个是聚类中心(codebook),第二个是损失distortion,即聚类后各数据点到其聚类中心的距离的加和.

# 4.vq(obs, code_book, check_finite=True) 
# 根据聚类中心将所有数据进行分类.obs为数据,code_book则是kmeans产生的聚类中心. 
# # 输出同样有两个:第一个是各个数据属于哪一类的label,第二个和kmeans的第二个输出是一样的,都是distortion

# Perform k-means clustering
k = 50
voc, variance = kmeans(descriptors, k, 1)       # 他这聚类是聚的.。。。
# 得出的结果是以50个特征点为中心

# plt.scatter(voc[:, 0], voc[:, 1])
# plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
# plt.show()

# Calculate the histogram of features
im_features = np.zeros((len(des_list), k), "float32")
words_list = []
for i in range(0,len(des_list)):
    words, distance = vq(des_list[i][1],voc) 
    words_list.append(words)
    for w in words:
        im_features[i][w] += 1

mark = ['or', 'ob', 'og', 'ok', 'oc', 'om', 'oy', 'ow', 
        'xr', 'xb', 'xg', 'xk', 'xc', 'xm', 'xy', 'xw', 
        'dr', 'db', 'dg', 'dk', 'dc', 'dm', 'dy', 'dw', 
        'sr', 'sb', 'sg', 'sk', 'sc', 'sm', 'sy', 'sw', 
        '*r', '*b', '*g', '*k', '*c', '*m', '*y', '*w', ]


for i in range(0 , len(kps_list)): 
    img1 = cv2.imread(des_list[i][0])
    # plt.subplot(4,6,i*2+1), plt.imshow(cv2.drawKeypoints(img1,kps_list[i],img1,color=(255,0,255)))  
    # plt.subplot(4,6,i*2+2)
    plt.subplot(3,4,i+1)
    plt.imshow(img1)
    ptr = 0
    for j in kps_list[i] :
        plt.plot(j.pt[0], j.pt[1], mark[ptr%10], markersize = 2)
        ptr += 1
plt.show()
# 后续再考虑对特征点查找进行优化


bin_des = [bin_des * 2 for bin_des in range(0,25+1)]
for i in range(0,len(des_list)):
    plt.subplot(3,4,i+1)
    plt.hist(im_features[i], bins=bin_des)
plt.show()

print ( len(des_list) )

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*1 + 1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


'''
# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)    
'''
