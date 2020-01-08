# sift
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os.path import splitext, join
import shutil
imgname1="packAirport/packMatch-bai/test img-C/test/1.png"
path='packAirport/packMatch-bai/test img-C/training/'
# os.chdir(path)
filelist=os.listdir(path)
length=0
list_len=len(filelist)
for i in filelist:
    imgname2=path+i 
    length+=1;
    print(i,end=": ")
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.imread(imgname1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
    kp1, des1 = sift.detectAndCompute(img1,None)   #des是描述子

    img2 = cv2.imread(imgname2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
    kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子

    # print("img1 key points:",len(kp1));
    # print("img2 key points:",len(kp2));

    img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
    img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈
    hmerge = np.hstack((img3, img4)) #水平拼接

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # print(matches);
    # print("matches:",len(matches),end=',');

    # 调整ratio
    good = []
    tmp = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
            tmp.append([m])

    # print("good:",len(good));
    # print("match ratio",format(len(good)/len(matches),'.4f'));
    if len(good) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
        ans=[]
        for j in range(len(status)):
            if status[j]!=0:
                ans.append(tmp[j])
        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,ans,None,flags=2)
        ax=plt.subplot(int(list_len+1)/2,2,length);
        ax.set_title(i+": "+format(len(ans)/len(matches),'.4f'));
        plt.imshow(img5);
    else:
        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,tmp,None,flags=2)
        ax=plt.subplot(int(list_len+1)/2,2,length);
        ax.set_title(i+": "+format(len(tmp)/len(matches),'.4f'));
        plt.imshow(img5);

plt.show();
plt.close('all') 
