
################################################################################
# 
#       2019.7.11
#       Video analysis (video module)
#       -----------
#       https://docs.opencv.org/master/da/dd0/tutorial_table_of_content_video.html
#       
################################################################################

# 似乎应该调一下参数，直接跑没有什么反应
'''
from __future__ import print_function
import cv2 as cv
import argparse

# localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/video0.mov'
# 此处实际上不太清楚用windows自带的照片软件将其截取为.mp4会不会失真
localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'

capture = cv.VideoCapture(localin)
print (capture.get(7))
cnt = 1

while True:
    if not capture.isOpened:
        print('Unable to open: ' + localin)
        exit(0)

    ret, frame = capture.read()
    if frame is None:
        break
    print ("ptr num of frame: "+str(cnt))
    cnt = cnt + 1
    # if cnt < int(capture.get(7)/4):
    #     continue
    backSub = cv.createBackgroundSubtractorMOG2()
    # backSub = cv.createBackgroundSubtractorKNN()
    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    cv.namedWindow("Frame",cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.namedWindow("FG Mask",cv.WINDOW_NORMAL)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(2)

    if keyboard == 'q' or keyboard == 27:
        break
'''

################################################################################
# 
#       2019.7.21
#       基于opencv的BackgroundSubtractorMOG2目标追踪
#       -----------
#       https://blog.csdn.net/zhangyonggang886/article/details/51638655
# 
################################################################################

'''

import numpy as np
import cv2
import time
import datetime

localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'
cap = cv2.VideoCapture(localin)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame1 = np.zeros((640,480))
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p")+'.avi',fourcc, 5.0, np.shape(frame1))

while(1):
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    (_,cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    for c in cnts:
        Area = cv2.contourArea(c)
        if Area < maxArea :
        #if cv2.contourArea(c) < 500:
            (x, y, w, h) = (0,0,0,0)
            continue
        else:
            if Area < 1000:
                (x, y, w, h) = (0,0,0,0)
                continue
            else:
                maxArea = Area
                m=c
                (x, y, w, h) = cv2.boundingRect(m)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(fgmask)
    
    cv2.namedWindow("fgmask",cv2.WINDOW_NORMAL)
    cv2.imshow('fgmask',fgmask)
    cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30)&0xff
    if k==27:
        break
out.release()
cap.release()
cv2.destoryAllWindows()

# 随着左下角的绿灯的照亮背景整体的亮度渲染会被突然提升，就会影响部分时间段的效果

'''


################################################################################
# 
#       2019.7.21
#       Meanshift and Camshift
#       -----------
#       https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html
# 
################################################################################

########## ---------- ########## Meanshift ########## ---------- ##########

'''
import numpy as np
import cv2 as cv
import argparse

localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'
cap = cv.VideoCapture(localin)
print ("the Width : ",cap.get(3)," and the Height : ",cap.get(4))
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
x, y, w, h = 1340, 750, 576, 324 # simply hardcoded the values
# 这个东西，箱子一出来就会到处跑（行李箱是一点点出来的），而且大小不能确定，位置也不容易设置
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        cv.namedWindow("img2",cv.WINDOW_NORMAL)
        cv.imshow('img2',img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

'''


########## ---------- ########## Camshift ########## ---------- ##########

'''
import numpy as np
import cv2 as cv
import argparse

localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'
cap = cv.VideoCapture(localin)
print ("the Width : ",cap.get(3)," and the Height : ",cap.get(4))

# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
x, y, w, h = 1340, 750, 576, 324 # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)

        cv.namedWindow("img2",cv.WINDOW_NORMAL)
        cv.imshow('img2',img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

# ||Φ|(|T|Д|T|)|Φ|| 我佛了，这个的实际效果虽然比上一个好一点但仍然很迷啊
# 由于履带部分磨损造成的图像细微的位移也会产生影响
# 对于video0行李必然是从右下角移动而出，因而其只会对几乎一秒内冲入方框界面的部分进行跟踪

'''



################################################################################
# 
#       2019.7.21
#       Optical Flow 光流法!! wryyyyyyyyyy 木大木大木大木大木大 卡兹-卡~兹
#       -----------
#       https://docs.opencv.org/master/d4/dee/tutorial_optical_flow.html
# 
################################################################################


#################### Lucas-Kanade Optical Flow in OpenCV ####################

'''
import numpy as np
import cv2 as cv
import argparse

localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'
cap = cv.VideoCapture(localin)
print ("the Width : ",cap.get(3)," and the Height : ",cap.get(4))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)

    cv.namedWindow("frame",cv.WINDOW_NORMAL)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

'''

############################### Dense Optical Flow in OpenCV ###############################

import numpy as np
import cv2 as cv

localin = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack1.mp4'
localin1 = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack3.mp4'
localin2 = 'D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/subvideo/video0_pack12.mp4'
cap = cv.VideoCapture(localin1)
print ("the Width : ",cap.get(3)," and the Height : ",cap.get(4))

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.namedWindow("frame2",cv.WINDOW_NORMAL)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next