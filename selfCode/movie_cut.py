
########################################################################################
# 
#       2019.7.17
#       使用moviepy进行视频剪切
#       https://www.jianshu.com/p/99bf9aad1624
#       http://zulko.github.io/moviepy/
#       https://blog.csdn.net/oJinShengTan/article/details/81080607
#       https://blog.csdn.net/HelloWorld_SDK/article/details/88088513
# 
########################################################################################



from moviepy.editor import *

path1 = "D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/video0.mov"
path2 = "D:/auxiliaryPlane/project/Python/packAirport/documents"

# 读取视频到内存
vfc = VideoFileClip(path1)   # path为输入视频路径

time_sum = vfc.duration
print( time_sum ) 

# ipython_display(vfc) # embeds a sound

time_1 = (0,0)
time_2 = (9,59.59)     # 可能应该是59.59，待商榷

for num in range(0, 18):
    time_start = (time_1[0]+10*num, time_1[1])
    time_end = (time_2[0]+10*num, time_2[1])
    if time_end[0]*60+time_end[1] >= time_sum:
        if time_start[0]*60+time_start[1] >= time_sum:
            break
        time_end = (int(time_sum/60), int(time_sum%60))
        vfc.subclip(time_start, time_end)
        vfc.write_videofile(path2+"/video0_"+str(num), codec='mpeg4', verbose=False, audio=False)
        print (num, time_start, time_end)
        break
    else :
        vfc.subclip(time_start, time_end)
        vfc.write_videofile(path2+"/video0_"+str(num), codec='mpeg4', verbose=False, audio=False)
        print (num, time_start, time_end)
        continue

# 哦。。。这个东西初始化就要把整个视频读进去，心累啊，还是把视频存成图片吧


########################################################################################
# 
#       2019.7.17
#       将视频分割成图片保存
#       --------------------- 
#       作者：等一杯咖啡 
#       原文：https://blog.csdn.net/bskfnvjtlyzmv867/article/details/79970146 
# 
########################################################################################

# coding=utf-8
'''
import os
import cv2

videos_src_path = "D:/auxiliaryPlane/project/Python/packAirport/video2019-3-14/"
video_formats = [".mp4", ".mov"]
frames_save_path = "D:/auxiliaryPlane/project/Python/packAirport/documents/video2019-3-14/"
width = 1280
height = 720
time_interval = 50


def video2frame(video_src_path, formats, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    videos = os.listdir(video_src_path)

    def filter_format(x, all_formats):
        if x[-4:] in all_formats:
            return True
        else:
            return False

    videos = filter(lambda x: filter_format(x, formats), videos)
    # print (videos)

    for each_video in videos:
        print ("正在读取视频：", each_video)

        each_video_name = each_video[:-4]
        os.mkdir(frame_save_path + each_video_name)
        each_video_save_full_path = os.path.join(frame_save_path, each_video_name) + "/"

        each_video_full_path = os.path.join(video_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        print ("共有 ",capture.get(7)," 帧")
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("读取失败!")

        while(success):
            success, frame = cap.read()
            print ("---> 正在读取第%d帧:" % frame_index, success)

            if frame_index % interval == 0:
                resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
                cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, resize_frame)
                frame_count += 1

            frame_index += 1

    cap.release()


if __name__ == '__main__':
    video2frame(videos_src_path, video_formats, frames_save_path, width, height, time_interval)
'''