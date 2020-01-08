
################################################################################
# 
#       2019.7.17
#       -----------
#       https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md
#       
################################################################################

'''
from imageai.Detection import VideoObjectDetection
import os

# execution_path = os.getcwd()
execution_path = "D:/auxiliaryPlane/project/Python/packAirport/"
print (execution_path)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(suitcase=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "video2019-3-14/video0.mov"),
                                output_file_path=os.path.join(execution_path, "detected")
                                , frames_per_second=20, log_progress=True)

print(video_path)
'''

################################################################################
#   
#   imageAI自动提取
#   https://www.jianshu.com/p/94d5edfaddd5
# 
################################################################################

from imageai.Detection import ObjectDetection
import os

# execution_path = os.getcwd()
execution_path = "D:/auxiliaryPlane/project/Python/packAirport/"

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# detections, objects_path = detector.detectObjectsFromImage(
#                     input_image=os.path.join(execution_path , "documents/video2019-3-14/video0/554.jpg"), 
#                     output_image_path=os.path.join(execution_path , "imageAinew.jpg"), 
#                     extract_detected_objects=True)

custom_objects = detector.CustomObjects(suitcase=True)

detections, objects_path = detector.detectCustomObjectsFromImage(
    custom_objects=custom_objects,
    input_image=os.path.join(execution_path , "documents/video2019-3-14/video0/554.jpg"),
    output_image_path=os.path.join(execution_path , "imageAinew.jpg"),
    minimum_percentage_probability=30)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(str(eachObject["name"]) + " : " + str(eachObject["percentage_probability"]) )

# 用这种封装好的机器学习识别时间也太长了
