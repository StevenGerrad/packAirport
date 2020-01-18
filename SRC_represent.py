
###################################################################################
# 
#       2020.1.18
#       ----------
#       SRC
# 
###################################################################################




# AR 数据集，包括100个人的正面人脸图像，每位个体包含14张无遮挡图像（可用于构建字典），和12张有遮挡图像（可用于测试）。

# 数据共100类，m代表男性样本，w代表女性样本，第一个三位数代表样本类别。最后的两位数代表该类别下的26张图，
# 其中图片id为1-7,14-20可以作为训练集，其余的为测试集（戴眼镜和蒙面） 

# 数据命名格式为：性别-个体id-图片id，例如m-001-01，表示第一个个体的第一张人脸图片，性别为男性。



import csv
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2

class dataMaker():
    def __init__(self, file_dir, name='AR', ):
        if name == 'AR':
            self.AR_dataSet(file_dir)
    
    def AR_dataSet(self, file_dir):
        ''' 提取文件夹下的地址+文件名，源文件设定排序规则 '''
        train_file = []
        test_file = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                f_name = file.split('-')
                id = f_name[2].split('.')
                id = int(id[0])
                if id <= 7 or (id >= 14 and id <= 20) :
                    train_file.append(os.path.join(root, file))
                else:
                    test_file.append(os.path.join(root, file))
        
        train_data = []
        test_data = []
        for i in train_file:
            img = Image.open(i)
            cv2.pyrDown()
            train_data.append(np.array(img))
        for i in test_file:
            img = Image.open(i)

            test_data.append(np.array(img))
        self.train_data = train_data        # 14*100
        self.test_data = test_data          # 12*100


class SRC():
    def __init__(self, train_data, test_data):
        '''
        1.初始化字典
        A. 将训练集图像进行降采样eg.(40.30), 并reshape成一列（120行1列）,并对该列进行归一化。 B. 将训练图像依次处理并排列成字典，
         1.1 其中Ai 是某一个人的特征集合。 1.2 其中"a" _i1是第i个人的第1张图像reshape的那一列（120行1列）。
        '''

    # 2.将测试数据用同样的参数降采样并reshape得到特征向量。并用OMP算法计算该测试数据的稀疏表达x；

    # 3.使用类似one-hot方法对x进行处理。

    # 4.应用字典将处理后的稀疏表达还原，并计算原后的向量和图像原始特征向量的距离

    # 5.对所有类别均用2.3、2.4的方法计算距离。距离最小的类，即为分类结果。


if __name__ == '__main__':
    # 1.检查数据集中的数据特征，确定图片分块大小 (120, 165)
    #   并将无遮挡的人脸作为训练数据，有遮挡的人脸作为测试数据。
    dataset = dataMaker('D:\\MINE_FILE\\dataSet\\AR', 'AR')
    
    # 2.应用SCR算法进行字典构建并对测试集进行基于分块投票的分类；
    src_algorithm = SRC(dataset.train_data, dataset.test_data)

    # 3.统计分类结果与准确率。




