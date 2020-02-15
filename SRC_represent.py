
###################################################################################
# 
#       2020.1.18
#       ----------
#       SRC
# 
###################################################################################


# AR 数据集，包括100个人的正面人脸图像，每位个体包含14张无遮挡图像（可用于构建字典），和12张有遮挡图像（可用于测试）。

# 数据共100类，m代表男性样本，w代表女性样本，第一个三位数代表样本类别。最后的两位数代表该类别下的26张图，
# 其中图片id为1-7,14-20可以作为训练集(14)，其余的为测试集（戴眼镜和蒙面） (12)

# 数据命名格式为：性别-个体id-图片id，例如m-001-01，表示第一个个体的第一张人脸图片，性别为男性。

import csv
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

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
        print('prepare file name...',end=' ')
        for i in train_file:
            img = Image.open(i)
            train_data.append(np.array(img))
        print('read in train data...',end=' ')
        for i in test_file:
            img = Image.open(i)
            test_data.append(np.array(img))
        print('read in test data...', end='\n')
        
        self.train_data = train_data        # 14*100
        self.test_data = test_data          # 12*100


class SRC():
    def __init__(self, train_data, test_data, max_iter=100, tol=1e-6, n_nonzero_coefs=None):
        '''
        1.初始化字典
        '''
        # (120,165) -> (120, 160) -> (30, 40)*16 -> (1200,)*16
        self.max_iter = max_iter
        self.tol = tol
        self.n_nonzero_coefs = n_nonzero_coefs
        self.train_data = train_data
        self.test_data = test_data
    
    def makeDictionary(self, newShape=(12,10), block=(1,1)):
        '''
        newShape为降采样后的图片大小，block为对降采样后的图片分别在行、列分为多少块

        A. 将训练集图像进行降采样eg.(40,30), 并reshape成一列（120行1列）,并对该列进行归一化。 
        B. 将训练图像依次处理并排列成字典，
         1.1 其中Ai 是某一个人的特征集合。 1.2 其中"a" _i1是第i个人的第1张图像reshape的那一列（120行1列）。
        C. 将测试数据用同样的参数降采样并reshape得到特征向量。
        '''
        self.divide = Divide(int(newShape[1] / block[1]),int(newShape[0] / block[0]))
        self.div_num = block[0] * block[1]
        num_row = int(newShape[0]*newShape[1]/self.div_num)

        # 处理训练集(字典)
        print('SRC->init: train_data', end='')
        self.dictionary = np.zeros((num_row, 100*14*self.div_num))
        for ind, i in enumerate(self.train_data):
            # 先重新降采样，规划图片大小
            img = cv2.resize(i, newShape, interpolation=cv2.INTER_CUBIC)
            # 按参数分块
            res = self.divide.encode(img)
            # self.dictionary = np.column_stack((self.dictionary, res))
            self.dictionary[:,self.div_num*ind:self.div_num*(ind+1)] = res
            if ind % 20 == 0:
                print('.', end='')
        
        # 处理测试集(test)
        print('\nSRC->init: test_data', end='')
        self.test_img = np.zeros((num_row, 100*12*self.div_num))
        for ind,i in enumerate(self.test_data):
            img = cv2.resize(i, newShape, interpolation=cv2.INTER_CUBIC)
            res = self.divide.encode(img)
            # self.test_img = np.column_stack((self.test_img, res))
            self.test_img[:,self.div_num*ind:self.div_num*(ind+1)] = res
            if ind % 20 == 0:
                print('.',end='')
        print('')

    def OMP(self, y):
        '''
        2.用OMP算法计算该测试数据的稀疏表达x；
        '''
        nrows = 3
        ncols = 4
        figsize = (8, 8)
        _, figs = plt.subplots(nrows, ncols, figsize=figsize)
        # l = []

        for i in range(12):
            yy = y[:, i * self.div_num: (i + 1) * self.div_num]
            xx = linear_model.orthogonal_mp(self.dictionary, yy, n_nonzero_coefs=self.n_nonzero_coefs)
            if len(xx.shape) == 1:
                xx = xx[:, np.newaxis]

            l = []
            print("[OMP]->i:{}: ".format(i))
            for j in range(100):
                # (1200, 14*16) * (14*16, 16)
                # (120,1) -
                dd = self.dictionary[:, j * 14 * self.div_num : (j + 1) * 14 * self.div_num]
                xxx = xx[j * 14 * self.div_num : (j + 1) * 14 * self.div_num, :]
                e = np.linalg.norm(yy - np.dot(dd, xxx))
                # print("[OMP]->i:{},j:{}->e:{}".format(i, j, e))
                print("\tj:{}->e:{}".format(j, e),end='')
                if e == 0.0:
                    e += 1e-6
                l.append(math.log(e))
            print()

            figs[int(i/4)][i%4].bar(list(range(100)), l)
            # figs[i][j].axes.get_xaxis().set_visible(False)
            # figs[i][j].axes.get_yaxis().set_visible(False)

            # plt.bar(list(range(100)), l)
            # plt.show()
        plt.show()
            
    
    def run(self):
        # 2.用OMP算法计算该测试数据的稀疏表达x；
        for i in range(100):
            self.OMP(self.test_img[:, i * self.div_num * 12 : (i + 1) * self.div_num * 12])
        print('size dic:{},test:{}'.format(self.dictionary.size(), self.test_img.size()))
        # 3.使用类似one-hot方法对x进行处理。
        # 4.应用字典将处理后的稀疏表达还原，并计算原后的向量和图像原始特征向量的距离
        # train: 100类*14张*16块*1200 test: 100*12*16*1200

        # 5.对所有类别均用3、4的方法计算距离。距离最小的类，即为分类结果。


class Divide:
    def __init__(self, b_w, b_h):
        '''
        b_w: block width
        b_h: block height
        '''
        self.block_width = b_w
        self.block_height = b_h

    def encode(self, mat):
        (W, H) = mat.shape
        # (192, 168)->(24,21)
        w_len = int(W / self.block_width)
        h_len = int(H / self.block_height)
        res = np.zeros((self.block_width * self.block_height, w_len * h_len))
        for i in range(h_len):
            for j in range(w_len):
                temp = mat[j * self.block_width:(j + 1) * self.block_width,
                           i * self.block_height:(i + 1) * self.block_height]
                temp = temp.reshape(self.block_width * self.block_height)
                res[:, i * w_len + j] = temp
        return res

    def decode(self, mat, W, H):
        '''
        mat.shape should be ( block_width*block_height, ~ = 24*21 )
        '''
        w_len = int(W / self.block_width)
        h_len = int(H / self.block_height)
        mat = mat.reshape(self.block_width * self.block_height, w_len * h_len)
        
        res = np.zeros((W, H))
        for i in range(h_len):
            for j in range(w_len):
                temp = mat[:, i * w_len + j]
                temp = temp.reshape(self.block_width, self.block_height)
                res[j * self.block_width:(j + 1) * self.block_width,
                    i * self.block_height:(i + 1) * self.block_height] = temp
        return res

if __name__ == '__main__':
    # 1.检查数据集中的数据特征，确定图片分块大小 (120, 165)
    #   并将无遮挡的人脸作为训练数据，有遮挡的人脸作为测试数据。
    dataset = dataMaker('D:\\MINE_FILE\\dataSet\\AR', 'AR')
    
    # 2.应用SCR算法进行字典构建并对测试集进行基于分块投票的分类；
    src_algorithm = SRC(dataset.train_data, dataset.test_data, max_iter=100, tol=1e-5)
    src_algorithm.makeDictionary(newShape=(60,50),block=(5,5))
    src_algorithm.run()
    # 3.统计分类结果与准确率。




