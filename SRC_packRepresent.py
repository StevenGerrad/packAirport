
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
import copy

from torchvision import transforms as tfs

im_aug = tfs.Compose([
    tfs.Resize(200),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(96)
])

class dataMaker():
    def __init__(self, file_dir, name='AR', use_num=None):
        if name == 'AR':
            if use_num == None:
                self.num_class = 100
            else:
                self.num_class = use_num
            self.train_item = 14
            self.test_item = 12
            self.AR_dataSet(file_dir)
        elif name == 'YaleB':
            if use_num == None:
                self.num_class = 39
            else:
                self.num_class = use_num
            self.train_item = 1
            self.test_item = 1
            self.YaleB_dataSet(file_dir)
        elif name == 'classed_pack':
            if use_num == None:
                self.num_class = 10
            else:
                self.num_class = use_num
                # 每类图片有三张(三个摄像头)取两个做训练集，一个做测试集
            self.train_item = 2
            self.test_item = 1

            self.classed_pack_dataset(file_dir)
            
    
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
        self.test_data = test_data  # 12*100

    def YaleB_dataSet(self, file_dir):
        '''
        读取YaleB数据集, 同时作为训练集和测试集
        TODO：好吧实际上YaleB数据集每类只有一张图片所以在SRC算法中无用
        '''
        src_img_w = 192
        src_img_h = 168

        # dataset = np.zeros((38,192,168), np.float)
        dataset = np.zeros((src_img_w * src_img_h, 38), np.float)
        cnt_num = 0
        img_list = sorted(os.listdir(file_dir))
        os.chdir(file_dir)

        self.train_data = []
        self.test_data = []
        for img in img_list:
            if img.endswith(".pgm"):
                # print(img.size)
                gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                # gray_img = cv2.resize(gray_img, (src_img_w, src_img_h),interpolation=cv2.INTER_AREA)
                # dataset[:, cnt_num] = gray_img.reshape(src_img_w * src_img_h, )
                cnt_num += 1
                self.train_data.append(gray_img)
                self.test_data.append(gray_img)
        print('...prepare train data finished')


    def classed_pack_dataset(self, file_dir):
        '''
        大创行李数据集，该数据集为'三个轨道的图像分别有两个做训练集，剩下的一个做测试集'
        TODO：考虑使用transforms
        '''
        # TODO：预定义哪个做测试集
        test_class = [3,]

        self.train_data = []
        self.test_data = []

        # 根据传入行李类别数判断读取.jpg文件
        for i in range(1, self.num_class + 1):
            temp = []
            sub_floder = file_dir + str(i) + '/'
            # 每件行李有三张图片
            for j in range(1, self.train_item + self.test_item + 1):
                # temp.append(sub_floder + str(j) + '.jpg')
                img_dir = sub_floder + str(j) + '.jpg'
                # TODO：直接以灰度图方式读出
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                if j in test_class:
                    self.test_data.append(img)
                else:
                    self.train_data.append(img)
        
        print('...prepare train data finished')
                


class SRC():
    def __init__(self, dataset, max_iter=100, tol=1e-6, n_nonzero_coefs=None):
        '''
        1.初始化字典
        '''
        # (120,165) -> (120, 160) -> (30, 40)*16 -> (1200,)*16
        self.max_iter = max_iter
        self.tol = tol
        self.n_nonzero_coefs = n_nonzero_coefs

        self.train_data = dataset.train_data
        self.test_data = dataset.test_data

        self.num_class = dataset.num_class
        self.train_item = dataset.train_item
        self.test_item = dataset.test_item
    
    def makeDictionary(self, newShape, block):
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
        self.dictionary = np.zeros((num_row, self.num_class*self.train_item*self.div_num))
        for ind, i in enumerate(self.train_data):
            # 先重新降采样，规划图片大小
            img = cv2.resize(i, newShape, interpolation=cv2.INTER_CUBIC)
            # 按参数分块
            # plt.imshow(img),plt.show()

            res = self.divide.encode(img)
            # self.dictionary = np.column_stack((self.dictionary, res))
            self.dictionary[:,self.div_num*ind:self.div_num*(ind+1)] = res
            if ind % 20 == 0:
                print('.', end='')
        
        # 处理测试集(test)
        print('\nSRC->init: test_data', end='')
        self.test_img = np.zeros((num_row, self.num_class*self.test_item*self.div_num))
        for ind,i in enumerate(self.test_data):
            img = cv2.resize(i, newShape, interpolation=cv2.INTER_CUBIC)
            res = self.divide.encode(img)
            # plt.imshow(self.divide.decode(res,newShape[1],newShape[0])),plt.show()

            # self.test_img = np.column_stack((self.test_img, res))
            self.test_img[:,self.div_num*ind:self.div_num*(ind+1)] = res
            if ind % 20 == 0:
                print('.',end='')
        print('')

        # Normalize the columns of A to have unit l2-norm
        print('dictionary', self.dictionary.shape, 'test_img', self.test_img.shape)
        
        # plt.imshow(self.dictionary), plt.show()
        # plt.imshow(self.test_img), plt.show()

        self.dictionary = self.l2_normalize(self.dictionary)
        self.test_img = self.l2_normalize(self.test_img)

        # plt.imshow(self.dictionary), plt.show()
        # plt.imshow(self.test_img), plt.show()
        print()

    def l2_normalize(self, x, axis=-1, order=2):
        l2 = np.linalg.norm(x, ord = order, axis=axis, keepdims=True)
        l2[l2==0] = 1
        return x/l2

    def dict_update(self, y, d, x, n_components):
        """
        使用KSVD更新字典的过程
        """
        for i in range(n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            # 更新第i列
            d[:, i] = 0
            # 计算误差矩阵
            r = (y - np.dot(d, x))[:, index]
            # 利用svd的方法，来求解更新字典和稀疏系数矩阵
            u, s, v = np.linalg.svd(r, full_matrices=False)
            # 使用左奇异矩阵的第0列更新字典
            d[:, i] = u[:, 0]
            # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
            for j,k in enumerate(index):
                x[i, k] = s[0] * v[0, j]
        return d, x

    def OMP(self, y):
        '''
        2.用OMP算法计算该测试数据的稀疏表达x；
        '''
        nrows = 3
        ncols = 4
        figsize = (8, 8)
        # _, figs = plt.subplots(nrows, ncols, figsize=figsize)
        # l = []

        # 共self.num_class类，每类图片测试集有...张图片
        for i in range(self.test_item):
            yy = y[:, i * self.div_num : (i + 1) * self.div_num]
            
            # 求解稀疏表达 ATTENTION: 这个写错了
            n_comp = self.num_class * self.train_item * self.div_num
            # max_iter = 10
            dic = copy.deepcopy(self.dictionary)

            for j in range(self.max_iter):
                # 稀疏编码
                x = linear_model.orthogonal_mp(dic, y)
                if len(x.shape) == 1:
                    x = x[:, np.newaxis]
                e = np.linalg.norm(y - np.dot(dic, x))
                print('dict_update->e:',e)
                if e < self.tol:
                    break
                dic, temp_x = self.dict_update(y, dic, x, n_comp)
                print('dict_update->dic.e:', np.linalg.norm(self.dictionary - dic))
                # print(x.transpose())
                # print(temp_x.transpose())

            xx = linear_model.orthogonal_mp(dic, yy)
            # xx = linear_model.orthogonal_mp(self.dictionary, yy)
            if len(xx.shape) == 1:
                xx = xx[:, np.newaxis]

            print(xx.transpose())
            # _, figs1 = plt.subplots(5, 5, figsize=figsize)
            for i in range(0, self.div_num):
                # TODO: 原来xx[]为0时log为-inf
                xx[xx==0] = self.tol
                t_y = np.log(xx[:,i])
                # t_y = xx[:,i]
                # t_x = list(range(35000))
                t_x = list(range(self.train_item*self.num_class*self.div_num))
                # figs1[i][j].bar(t_x,t_y)

                # 这个占用内存太大了似乎出不来啊
                plt.subplot(1, 4, 1), plt.imshow(self.divide.decode(yy, 50, 60))
                plt.subplot(1, 4, 2), plt.bar(t_x, t_y)
                plt.subplot(1, 4, 3), plt.imshow(self.divide.decode(self.dictionary[:, 15 * self.div_num : (15 + 1) * self.div_num], 50, 60))
                plt.subplot(1, 4, 4), plt.imshow(self.divide.decode(np.dot(dic, x),50,60))
                plt.show()
            # plt.show()

            l = []
            print("[OMP]->i:{}: ".format(i))
            for j in range(self.num_class):
                # 取每类的字典部分和每类的稀疏表达部分计算误差
                dd = self.dictionary[:, j * self.train_item * self.div_num : (j + 1) * self.train_item * self.div_num]
                xxx = xx[j * self.train_item * self.div_num : (j + 1) * self.train_item * self.div_num, :]
                e = np.linalg.norm(yy - np.dot(dd, xxx))
                # print("[OMP]->i:{},j:{}->e:{}".format(i, j, e))
                print("\tj:{}->e:{}".format(j, e),end='')
                if e == 0.0:
                    e = self.tol
                l.append(math.log(e))
            print()

            # figs[int(i/4)][i%4].bar(list(range(100)), l)
            # figs[i][j].axes.get_xaxis().set_visible(False)
            # figs[i][j].axes.get_yaxis().set_visible(False)
            
            plt.bar(list(range(self.num_class)), l)
            plt.show()
        
        # plt.show()
            
    
    def run(self):
        # 2.用OMP算法计算该测试数据的稀疏表达x；
        for i in range(self.num_class):
            self.OMP(self.test_img[:, i * self.div_num * self.test_item : (i + 1) * self.div_num * self.test_item])
        print('size dic:{},test:{}'.format(self.dictionary.shape, self.test_img.shape))
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

    # dataset = dataMaker('D:\\MINE_FILE\\dataSet\\AR', 'AR', use_num=40)
    # dataset = dataMaker('D:\\MINE_FILE\\dataSet\\YaleB', 'YaleB')
    dataset = dataMaker('./image/classed_pack/', 'classed_pack')
    
    # 2.应用SCR算法进行字典构建并对测试集进行基于分块投票的分类；
    src_algorithm = SRC(dataset, max_iter=100, tol=1e-5)

    # AR
    src_algorithm.makeDictionary(newShape=(60,50),block=(1,1))
    # src_algorithm.makeDictionary(newShape=(120,160),block=(5,5))

    # YaleB
    # src_algorithm.makeDictionary(newShape=(120, 160), block=(5, 5))

    src_algorithm.run()
    # 3.统计分类结果与准确率。




