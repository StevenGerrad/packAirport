
################################################################################################
# 
#       2020.1.8
#       -----------
#       siameseNet 网络test
#       1. 图像增强 https://blog.csdn.net/weixin_40793406/article/details/84867143
#       https://www.pytorchtutorial.com/pytorch-one-shot-learning/#Contrastive_Loss_function
# 
################################################################################################

import sys
sys.path.append('..')

import PIL
from PIL import Image

from torchvision import transforms as tfs
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

image_width = 200
image_height = 200

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 输入为 200 x 200
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            # 对每个 channel 按照概率设置为 0
            nn.Dropout2d(p=.2),
            # 输出为 4 * 200 * 200
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
            # 输出为 8 * 200 * 200

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            # 输出为 8 * 200 * 200
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*image_width*image_height, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = torch.tensor([margin])

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = ((1 - label) * torch.pow(euclidean_distance, 2)
                        + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))/2
        return loss_contrastive

class SiameseNetworkDataset():
    __set_size__ = 90
    __batch_size__ = 10

    def __init__(self,set_size=90,batch_size=10,transform=None,should_invert=False):
        self.imageFolderDataset = []
        self.train_dataloader = []

        self.__set_size__ = set_size
        self.__batch_size__ = batch_size

        self.transform = tfs.Compose([
                            tfs.Resize((image_width,image_height)),
                            tfs.ToTensor()
                        ])
        self.should_invert = should_invert
        
    def __getitem__(self, class_num=40):
        '''
        如果图像来自同一个类，标签将为0，否则为1
        TODO: 实际上: Y值为1或0。如果模型预测输入是相似的，那么Y的值为0，否则Y为1。
        TODO: 由于classed_pack 每类可能有2-3张, 此时参数item_num无效,故删去参数中的item_num
        '''
        data0 = torch.empty(0, 3, image_width, image_height)
        data1 = torch.empty(0, 3, image_width, image_height)

        should_get_same_class = random.randint(0,1)
        for i in range(self.__batch_size__):
            img0_class = random.randint(0,class_num-1)
            # we need to make sure approx 50% of images are in the same class
            
            if should_get_same_class:
                item_num = len(self.imageFolderDataset[img0_class])
                temp = random.sample(list(range(0,item_num)), 2)
                img0_tuple = (self.imageFolderDataset[img0_class][temp[0]], img0_class)
                img1_tuple = (self.imageFolderDataset[img0_class][temp[1]], img0_class)
            else:
                img1_class = random.randint(0, class_num - 1)
                # 保证属于不同类别
                while img1_class == img0_class:
                    img1_class = random.randint(0, class_num - 1)
                item_num = len(self.imageFolderDataset[img0_class])
                img0_tuple = (self.imageFolderDataset[img0_class][random.randint(0, item_num - 1)], img0_class)
                item_num = len(self.imageFolderDataset[img1_class])
                img1_tuple = (self.imageFolderDataset[img1_class][random.randint(0, item_num - 1)], img1_class)

            img0 = Image.open(img0_tuple[0])
            img1 = Image.open(img1_tuple[0])

            if self.should_invert:
                # 二值图像黑白反转,默认不采用
                img0 = PIL.ImageOps.invert(img0)
                img1 = PIL.ImageOps.invert(img1)

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            
            img0 = torch.unsqueeze(img0, dim=0).float()
            img1 = torch.unsqueeze(img1, dim=0).float()

            data0 = torch.cat((data0, img0), dim=0)
            data1 = torch.cat((data1, img1), dim=0)
        
        # XXX: 注意should_get_same_class的值
        return data0, data1, torch.from_numpy(np.array([should_get_same_class ^ 1], dtype=np.float32))
    
    def classed_pack(self):
        local = 'image/classed_pack/2019-03-14 22-19-img/'
        local1 = 'image/classed_pack/2019-03-14 16-30-img/'
        self.imageFolderDataset = []

        # floader1
        subFloader = os.listdir(local)
        for i in subFloader:
            temp = []
            sub_dir = local + i + '/'
            subsubFloader = os.listdir(sub_dir)
            for j in subsubFloader:
                temp.append(sub_dir + j)
            self.imageFolderDataset.append(temp)
        # floader2
        subFloader = os.listdir(local1)
        for i in subFloader:
            temp = []
            sub_dir = local + i + '/'
            subsubFloader = os.listdir(sub_dir)
            for j in subsubFloader:
                temp.append(sub_dir + j)
            self.imageFolderDataset.append(temp)

        # 为数据集添加数据
        for i in range(self.__set_size__):
            img0, img1, label = self.__getitem__(len(self.imageFolderDataset))
            self.train_dataloader.append((img0, img1, label))



class siamese_match():
    def __init__(self):
        self.net = SiameseNetwork()
        self.net.load_state_dict(torch.load('net031402_params.pkl'))
        self.net.eval()

        self.image_width = 200
        self.image_height = 200
        self.transform = tfs.Compose([
                            tfs.Resize((self.image_width,self.image_height)),
                            tfs.ToTensor()
                        ])
    
    def im_match(self, img0, img1):
        '''
        传入数据: 为numpy类型: [height, width, channel]
        返回数据: result(匹配为0, 不匹配为1), 误差(不匹配的误差大)
        '''
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        # 扩充维度, 3维 -> 4维
        img0 = torch.unsqueeze(img0, dim=0).float()
        img1 = torch.unsqueeze(img1, dim=0).float()

        # 调整维度顺序
        img0 = img0.permute(0, 2, 3, 1)
        img1 = img1.permute(0, 2, 3, 1)

        output1,output2 = self.net(img0,img1)
        euclidean_distance = F.pairwise_distance(output1, output2)

        # TODO: 这两个参数是根据下面的调试结果定的
        if euclidean_distance < 1.0:
            return 0, euclidean_distance.cpu().data.numpy()[0]
        elif euclidean_distance >= 1.0:
            return 1, euclidean_distance.cpu().data.numpy()[0]

if __name__ == '__main__':
    batch_size = 1 
    data_num = 600      # 训练集总数

    print('start preparing the data...')
    train_data = SiameseNetworkDataset(set_size=data_num, batch_size=batch_size)
    train_data.classed_pack()
    print('finish preparing the data...')

    criterion = ContrastiveLoss()

    # 测试接口
    siaMatch = siamese_match()

    match_err = [0, 0]
    e_dis = [[],[]]
    for i, data in enumerate(train_data.train_dataloader):
        img0, img1, label = data
        label = int(label)

        res, euclidean_distance = siaMatch.im_match(img0, img1)
        print('Item {}, label {}->{}, euclidean_distance {:.6f}'.format(i, label, res, euclidean_distance))
        e_dis[label].append(euclidean_distance)
        
        # 统计判断错误个数
        if label != res:
            match_err[label] += 1

        '''
        # XXX:可以用来看看反常图像
        if label != res:
            plt.subplot(121)
            plt.imshow((img0.squeeze(0)).permute(1, 2, 0))
            plt.subplot(122)
            plt.imshow((img1.squeeze(0)).permute(1, 2, 0))
            plt.show()
        '''

    print('error numbers for label 0 and 1: ',match_err)

    # 查看分布的四分位数
    temp = np.percentile(e_dis[0], (75,80,85,90,95,97), interpolation='midpoint')
    print('label 0 quartile:', temp)
    temp = np.percentile(e_dis[1], (3,5,10,15,20,25), interpolation='midpoint')
    print('label 1 quartile:', temp)

    # 查看分布图
    plt.subplot(121)
    plt.hist(x=e_dis[0], bins='auto',density=True)
    plt.subplot(122)
    plt.hist(x=e_dis[1], bins='auto',density=True)
    plt.show()
    
    

