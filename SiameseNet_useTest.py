
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
        # 输入为 100 x 100
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            # 对每个 channel 按照概率设置为 0
            nn.Dropout2d(p=.2),
            # 输出为 4 * 100 * 100
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
            # 输出为 8 * 100 * 100

            # nn.ReflectionPad2d(1),
            # nn.Conv2d(8, 8, kernel_size=3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(8),
            # nn.Dropout2d(p=.2),

            # 输出为 8 * 100 * 100
        )

        self.fc1 = nn.Sequential(
            # nn.Linear(8*100*100, 500),
            nn.Linear(8*image_width*image_height, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)
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
        # self.margin = torch.from_numpy(margin)
        self.margin = torch.tensor([margin])

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # temp = self.margin - euclidean_distance
        # loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2)
        #                             (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
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
                            # tfs.RandomHorizontalFlip(),
                            # tfs.RandomCrop(128),
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
            #we need to make sure approx 50% of images are in the same class
            
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
            # 用以指定一种色彩模式, "L"8位像素，黑白
            # img0 = img0.convert("L")
            # img1 = img1.convert("L")

            # img0 = img0.resize((100,100),Image.BILINEAR)
            # img1 = img1.resize((100,100),Image.BILINEAR)

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
        return data0, data1, should_get_same_class ^ 1

    def att_face_data(self):
        ''' AT&T 数据集: 共40类, 每类十张图像 '''
        local = 'D:/MINE_FILE/dataSet/att_faces/'
        self.imageFolderDataset = []
        for i in range(1, 40 + 1):
            temp = []
            sub_floder = local + 's' + str(i) + '/'
            for j in range(1, 10 + 1):
                temp.append(sub_floder + str(j) + '.pgm')
            self.imageFolderDataset.append(temp)
        # 为数据集添加数据
        for i in range(self.__set_size__):
            img0, img1, label = self.__getitem__()
            self.train_dataloader.append((img0, img1, label))
            # print("\r" + 'Cnt: ' + str(i)  + '/' + str(self.__epoch_size__) + '[' +">>" * i + ']',end=' ')

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
    

def show_plot(x, y):
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    batch_size = 1 
    data_num = 600      # 训练集总数
    train_number_epochs = 100

    # net = SiameseNetwork().cuda()
    print('start preparing the data...')
    train_data = SiameseNetworkDataset(set_size=data_num, batch_size=batch_size)
    # train_data.att_face_data()
    train_data.classed_pack()
    print('finish preparing the data...')

    net = SiameseNetwork()
    print(net)

    criterion = ContrastiveLoss()
    # optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    
    # counter = []
    loss_history = [[],[]]
    iteration_number= 1     # 实验数据记录的步长

    # 加载模型
    net.load_state_dict(torch.load('net0313_params.pkl'))
    net.eval()

    # train
    match_err = [0,0]
    for i, data in enumerate(train_data.train_dataloader):
        img0, img1, label = data

        output1,output2 = net(img0,img1)
        # optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)

        euclidean_distance = F.pairwise_distance(output1, output2)
        if i % iteration_number == 0:
            print("Step {}, label {}, loss {:.4f}".format(i, label, loss_contrastive.data[0]), end='')
            print(', euclidean_distance {:.6f}'.format(euclidean_distance.cpu().data.numpy()[0]))
            # 为后续图形化训练过程总结准备
            # counter.append(iteration_number)
            loss_history[label].append(loss_contrastive.data[0])
                
        if label == 0 and loss_contrastive.data[0] > 1.0 :
            match_err[0] += 1
        elif label == 1 and loss_contrastive.data[0] < 1.0 :
            match_err[1] += 1

        # if loss_contrastive.data[0] > 2.0:
        #     plt.subplot(121)
        #     plt.imshow((img0.squeeze(0)).permute(1, 2, 0))
        #     plt.subplot(122)
        #     plt.imshow((img1.squeeze(0)).permute(1, 2, 0))
        #     plt.show()

    print(match_err)

    # show_plot(counter, loss_history)
