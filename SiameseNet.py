
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


# 先对数据进行预处理

im_aug = tfs.Compose([
    tfs.Resize((100,100)),
    tfs.ToTensor()
])

im_aug1 = tfs.Compose([
    tfs.Resize(200),
    tfs.RandomHorizontalFlip(),
    tfs.RandomCrop(128),
    tfs.ToTensor()
])


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

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
            # 输出为 8 * 100 * 100
        )

        self.fc1 = nn.Sequential(
            # nn.Linear(8*100*100, 500),
            nn.Linear(8*128*128, 512),
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
    __epoch_size__ = 200

    def __init__(self,transform=None,should_invert=False):
        self.imageFolderDataset = []
        self.train_dataloader = []
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self, class_num=40, item_num=10):
        '''
        如果图像来自同一个类，标签将为0，否则为1
        '''
        img0_class = random.randint(0,class_num-1)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            temp = random.sample(list(range(0,item_num)), 2)
            img0_tuple = (self.imageFolderDataset[img0_class][temp[0]], img0_class)
            img1_tuple = (self.imageFolderDataset[img0_class][temp[1]], img0_class)
        else:
            img1_class = random.randint(0, class_num - 1)
            # 保证属于不同类别
            while img1_class == img0_class:
                img1_class = random.randint(0, class_num - 1)
            img0_tuple = (self.imageFolderDataset[img0_class][random.randint(0,item_num-1)], img0_class)
            img1_tuple = (self.imageFolderDataset[img1_class][random.randint(0,item_num-1)], img1_class)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # 用以指定一种色彩模式, "L"8位像素，黑白
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # img0 = img0.resize((100,100),Image.BILINEAR)
        # img1 = img1.resize((100,100),Image.BILINEAR)

        if self.should_invert:
            # 二值图像黑白反转
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        # if self.transform is not None:
        #     # 不知道是做什么用的
        #     img0 = self.transform(img0)
        #     img1 = self.transform(img1)
        
        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        return img0, img1, should_get_same_class

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
        for i in range(self.__epoch_size__):
            img0, img1, label = self.__getitem__()
            self.train_dataloader.append((img0, img1, label))
            # print("\r" + 'Cnt: ' + str(i)  + '/' + str(self.__epoch_size__) + '[' +">>" * i + ']',end=' ')

    def classed_pack(self):
        local = 'image/classed_pack/'
        self.imageFolderDataset = []
        for i in range(1, 10 + 1):
            temp = []
            sub_floder = local + str(i) + '/'
            for j in range(1, 3 + 1):
                temp.append(sub_floder + str(j) + '.jpg')
            self.imageFolderDataset.append(temp)
        # 为数据集添加数据
        for i in range(self.__epoch_size__):
            img0, img1, label = self.__getitem__(10,3)
            self.train_dataloader.append((img0, img1, label))
    

def show_plot(x, y):
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    # net = SiameseNetwork().cuda()
    print('start preparing the data...')
    train_data = SiameseNetworkDataset()
    # train_data.att_face_data()
    train_data.classed_pack()
    print('finish preparing the data...')

    net = SiameseNetwork()
    print(net)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    
    counter = []
    loss_history = []
    iteration_number= 0
    train_number_epochs = 100

    for epoch in range(0,train_number_epochs):
        for i, data in enumerate(train_data.train_dataloader):
            img0, img1, label = data
            img0 = im_aug1(img0)
            img1 = im_aug1(img1)
            # img0 = torch.unsqueeze(img0, 0)
            # img1 = torch.unsqueeze(img1, 0)
            img0 = torch.unsqueeze(img0, dim=0).float()
            img1 = torch.unsqueeze(img1, dim=0).float()
            # img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1,output2 = net(img0,img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i%10 == 0 :
                print("Epoch: {} step: {} Current loss {}".format(epoch,i,loss_contrastive.data[0]))
                iteration_number = 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
    # show_plot(counter,loss_history)
    show_plot(counter, loss_history)
