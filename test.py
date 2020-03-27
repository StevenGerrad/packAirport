
# 尝试矩阵稀疏表达

'''

import numpy as np
from sklearn import linear_model

def dict_update(y, d, x, n_components):
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

if __name__ == '__main__':
    # 3x1 = 3x10 * 10x1
    n_comp = 10

    y = np.random.rand(3, 1)+1
    print(y)
    dic = np.random.rand(3, n_comp)
    print(dic)

    xx = linear_model.orthogonal_mp(dic, y)

    max_iter = 10
    dictionary = dic
    tolerance = 1e-6
    for i in range(max_iter):
        # 稀疏编码
        x = linear_model.orthogonal_mp(dictionary, y)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        e = np.linalg.norm(y - np.dot(dictionary, x))
        print('e:',e)
        if e < tolerance:
            break
        dictionary,_ = dict_update(y, dictionary, x, n_comp)

    sparsecode = linear_model.orthogonal_mp(dictionary, y)
    print(sparsecode)
'''

'''


import os,datetime

def takeSecond(elem):
    return elem[1]

base_dir = 'image/2019-03-14 16-30-img'
# list = os.listdir(base_dir)
list = []
for i in range(1, 88 + 1):
    list.append(str(i))

filelist = []

for i in range(0, len(list)):
    floader_path = os.path.join(base_dir, list[i])
    sub_list = os.listdir(floader_path)

    num_floader = []
    for i in range(3):
        file_path = os.path.join(floader_path, sub_list[i])
        num_floader.append(file_path)
    filelist.append(num_floader)

for i in filelist:
    # path = os.path.join(base_dir, filelist[i])
    # if os.path.isdir(path):
    #     continue
    acc = []
    for j in i:
        timestamp = os.path.getmtime(j)
        # print timestamp
        ts1 = os.stat(j).st_mtime
        # print ts1

        date = datetime.datetime.fromtimestamp(timestamp)
        p_str = date.strftime('%Y-%m-%d %H:%M:%S')
        # print(p_str)
        acc.append((j, p_str))

    acc.sort(key=takeSecond)
    for index,j in enumerate(acc):
        l = j[0].split('\\', 2)
        os.rename(j[0], l[0]+'\\'+l[1]+'\\'+str(index+1)+'.jpg')
        # print()
    # print list[i],' 最近修改时间是: ',date.strftime('%Y-%m-%d %H:%M:%S')

'''

'''

import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# x = torch.unsqueeze(torch.linspace(0, 1, 140), dim=1)
# y = torch.zeros(140, 1, dtype=torch.float)


# 1880-2019
x = torch.zeros(140, 1, dtype=torch.float)
y = torch.zeros(29, 1, dtype=torch.float)

max_i = 0.0
with open("sea_level.txt", "r") as f:
    l_cnt = 0
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        t = float(line)
        # print(line)
        max_i = max(max_i, abs(t))
        x[l_cnt,0] = t
        l_cnt += 1

x = x[-30:-1, :]
max_i = max(abs(x))
# TODO:先这样定着
# max_i = 92.15

# x = torch.div(x, max(abs(x)))

# 1990-2018
max_e = 0.0
with open("edps_world.txt", "r") as f:
    l_cnt = 0
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        t = float(line)
        # print(line)
        max_e = max(max_e, abs(t))
        y[l_cnt,0] = t
        l_cnt += 1

# y = torch.div(y, max_e)

x = x[-20:-1, :]
max_i = max(abs(x))
x = torch.div(x, max_i)

y = y[-20:-1, :]
max_e = max(abs(y))
y = torch.div(y, max_e)


# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.l1 = torch.nn.Linear(n_hidden, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        # x = F.sigmoid(self.hidden(x))
        x = self.l1(x)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)


plt.ion()   # 画图
plt.show()

for t in range(1000):
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters上
    
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        # plt.scatter(x.data.numpy()*max_i, y.data.numpy()*max_e)
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.plot(x.data.numpy()*max_i, prediction.data.numpy()*max_e, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        print('Epoch:{},loss:{}'.format(t, loss))



plt.ioff()
plt.show()


# temp = torch.linspace(1, 100, 100)
future = torch.div(torch.unsqueeze(torch.linspace(1, 100, 100), dim=1), max_i)
future += 1

future = torch.cat((x, future), 0)

yy = net(future)

plt.plot(future.data.numpy(), yy.data.numpy(), 'r-', lw=5)
plt.show()




'''
import random
import matplotlib.pyplot as plt

l = []
for i in range(1000):
    l.append(random.normalvariate(2, 10))

plt.hist(l, bins='auto')
plt.show()