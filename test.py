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

