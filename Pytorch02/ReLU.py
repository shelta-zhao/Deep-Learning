# @ Time : 2022/3/23,17:55
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:

import numpy as np
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        r'''
        ReLU激活函数的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, d1, d2, ..., dk)
        '''

        self.mask = (x <= 0)  # true=1,false=0
        out = x.copy()  # out[0]=x,将另一个对象关联到这个对象的副本,关联到[0]的位置,也就是副本的位置
        out[self.mask] = 0
        return out  # 总是输出out[1]


x= np.array([[-2, -1, 0], [-1, 2, -3]]).astype(np.float32)  # 把二维列表转化成数组，float32数位越高浮点数的精度越高


print()
print(ReLU().forward(x))
