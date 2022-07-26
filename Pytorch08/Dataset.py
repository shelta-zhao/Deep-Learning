# @ Time : 2022/7/18,20:45
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:

import torch
from torch.utils.data import TensorDataset
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def show_figure(np_img):
    plt.figure()
    plt.imshow(np_img)
    plt.axis('on')
    plt.title('Numpy Image')
    plt.show()
    pass

class HumanDataset(TensorDataset):
    # TensorDataset 支持transform转换
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            # permute 维度交换 交换前[100000, 64, 64, 3], 交换后[100000, 3, 64, 64]
            self.tensors = tensors.permute(0, 3, 1, 2)

        # 加入transform的意义是将pixel从[0, 255]转化为[-1, 1]
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x:x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x/255. - 1.)
        ])

    def __getitem__(self, item):
        img = self.tensors[item]
        if self.transform:
            # 将pixel从[0, 255]转化为[-1, 1]
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.tensors)


if __name__ == "__main__":

    # 读取数据
    train_set = np.load('./data/trainingset.npy', allow_pickle=True)
    test_set = np.load('./data/testingset.npy', allow_pickle=True)

    # 查看大小
    print(train_set.shape)  # [100000, 64, 64, 3]
    print(test_set.shape)   # [19636, 64, 64, 3]

    # 查看图片
    # show_figure(test_set[100])

    # tensor转化测试
    tensor = torch.from_numpy(train_set)
    print(tensor.shape)  # [100000, 64, 64, 3]
    tensor = tensor.permute(0, 3, 1, 2)
    print(tensor.shape)  # [100000, 3, 64, 64]
    tensor = tensor.contiguous().view(tensor.size()[0], -1)
    print(tensor.shape)  # [100000, 12288]

