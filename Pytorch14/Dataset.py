# @ Time   : 2022/8/8 13:49
# @ Author : Super_XIAOTANG
# @ File   : Dataset.py
# @ IDE    : PyCharm
# @ Brief  :

import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

def rotate_image(image, angle):
    # 旋转图片
    if angle is None:
        return image

    image = transforms.functional.rotate(image, angle=angle)
    return image

def get_transform(angle=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: rotate_image(x, angle)),
        Pad(28),  # MNIST数据集的大小就是28 经过旋转之后要进行padding保证大小不变
    ])
    return transform

class Pad(object):
    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # 补充图片
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)  # 断言 在表达式为false的时候直接触发异常退出程序
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)

class Data:
    def __init__(self, path, train=True, angle=None):
        transform = get_transform(angle)
        # train如果是True 就从training.pt建立数据集，否则从test.pt建立数据集
        self.dataset = datasets.MNIST(root=os.path.join(path, 'MNIST'), transform=transform, train=train, download=True)


if __name__ == '__main__':

    # 查看五个task对应的数据集
    angle_list = [20 * x for x in range(5)]
    sample = [Data('data', angle=angle_list[index]) for index in range(5)]

    plt.figure(figsize=(30, 10))
    for task in range(5):
        target_list = []
        cnt = 0
        while len(target_list) < 10:
            img, target = sample[task].dataset[cnt]
            cnt += 1
            if target in target_list:
                continue
            else:
                target_list.append(target)

            plt.subplot(5, 10, task * 10 + target + 1)
            curr_img = np.reshape(img, (28, 28))
            plt.matshow(curr_img, cmap=plt.get_cmap('gray'), fignum=False)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.title("task:" + str(task + 1) + " " + "label" + str(target), y=1)
            pass
        pass
    plt.show()
