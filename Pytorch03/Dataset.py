# @ Time : 2022/3/24,10:40
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import os
from torch.utils.data import Dataset
from PIL import Image

class FoodDataset(Dataset):

    def __init__(self, path, tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])  # 读取所有文件的名字
        if files:
            self.files = files
        print(f'One {path} sample', self.files[0])
        self.transform = tfm

    def __getitem__(self, item):
        fname = self.files[item]
        img = Image.open(fname)
        img = self.transform(img)
        try:
            label = int(fname.split("\\")[-1].split('_')[0])
        except:
            label = -1
        return img, label

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    fname = '0_48.jpg'
    print(fname.split('_'))  # 可以看到分割结果为[0][48.jpg],可以得到标签

    path = './data/training'
    print(os.path.join(path, '0_0.jpg'))  # 组合地址
    print(os.path.join(path, '0_0.jpg').split("/")[-1].split("\\")[1].split('_')[0])
    # print(os.listdir(path))  # 输出该地址下所有文件的名字
    for x in os.listdir(path):
        if x.endswith('.jpg'):
            print(x)
            break
    print(type(sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])))  # 类型为list
    print(type(sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])[0]))  # 元素的类型为str
    print(sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')])[0])  # 输出为‘./data/test/0001.jpg’
