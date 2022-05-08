# @ Time : 2022/4/2,15:58
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import glob
import os
import torchvision.io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, item):
        fname = self.fnames[item]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    pass

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))  # 读取数据文件夹中的所有图片 返回一个长为71314的list
    compose = [
        transforms.ToPILImage(),   # 转化为PIL图像格式
        transforms.Resize((64, 64)),  # 转化大小为 64 * 64 * 3, 原图都是96 * 96 * 3的
        transforms.ToTensor(),  # 将处理好的数据转化为Tensor, 此时Tensor的大小为64 * 64 *3
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))  # 归一化,加快模型收敛速度
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


if __name__ == '__main__':
    prefix = './data/'
    # 读取数据集
    dataset = get_dataset(os.path.join(prefix, 'faces'))
    # 输出查看
    images = [(dataset[i]+1)/2 for i in range(16)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()