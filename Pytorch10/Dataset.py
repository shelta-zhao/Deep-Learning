# @ Time   : 2022/7/29 19:39
# @ Author : Super_XIAOTANG
# @ File   : Dataset.py
# @ IDE    : PyCharm
# @ Brief  :

from torch.utils.data import Dataset
import os
import glob
from torchvision.transforms import transforms
from PIL import Image

class AdvDataset(Dataset):

    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        self.transform = transform

        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            # enumerate括号里对部分为data下的各个子文件夹
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]

        pass

    def __getitem__(self, item):
        image = self.transform(Image.open(self.images[item]))
        label = self.labels[item]
        return image, label

    def __getname__(self):
        return self.names

    def __len__(self):
        return len(self.images)




if __name__ == '__main__':

    # 测试函数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
    ])

    adv_set = AdvDataset('./data', transform=transform)
    print(adv_set.__getitem__(0)[0].shape)  # 【3， 32，32】





