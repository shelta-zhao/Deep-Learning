# @ Time   : 2022/8/11 17:26
# @ Author : Super_XIAOTANG
# @ File   : Dataset.py
# @ IDE    : PyCharm
# @ Brief  :

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import glob
import numpy as np


class Omniglot(Dataset):

    def __init__(self, data_dir, k_shot, q_query, task_num=None):
        self.file_list = [file for file in glob.glob(data_dir + '**/character*', recursive=True)]  # recursive代表递归调用

        if task_num is not None:
            # 限制task的数量
            self.file_list = self.file_list[:min(len(self.file_list), task_num)]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_shot + q_query
        pass

    def __getitem__(self, item):
        sample = np.arange(20)  # [0,1,2,...,19]

        np.random.shuffle(sample)  # 打乱
        img_path = self.file_list[item]
        img_list = [file for file in glob.glob(img_path + '**/*.png', recursive=True)]
        img_list.sort()  # 某一个character下的20个png的列表

        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        imgs = torch.stack(imgs)[sample[: self.n]]  # 拼接后返回前n个 由于sample被打乱过 所以这里是随机在20个里选择n个
        return imgs  # imgs的形状为[self.n, 1, 28, 28]

    def __len__(self):
        return len(self.file_list)


def dataloader_init(datasets, n_way, shuffle=True):

    train_set, valid_set = datasets
    train_loader = DataLoader(
        train_set,
        batch_size=n_way,
        shuffle=shuffle,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=n_way,
        shuffle=shuffle,
        drop_last=True,
    )

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)

    return (train_loader, valid_loader), (train_iter, valid_iter)




if __name__ == '__main__':
    dataset = Omniglot('Omniglot/images_background/', 1, 1, 10)
    print(len(dataset))
    print(dataset[0].shape)


