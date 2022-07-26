# @ Time : 2022/7/23,10:22
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch

class FoodDataset(Dataset):

    def __init__(self, paths, labels, mode):
        # mode分为eval和train
        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            # 针对训练图片做transform处理，学到更加深层次的特征
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])

        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])

        self.transform = trainTransform if mode == 'train' else evalTransform
        pass

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        X = Image.open(self.paths[item])
        X = self.transform(X)
        Y = self.labels[item]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


def get_paths_labels(path):
    # 便于获取data和label
    def my_key(name):
        return int(name.replace(".jpg","").split("_")[1])+1000000*int(name.split("_")[0])

    imgnames = os.listdir(path)
    imgnames.sort(key=my_key)
    imgpaths = []
    labels = []

    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))

    return imgpaths, labels



if __name__ == '__main__':
    # 测试函数功能
    train_paths, train_labels = get_paths_labels('./data/food')

    print(train_paths)
    print(train_labels)