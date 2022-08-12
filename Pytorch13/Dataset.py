# @ Time   : 2022/8/7 15:29
# @ Author : Super_XIAOTANG
# @ File   : Dataset.py
# @ IDE    : PyCharm
# @ Brief  :

import os
from torch.utils.data import Dataset
from PIL import Image as Img

class FoodDataset(Dataset):

    def __init__(self, path, tfm, files=None):
        super().__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if None != files:
            self.files = files
        print(f"One {path} Sample: {self.files[0]}")
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fname = self.files[item]
        img = Img.open(fname)
        img = self.tfm(img)

        try:
            label = int(fname.split('/')[-1].split("_")[0])
        except:
            label = -1  # test数据集没有label

        return img, label

if __name__ == '__main__':
    print('Hello ')