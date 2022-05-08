# @ Time : 2022/3/24,10:40
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:



import gc
import random
import numpy as np
import torchvision.transforms as transforms
from functools import partial
from Dataset import *
from Model import *

from torch.utils.data import DataLoader


# 固定时间种子
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        pass
    pass


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    data_path = './data'

    # transforms 是常用的图像预处理函数 可以进行一些诸如裁剪、缩放等操作提高泛化能力 防止过拟合  即Data Augmentation过程
    transforms_list = [partial(transforms.Grayscale),  # 变灰度图
                       partial(transforms.RandomRotation, 45),  # 随机旋转
                       partial(transforms.RandomVerticalFlip, p=1),  # 随机水平旋转
                       partial(transforms.RandomHorizontalFlip, p=1),  # 随机水平旋转
                       partial(transforms.ColorJitter, brightness=0.5),  # 调整亮度
                       partial(transforms.CenterCrop,128)  # 不做任何处理
                       ]
    test_tfm = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor(),])
    train_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        # 对图片随机产生一种影响
        random.choice(transforms_list)(),
        transforms.ToTensor(),
    ])
    config ={
        # train parameters
        'seed':1127,
        'batch_size':64,
        'n_epochs':5,
        'patience':300,
        'learning_rate':1e-4,
        'model_path':'./models/model.ckpt'
    }

    # 由于数据集已经划分好了，直接读取数据集
    train_set = FoodDataset(os.path.join(data_path,'training'), train_tfm)
    valid_set = FoodDataset(os.path.join(data_path,'validation'), test_tfm)



    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)  # num_worker 是主动将batch加载进内存的workers数,一般设置与CPU核心数一致
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0)  # pin_memory 锁页内存 将Tensor从内存移动到GPU的速度会变快，高端设备才行

    # 内存回收 善待我们的GPU
    del train_set, valid_set
    gc.collect()

    # 运行model
    model = Classifier().to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # 计算模型的参数个数
    # print(len(list((model.parameters())))) 输出model的参数个数 本模型一共26组
    # for tensor in list(model.parameters()):
    #     # 一层conv2包括两组参数 以第一层为例 64 * 3 * 3  和 64(bias)
    #     # 一层BatchNorm2d包含两组参数 以第一层为例 64  和 64 详见 https://blog.csdn.net/bigFatCat_Tom/article/details/91619977
    #     # 一层Linear包括2组参数 以fc的第一层为例  1024*8192 和 1024(bias)
    #     # 所以对于本模型一共 4*5 + 2*3 = 26组参数需要gradient
    #     print(tensor.shape)

    pass
