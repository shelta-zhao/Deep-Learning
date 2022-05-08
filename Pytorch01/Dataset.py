# @ Time : 2022/3/21,10:21
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: This file is used to define data

import numpy as np
import torch
import tqdm as tqdm
from torch.utils.data import Dataset, random_split

# 定义四个操作函数
def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility
    固定时间种子。由于cuDNN会自动从几种算法中寻找最适合当前配置的算法，为了使选择的算法固定，所以固定时间种子
    :param seed: 时间种子
    :return: None
    """
    torch.backends.cudnn.deterministic = True  # 解决算法本身的不确定性，设置为True 保证每次结果是一致的
    torch.backends.cudnn.benchmark = False  # 解决了算法选择的不确定性，方便复现，提升训练速度
    np.random.seed(seed)  # 按顺序产生固定的数组，如果使用相同的seed，则生成的随机数相同， 注意每次生成都要调用一次
    torch.manual_seed(seed)  # 手动设置torch的随机种子，使每次运行的随机数都一致
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为GPU设置唯一的时间种子
        pass
    pass

def train_valid_split(data_set, valid_ratio, seed):
    """
    Split provided training data into training set and validation set
    将数据集随机分割为 训练集 和 验证集
    :param data_set: 源数据
    :param valid_ratio: 验证集比例
    :param seed: 时间种子
    :return: 训练集 验证集
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set,[train_set_size,valid_set_size],generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def select_feature(train_data, valid_data, test_data, select_all=True):
    """
    Select useful features to perform regression
    :param train_data: 训练集
    :param valid_data: 验证集
    :param test_data: 测试集
    :param select_all: 选择前五行还是全部数据
    :return:
    """
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    row_x_train, row_x_valid, row_x_test = train_data[:,:-1], valid_data[:,:-1], test_data
    if select_all:
        feat_idx = list(range(row_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4]  # 选择前五行数据
    return row_x_train[:,feat_idx], row_x_valid[:,feat_idx], row_x_test[:,feat_idx], y_train, y_valid

def predict(test_loader, model, device):
    """
    Use model to predict the task based on the test_set
    在测试集上完成预测
    :param test_loader:
    :param model: 模型
    :param device: CPU or GPU
    :return: preds 预测结果
    """
    model.eval()  # 将模型调整为评价模式
    preds = []
    # tqdm是进度条库 可以在python长循环中添加进度提示信息
    for x in tqdm(test_loader):
        x = x.to(device)
        # torch.no_grad 停止梯度下降 防止继续更新模型
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu()) # detach阻断反向传播，并把数值转到CPU中，此处只需要记录数据即可
            pass
        pass
    preds = torch.cat(preds, dim=0).numpy()  # 这里为什么要用cat呢
    return preds

# 定义dataset类
class COVIDDataset(Dataset):
    """
    x: Features
    y: Target, if none, do prediction
    """
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)  # 转化为Tensor类型 单精度浮点数
        self.x = torch.FloatTensor(x)
        pass

    def __getitem__(self, item):
        """
        Return one sample at one time
        每次返回一个样本
        :param item:
        :return: 返回一个样本
        """
        if self.y is None:
            return self.x[item]
        else:
            return self.x[item],self.y[item]

    def __len__(self):
        return len(self.x)

    pass