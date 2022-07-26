# @ Time : 2022/7/18,20:37
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Dataset import HumanDataset
from Model import FCN_AutoEncoder, Conv_AutoEncoder, VAE, trainer, anomaly_detection
import numpy as np
import gc


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pass



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 1127,
        'n_epochs':50,
        'batch_size':2000,
        'learning_rate':1e-3,
        'model_type':'fcn',  # fcn/conv/vae
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 读取数据集
    train_set = np.load('./data/trainingset.npy', allow_pickle=True)
    train_tensor = torch.from_numpy(train_set)
    train_dataset = HumanDataset(train_tensor)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['batch_size'])

    test_set = np.load('./data/testingset.npy', allow_pickle=True)
    test_tensor = torch.tensor(test_set, dtype=torch.float32)
    test_dataset = HumanDataset(test_tensor)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)


    # 清除内存
    del train_set
    gc.collect()

    # 模型设置
    model_classes = {'fcn':FCN_AutoEncoder(), 'conv':Conv_AutoEncoder(), 'vae': VAE()}
    model = model_classes[config['model_type']].to(device)

    # 模型训练
    # trainer(model, train_loader, config, device)

    # 异常检测
    anomaly_detection(test_loader, config['model_type'], device)
