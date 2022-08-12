# @ Time   : 2022/8/11 17:23
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :


import torch
import numpy as np
import gc
from Dataset import Omniglot,dataloader_init
from Model import Classifier, trainer


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pass



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed':1127,
        'n_way':5,
        'k_shot':1,
        'q_query':1,

        'train_inner_train_step':1,
        'val_inner_train_step':3,
        'inner_lr':0.4,

        'meta_lr':1e-3,
        'meta_batch_size':32,
        'n_epochs':5,

        'eval_batches':20,
        'train_data_path':'./Omniglot/images_background/'
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 生成数据集
    train_set, valid_set = torch.utils.data.random_split(
        Omniglot(config['train_data_path'], config['k_shot'], config['q_query'], task_num=10),
        #  [5, 5]代表将数据按照1：1划分训练集和验证集
        [5, 5]
    )
    (train_loader, valid_loader), (train_iter, valid_iter) = dataloader_init((train_set, valid_set), config['n_way'], shuffle=False)

    del train_set, valid_set
    gc.collect()

    # 生成模型
    model = Classifier(1, config['n_way']).to(device)
    trainer(train_loader, train_iter, valid_loader, valid_iter, model, config, device)
