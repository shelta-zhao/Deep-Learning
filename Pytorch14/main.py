# @ Time   : 2022/8/8 13:46
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :


import torch
import numpy as np
from torch.utils.data import DataLoader
from Dataset import Data
from Model import Model, trainer, Baseline
import tqdm

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
        'batch_size':128,
        'lr':1e-4,
        'test_size':8192,
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 生成数据集
    angle_list = [20 * x for x in range(5)]  # 将不同的旋转角度对应不同的任务，一共五个task

    train_datasets = [Data('data', angle=angle_list[index]) for index in range(5)]
    train_dataloaders = [DataLoader(train_dataset.dataset, batch_size=config['batch_size'], shuffle=True) for train_dataset in train_datasets]

    test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(5)]
    test_dataloaders = [DataLoader(test_dataset.dataset, batch_size=config['test_size'], shuffle=True) for test_dataset in test_datasets]


    # 建立模型
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    lll_obj = Baseline(model=model, dataloader=None, device=device)
    lll_lambda = 0.0
    baseline_acc = []
    task_bar = tqdm.auto.trange(len(train_dataloaders), desc="Task  1")

    # 训练模型
    for train_indexes in task_bar:
        model, _, acc_list = trainer(train_dataloaders[train_indexes], test_dataloaders, optimizer, lll_obj, lll_lambda, model, device)

        lll_obj = Baseline(model=model, dataloader=train_dataloaders[train_indexes], device=device)

        optimizer = torch.optim.Adam(model.parameters(), config['lr'])

        baseline_acc.extend(acc_list)

        task_bar.set_description_str(f'Task  {train_indexes+2:2}')

    print(baseline_acc)


