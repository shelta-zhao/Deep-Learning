# @ Time : 2022/3/21,10:22
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: This file defines the structure of the model


import torch
import os
import math
from tqdm import *
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义模型
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
        pass

    def forward(self, x):
        """
        python call函数会调用forward 所以model.(data) = model.forward(data)
        :param x:
        :return:
        """
        x = self.layers(x)
        x = x.squeeze(1)  # 只保留第0维
        return x

    pass

# 定义训练循环
def trainer(train_loader, valid_loader, model, config, device):
    """
    Define train loop
    :param train_loader:按batch分割的训练集
    :param valid_loader:按batch分割的验证集
    :param model:模型
    :param config:模型参数
    :param device:CPU or GPU
    :return:
    """

    # 定义损失函数 迭代器 记录
    criterion = nn.MSELoss(reduction='mean')  # 定义损失函数为MSE mean代表loss会除以elements个数
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)  # momentum 定义惯性，加速收敛
    writer = SummaryWriter()  # 自动生成文件夹名称"runs"，以及随机的events名称

    # 定义model文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')  # 创建model文件夹

    # 迭代次数 最低损失率 步骤
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    # 定义迭代
    for epoch in range(n_epochs):
        model.train()  # 进入训练模式  model.eval()为评价模式
        loss_record = []  # 损失函数

        # tqdm is a packet to visualize the training progress
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x,y in train_pbar:
            optimizer.zero_grad()  # 数值清零 防止前序计算影响
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 更新参数
            step += 1
            loss_record.append(loss.detach().item())
            # Display current epoch number and loss on tqdm progress bar
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss':loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train',mean_train_loss, step)

        model.eval()
        loss_record = []

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

    pass