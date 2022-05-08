# @ Time : 2022/3/22,20:34
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: This file is used to define the model

import torch
import torch.nn as nn
import os
import math
from tqdm import *

class BasicBlock(nn.Module):
    # 定义一个包括一个线性层和ReLU的基本单元
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x

    pass


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        # 第二行的写法可以动态增加隐藏层的层数
        # 没有定义Softmax是因为Pytorch中将损失函数Cross-entropy与Softmax绑定
        self.fc = nn.Sequential(
            BasicBlock(input_dim,hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    pass


# 定义训练trainer
def trainer(train_loader, valid_loader, model, config, device):
    # 定义损失函数Cross-Entropy，优化器AdamW
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 定义model文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 定义迭代次数 最低损失率 步骤 和 自动停止阈限
    n_epochs, best_loss, early_stop_count = config['n_epochs'], math.inf, 0

    # 定义迭代
    for epoch in range(n_epochs):
        train_len = 0
        valid_len = 0

        # training
        model.train()  # 设置model为训练模式

        train_loss_record = []  # 记录损失函数的值
        train_acc_record = []  # 记录准确率

        # 定义每个batch的训练
        for i, batch in enumerate(tqdm(train_loader)):
            # batch是一个list 其中batch[0]为512*39的一个datas, batch[1]是512*1的labels
            datas, labels = batch
            datas = datas.to(device)
            labels = labels.to(device)

            # 梯度下降五步走
            optimizer.zero_grad()  # 梯度置零
            preds = model(datas)  # 模型计算  512 * 41
            loss = criterion(preds, labels)  # 计算损失函数
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 记录数据
            _, train_pred = torch.max(preds, 1)  # 对softmax的结果preds 找出每列最大值的索引 即预测结果
            train_loss_record.append(loss.detach().item())
            train_acc_record.append((train_pred.detach() == labels.detach()).sum().item())
            train_len += len(labels)
            pass

        # validating
        model.eval()
        valid_loss_record = []  # 记录损失函数的值
        valid_acc_record = []  # 记录准确率

        # 用with这个写法更好
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader)):
                datas, labels = batch
                datas = datas.to(device)
                labels = labels.to(device)

                # 模型预测
                preds = model(datas)
                loss = criterion(preds, labels)

                # 记录数据
                _, valid_pred = torch.max(preds, 1)  # 对softmax的结果preds 找出每列最大值的索引 即预测结果
                valid_loss_record.append(loss.detach().item())
                valid_acc_record.append((valid_pred.detach() == labels.detach()).sum().item())
                valid_len += len(labels)

            # 输出epoch结果
            train_acc = sum(train_acc_record)/train_len
            valid_acc = sum(valid_acc_record)/valid_len
            train_loss = sum(train_loss_record)/ len(train_loss_record)
            valid_loss = sum(valid_loss_record) / len(valid_loss_record)
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Valid Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_acc,train_loss,valid_acc,valid_loss
            ))

        # 记录模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), config['model_path'])  # 保存valid_loss最低的模型
            print('Saving model with loss {:.3f}...'.format(best_loss))

    pass








