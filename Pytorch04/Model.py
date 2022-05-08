# @ Time : 2022/3/27,22:06
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_speakers=600, dropout=0.1):
        super().__init__()
        # 设置前置网络prenet 将feature变为80
        self.prenet = nn.Linear(40, d_model)
        # TODO：
        # Change Transformer to Conformer
        #
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_speakers),
        )
        pass

    def forward(self, mels):
        # mels 32 * 10 * 40
        out = self.prenet(mels)  # out:32 * 10 * 80
        out = out.permute(1, 0, 2)  # permute 将tensor维度换位  变为10 * 32 * 80
        out = self.encoder_layer(out)  # 10 * 32 * 80
        out = out.transpose(0, 1)  # 32 * 10 * 80
        stats = out.mean(dim=1)  # stats 32 * 80  对batch的每个数据的长度求平均值

        # 预测
        out = self.pred_layer(stats)  # 32 * 600
        return out

# 定义学习率时间表
# 对于 Transformer 架构，学习率调度的设计与 CNN 不同。
# 以前的工作表明，学习率的预热对于训练具有变压器架构的模型很有用。
# 详见 https://zhuanlan.zhihu.com/p/410971793
def get_conine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1,
                                    ):
    """
    创建一个学习率随着优化器中设置的初始 lr 到 0 之间的余弦函数值而减小的计划，在预热期之后，它在 0 和优化器中设置的初始 lr 之间线性增加。
    :param optimizer: 为其安排学习率的优化器。
    :param num_warmup_steps:预热阶段的步数。
    :param num_training_steps:训练步骤的总数。
    :param num_cycles:余弦调度中的波数（默认值是从最大值减少到 0
    :param last_epoch:恢复训练时最后一个 epoch 的索引
    :return:学习了时间表
    """
    def lr_lambda(current_step):
        # 预热 Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 下降 Decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def trainer(train_loader, valid_loader, model, config, device):
    # 定义损失函数、迭代器、学习表
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_conine_schedule_with_warmup(optimizer, config['warmup_steps'], config['n_epochs'])

    # 定义models文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 定义迭代次数 最低损失率 步骤 和 自动停止阈限
    n_epochs, best_acc, patient = config['n_epochs'], 0, 0

    # 定义迭代
    for epoch in range(n_epochs):
        # training
        model.train()

        train_loss_record = []  # 记录损失函数的值
        train_acc_record = []  # 记录准确率

        for i, batch in enumerate(tqdm(train_loader)):
            mels, labels = batch  # mels 32 * 10 * 40
            mels = mels.to(device)
            labels = labels.to(device)

            # 梯度下降五步走
            optimizer.zero_grad()
            preds = model(mels)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 输出当前的学习率
            # print(optimizer.state_dict()['param_groups'][0]['lr'])  可见实现了先上升后缓步下降的过程

            # 记录数据
            acc = (preds.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss_record.append(loss.item())
            train_acc_record.append(acc)

        train_loss = sum(train_loss_record) / len(train_loss_record)
        train_acc = sum(train_acc_record) / len(train_acc_record)

        # validating
        model.eval()
        valid_loss_record = []  # 记录损失函数的值
        valid_acc_record = []  # 记录准确率

        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                mels, labels = batch
                mels = mels.to(device)
                labels = labels.to(device)

                # 预测
                preds = model(mels)
                loss = criterion(preds, labels)

                # 记录数据
                acc = (preds.argmax(dim=-1) == labels.to(device)).float().mean()
                valid_loss_record.append(loss.item())
                valid_acc_record.append(acc)

            valid_loss = sum(valid_loss_record) / len(valid_loss_record)
            valid_acc = sum(valid_acc_record) / len(valid_acc_record)

        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Valid Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_acc, train_loss, valid_acc, valid_loss
        ))

        # 记录最好模型 每1000次考虑一次
        if valid_acc > best_acc and (epoch + 1) % config['save_steps'] == 0:
            best_acc = valid_acc
            torch.save(model.state_dict(), config['model_path'])
            print(f"Best model found at epoch {epoch + 1}, best acc{best_acc},saving model")
            patient = 0
        else:
            patient += 1
            if patient > config['patience']:
                print(f"No improvement {patient} consecutive epochs, early stopping")
                break
        pass
    pass
