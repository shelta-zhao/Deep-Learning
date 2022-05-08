# @ Time : 2022/3/24,11:16
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import torch
import torch.nn as nn
import os
import math
from tqdm import *

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            # nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=0...)
            # 默认kernel是正方形 所以下面这行代码代表 一共 3*3*3的kernel 有64个 padding为1代表上下左右各填充一行
            # 输入的img 是 128*128*3 ， 经过padding变成 130*130*3 所以经过kernel卷积之后 变成 128*128*64
            # nn.BatchNorm2d(num_features, eps=1e-5, momentum==0.1,affine=True,...) 对数据做归一化处理，使其分布均匀，防止梯度消失
            # nn.MaxPool2d(kernel_size, stride, padding,...)最大池化，减小图像尺寸，提高训练速度,变成64 * 64 * 64
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        # 全连接层 FullConnected Layer
        self.fc = nn.Sequential(
            # 输入是512 * 4 * 4 ,把这个Tensor拉直成一个向量，作为输入
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,11)
        )

    # forward相当于定义训练过程
    def forward(self, x):
        out = self.cnn(x)
        # view相当于reshape 把输出拉成一个vector, -1代表自动计算相应长度保证总体元素个数不变
        # out.size()[0]是batch大小 相当于将batch中的每一个sample的特征拉成一个vector 用作全连接层的输入 也可以用flatten函数
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    pass


def trainer(train_loader, valid_loader, model, config, device):

    # 定义损失函数Cross-Entropy和优化器AdamW
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)  # weight_decay就是正则项系数

    # 定义model文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # 定义迭代次数 最低损失率 步骤 和 自动停止阈限
    n_epochs, best_acc, patient = config['n_epochs'], 0, 0

    # 定义迭代
    for epoch in range(n_epochs):

        # training
        model.train()  # 设置model为训练模式

        train_loss_record = []  # 记录损失函数的值
        train_acc_record = []  # 记录准确率

        for i,batch in enumerate(tqdm(train_loader)):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 梯度下降五步走 + 1
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # 剪裁梯度范数以进行稳定训练。
            optimizer.step()

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
            for i,batch in enumerate(valid_loader):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                # 预测
                preds = model(imgs)
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

        # 记录最好模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), config['model_path'])
            print(f"Best model found at epoch {epoch}, best acc{best_acc},saving model")
            patient = 0
        else:
            patient += 1
            if patient > config['patience']:
                print(f"No improvement {patient} consecutive epochs, early stopping")
                break
        pass
    pass



if __name__ == '__main__':
    v1 = torch.range(1,16)
    v2 = v1.view(4, -1)
    print(v1)  # 1 * 16
    print(v2)  # 4 * 4



