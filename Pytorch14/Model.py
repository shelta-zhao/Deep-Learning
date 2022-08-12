# @ Time   : 2022/8/8 14:44
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :
import numpy as np
import torch.nn as nn
import torch.optim.optimizer
import tqdm

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1*28*28, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10),
            nn.ReLU()
        )

    def forward(self, img):
        img = img.view(-1, 1 * 28 * 28)
        return self.fc(img)


# 基础方法
class Baseline:
    # Select Synaptic Plasticity
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # 获取模型的所有参数

        self.params_old = {}
        for n, p in self.params.items():
            self.params_old[n] = p.clone().detach()

        self._precision_matrices = self._calculate_importance()

    def _calculate_importance(self):
        precision_matrices = {}
        # 用0填充矩阵, 可以改用1对比结果
        for n,p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        return precision_matrices

    def penalty(self, model:nn.Module):
        # 定义新的loss L'(θ) = L(θ) + a * ∑ b_i *(θ - θ^b)^2
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.params_old[n]) ** 2
            loss += _loss.sum()

        return loss

    def update(self, model):
        return


def evaluate(model, test_dataloader, device):
    # 计算模型的准确率acc
    model.eval()
    correct_cnt = 0
    total = 0

    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        y_pred = model(imgs)

        _, label_pred = torch.max(y_pred.data, 1)

        correct_cnt += (label_pred == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()

    return correct_cnt/total

def trainer(train_dataloader, test_dataloaders, optimizer, lll_obj, lll_lambda, model, device):

    # 定义损失函数和迭代器
    criterion = nn.CrossEntropyLoss()

    # 定义训练变量
    acc_list = []
    loss = 1.0

    bar = tqdm.auto.trange(10, leave=False, desc=f'Epoch 1, Loss: {loss:.7f}')

    # 定义迭代
    for epoch in bar:
        model.train()

        for imgs, labels in tqdm.auto.tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            y_pred = model(imgs)
            loss = criterion(y_pred, labels)
            lll_loss = lll_obj.penalty(model)
            total_loss = loss + lll_lambda * lll_loss

            lll_obj.update(model)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss += total_loss.item()
            bar.set_description_str(desc=f'Epoch {epoch+1:2}, Loss: {loss:.7f}', refresh=True)

        acc_average = []
        for test_dataloader in test_dataloaders:
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)

        acc_list.append(np.mean(np.array(acc_average)) * 100.0)
        bar.set_description_str(desc=f'Epoch {epoch + 2:2}, Loss: {loss:.7f}', refresh=True)

    return model, optimizer, acc_list

