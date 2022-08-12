# @ Time   : 2022/8/11 22:20
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm.auto import tqdm

def ConvBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )

def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


def calculate_accuracy(logits, labels):
    acc = np.asarray((torch.argmax(logits, -1).cpu().numpy() == labels.cpu().numpy())).mean()
    return acc

def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()  # tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


class Classifier(nn.Module):

    def __init__(self, in_ch, n_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, n_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        # 这里就是模拟了原模型的操作 具体用途明天再看吧
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(
                x,
                params[f'conv{block}.0.weight'],
                params[f'conv{block}.0.bias'],
                params.get(f'conv{block}.1.weight'),
                params.get(f'conv{block}.1.bias'),

            )
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x


def BaseSolver(
    model,
    optimizer,
    x,
    n_way,
    k_shot,
    q_query,
    loss_fn,
    device,
    inner_train_step=1,
    inner_lr=0.4,
    train=True,
    return_labels=False,
):
    criterion, task_loss, task_acc = loss_fn, [], []
    labels = []

    for meta_batch in x:
        # 划分support set 和 query set 两者一样大
        support_set = meta_batch[: n_way * k_shot]  # [5, 1, 28, 28]
        query_set = meta_batch[n_way * k_shot:]  # [5, 1, 28, 28]

        if train:
            """ training loop """
            # 利用support set 计算当前模型的损失
            labels = create_label(n_way, k_shot).to(device)
            logits = model.forward(support_set)
            loss = criterion(logits, labels)

            task_loss.append(loss)
            task_acc.append(calculate_accuracy(logits, labels))
        else:
            """ validation / testing loop """
            # inner loop 利用support set更新模型
            fast_weights = OrderedDict(model.named_parameters())


            for inner_step in range(inner_train_step):
                # Simply training
                train_label = create_label(n_way, k_shot).to(device)
                logits = model.functional_forward(support_set, fast_weights)
                loss = criterion(logits, train_label)

                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # SGD梯度下降
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grads)
                )

            if not return_labels:
                # 利用query set 计算loss
                val_label = create_label(n_way, q_query).to(device)
                logits = model.functional_forward(query_set, fast_weights)
                loss = criterion(logits, val_label)

                task_loss.append(loss)
                task_acc.append(calculate_accuracy(logits, val_label))
            else:
                # 利用query set 测试当前参数的表现
                logits = model.functional_forward(query_set, fast_weights)
                labels.extend(torch.argmax(logits, -1).cpu().numpy())

    if return_labels:
        return labels

    batch_loss = torch.stack(task_loss).mean()
    task_acc = np.mean(task_acc)

    if train:
        # 更新模型
        model.train()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return batch_loss, task_acc

def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator, device):
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = iterator.next()  # [n_way, k_shot+q_query, 1, 28, 28]
        except StopIteration:
            iterator = iter(data_loader)
            task_data = iterator.next()
        train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
        val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)

    return torch.stack(data).to(device), iterator

def trainer(train_loader, train_iter, valid_loader, valid_iter, model, config, device):

    # 设定迭代器 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['meta_lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # 定义相关变量
    n_epochs = config['n_epochs']
    meta_batch_size = config['meta_batch_size']
    n_way = config['n_way']
    k_shot = config['k_shot']
    q_query = config['q_query']
    eval_batches = config['eval_batches']

    for epoch in range(n_epochs):
        print("Epoch %d" % (epoch + 1))
        train_meta_loss = []
        train_acc = []

        for step in tqdm(range(max(1, len(train_loader) // meta_batch_size))):
            # x [32, 10, 1, 28, 28]
            x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter, device)

            meta_loss, acc = BaseSolver(
                model,
                optimizer,
                x,
                n_way,
                k_shot,
                q_query,
                criterion,
                device
            )

            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)

        print("  Loss    : ", "%.3f" % (np.mean(train_meta_loss)), end="\t")
        print("  Accuracy: ", "%.3f %%" % (np.mean(train_acc) * 100))

        # validate
        valid_acc = []
        for step in tqdm(range(max(1, len(valid_loader) // eval_batches))):
            x, valid_iter = get_meta_batch(eval_batches, k_shot, q_query, valid_loader, valid_iter, device)

            _, acc = BaseSolver(
                model,
                optimizer,
                x,
                n_way,
                k_shot,
                q_query,
                criterion,
                device,
                inner_train_step=3,
                train=False
            )

            valid_acc.append(acc)

        print("  Validation accuracy: ", "%.3f %%" % (np.mean(valid_acc) * 100))








