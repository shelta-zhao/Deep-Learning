# @ Time   : 2022/7/29 20:44
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :


import os.path
import shutil
from PIL import Image
import torch
import numpy as np

def FGSM(model, x, y, loss_fn, device):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    loss = loss_fn(model(x_adv), y)
    loss.backward()

    CIFAR_STD = torch.tensor((0.202, 0.199, 0.201)).to(device).view(3, 1, 1)
    epsilon = 8 / 255 / CIFAR_STD

    # 获得梯度
    grad = x_adv.grad.detach()
    # 计算获得 adversarial sample
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv

def I_FGSM(model, x, y, loss_fn, device, num_iter=20):
    CIFAR_STD = torch.tensor((0.202, 0.199, 0.201)).to(device).view(3, 1, 1)
    epsilon = 8 / 255 / CIFAR_STD
    alpha = 0.8 / 255 / CIFAR_STD

    x_adv = x
    # 迭代循环
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x+epsilon), x - epsilon)

    return x_adv

def gen_adv_examples(model, loader, attack, loss_fn, device, std, mean):
    model.eval()
    train_acc, train_loss = 0.0, 0.0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn, device)
        y_pred = model(x_adv)

        loss = loss_fn(y_pred, y)
        train_acc += (y_pred.argmax(dim=1) == y).sum().item()   # item可以得到一个纯粹的数值
        train_loss += loss.item() * x.shape[0]

        # 保存adversarial sample
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)  # clamp函数可以将值限定在指定范围内
        adv_ex = (adv_ex * 255).clamp(0, 255)
        adv_ex = adv_ex.detach().cpu().numpy().round()  # round可以去除小数部分
        adv_ex = adv_ex.transpose((0, 2, 3, 1))
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]

    return adv_examples, train_acc/len(loader.dataset), train_loss/len(loader.dataset)

def save_img(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        # 将data_dir的文件目录复制到adv_dir中
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        img = Image.fromarray(example.astype(np.uint8))
        img.save(os.path.join(adv_dir, name))
    pass



