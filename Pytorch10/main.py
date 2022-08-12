# @ Time   : 2022/7/27 14:47
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :


import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader
from Model import gen_adv_examples, FGSM, I_FGSM, save_img
from Dataset import AdvDataset

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pass

def epoch_benign(model, data_loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_acc += (y_pred.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]

    return train_acc/len(data_loader.dataset), train_loss/len(data_loader.dataset)


if __name__ == '__main__':

    # 模型配置信息
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CIFAR_MEAN = torch.tensor((0.491, 0.482, 0.447)).to(device).view(3, 1, 1)
    CIFAR_STD = torch.tensor((0.202, 0.199, 0.201)).to(device).view(3, 1, 1)
    epsilon = 8/255/CIFAR_STD
    alpha = 0.8/255/CIFAR_STD
    config = {
        'seed': 1127,
        'batch_size': 8,
    }

    # 对图像进行transform处理，先转化为tensor，再进行normalize
    transform = transforms.Compose([
        # ToTensor 可以将[0, 255] 转化为[0, 1], 还将(H * W * C)转化为(C * H * W)
        transforms.ToTensor(),
        # 归一化处理
        transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
    ])

    # 固定时间种子
    same_seed(config['seed'])

    # 生成数据集
    adv_set = AdvDataset('./data', transform=transform)
    adv_names = adv_set.__getname__()
    adv_loader = DataLoader(adv_set, batch_size=config['batch_size'], shuffle=False)

    # 在benign images上测试pretrained模型的表现 benign image 初始数据/adversarial images 攻击数据
    model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()

    benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
    print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')  # benign_acc = 0.95000, benign_loss = 0.22678

    # FGSM/I_FGSM 攻击
    adv_examples, FGSM_acc, FGSM_loss = gen_adv_examples(model, adv_loader, FGSM, loss_fn, device, CIFAR_STD, CIFAR_MEAN)
    print(f'FGSM_acc = {FGSM_acc:.5f}, FGSM_loss = {FGSM_loss:.5f}')
    save_img('./data', 'FGSM', adv_examples, adv_names)  # FGSM_acc = 0.59000, FGSM_loss = 2.49236

    adv_examples, I_FGSM_acc, I_FGSM_loss = gen_adv_examples(model, adv_loader, I_FGSM, loss_fn, device, CIFAR_STD, CIFAR_MEAN)
    print(f'I_FGSM_acc = {I_FGSM_acc:.5f}, I_FGSM_loss = {I_FGSM_loss:.5f}')
    save_img('./data', 'I_FGSM', adv_examples, adv_names)  # I_FGSM_acc = 0.00500, I_FGSM_loss = 17.37221



