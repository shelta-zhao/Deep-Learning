# @ Time   : 2022/8/7 14:31
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :



import numpy as np
import torch
import os
from torchvision.transforms import transforms
from Dataset import FoodDataset
from torch.utils.data import DataLoader
import gc
from Model import StudentNet, trainer, loss_fn_kd
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pass

def log(text, save_path):
    # 记录模型参数
    log_fw = open(f'{save_path}/log.txt', 'w')
    log_fw.write(str(text)+'\n')
    log_fw.flush()
    pass

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed':1127,
        'data_path':'./data',
        'save_dir':'./outputs',
        'exp_name':'simple_baseline',

        'batch_size':64,
        'lr':3e-4,
        'weight_decay':1e-5,
        'grad_norm_max':10,
        'n_epochs':10,
        'patience':300
    }

    # 记录模型参数
    save_path = os.path.join(config['save_dir'], config['exp_name'])
    os.makedirs(save_path, exist_ok=True)
    log(config, save_path)

    # 固定时间种子
    same_seed(config['seed'])

    # 生成数据集
    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 从图像中心向两边裁剪，裁剪后的大小为224 * 224
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = FoodDataset(os.path.join(config['data_path'], 'training'), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(config['data_path'], 'validation'), tfm=train_tfm)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=False)

    del train_set, valid_set
    gc.collect()

    # 加载模型
    teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
    teacher_model.load_state_dict(torch.load('./resnet18_teacher.ckpt', map_location='cpu'))

    # 训练模型
    student_model = StudentNet().to(device)
    teacher_model.to(device)

    trainer(train_loader, valid_loader,student_model, teacher_model, config, device)
