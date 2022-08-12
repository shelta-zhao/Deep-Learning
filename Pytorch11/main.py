# @ Time   : 2022/7/31 20:16
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :  DaNN Domain Adversarial Training of Neural NetWorks


import torch
import cv2
import gc
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Model import FeatureExtractor, LabelPredictor, DomainClassifier, trainer, predictor

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
        'seed': 1127,
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 生成数据集
    source_transform = transforms.Compose([
        transforms.Grayscale(),  # 转化为灰度图，因为cv2没法处理RGB图片
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),  # Canny函数用于边缘检测
        transforms.ToPILImage(),  # 将np array 转回 skimage.Image
        transforms.RandomHorizontalFlip(),  # Augmentation 水平翻转
        transforms.RandomRotation(15, fill=(0,)),  # Augmentation  随机旋转
        transforms.ToTensor()  # 转化为Tensor
    ])
    target_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),  # 将target image 从28 * 28 转化为 32 * 32
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        transforms.ToTensor()
    ])

    source_dataset = ImageFolder('./real_or_drawing/train_data', transform=source_transform)
    target_dataset = ImageFolder('./real_or_drawing/test_data', transform=target_transform)

    source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    del source_dataset, target_dataset
    gc.collect()

    # 建立模型
    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)

    # 训练
    trainer(feature_extractor, label_predictor, domain_classifier, source_dataloader, target_dataloader, device)

    # 测试
    predictor(feature_extractor, label_predictor, test_dataloader)


