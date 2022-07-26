# @ Time : 2022/7/22,17:47
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:


import torch.nn as nn
import torch
from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from FoodDataset import FoodDataset, get_paths_labels
from torch.autograd import Variable


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(input_dim, output_dim):
            # 一个卷积层block， 包括conv、batch_norm、 ReLU
            return [
                nn.Conv2d(input_dim, output_dim, 3, 1, 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
            ]

        def stack_blocks(input_dim, output_dim, block_num):
            # 生成stack block 包含block_num 个卷积层
            layers = building_block(input_dim, output_dim)
            for i in range(block_num - 1):
                layers += building_block(output_dim, output_dim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 128, 3)
        cnn_list += stack_blocks(128, 128, 3)
        cnn_list += stack_blocks(128, 256, 3)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            # nn.Dropout是为了防止过拟合而设置，p=0.3代表每个神经元都有0.3的概率不被激活
            nn.Dropout(p=0.3),
            nn.Linear(1024, 11),
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size()[0], -1)
        return self.fc(out)


# LIME分析法：局部可解释性分析，通过用简单的模型来模拟复杂模型的行为
# 过程:
#      1. 选择给定的数据样本
#      2. 将样本进行segment分割，分割的数量为M
#      3. 随机拿走一些segment，通过这种方式可以产生一个新的数据集
#      4. 进行特征降维，得到特征x = {1， segment被拿走/ 0， segment保留} 易知x的长度为M
#      5. 以x为输入，利用Linear模型拟合，得到M的权重w，若w≈0，则代表这一个seg不重要，若w>0，则代表seg很重要，w<0, 则与预测结果负相关

def predict(input):
    # 根据模型得到预测结果
    # 输入: [batch_size, height, weight, channels]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    output = model(input.to(device))
    return output.detach().cpu().numpy()

def segmentation(input):
    # 按照LIME的方法，首先对input进行切片
    # compactness 浮点数的使用
    # start_label 索引开始数值
    return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)

def LIME(images, labels):

    # 固定随机种子
    np.random.seed(1127)

    # 绘制分析结果图
    fig, axs = plt.subplots(1, 10, figsize=(15, 8))

    # LIME 分析
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):  # zip函数 返回一个元组列表
        x = image.astype(np.double)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        lime_img, mask = explanation.get_image_and_mask(
            label=label.item(),
            positive_only=False,
            hide_rest=False,
            num_features=11,
            min_weight=0.05
        )

        # 绘图
        axs[idx].imshow(lime_img)

    plt.show()
    plt.close()
    pass


# Saliency Map : 当改变图像的像素值时，损失到图像的偏微分值显示了损失的变化。 可以说它意味着像素的重要性。

def normalize(img):
    # 最大最小值归一化
    return (img - img.mean()) / (img.max() - img.min())

def compute_saliency_maps(x, y, model):

    model.eval()
    x = x.cuda()
    y = y.cuda()

    # 需要在计算中保留对应的梯度信息
    x.requires_grad_()

    # 梯度下降
    y_preds = model(x)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_preds, y)
    loss.backward()

    # saliencies = x.grad.abs().detach().cpu()
    saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(), dim=1)  # torch.max(, dim=1)返回tensor中每一行的最大值
    # x.grad() [10, 3, 128, 128]  saliencies[10, 128, 128]

    # 归一化， 因为saliencies的梯度可能在不同的scale
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

def Saliency_Maps(images, labels, model):

    saliencies = compute_saliency_maps(images, labels, model)

    # 可视化
    fig, axs = plt.subplots(2, 10, figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            if row == 0:
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
            else:
                axs[row][column].imshow(img.numpy(), cmap=plt.cm.hot)

    plt.show()
    plt.close()
    pass

def compute_smooth_grad(image, lable, model, epoch, param_sigma_multiplier):

    model.eval()

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(image) - torch.min(image)).item()
    smooth = np.zeros(image.cuda().unsqueeze(0).size())  # [1, 3, 128, 128]  unsqueeze函数起升维作用

    #
    for i in range(epoch):
        # 每一个图片计算了epoch组，再进行求平均
        noise = Variable(image.data.new(image.size()).normal_(mean, sigma**2))  # 随机生成噪声数据
        x_mod = (image + noise).unsqueeze(0).cuda()
        x_mod.requires_grad_()

        y_preds = model(x_mod)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(y_preds, lable.cuda().unsqueeze(0))
        loss.backward()

        smooth += x_mod.grad.abs().detach().cpu().data.numpy()

    # 标准化
    smooth = normalize(smooth / epoch)
    return smooth

def Smooth_Grad(images, labels, model):
    smooth = []

    for image, label in zip(images, labels):
        smooth.append(compute_smooth_grad(image, label, model, 500, 0.4))
    smooth = np.stack(smooth)  # 连接每个图片的smooth，连接后大小为[10, 1, 3, 128, 128] 

    # 绘图
    fig, axs = plt.subplots(2, 10, figsize=(15, 8))
    for row, target in enumerate([images, smooth]):
        for column, img in enumerate(target):
            axs[row][column].imshow(np.transpose(img.reshape(3, 128, 128), (1, 2, 0)))

    plt.show()
    pass


if __name__ == '__main__':

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier().to(device)
    checkpoint = torch.load('./data/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载数据
    train_paths, train_labels = get_paths_labels('./data/food')
    train_set = FoodDataset(train_paths, train_labels, mode='eval')

    images, labels = train_set.getbatch([i for i in range(10)])  # images[10, 3, 128, 128]

    # 查看图片
    # fig, axs = plt.subplots(1, 10, figsize=(15, 8))
    # for i, img in enumerate(images):
    #     axs[i].imshow(img.cpu().permute(1, 2, 0))
    #     pass
    # plt.show()

    # LIME 分析
    # LIME(images, labels)

    # Saliency Map
    # Saliency_Maps(images, labels, model)

    # Smooth Grad
    Smooth_Grad(images, labels, model)

