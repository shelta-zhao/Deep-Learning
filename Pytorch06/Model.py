# @ Time : 2022/4/2,16:37
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import torch
import torch.nn as nn
import os
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from qqdm.notebook import qqdm
from tqdm import tqdm

def weights_init(m):
    '''
    还没看懂这个是干啥

    :param m:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass

class Generator(nn.Module):
    """
    Input : batch_size * input_dim
    Output: batch_size * 3 * 64 * 64
    """
    def __init__(self, input_dim, dim=64):
        super(Generator, self).__init__()

        # 应用DCGAN
        # 将空间池化层用卷积层替代，只需将步长stride设置为大于1的值
        # 删除全连接层
        # 采用BN层 用于卷积后归一化 可以帮助模型收敛
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                # nn.ConvTranspose2d 转置卷积，转置卷积即从低维转到高维（以前的CNN都是从高维到低维）
                # kernel_size:5 stride:2
                # padding代表先对输入进行补充 此处输入为64 * 512 * 4 * 4
                # output_padding 对输出进行补充 只加一倍（对输入的是加两倍）
                # 计算可得 output = (input-1)stride+output_padding -2padding+kernel_size
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                # Batch Normalization 对输出进行归一 加快模型收敛
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        # 线性层
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        # 卷积层
        self.layer2 = nn.Sequential(
            # 输入 [100, 512, 4, 4]
            dconv_bn_relu(dim * 8, dim * 4),  # [64, 256, 8, 8]
            dconv_bn_relu(dim * 4, dim * 2),  # [64, 128, 16, 16]
            dconv_bn_relu(dim * 2, dim),      # [64, 256, 32, 32]
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),  # [64, 3, 64, 64]
            nn.Tanh()  # 激活函数Tanh
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layer1(x)
        y = y.view(y.size(0), -1, 4, 4)  # resize过程 将100*8192的输出 变成 100 * 512 * 4 * 4
        y = self.layer2(y)
        return y

    pass

class Discriminator(nn.Module):
    """
    Input : N * 3 * 64 * 64
    Output: N
    """

    def __init__(self, input_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )

        self.layer = nn.Sequential(
            # out size = (input_dim - kernel_size + 2 * padding)/stride + 1
            # 输入为 [64, 3, 64, 64]
            nn.Conv2d(input_dim, dim, 5, 2, 2),  # [64, 64, 32, 32]
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),  # [64, 128, 16, 16]
            conv_bn_lrelu(dim * 2, dim * 4),  # [64, 256, 8, 8]
            conv_bn_lrelu(dim * 4, dim * 8),  # [64, 512, 4, 4]
            nn.Conv2d(dim * 8, 1, 4),  # [64, 1, 1, 1]
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.layer(x)
        y = y.view(-1)
        return y

    pass

def trainer(dataloader, generator, discriminator, config, device):


    # 定义损失函数和迭代器
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999))

    # 生成记录文件夹
    log_dir = os.path.join(config['prefix'], 'logs')
    ckpt_dir = os.path.join(config['prefix'], 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 定义迭代次数 最低损失率 步骤 和 自动停止阈限
    n_epochs, best_acc, steps, z_dim, batch_size = config['n_epochs'], 0, 0, config['z_dim'], config['batch_size']
    z_sample = Variable(torch.randn(100, z_dim)).cuda()

    # 定义训练循环
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            imgs = data  # data [64, 3, 64, 64]
            imgs = imgs.to(device)

            # 训练Discriminator
            # Variable 是 pytorch的两种基本类型之一（Tensor） Variable是不断变化的量，适合反向传播
            z = Variable(torch.randn(batch_size, z_dim)).to(device)   # z[64, 100]
            r_imgs = Variable(imgs).cuda()
            f_imgs = generator(z)  # 使用generator 生成图片

            # 标签 给真实的图片打1  generator生成的图片打0
            r_label = torch.ones(batch_size).cuda()  # 全1
            f_label = torch.zeros(batch_size).cuda()  # 全0

            # 模型训练
            r_logit = discriminator(r_imgs.detach())
            f_logit = discriminator(f_imgs.detach())

            # 计算discriminator的损失函数
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # 模型参数更新
            discriminator.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # 训练Generator  discriminator每更新n_critic次，generator更新一次
            if steps % config['n_critic'] == 0:
                # 生成图片
                z = Variable(torch.randn(batch_size, z_dim)).cuda()
                f_imgs = generator(z)

                # 计算generator的损失函数
                f_logit = discriminator(f_imgs)
                loss_G = criterion(f_logit, r_label)

                # 模型参数更新
                generator.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            steps += 1
            # progress_bar.set_infos({
            #     'Loss_D': round(loss_D.item(), 4),
            #     'Loss_G': round(loss_G.item(), 4),
            #     'Epoch': epoch + 1,
            #     'Step': steps,
            # })

        # 模型评价
        generator.eval()
        f_imgs_sample = (generator(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # 输出生成的图像
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 储存模型
            torch.save(generator.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

    pass




