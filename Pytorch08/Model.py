# @ Time : 2022/7/19,14:19
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:


import torch.nn as nn
import torch
import os
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from Dataset import show_figure
import pandas as pd

class FCN_AutoEncoder(nn.Module):

    def __init__(self):
        super(FCN_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64 * 3),
            # Tanh是双曲正切函数激活函数 Tanh = 2 * sigmoid(2 * x) - 1
            # Tanh 是奇函数，在I、III象限严格单调递增，其图像在[-1, 1]之间，可以提高收敛速度
            nn.Tanh()
        )
        pass

    def forward(self, img):

        img = self.encoder(img)
        img = self.decoder(img)
        return img

class Conv_AutoEncoder(nn.Module):

    def __init__(self):
        super(Conv_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        pass

    def forward(self, img):
        img = self.encoder(img)
        img = self.decoder(img)
        return img

class VAE(nn.Module):
    # VAE:Variational Autoencoder 变分自编码器
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        pass

    def encode(self, img):
        h1 = self.encoder(img)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

def trainer(model, train_loader, config, device):

    # 设置迭代器和损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 定义相关参数
    best_loss = np.inf
    n_epochs = config['n_epochs']

    # 定义训练过程
    step = 0
    for epoch in range(n_epochs):
        model.train()
        train_loss = []

        for data in tqdm(train_loader):
            img = data.float().to(device)  # img [2000, 3, 64, 64]
            if config['model_type'] in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)

            if config['model_type'] == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            train_loss.append(loss.item())

            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # 展示生成的图像
            if step % 100 == 0:
                show_figure(output[0].reshape((3, 64, 64)).permute(1, 2, 0).detach().numpy())

        mean_loss = np.mean(train_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, f'best_mode_{config["model_type"]}.pt')

        print(f'{epoch +1:.0f}/{n_epochs:.0f}: {mean_loss:.4f}')
    pass


def anomaly_detection(test_loader, model_type, device):

    eval_loss = nn.MSELoss(reduction='none')

    # 加载模型
    checkpoint_path = f'best_mode_{model_type}.pt'
    model = torch.load(checkpoint_path)
    model.eval()

    # 预测结果
    out_file = 'prediction.csv'

    # 开始预测
    anomality = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img = data.float().to(device)
            if model_type in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)
            if model_type in ['vae']:
                output = output[0]
            if model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])

            anomality.append(loss)

    # 将损失值开平方后的结果作为判断的依据
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(19636, 1).cpu().numpy()

    df = pd.DataFrame(anomality, columns=['score'])
    df.to_csv(out_file, index_label='ID')
    pass
