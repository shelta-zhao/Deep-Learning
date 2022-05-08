# @ Time : 2022/4/3,12:27
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:

import torch
import os
import torchvision.utils
from Model import Generator
from torch.autograd import Variable
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 加载模型
    generator = Generator(100)
    generator.load_state_dict(torch.load(os.path.join('./data/checkpoints', 'G.pth')))
    generator = generator.cuda()
    generator.eval()

    # 生成图片
    n_output = 64
    z_sample = Variable(torch.randn(n_output, 100)).cuda()
    print(z_sample.shape)
    imgs_sample = (generator(z_sample).data + 1)/2

    # 展示前40个图片
    grid_img = torchvision.utils.make_grid(imgs_sample[:40].cpu(), nrow=8)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

