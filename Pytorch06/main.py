# @ Time : 2022/4/2,15:53
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: Unconditional Generative Adversarial Network

import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader


# 固定时间种子
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
        # 模型参数
        'seed':1127,
        'batch_size':64,
        'learning_rate':1e-4,
        'n_epochs':1,
        'n_critic':1,

        # 数据参数
        'z_dim':100,
        'prefix':'./data',
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 生成DataLoader
    dataset = get_dataset(os.path.join(config['prefix'], 'faces'))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)  # len(dataloader)=1115

    # 生成模型
    generator = Generator(input_dim=config['z_dim']).to(device)  # generator.parameters 共有14组 Linear[8192, 100]\[8192, 1]
    discriminator = Discriminator(3).to(device)

    # # 运行模型
    trainer(dataloader, generator, discriminator, config, device)

