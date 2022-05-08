# @ Time : 2022/3/22,20:16
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: The purpose of this project is to perform framewise phoneme classification using pre-extracted MFCC features



import gc
import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader


# 定义函数固定时间种子 保证每次实验结果一致
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        pass
    pass




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        # data parameters
        'concat_nframes':1,
        'train_ratio':0.8,

        # train parameters
        'seed':1127,
        'batch_size':512,
        'n_epochs':5,
        'learning_rate':1e-4,

        # model parameters
        'input_dim':39,
        'hidden_layers':1,
        'hidden_dim':256,
        'model_path': './models/model.ckpt'
    }

    # 设置时间种子
    same_seed(config['seed'])

    # 划分数据集  每个数据39个特征 对应分类为41种
    x_train, y_train = preprocess_data(split='train', feat_dir='./data/libriphone/feat', phone_path='./data/libriphone',concat_nframes=config['concat_nframes'], train_ratio=config['train_ratio'])
    x_valid, y_valid = preprocess_data(split='val', feat_dir='./data/libriphone/feat', phone_path='./data/libriphone', concat_nframes=config['concat_nframes'], train_ratio=config['train_ratio'])
    train_set, valid_set = LibriDataset(x_train, y_train), LibriDataset(x_valid, y_valid)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True)

    # 内存回收 善待我们的GPU
    del x_train,y_train,x_valid,y_valid
    gc.collect()

    # 运行model
    model = Classifier(input_dim=config['input_dim'], hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim']).to(device)
    # model的参数个数为 每个batch : 256*39 + 256+256 * 256 + 256 + 256*41 + 41 ReLU没有参数， 也不改变Tensor的大小
    trainer(train_loader, valid_loader, model, config, device)



