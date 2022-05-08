# @ Time : 2022/3/31,14:50
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: prefix 前缀 suffix 后缀


import torch
import numpy as np
from torch.utils.data import DataLoader
from Preprocess import *
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
    data_prefix = './data/train_dev.raw'
    test_prefix = './data/test.raw'
    config = {
        # 数据集参数
        'seed':1127,
        'src_lang':'en',
        'tgt_lang':'zh',
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 预处理数据集
    clean_corpus(data_prefix, config['src_lang'], config['tgt_lang'])
    clean_corpus(test_prefix, config['src_lang'], config['tgt_lang'], ratio=-1, min_len=-1, max_len=-1)

    # 划分训练集 验证集
    data_split()
    Subword_Units()


    # print("Hello Pytorch05")
