# @ Time : 2022/7/22,17:39
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:


import torch

if __name__ == '__main__':

    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    config = {
        'check_path':'./data/checkpoint.pth',
        'data_dir':'./data/food/'

    }

