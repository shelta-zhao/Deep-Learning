# @ Time : 2022/3/25,22:11
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: Classify the speakers of given features


import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from Dataset import *
from Model import *

# 固定时间种子
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        pass
    pass

# 处理批处理的特征 由于我们是逐批次处理数据 所以需要把数据补齐到同一长度
def collate_batch(batch):
    # 根据dataset的__getitem__函数可知，返回的mel的长度可能为segment， 也可能小于segment 所以这里要进行补充到同一个长度batch_size * segment * 40
    mel, speaker = zip(*batch)  # zip是打包返回一个元组列表 而zip(*)是解压
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # 默认batch在第一维度
    return mel, torch.FloatTensor(speaker).long()

# 获得DataLoader
def get_dataloader(data_dir, batch_size):
    dataset = SpeakDataset(data_dir)
    speck_num = dataset.get_speaker_number()  # 获得演讲者总数

    # 分割数据为训练集0.9和验证集0.1
    train_len = int(0.9 * len(dataset))
    lengths = [train_len, len(dataset) - train_len]
    train_set, valid_set = random_split(dataset, lengths)

    # 制作train_loader, valid_loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=collate_batch)   # drop_last 舍弃数据末尾不足一个batch的数据
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True,collate_fn=collate_batch)   # collate_fn 相当于对每个batch的处理方式

    return train_loader, valid_loader, speck_num

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        # 数据参数
        "data_dir": "./data/Dataset",
        "model_path": "./models/model.ckpt",

        # 模型参数
        "batch_size": 32,
        "n_epochs": 70000,
        "warmup_steps": 1000,
        "patience":500,
        "save_steps":1000,
    }

    # 获得dataloader
    train_loader, valid_loader, speaker_num = get_dataloader(config['data_dir'], config['batch_size'])

    # for i,batch in enumerate(train_loader):
    #     print(len(batch))
    #     print(batch[0].shape)  # 可见大小为32*10*40  即batch_size * segment * 40
    #     print(batch[1].shape)  # 32
    #     break


    # 运行模型
    model = Classifier(n_speakers=speaker_num).to(device)
    trainer(train_loader, valid_loader, model, config, device)
    pass
