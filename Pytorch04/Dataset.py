# @ Time : 2022/3/25,22:04
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:


import json
import random
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset


class SpeakDataset(Dataset):
    def __init__(self, data_dir, segment_len=10):
        self.data_dir = data_dir
        self.segment_len = segment_len  # 为了提高训练效率，将数据分割

        # 加载mapping数据 里面是speaker2id、id2speaker的信息
        mapping_path = Path(data_dir) / 'mapping.json'
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']

        # 加载元数据
        metadata_path = Path(data_dir) / 'metadata.json'
        meta_data = json.load(metadata_path.open())['speakers']

        # 获取speaker的数量 utterance 不同的录音片段
        self.speaker_num = len(meta_data.keys())
        self.data = []
        for speaker in meta_data.keys():
            for utterances in meta_data[speaker]:
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feat_path, speaker = self.data[item]
        # 读取数据
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # 切割数据
        if len(mel) > self.segment_len:
            # 随机选择切割点 取一段长度为segment_len的数据
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num

    pass


if __name__ == '__main__':


    #  测试读到的json数据是什么结构
    data_dir = './data/Dataset'

    # json文件是key-value形式
    mapping_path = Path(data_dir) / 'mapping.json'
    mapping = json.load(mapping_path.open())
    print(type(mapping))  # 可以看到类型是dict字典
    print(len(mapping['id2speaker']))  # 只包括speaker2id  id2speaker两个 一共有600个id

    metadata_path = Path(data_dir) / 'metadata.json'
    meta_data = json.load(metadata_path.open())['speakers']
    print(meta_data['id03074'])  # keys为id，values为多个feature_path 和 mel_len

    mel = torch.load(os.path.join(data_dir, 'uttr-18e375195dc146fd8d14b8a322c29b90.pt'))
    print(mel.shape)       # 435 * 40的tensor 435为mel_len
    print(len(mel))        # 长度为435

    # 看看训练数据一共多少
    dataset = SpeakDataset(data_dir)
    print(len(dataset))   # 可见长度为56666
    



