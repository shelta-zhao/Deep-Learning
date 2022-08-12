# @ Time   : 2022/8/1 20:50
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim
import pandas as pd

def figure_show(img, title="", cmap=None):
    fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
    # 取消坐标轴的显示
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    pass

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

    pass

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    pass

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    pass

def trainer(feature_extractor, label_predictor, domain_classifier, source_dataloader, target_dataloader, device):

    # 定义损失函数和迭代器
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = torch.optim.Adam(feature_extractor.parameters())
    optimizer_C = torch.optim.Adam(label_predictor.parameters())
    optimizer_D = torch.optim.Adam(domain_classifier.parameters())

    # 定义相关变量
    # D loss： Domain Classifier 的loss
    # F loss： Feature Extractor 和 Label Predictor的端到端的loss

    for epoch in range(200):

        running_D_loss, running_F_loss = 0.0, 0.0
        total_acc, total_num = 0.0, 0.0
        count = 0

        for _, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
            # 由于source_dataloader和target_dataloader的大小不一致， zip会以小的为基准，舍弃大的多余的部分
            source_data = source_data.to(device)  # [32, 1, 32, 32]
            source_label = source_label.to(device)  # [32, 1]
            target_data = target_data.to(device)  # [32, 1, 32, 32]

            # 混合数据
            mixed_data = torch.cat([source_data, target_data], dim=0)  # [64, 1, 32, 32]
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)  # [64, 1]
            domain_label[:source_data.shape[0]] = 1  # 将source data设置为1， target data 设置为0

            # 第一步 训练domain classifier
            feature = feature_extractor(mixed_data)  # [64, 512]
            domain_logits = domain_classifier(feature.detach())  # 第一步不需要训练feature extractor 所以detach数据避免反向传播
            loss = domain_criterion(domain_logits, domain_label)
            running_D_loss += loss.item()
            loss.backward()
            optimizer_D.step()

            # 第二步 训练feature extractor 和 label predictor
            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            loss = class_criterion(class_logits, source_label) - 0.1 * domain_criterion(domain_logits, domain_label)
            running_F_loss += loss.item()
            loss.backward()
            optimizer_F.step()
            optimizer_C.step()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            total_acc += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]
            count += 1

        print(f'epoch: {epoch + 1}: train_D_loss:{running_D_loss/count:.5f}, train_F_loss:{running_F_loss/count:.5f}, acc{total_acc/total_num:.5f}')

    pass

def predictor(feature_extractor, label_predictor, test_dataloader, device):

    result = []
    feature_extractor.eval()
    label_predictor.eval()

    for _, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.to(device)
        class_logits = label_predictor(feature_extractor(test_data))

        # 得到结果
        y_pred = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(y_pred)

    # 保存结果
    result = np.concatenate(result)

    df = pd.DataFrame({'id':np.array(0, len(result)), 'label':result})
    df.to_csv('DaNN_result.csv', index=False)
    pass







if __name__ == '__main__':

    titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
    plt.figure(figsize=(18, 18))

    for i in range(10):
        plt.subplot(1, 10, i+1)
        figure_show(plt.imread(f'real_or_drawing/train_data/{i}/{500 * i}.bmp'), title=titles[i])

