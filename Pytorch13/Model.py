# @ Time   : 2022/8/7 16:03
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :


import torch.nn as nn
import torch.optim
from torchsummary import summary
from tqdm import tqdm

def dwpw_conv(input_channels, output_channels, kernel_size, stride=1, padding=0):
    # 应用Architecture Design
    return nn.Sequential(
        # group参数可以将输入的channels进行分组
        nn.Conv2d(input_channels, input_channels, kernel_size, stride=stride, padding=padding, groups=input_channels), # depthwise convolution
        nn.Conv2d(input_channels, output_channels, 1)  # pointwise convolution
    )

def loss_fn_kd(student_logits, teacher_logits, labels, alpha=0.5, temperature=1):
    CE = nn.CrossEntropyLoss()
    loss_CE = CE(student_logits, labels)

    KL = nn.KLDivLoss()
    student_logits = (student_logits / temperature).log_softmax(dim=1)
    teacher_logits = (teacher_logits / temperature).softmax(dim=1)
    loss_KL = KL(student_logits, teacher_logits)

    return alpha * temperature * temperature * loss_KL + (1 - alpha) * loss_CE

class StudentNet(nn.Module):

    def __init__(self):
        super(StudentNet, self).__init__()
        self.cnn = nn.Sequential(
            dwpw_conv(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            dwpw_conv(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            dwpw_conv(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            dwpw_conv(64, 100, 3),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.AdaptiveAvgPool2d((1, 1)),  # 经过这个池化后 输入大小变为 100 * 1 * 1
        )

        self.fc = nn.Sequential(
            nn.Linear(100, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def trainer(train_loader, valid_loader, student_model, teacher_model, config, device):

    # 定义optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    stale = 0
    best_acc = 0
    n_epochs = config['n_epochs']

    # 定义迭代
    for epoch in range(n_epochs):
        student_model.train()

        train_loss = []
        train_accs = []
        train_lens = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            student_logits = student_model(imgs)
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)

            loss = loss_fn_kd(student_logits, teacher_logits, labels)

            optimizer.zero_grad(),
            loss.backward()
            optimizer.step()

            acc = (student_logits.argmax(dim=1) == labels).float().sum()

            # 记录过程
            train_batch_len = len(imgs)
            train_loss.append(loss.item() * train_batch_len)
            train_accs.append(acc)
            train_lens.append(train_batch_len)

        train_loss = sum(train_loss)/len(train_lens)
        train_acc = sum(train_accs)/len(train_lens)

        print(f'{epoch}: train_acc:{train_acc:.5f}, train_loss: {train_loss:.5f}')

        # 模型评价
        student_model.eval()

        valid_loss = []
        valid_accs = []
        valid_lens = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                student_logits = student_model(imgs)
                teacher_logits = teacher_model(imgs)

            loss = loss_fn_kd(student_logits, teacher_logits, labels)
            acc = (student_logits.argmax(dim = -1) == labels).float().sum()

            batch_len = len(imgs)
            valid_loss.append(loss)
            valid_accs.append(acc)
            valid_lens.append(batch_len)

        valid_loss = sum(valid_loss) / len(valid_lens)
        valid_acc = sum(valid_accs) / len(valid_lens)

        print(f'{epoch}: valid_acc:{valid_acc:.5f}, valid_loss: {valid_loss:.5f}')

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(student_model.state_dict(), f'./outputs/student_best.ckpt')
            stale = 0
        else:
            stale += 1
            if stale > config['patience']:
                break
    pass


if __name__ == '__main__':

    student_model = StudentNet()
    summary(student_model, (3, 224, 224), device='cpu')  # 查看模型的参数情况，以及输入的tensor变化

    a = torch.tensor([[-0.1013,  0.2261,  0.1859, -0.6944,  0.6122,  0.6133,  0.1575, -0.6331, -0.0267, -0.4083, -0.2460]])
    b = torch.tensor([[-7.4063, -13.4838,  -8.3455,  -9.9323,  -3.4067,   3.6421,  -6.7755, -9.5859,  -3.4213,  14.6621,  -4.2272]])

    a = a.log_softmax(dim=1)
    b = b.softmax(dim=1)
    print(a)
    KL = nn.KLDivLoss()
    print(KL(a, b))
