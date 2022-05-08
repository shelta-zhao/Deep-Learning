# @ Time : 2022/3/22,22:29
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description: This file is used to test the model saved


import numpy as np
from torch.utils.data import DataLoader
from Dataset import *
from Model import *

# 定于预测函数
def predict(test_loader, model, device, save=False):
    test_acc = []
    test_len = 0
    pred = np.array([],dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            datas = batch
            datas = datas.to(device)

            # 预测
            preds = model(datas)

            # 记录
            _, test_pred = torch.max(preds, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
            pass
    if save:
        if not os.path.isdir('./predict'):
            os.mkdir('./predict')
        with open('./predict/prediction.csv', 'w') as of:
            of.write('Id,Class\n')
            for index, label in enumerate(pred):
                of.write('{},{}\n'.format(index,label))
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    x_test = preprocess_data(split='test', feat_dir='./data/libriphone/feat', phone_path='./data/libriphone',concat_nframes=1)
    test_set = LibriDataset(x_test, None)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    # 加载模型
    model = Classifier(input_dim=39).to(device)
    model.load_state_dict(torch.load('./models/model.ckpt'))

    # 预测
    predict(test_loader, model, device, save=True)