# @ Time : 2022/3/28,17:51
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description:use model to predict on testdata

import json
import csv
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from Model import *

class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)

if __name__ == '__main__':
    data_dir = './data/Dataset'
    model_path = './models/model.ckpt'
    output_path = './prediction/predict.csv'

    # 观察test数据
    # testdata_path = Path(data_dir) / "testdata.json"
    # metadata = json.load(testdata_path.open())
    # print(metadata.keys())  # 只包含n_mels 和 utterances
    # print(metadata['n_mels'])

    # 加载数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=inference_collate_batch)

    speaker_num = len(mapping["id2speaker"])
    model = Classifier(n_speakers=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 预测并记录结果
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
        pass
    pass

