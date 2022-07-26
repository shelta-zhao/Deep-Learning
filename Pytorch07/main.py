# @ Time : 2022/7/14,20:17
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:



import numpy as np
import torch.backends.cudnn
import gc
from torch.utils.data import DataLoader
from Dataset import read_data, QADataset
from Model import trainer, predictor
from transformers import BertTokenizerFast, BertForQuestionAnswering


def same_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        pass
    pass


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed':1127,
        'batch_size':32,
        'n_epochs':1,
        'logging_step':100,
        'learning_rate':1e-4,
        'save_dir':'./models'
    }

    # 固定时间种子
    same_seed(config['seed'])

    # 读取数据，返回的数据类型都是list
    train_questions, train_paragraphs = read_data('./data/hw7_train.json')
    dev_questions, dev_paragraphs = read_data('./data/hw7_dev.json')
    test_questions, test_paragraphs = read_data('./data/hw7_test.json')

    # Tokenizer处理
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False)

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    # 获得Dataloader
    train_set = QADataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QADataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QADataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    # train_loader:包含5个对象的列表，分别是input_ids[32, 193]\token_type_ids[32,193]\attention_mask[32,193]\
    # answer_token_start[32]\answer_token_end[32] 这两项可视为label
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 清理数据
    del train_set, dev_set, test_set
    gc.collect()

    # 设定模型
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    trainer(model, train_loader, dev_loader, config, device, dev_questions)

    # 模型预测
    predictor(model, test_loader, test_questions, device)