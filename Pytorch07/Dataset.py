# @ Time : 2022/7/14,20:25
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:

import json
import torch
from transformers import BertTokenizerFast
from torch.utils.data import Dataset


def read_data(file):
    """
    读取数据集中的数据，同时对数据进行tokenize处理
    """
    with open(file, 'r', encoding='UTF-8') as reader:
        data = json.load(reader)
    return data['questions'], data['paragraphs']

class QADataset(Dataset):

    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split  # 用于判断类型：train\validate\test
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150

        # 可以调整doc_stride的值 doc_stride可以理解为窗口大小
        self.doc_stride = 150

        # 设置模型同时处理的seq最大长度 length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def padding(self, input_ids_question, input_ids_paragraph):
        # 对于模型的处理，每个tensor内各vector的长度必须相同，因此要进行padding处理
        # 如果seq的长度小于max_seq_len，就进行补0操作
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # input_ids 根据padding_len的长度进行补0的结果
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # token_type_ids: 用于标注不同segment
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # attention_mask:由于进行padding操作，padding的部分是不需要处理的，因此设置1为需要处理，0为不需要处理
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

    def __getitem__(self, item):
        question = self.questions[item]
        tokenized_question = self.tokenized_questions[item]
        tokenized_paragraph = self.tokenized_paragraphs[question['paragraph_id']]

        # 预处理过程 Preprocessing
        if self.split == 'train':
            # 将answer中的start/end位置转化为tokenized后的位置
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
            # print(f"answer_start:{question['answer_start']},  answer_start_token:{answer_start_token}")
            # print(f"answer_end:{question['answer_end']},  answer_end_token:{answer_end_token}")

            # 通过切片包含答案的段落部分获得单个窗口 //为取整除法 如9 // 2=4
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            # print(f'mid:{mid}')
            # print(f'{mid - self.max_paragraph_len // 2},  {len(tokenized_paragraph) - self.max_paragraph_len}')
            # print(f'paragraph_start:{paragraph_start},  paragraph_end:{paragraph_end}')

            # 划分question和paragraph 并添加特殊值 [101]:[CLS]\[102]:[SEP]
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]

            # 将 tokenized_paragraph 中答案的开始/结束位置转换为窗口中的开始/结束位置
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            # print(f"answer_start_token:{answer_start_token}, answer_end_token:{answer_end_token}")

            # padding，得到模型的输入
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            # 段落被分成几个窗口，每个窗口的起始位置由 step "doc_stride" 分隔
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # 划分question和paragraph 并添加特殊值 [101]:[CLS]\[102]:[SEP]
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]

                # padding，得到模型的输入
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                # 以列表的形式返回input_ids\token_type_ids\attention_mask
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
                # 加上二维列表后转换为tensor，batch_size则为1
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    pass


if __name__ == '__main__':

    # 读取数据, 返回的数据类型为list
    train_questions, train_paragraphs = read_data('./data/hw7_train.json')
    dev_questions, dev_paragraphs = read_data('./data/hw7_dev.json')
    test_questions, test_paragraphs = read_data('./data/hw7_test.json')
    print(train_paragraphs[3884])
    # print(train_questions[0])
    # print(train_paragraphs[3884])

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    # token实验
    token = tokenizer("函数测试，赵小棠", add_special_tokens=False)  # 样例语句进行tokenize处理
    print(token)  # 由三部分组成，input_ids\token_type_ids\attention_mask 语句被转化成数字token
    print(tokenizer.decode(token['input_ids']))  # 将token进行解码可以得到初始语句”函数测试，赵小棠“
    print(token.char_to_token(2))

    print(float('-inf'))

