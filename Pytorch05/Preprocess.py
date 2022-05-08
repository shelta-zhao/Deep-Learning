# @ Time : 2022/3/31,19:47
# @ Author : 小棠
# @ Version : python = 3.8.12, torch = 1.11
# @ Encoding : UTF-8
# @ Description: This file is used to define some functions for preprocessor

import re  # 正则表达式
from pathlib import Path
import random
import os
import sentencepiece as spm


# 预处理函数
def strQ2B(ustring):
    """将中文中占两个字节的标点符号转化为英文的只有一个字节的标点符号"""
    ans = []
    for str in ustring:
        res = ""
        for uchar in str:
            inside_code = ord(uchar)  # uchar的汉字的码 如12288为空格
            if inside_code == 12288:  # 将中文的空格转化为英文的空格
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 将中文标点符号转化为英文标点符号 如 ！转化为 !
                inside_code -= 65248
            res += chr(inside_code)
        ans.append(res)
    return ''.join(ans)

def clean_str(str, lang):
    """清除字符串中的标点符号"""
    if lang == 'en':
        str = re.sub(r"\([^()]*\)", "", str)  # 去除 ([text])
        str = str.replace('-', '')  # 去除 '-'
        str = re.sub('([.,;!?()\"])', r' \1 ', str)  # 保持标点符号，在两边加空格
    elif lang == 'zh':
        str = strQ2B(str)  # Q2B
        str = re.sub(r"\([^()]*\)", "", str)  # 去除 ([text])
        str = str.replace(' ', '')
        str = str.replace('—', '')
        str = str.replace('“', '"')
        str = str.replace('”', '"')
        str = str.replace('_', '')
        str = re.sub('([。,;!?()\"~「」])', r' \1 ', str)  # 保持标点符号，在两边加空格
    str = ' '.join(str.strip().split())
    return str

def len_str(str, lang):
    """获得单词个数 或者中文字符个数（包括标点）"""
    if lang == 'zh':
        return len(str)
    return len(str.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    """清理数据语料库，对数据集进行预处理"""

    # 判断清洗后的文件是否已经存在，如果已经存在则跳过这一步
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return

    with open(f'{prefix}.{l1}', 'r', encoding='UTF-8') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r', encoding='UTF-8') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w', encoding='UTF-8') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w', encoding='UTF-8') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()  # 删除字符串开头和结尾的空格
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_str(s1, l1)
                        s2 = clean_str(s2, l2)
                        s1_len = len_str(s1, l1)
                        s2_len = len_str(s2, l2)
                        if min_len > 0:  # 移除过短的句子 如果min_len < 0 代表处理test数据 则不移除
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # 移除过长的句子 如果max_len < 0 代表处理test数据 则不移除
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # 移除不同语言的两句话比例异常的语句 如果ratio < 0 代表处理test数据 则不移除
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

def data_split():
    src_lang = 'en'
    tgt_lang = 'zh'
    data_dir = './data'
    if Path(f'./data/train.clean.{src_lang}').exists() \
            and Path(f'./data/train.clean.{tgt_lang}').exists() \
            and Path(f'./data/valid.clean.{src_lang}').exists() \
            and Path(f'./data/valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
        return
    else:

        line_num = sum(1 for line in open(f'./data/train_dev.raw.clean.{src_lang}', encoding="UTF-8"))  # 获得源英文数据的行数
        labels = list(range(line_num))
        random.shuffle(labels)
        for lang in [src_lang, tgt_lang]:
            train_f = open(os.path.join(data_dir, f'train.clean.{lang}'), 'w', encoding="UTF-8")
            valid_f = open(os.path.join(data_dir, f'valid.clean.{lang}'), 'w', encoding="UTF-8")
            count = 0
            for line in open(f'./data/train_dev.raw.clean.{lang}', 'r', encoding="UTF-8"):
                # 根据编号labels大小判断该分配至哪个数据集
                if labels[count] / line_num < 0.99:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
            train_f.close()
            valid_f.close()

def Subword_Units():
    src_lang = 'en'
    tgt_lang = 'zh'
    prefix = Path('./data')
    vocab_size = 8000
    if (prefix / f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
        return
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix / f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram',  # 'bpe' works as well
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )
        spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
        in_tag = {
            'train': 'train.clean',
            'valid': 'valid.clean',
            'test': 'test.raw.clean',
        }
        for split in ['train', 'valid', 'test']:
            for lang in [src_lang, tgt_lang]:
                out_path = prefix / f'{split}.{lang}'
                if out_path.exists():
                    print(f"{out_path} exists. skipping spm_encode.")
                else:
                    with open(prefix / f'{split}.{lang}', 'w', encoding="UTF-8") as out_f:
                        with open(prefix / f'{in_tag[split]}.{lang}', 'r', encoding="UTF-8") as in_f:
                            for line in in_f:
                                line = line.strip()
                                tok = spm_model.encode(line, out_type=str)
                                print(' '.join(tok), file=out_f)

def Binarize():
    pass


if __name__ == '__main__':
    # prefix = './data'
    #
    # src_lang = 'en'
    # tgt_lang = 'zh'
    #
    # data_prefix = f'{prefix}/train_dev.raw'
    # test_prefix = f'{prefix}/test.raw'
    #
    # Str = "非常謝謝你，克里斯。能有這個機會第二度踏上這個演講台！"
    # Str2 = "Thank you so much, Chris."
    # print(clean_str(Str, tgt_lang))
    # print(len_str(Str, tgt_lang))
    #
    # print(clean_str(Str2, src_lang))
    # print(len_str(Str2, src_lang))

    # 测试划分数据集
    # data_split()
    # Subword_Units()
    pass
