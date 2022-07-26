# @ Time : 2022/7/15,15:18
# @ Author : 小棠
# @ Version : 1.0.0
# @ Encoding : UTF-8
# @ Description:

import torch
from tqdm import tqdm
import os
from transformers import AdamW

def evaluate(data, output, tokenizer):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # 选择最合适的起止位置
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim = 0)

        # 计算prob
        prob = start_prob + end_prob

        # 更替
        if prob > max_prob:
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index:end_index + 1])

        # 去除answer中的空格
        return answer.replace(' ', '')

def trainer(model, train_loader, dev_loader, config, device, dev_questions):

    # 定义迭代器，和scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    n_epochs = config['n_epochs']
    logging_step = config['logging_step']

    # 定义models文件夹
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    for epoch in range(n_epochs):
        model.train()
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):
            data = [i.to(device) for i in data]
            # 模型输入: input_ids, token_type_ids, attentionmask, start_position, end_position
            # 模型输出: start_logits, end_logits, loss

            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # 选则最好的输出
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # 计算acc 和 loss
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            # 梯度下降
            # print(optimizer.state_dict()['param_groups'][0]['lr'])  # 输出实时学习率
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1



            # 打印训练过程
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        # Validate
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device),
                               token_type_ids=data[1].squeeze(dim=0).to(device),
                               attention_mask=data[2].squeeze(dim=0).to(device))
                dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]

            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")

    # 保存模型
    model.save_pretrained(config['save_dir'])

def predictor(model, test_loader, test_questions, device):
    """
    加载训练好的模型用于预测
    """
    model.load_state_dict(torch.load('./models/'))
    model.eval()

    result = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
        result.append(evaluate(data, output))

    with open('./models/result.csv', 'w', encoding='UTF-8') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            f.write(f"{test_question['id']},{result[i].replace(',', '')}\n")
            pass
    pass



