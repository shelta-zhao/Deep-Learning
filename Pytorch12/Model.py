# @ Time   : 2022/8/6 19:31
# @ Author : Super_XIAOTANG
# @ File   : Model.py
# @ IDE    : PyCharm
# @ Brief  :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np

class PolicyGradientNetwork(nn.Module):
    # 输入是8个纬度的observation 输出4个动作
    def __init__(self):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(state))
        return F.softmax(self.fc3(hid), dim=1)

class PolicyGradientAgent:

    def __init__(self, network):
        self.network = network
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1e-3)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()  # version 0的Policy Gradient

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)  # 根据action_prob的概率建立分布，并从中随机抽取
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, PATH):
        # 保存模型
        Agent_Dict = {
            'network':self.network.state_dict(),
            'optimizer':self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)
        pass

    def load(self, PATH):
        # 载入模型
        check_point = torch.load(PATH)
        self.network.load_state-dict(check_point['network'])
        self.optimizer.load_state_dict(check_point['optimizer'])
        pass


def trainer(agent, env, config):

    agent.network.train()

    # 定义reward
    avg_total_rewards, avg_final_rewards = [], []  # final_reward可以用于观察飞船的落地是否顺利
    n_epochs = config['n_epochs']

    # 定义迭代
    for batch in tqdm(range(n_epochs)):
        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        # 收集数据， 每五个episode更新一次agent
        for episode in range(5):

            state = env.reset()
            total_reward, total_step = 0, 0
            seq_rewards = []

            while True:
                action, log_prob = agent.sample(state)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                seq_rewards.append(reward)
                rewards.append(reward)

                state = next_state
                total_reward += reward
                total_step += 1

                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    break

        print(f'rewards: {np.shape(rewards)}')
        print(f'log_probs: {np.shape(log_probs)}')

        # 记录训练过程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)

        # 更新网络
        # rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        print("logs prob looks like ", torch.stack(log_probs).size())
        print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

        pass


if __name__ == '__main__':

    probs = torch.FloatTensor([0.1, 0.3, 0.2, 0.4])
    D = Categorical(probs)
    sample = D.sample()
    print(sample, D.log_prob(sample))
