# @ Time   : 2022/8/5 15:32
# @ Author : Super_XIAOTANG
# @ File   : main.py
# @ IDE    : PyCharm
# @ Brief  :


import torch
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from Model import PolicyGradientNetwork, PolicyGradientAgent, trainer

def same_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cdunn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pass

def show_game(env):
    # 可视化展示一次游戏的过程
    env.reset()

    img = plt.imshow(env.render(mode='rgb_array'))
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        img.set_data(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)

    pass

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('LunarLander-v2')  # 模拟登月小游戏 让登月小艇降落在两个旗帜之间
    initial_state = env.reset()
    print(initial_state)
    config = {
        'seed':1127,
        'n_epochs':400,

    }


    # 固定时间种子
    same_seed(env, config['seed'])

    # 建立模型
    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network)

    # 训练模型
    trainer(agent, env, config)