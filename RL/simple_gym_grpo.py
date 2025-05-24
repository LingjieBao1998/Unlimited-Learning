# ref:https://superb-makemake-3a4.notion.site/group-relative-policy-optimization-GRPO-18c41736f0fd806eb39dc35031758885
import gym
import torch
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt  # 导入绘图库

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def collect_trajectory(env, net):
    observation = env.reset()[0]
    log_probs = []
    observations = []
    chosen_actions = []
    episode_reward = 0

    for t in range(200):
        observations.append(observation)
        logits = net(torch.from_numpy(observation).float())
        probs = torch.nn.functional.softmax(logits, dim=0)
        action = torch.multinomial(probs, 1).item()

        observation, reward, done, truncated, info = env.step(action)
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob.item())
        chosen_actions.append(action)
        episode_reward += reward

        if done:
            break

    normalized_reward = episode_reward / 200.0  # 因为这个环境中最大奖励为200
    return observations, log_probs, chosen_actions, normalized_reward

def grpo_update(trajectories, net, optimizer, n_iterations=20):
    rewards = [r for o, l, a, r in trajectories]
    mean_reward = sum(rewards) / len(rewards)
    std_reward = np.std(rewards) + 1e-8
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    for i_iter in range(n_iterations):
        loss = 0
        for traj, advantage in zip(trajectories, advantages):
            (observations, log_probs, chosen_actions, _) = traj
            trajectory_loss = 0
            for t in range(len(observations)):
                new_policy_probs = torch.nn.functional.softmax(net(torch.from_numpy(observations[t]).float()), dim=0)
                new_log_probs = torch.log(new_policy_probs)[chosen_actions[t]]

                ratio = torch.exp(new_log_probs - log_probs[t])
                clipped_ratio = torch.clamp(ratio, min=1 - eps, max=1 + eps)
                trajectory_loss += -clipped_ratio * advantage
            trajectory_loss /= len(observations)
            loss += trajectory_loss
        loss /= len(trajectories)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

env = gym.make('CartPole-v0')
net = PolicyNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
episode_reward_window = deque(maxlen=100)

# GRPO特定参数
trajectories_per_update = 5  # 组大小
eps = 0.2  # 剪切epsilon

# 用于绘图的平均奖励列表
avg_rewards = []

# 训练循环
for i_episode in range(5000):
    trajectories = []
    episode_rewards = []

    for _ in range(trajectories_per_update):
        observations, log_probs, chosen_actions, normalized_reward = collect_trajectory(env, net)
        trajectories.append((observations, log_probs, chosen_actions, normalized_reward))
        episode_rewards.append(normalized_reward * 200)  # 反归一化以便追踪

    # 使用收集的轨迹更新策略
    grpo_update(trajectories, net, optimizer)

    episode_reward_window.extend(episode_rewards)
    avg_reward = sum(episode_reward_window) / len(episode_reward_window)
    avg_rewards.append(avg_reward)  # 记录平均奖励

    if avg_reward > 195:
        print('solved at episode', i_episode)
        break

    if i_episode % 10 == 0:
        print(f'episode {i_episode}, avg reward: {avg_reward:.2f}')

# 绘制平均奖励曲线
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward (Last 100 Episodes)')
plt.title('Average Reward Over Episodes')
plt.grid()
plt.savefig('average_reward_plot(grpo).png')  # 保存图表
plt.show()

env.close()