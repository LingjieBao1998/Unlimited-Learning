# ref:https://superb-makemake-3a4.notion.site/group-relative-policy-optimization-GRPO-18c41736f0fd806eb39dc35031758885
import gym
import torch
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('CartPole-v0')

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = PolicyNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
gamma = 0.99

episode_reward_window = []
avg_rewards = []  # 用于存储每个episode的平均奖励

for i_episode in range(5000):
    observation = env.reset()[0]
    log_probs = []
    episode_reward = 0

    for t in range(200):
        logits = net(torch.from_numpy(observation).float())
        probs = torch.nn.functional.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        observation, reward, done, truncated, info = env.step(action)
        log_prob = torch.log(probs[action])
        log_probs.append(log_prob)
        episode_reward += reward
        if done:
            break

    normalized_reward = episode_reward / 200.0
    policy_loss = []
    for log_prob in log_probs:
        policy_loss.append(-log_prob * normalized_reward)

    policy_loss = torch.stack(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    episode_reward_window.append(episode_reward)
    if len(episode_reward_window) > 100:
        episode_reward_window.pop(0)
    avg_reward = sum(episode_reward_window) / len(episode_reward_window)
    avg_rewards.append(avg_reward)  # 记录平均奖励

    if avg_reward > 195:
        print('solved at episode', i_episode)
        break

    if i_episode % 100 == 0:
        print('episode', i_episode, 'avg_reward', avg_reward)

# 绘制平均奖励曲线
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward (Last 100 Episodes)')
plt.title('Average Reward Over Episodes')
plt.grid()

# 保存图表为文件
plt.savefig('average_reward_plot(ppo).png')  # 可以选择文件格式，如 .png, .jpg, .pdf 等
plt.show()

env.close()