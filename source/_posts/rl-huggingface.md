---
title: 基于 PyTorch 与 Gym 的 PPO 算法复现
date: 2025-01-19 15:53:39
tags: 
    - 强化学习
    - PyTorch
categories: 
    - 强化学习
description: |
    ❓ 如何用 PyTorch 复现 PPO(Proximal Policy Optimization，近端策略优化) 算法
---
> 🤞 这段时间在师姐的推荐下，学习了网上关于强化学习的 PyTorch 实战视频，小生自己也是跟着视频复现了一遍 PPO 算法，于是准备写一篇 Blog 来记录学习心得
>
> 框架：**PyTorch**
>
> 游戏：**LunarLander-v2**
>
> 上一次运行成功时间：**2025-01-20**
>
> （本文里的代码是小生重新手打的，未复制粘贴，可能存在拼写错误，望见谅）

![pic1](../images/rl_practice/pic1)

# 1. 📦 导入必要的包
```Python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gym
import numpy as np

# 导完包之后顺手定义一下 device
device = torch.device(“cuda:0” if torch.cuda.is_available() else "cpu")
```

⚠️ 注意：下载 gym 的时候一定要记得安装一下 `box2d` 包，否则会报错。如下所示：

```Terminal
pip install box2d box2d-kengz
```

这里顺便列一下 requirements:
- **pytorch**: 2.5.0
- **gym**: 0.26.2
- **box2d**: 2.3.10
- **box2d**-kengz: 2.3.3
- **numpy**: 2.1.3


# 2. 🫙 构建缓存区
```Python
class Memory:
    def __init__(self):
        # 缓存区，用于存储当前一个 episode 中的经验
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # 清空缓存
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
```

# 3. 🧱 构建 A2C 策略网络
```Python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # Actor 网络：
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),  # 你问我为什么不用其他激活函数？我也不清楚，待会儿查一下 QWQ
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)  # 这里 dim=-1 相当于 dim=2，对 actions 进行归一化处理
        )

        # Critic 网络：
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )
    
    def forward(self):
        # 你问这里为什么用不到 forward？大笨蛋！这里有两个网络你怎么前向传播？
        raise NotImplementedError

    def act(self, state, memory):
        # 根据当前状态 s 生成下一步动作 a
        state = torch.from_numpy(state).float().to(device)
        actions_probs = self.action_layer(state)
        dist = Categorical(actions_probs)
        action = dist.sample()

        # 压入缓存区
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        # 评估模式
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
```

`ActorCritic` 类是 PPO 算法的核心，它由两个部分组成：
- **Actor**：用于根据当前状态选择动作。通过一个神经网络来预测动作的概率分布，并通过 `Categorical` 类来抽样动作。
- **Critic**：用于评估当前状态的价值，通过一个神经网络输出状态的价值函数。

`act` 方法接受当前状态，计算出动作概率分布并根据其进行抽样，保存当前的状态、动作和对应的 log 概率。

`evaluate` 方法用于计算动作的 log 概率、状态值和分布的熵。


# 4. 👷 构建 PPO 算法
```Python
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 创建策略网络
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # 这里使用 off policy 思想，给我们的 policy 召唤一个分身
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict)

        self.MseLoss = nn.MSELoss()
    

    def update(self, memory):
        # 利用蒙特卡洛算法计算 state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # 归一化 rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 把 list 全部转换为 tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 优化策略
        for _ in range(self.K_epochs):
            # 评估 old actions 和 values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_state, old_actions)

            # 计算 ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算 surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0-self.eps_clip, 1.0+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # 更新权重
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        
        # 将新的权重赋给我们的 old_policy
        self.policy_old.load_state_dict(self.policy.state_dict())
```

`PPO` 类封装了强化学习训练的主要逻辑。
`update` 方法通过经验回放更新策略：
- 首先，计算蒙特卡洛奖励。
- 然后对奖励归一化处理。
- 接着，通过多次优化（K-epochs），计算损失函数并进行梯度更新。
- 最后，将新的策略参数复制到旧策略中。

损失函数由三部分组成：
- **Surrogate Loss**：确保新的策略和旧的策略之间的比率 ratio 不便，以避免过大的策略变化。
- **Value Loss**：通过均方误差来最小化状态价值的误差。
- **Entropy**：通过熵来避免策略过于确定，保持探索性。


# 5. 🀄 主函数
```Python
def main():
    # 创建环境
    env_name = "LunarLander-V2"
    env = gym.make(env_name, render_name="human")

    #############################################################################
    # 初始化参数
    state_dim = env.observation_space.shape[0]  # state_dim = 8
    action_dim = 4  # 上下左右
    render = True   # 是否展示（建议设置为 False,这样可以加快训练速度）
    solved_reward = 230 # 训练成功的指标
    log_interval = 20
    max_episodes = 50000
    max_timesteps = 300
    n_latent_var = 64
    update_timestep = 2000
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    random_seed = None
    #############################################################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    running_reward = 0
    avg_length = 0
    timestep = 0

    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        state = np.array(state[0])
        for t in range(max_timesteps):
            timestep += 1

            action = ppo.policy_old.act(state, memory)
            state, reward, done, _, _ = env.step(action)
            # 注意：新版的 gym 在 env.step 时会返回 5 个值，具体请查阅官方文档或源码注释

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break
            
        avg_length += t

        if running_reward > (log_interval*solved_reward):
            print("#################### Solved ! ####################")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env.name))
            break

        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int(running_reward/log_interval)

            print("Episode {} \t avg length: {} \t reward:{}".format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
```

# 6. 🚗 加载模型文件展示效果：
> 这里要新建一个 `ppo_show.py` 文件
```Python
from ppo import PPO, Memory, ActorCritic
import gym
import torch
import numpy as np

env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 4
render = True
max_timesteps = 300
n_latent_var = 64
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 4
eps_clip = 0.2
n_episodes = 3
filename = "PPO_LunarLander-v2.pth"
directory = "./"

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip=eps_clip)
ppo.policy.load_state_dict(torch.load(filename))
ppo.policy_old.load_state_dict(torch.load(filename))

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

# training loop
for i_episode in range(1, 20):
    state = env.reset()
    state = np.array(state[0])
    for t in range(max_timesteps):
        timestep += 1

        action = ppo.policy_old.act(state, memory)
        state, reward, done, _, _ = env.step(action)

        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t
```
> **如有报错且找不到报错原因的，可以私信小生** 🥺