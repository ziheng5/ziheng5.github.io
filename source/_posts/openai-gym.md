---
title: OpenAI Gym 的使用
date: 2024-12-10 13:49:33
tags: 
    - RL（强化学习）
categories: 
    - RL（强化学习）实战笔记
description: |
    如何使用 OpenAI Gym
---
> OpenAI Gym 是一个能够提供智能体统一 API 以及很多 RL 环境的库。有了它，我们就不需要写大把大把的样板代码了
>
> 在这篇文章中，我们会学习如何写下第一个有随机行为的智能体，并借此来进一步熟悉 RL 中的各种概念。在这篇文章结束时，你可能会理解以下内容：
> - 将智能体插入 RL 框架所需的高层次要求
> - 基本、纯 Python 实现的随机 RL 智能体
> - OpenAI Gym

## 1. 剖析智能体
RL 世界中包含许多实体：
- **智能体**：主动行动的人或物。实际上，智能体只是实现了某些策略的代码片段而已。这个策略根据观察决定每个时间点执行什么动作。
- **环境**：某些世界的模型，它在智能体外部，负责提供观察并给予奖励。而且环境会根据智能体的动作改变自己的状态。

接下来，我们来探究如何实现它们。我们先从环境入手，先定义一个环境，限定交互步数，并且不管智能体执行任何动作，它都能给智能体返回随机奖励（先写一个简单的示例）：

```Python
import random
from typing import List

class Environment:
    def __init__(self):
        # 环境初始化的内部状态
        self.step_left = 10
    
    def get_observation(self) -> List[float]:
        # 该方法用于给智能体返回当前环境的观察
        # 这里输出的观察向向量都是0,是因为我们没有给环境任何内部状态
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        # 该方法允许智能体查询自己能执行的动作集
        # 某些条件下，当环境变化的时候，智能体能执行的动作集也会发生改变
        return [0, 1]

    def is_done(self) -> bool:
        # 给予智能体片段结束的信号
        return self.steps_left == 0

    def action(self, action: int) -> float:
        # 环境的核心功能
        # 它用于处理智能体的动作、返回该动作的奖励、更新已经执行的步数、拒绝执行已执行的片段
        if self.is_done():
            raise Exception("Game is over!")
        self.steps_left -= 1
        return random.random()
```

接下来我们来看看智能体部分，它更简单，只包含了两个部分：构造函数以及在环境中执行一步的方法：

```Python
class Agent:
    def __init__(self):
        # 初始化计数器，用来保存片段中智能体积累的总奖励
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_oba = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward
```
在上面这段代码中，`step()` 函数接受环境实例作为参数，并允许智能体执行以下操作：
- 观察环境
- 基于观察决定动作
- 向环境提交动作
- 获取当前步骤的奖励

对于我们这个例子，智能体比较笨，它在决定执行什么动作的时候会忽略得到的观察。取而代之的是，随机选择动作。下面还有一段胶水代码，它创建两个类并执行一次片段：

```Python
if __name__ == "__main__":
    env = Environment()
    agent = Agent()

while not env.is_done():
    agent.step(env)

print("Total reward got: %.4f" % agent.total_reward)
```

前面这些简单的代码展示了 RL 模型的重要的基本概念。环境可以是极其复杂的物理模型，智能体也可以轻易地变成一个实现了最新 RL 算法的大型神经网络（NN），但是基本思想还是一致的——每一步，智能体都会从环境中得到观察，进行一番计算，最后选择要执行的动作。这个动作的结果就是奖励和新的观察。

你可能会问，如果基本思想是一样的，为什么还要从头开始实现呢？是否有人已经将其实现为一个通用库了呢？答案是肯定的，这样的框架已经存在了，但是在花时间讨论它们之前，先把你的开发环境准备好吧！

requirements如下所示（⚠️：其中gym版本尽量一致，或者不要用高版本；其他库可用最新版本）：
```Terminal
atari-py==0.2.6
gym==0.15.3
numpy==1.17.2
opencv-python==4.1.1.26
tensorboard==2.0.1
torch==1.3.0
torchvision==0.4.1
pytorch-ignite==0.2.1
tensorboardX==1.9
tensorflow==2.0.0
ptan==0.6
```

## 2. OpenAI Gym API
### 2.1 动作空间
### 2.2 观察空间
### 2.3 环境
### 2.4 创建环境
### 2.5 车摆控制
我们来应用学到的知识探索 Gym 提供的最简单的 RL 环境：

```Python
import gym

e = gym.make("CartPole-v0")
```