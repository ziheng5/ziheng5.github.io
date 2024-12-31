---
title: OpenAI Gym 的使用
date: 2024-12-10 13:49:33
tags: 
    - 强化学习
categories: 
    - 强化学习
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
>>> import gym

>>> e = gym.make("CartPole-v0")
```

这里我们导入了 gym 库，创建了一个叫做 CartPole（车摆系统）的环境。该环境来自经典的控制问题，其目的是控制底部附有木棒的平台。

这里的难点是，木棒会向左或向右倒，你需要在每一步，通过让平台往左或往右移动来保持平衡。

这个环境的观察是4个浮点数，包含了木棒质点的 **$x$ 坐标**、**速度**、**与平台的角度**以及**角速度**的信息。当然，通过应用一些数学和物理知识，将这些数字转换为动作来平衡木棒并不复杂，但问题是如何在不知道这些数字的确切含义、只知道奖励的情况下，学会平衡该系统？这个环境每执行一步，奖励都是1。片段会一直持续，直到木棒掉落为止，因此为了获得更多的累积奖励，我们需要以某种避免木棒掉落的方式平衡平台。

我们继续来编写代码：

```Python
>>> obs = e.reset()
>>> obs
array([ 0.04373323, -0.0498781 , -0.03481512,  0.01108434])
```

这里，先重置一下环境并获得第一个观察（在新创建环境时，总会重置一下它）。正如上文所说，观察结果是4个数字，我们来看一下如何提前知道这个信息。

```Python
>>> e.action_space
Discrete(2)
>>> e.observation_space
Box(4,)
```

`action_space` 字段是 `Discrete` 类型，所以动作只会是 0 或 1 ，其中 0 代表将平台推向左边，1代表推向右边。观察空间是 `Box(4,)`，这代表大小为 4 的向量，其值在 [-inf, inf] 区间内。

```Python
>>> e.step(0)
(array([ 0.04273567, -0.24448391, -0.03459343,  0.29258258]), 1.0, False, {})
```

现在，通过执行动作 0 可以将平台推向左边，然后会获得包含 4 个元素的元组：
- 一个新的观察，即包含 4 个数字的新向量。
- 值为 1.0 的奖励。
- done 的标记为 False，表示片段还没有结束，目前的状态多少还是可以的。
- 环境的额外信息，在这里是一个空的字典。

接下来，对 `action_space` 和 `observation_space` 调用 `Space` 类的 `sample()` 方法。

```Python
>>> e.action_space.sample()
0
>>> e.action_space.sample()
1
>>> e.observation_space.sample()
array([ 2.4720373e+00, -5.7555515e+37, -2.8382450e-01, -9.5865417e+37],
      dtype=float32)
>>> e.observation_space.sample()
array([-6.08082891e-01,  2.65997636e+38,  1.09545745e-01, -1.21019449e+38],
      dtype=float32)
```

这个方法从底层空间返回一个随机样本，在 Discrete 动作空间的情况下，这意味着为 0 或 1 的随机数，而对于观察空间来说，这意味着包含 4 个数字的随机向量。对观察空间的随机采样看起来没什么用，确实是这样的，但当不知道如何执行动作的时候，从动作空间进行采样是有用的。在还不知道任何 RL 方法，却仍然想试一下 Gym 环境的时候，这个方法尤其方便。现在你知道如何为 CartPole 环境实现第一个行为随机的智能体了，我们来试一试。

## 3. 随机 CartPole 智能体
尽管这个环境比开头的那个例子里的环境复杂得多，但是智能体的代码却更短了。这就是重用性、抽象性以及第三方库的强大力量。

代码如下：

```Python
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" %(total_steps, total_reward))
```

在该循环中，我们从动作空间中随机采样一个动作，然后让环境执行并返回下一个观察(obs)、reward 和 done 标记。如果片段结束，停止循环并展示执行了多少步以及累积获得了多少奖励。

随机智能体在木棒落地、片段结束之前，平均会执行 12～15 步。大部分 Gym 环境有一个“奖励边界”，它是智能体在 100 个连续片段中，为解决环境而应该得到的平均奖励。对于 CartPole 来说，这个边界是 195 ，这意味着，平均而言，智能体必须将木棒保持 195 个时间步长或更多。从这个角度来看，随机智能体貌似表现得很差。但是，不要失望，我们才刚刚起步，很快你就能解决 CartPole 以及其他很多有趣且富有挑战的环境了。

## 4. Gym 的额外功能：包装器和监控器
### 4.1 包装器
很多时候，你希望以某种通用的方式扩展环境的功能。例如，想象一个环境，它给了你一些观察，但是你想将它们累积缓存起来，用以提供智能体最近的 N 个观察。这在动态计算机游戏中是一个很常见的场景，比如单一一帧不足以了解游戏状态的完整信息。例如，你希望能够裁剪或预处理一些图像素以便智能体来消化这些信息，又或者你想以某种方式归一化奖励值。有相同结构的场景太多了，你可能想要将现有的环境“包装”起来并附加一些额外的逻辑。Gym 为这些场景提供了一个方便使用的框架——`Wrapper` 类。

`Wrapper` 类继承自 `Env` 类。它的构造函数只有一个参数，即要被包装的 Env 类的实例。为了附加额外的功能，需要重新定义想扩展的方法，例如 step() 或 reset()。唯一的要求就是需要调用超类中的原始方法。

为了处理更多特定的要求，例如 Wrapper 类只想要处理环境返回的观察或只处理动作，那么用 Wrapper 的子类过滤特定的信息即可。它们分别是：
- ObservationWrapper：需要重新定义父类的 observation(obs) 方法。obs 参数是被包装的环境给出的观察，这个方法需要返回给予智能体的观察。
- RewardWrapper：它暴露了一个 reward(rew) 方法，可以修改给予智能体的奖励值。
- ActionWrapper：需要覆盖 action(act) 方法，它能修改智能体传给包装环境的动作。

为了让它更实用，假设有一个场景，我们想要以 10% 的概率干涉智能体发出的动作流，将当前动作替换成随机动作。这看起来不是一个明智的决定，但是这个小技巧可以解决利用与探索问题，它是最实用、最强大的方法之一。通过发布随机动作，让智能体探索环境，时不时地偏离它原先的策略的轨迹。使用 ActionWrapper 类很容易就可以实现：

```Python
import gym
from typing import TypeVar
import random

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon
```

先通过调用父类的 `__init__` 方法初始化包装器，并保存 epsilon（随机动作的概率）。

```Python
def action(self, action: Action) -> Action:
    if random.random() < self.epsilon:
        print("Random!")
        return self.env.action_space.sample()
    return action
```

我们需要覆盖这个方法，并通过它来修改智能体的动作。每一次都先掷骰子，都会有 epsilon 的概率从动作空间采样一个随机动作，用来替换智能体传给我们的动作。注意，这里用了 `action_space` 和包装抽象，这样就能写抽象的代码了，这适用于 Gym 的任意一个环境。另外，每次替换动作的时候必须将消息打印出来，以验证包装器是否生效。当然，在生产代码中，这不是必需的。

```Python
if __name__ == "__main__":
    env = RandomActionWrapper(gym.make('CartPole-v0'))
```

是时候应用一下包装器了。创建一个普通的 CartPole 环境，并将其传入 Wrapper 构造函数。然后，将 Wrapper 类当成一耳光普通的 Env 实例，用它来取代原始的 CartPole。因为 Wrapper 类继承自 Env 类，并且暴露了相同的接口，我们可以任意地嵌套包装器。

```Python
obs = env.reset()
total_reward = 0.0

while True:
    obs, reward, done, _ = env.step(0)
    total_reward += reward
    if done:
        break

print("Reward got: %.2f" % total_reward)
```

除了智能体比较笨，每次都选择同样的 0 号动作外，代码几乎相同。通过运行代码，应该能看到包装器确实生效了。

如果愿意，可以在包装器创建时指定 epsilon 参数，验证这样的随机性平均下来，会提升智能体得到的分数。

### 4.2 监控器
它的实现方式与 Wrapper 类似，可以将智能体的性能信息写入文件，也可以选择将智能体的动作录下来。

下面来看一下如何将 Monitor 加入随机 CartPole 智能体中，唯一的区别就是下面这段代码：

```Python
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = gym.wrappers.Monitor(env, "recording")
```

传给 Monitor 类的第二个参数是监控结果存放的目录名。目录不应该存在，否则程序会抛出异常。