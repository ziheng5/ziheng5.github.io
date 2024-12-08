---
title: torchrl 入门 2
date: 2024-12-07 22:03:37
tags: 
    - RL（强化学习）
    - PyTorch
    - TorchRL
categories: 
    - TorchRL
description: |
    如何使用 TorchRL 的模块
---
# 开始使用 TorchRL 的模块
## 1. TensorDict 模块
与环境与实例交互的方式类似 `TensorDict`，用于表示策略和值函数的模块也执行相同的操作。核心思想很简单：将标准 `Module` （或任何其他函数）封装在一个类中，该类知道需要读取哪些条目并将其传递给模块，然后使用分配的条目记录结果。为了说明这一点，我们将使用最简单的策略：**从观察空间到动作空间的确定性映射**。为了获得最大的通用性，我们将使用一个带有 `LazyLinear` 的我们在上一个教程中实例化的 `Pendulum` 环境的模块。

```Python
import torch

from torchrl.envs import GymEnv
from tensordict.nn import TensorDictModule

env = GymEnv("Pendulum-v1")
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)
```

这就是执行我们的策略所需的全部内容！使用惰性模块可以让我们绕过获取观察空间形状的需要，因为模块会自动确定它。此策略现已准备好在环境中运行：

```Python
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

## 2. 专用的 wrappers
为了简化 `Actor` 的创建，我们可以使用 `ProbabilisticActor`、`ActorValueOperator` 或者 `ActorCriticOperator`。例如，`Actor` 为 `in_keys` 和 `out_keys` 提供默认值，从而可以直接与许多常见环境集成：

```Python
from torchrl.modules import Actor

policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

具体的调用参数可以在 [API文档](https://pytorch.org/rl/stable/reference/modules.html#tdmodules) 中找到。

## 3. Network
TorchRL 还提供了常规模块，无需重复使用 `tensordict` 特征即可使用。您将遇到的两个最常见的网络是 `CNN` 和 `MLP` 模块。我们可以用以下之一替换我们的策略模块：

```Python
from torchrl.modules import MLP

module = MLP(
    out_features=env.action_spec.shape[-1],
    num_cells=[32, 64],
    activation_class=torch.nn.Tanh,
)
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
```

这里我们创建了一个简单的 `MLP` 网络结构，下面我们来具体看一看它的结构：

```Python
print(module)
```

输出如下：

> MLP(
> 
>  (0): Linear(in_features=3, out_features=32, bias=True)
> 
>  (1): Tanh()
> 
>  (2): Linear(in_features=32, out_features=64, bias=True)
> 
>  (3): Tanh()
> 
>  (4): Linear(in_features=64, out_features=1, bias=True)
> 
>)

可以看到我们的网络一共有**三层**，`num_cells=[32, 64]` 代表的是：第一层输出有32个神经元，第二层输出有64个神经元。

## 4. 概率策略
策略优化算法（如 `PPO`）要求策略具有随机性：与上述示例不同，模块现在编码了从观察空间到参数空间的映射，该映射对可能的操作的分布进行了编码。TorchRL 通过将各种操作（例如从参数构建分布、从该分布中采样以及检索对数概率）归入单个类来促进此类模块的设计。在这里，我们将使用三个组件构建一个依赖于常规正态分布的 actor：

- 一个`MLP`主干读取大小为3的观测值并输出大小为2的单个张量；

- NormalParamExtractor将此输出分成两块的模块，即大小为的平均值和标准差[1]；

- 它将ProbabilisticActor读取这些参数in_keys，用它们创建一个分布，并用样本和对数概率填充我们的张量字典。

```Python
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor

backbone = MLP(in_features=3, out_features=2)   # 默认有4层网络，除了最后一层，每层输出都是32个神经元
extractor = NormalParamExtractor()
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor(
    td_module,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True,
)

rollout = env.rollout(max_steps=10, policy=policy)
print(rollout)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        loc: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
        sample_log_prob: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        scale: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

这里我们可以查看一下具体参数：

```Python
print(module)
```

输出如下：

>Sequential(
>
>  (0): MLP(
> 
>    (0): Linear(in_features=3, out_features=32, bias=True)
> 
>    (1): Tanh()
> 
>    (2): Linear(in_features=32, out_features=32, bias=True)
> 
>    (3): Tanh()
> 
>    (4): Linear(in_features=32, out_features=32, bias=True)
> 
>    (5): Tanh()
> 
>    (6): Linear(in_features=32, out_features=2, bias=True)
> 
>  )
> 
>  (1): NormalParamExtractor(
> 
>    (scale_mapping): biased_softplus()
> 
>  )
> 
>)

这里还有几点需要注意：

- 由于我们在构建 actor 时就要求这样做，因此还会写入当时分布下**动作的对数概率**。这对于 PPO 之类的算法来说是必要的。

- 分布的参数也在"`loc`"和"`scale`"条目下的输出张量字典中返回。

如果应用程序需要，您可以控制操作的采样以使用预期值或分布的其他属性，而不是使用随机样本。这可以通过以下 `set_exploration_type()` 函数进行控制：

```Python
from torchrl.envs.utils import ExplorationType, set_exploration_type

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # takes the mean as action
    rollout = env.rollout(max_steps=10, policy=policy)
with set_exploration_type(ExplorationType.RANDOM):
    # Samples actions according to the dist
    rollout = env.rollout(max_steps=10, policy=policy)
```

## 5. exploration
像这样的随机策略在某种程度上自然地在探索和利用之间进行权衡，但确定性策略则不会。幸运的是，TorchRL 还可以通过其探索模块来缓解这种情况。我们将以探索 `EGreedyModule` 模块为例。要查看此模块的实际操作，让我们恢复到确定性策略：

```Python
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

policy = Actor(MLP(3, 1, num_cells=[32, 64]))
```

在使用 $\varepsilon$-greedy exploration 模块时，我们需要定义一些退火方法和一个初始值 $\varepsilon$ 参数。值为 $\varepsilon=1$ 的策略意味着采取的每一个行动都是随机的；而值为 $\varepsilon=0$ 的策略意味着根本没有探索，每一个步骤都是固定的。如果需要退火探索因子，`step()` 需要调用：

```Python
exploration_module = EGreedyModule(
    spec=env.action_spec, annealing_num_steps=1000, eos_init=0.5
)
```

为了构建我们的探索性策略，我们只需要将确定性策略模块与 TensorDictSequential模块内的探索模块连接起来（这类似于Sequential张量字典领域）。

```Python
exploration_policy = TensorDictSequential(policy, exploration_module)

with set_exploration_type(ExplorationType.DETERMINISTIC):
    # 不探索
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
    # 探索
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
```

因为它必须能够在动作空间中对随机动作进行采样，所以 `EGreedyModule` 必须配备来自环境的 `action_space` 才能知道使用什么策略来随机采样动作。

## 6. Q-Value actors
在某些情况下，策略不是独立模块，而是构建在另一个模块之上。Q值参与者就是这种情况。简而言之，这些参与者需要估计动作值（大多数时候是离散的），并会贪婪地选择具有最高值的动作。在某些情况下（有限离散动作空间和有限离散状态空间），人们可以只存储一个 2D 状态动作对表并选择具有最高值的动作。DQN 带来的创新 是利用神经网络对值图进行编码，将其扩展到连续状态空间 。为了更清楚地理解，让我们考虑另一个具有离散动作空间的环境：Q(s, a)

```Python
env = GymEnv("CartPole-v1")
print(env.action_spec)
```

输出如下：

> OneHot(
    shape=torch.Size([2]),
    space=CategoricalBox(n=2),
    device=cpu,
    dtype=torch.int64,
    domain=discrete)

我们构建一个价值网络，当它从环境中读取状态时，每个动作都会产生一个价值：

```Python
num_actions = 2
value_net = TensorDictModule(
    MLP(out_features=num_actions, num_cells=[32, 32]),
    in_keys=["observation"],
    out_keys=["action_value"],
)
```

`QValueModule` 我们可以通过在价值网络后添加以下内容来轻松构建我们的 Q-Value actor ：

```Python
from torchrl.modules import QValueModule

policy = TensorDictSequential(
    value_net,
    QValueModule(spec=env.action_spec),
)
```

让我们检查一下。我们运行该策略几个步骤并查看输出。我们应该在获得的 rollout 中找到一个"action_value"以及一个 条目："chosen_action_value"

```Python
rollout = env.rollout(max_steps=3, policy=policy)
print(rollout)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.int64, is_shared=False),
        action_value: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        chosen_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([3]),
    device=None,
    is_shared=False)

由于它依赖于 `argmax` 操作，因此该策略是确定性的。在数据收集期间，我们需要探索环境。为此，我们EGreedyModule 再次使用：

```Python
policy_exploration = TensorDictSequential(policy, EGreedyModule(env.action_spec))

with set_exploration_type(ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)
```
