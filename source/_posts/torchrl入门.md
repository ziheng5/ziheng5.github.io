---
title: TorchRL 入门 1
date: 2024-12-07 20:00:00
tags: 
    - 强化学习
    - PyTorch
    - TorchRL
categories: 
    - 强化学习
description: |
    Environment, TED 和 transforms
---
> [TorchRL](https://pytorch.org/rl/stable/index.html) 是 PyTorch 下的一个用于强化学习的包
> 
> 使用前请先安装 torchrl：
>
> ```Terminal
> pip install torchrl
> ```

# 强化学习中的环境
## 1. 创建环境
实际上，TorchRL 并不直接提供环境，而是为封装模拟器的其他库提供包装器，该 `envs` 模块可以被视为通用环境 API 的提供者，以及 [gym](https://gymnasium.farama.org/) ( `GymEnv` )、 Brax ( `BraxEnv` ) 或 DeepMind Control Suite ( `DMControlEnv` ) 等模拟后端的中央枢纽。

创建环境通常与底层后端 API 允许的一样简单。下面是使用 gym 的示例：

```Python
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
```
## 2.  运行环境
TorchRL 中的环境有两个关键方法： `reset()`，用于启动情节，以及 `step()`，用于执行参与者选择的动作。在 TorchRL 中，环境方法读取和写入 `TensorDict` 实例。本质上，`TensorDict` 是张量的基于密钥的通用数据载体。与普通张量相比，使用 TensorDict 的好处是：它使我们能够交替处理简单和复杂的数据结构，且它消除了适应不同数据格式的难题。

话不多说，我们来看看 `tensordict` 实例是什么样子的：

```Python
reset = env.reset()
print(reset)
```

输出如下：

> TensorDict(
    fields={
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)

现在让我们在动作空间中**随机**采取一个动作。首先，**对动作进行采样**：

```Python
reset_with_action = env.rand_action(reset)
print(reset_with_action)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)

此 `tensordict` 的结构与从中获得的结构相同， 但 `EnvBase()` 多了一个 "`action`" 条目。你可以轻松访问操作，用法和 Python 自带的**字典**结构基本一致：

```Python
print(reset_with_action)
```

输出如下：

> tensor([0.0635])

接下来，我们需要将把整个 `tensordict` 传递给该 `step` 方法，因为在更高级的情况下（如**多智能体强化学习**或**无状态环境**），可能需要读取多个张量：

```Python
stepped_data = env.step(reset_with_action)
print(stepped_data)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)

这里需要指出，这个新的张量字典与前一个完全相同，只是它有一个 "`next`" 条目（本身就是一个张量字典！），其中包含由我们的行动产生的观察、奖励和完成状态。

我们将这种格式称为 **TED**，即 TorchRL Episode Dictory 数据格式。它是库中表示数据的普遍方式，既可以像这里一样动态表示，也可以使用离线数据集静态表示。

在环境中运行部署所需的最后一点信息是如何将该 "`next`" 条目置于根目录以执行下一步。TorchRL 提供了一个专门的 `step_mdp()` 功能来执行此操作：它会过滤掉您不需要的信息，并在马尔可夫决策过程 (MDP) 中的某个步骤之后提供与您的观察结果相对应的数据结构。

```Python
from torchrl.envs import step_mdp

data = step_mdp(stepped_data)
print(data)
```

输出如下：

> TensorDict(
    fields={
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)

## 3. 环境推出
写下这三个步骤（计算动作、采取步骤、在 MDP 中移动）可能有点繁琐和重复。幸运的是，TorchRL 提供了一个很好的 `rollout()` 函数，允许你随意在闭环中运行它们：

```Python
rollout = env.rollout(max_steps=10)
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

`stepped_data` 除了批处理大小之外，此数据与上面的数据非常相似，批处理大小现在等于我们通过 `max_steps` 参数提供的步骤数。tensordict 的魔力不止于此：如果你对此环境的单个转换感兴趣，则可以像索引张量一样对 tensordict 进行索引：

```Python
transition = rollout[3]
print(transition)
```

输出如下：

> TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
        done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)

`TensorDict` 将自动检查您提供的索引是否是键（在这种情况下，我们沿着键维度进行索引）或像这样的空间索引。

按照这种方式执行（没有策略），该 `rollout` 方法可能看起来相当无用：它只是运行**随机操作**。如果有策略可用，则可以**将其传递给该方法并用于收集数据**。

尽管如此，首先运行一个简单的、无策略的部署来检查对环境的期望是有用的。

要了解 TorchRL API 的多功能性，请考虑这样一个事实：`rollout` 方法具有普遍适用性。它适用于所有用例，无论你使用的是像这样的单一环境、跨各种流程的多个副本、多代理环境，甚至是无状态版本！

## 4. 改变环境
大多数情况下，你需要修改环境的输出以更好地满足您的要求。

例如，你可能想要监控自上次重置以来执行的步骤数、调整图像大小或将连续的观察结果堆叠在一起。

在本节中，我们将研究一种简单的变换，即 `StepCounter` 变换。完整的变换列表可在[这里](https://pytorch.org/rl/stable/reference/envs.html#id2)找到。

变换通过以下方式与环境集成 `TransformedEnv`：

```Python
from torchrl.envs import StepCounter, TransformedEnv

transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
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
                step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([10, 3]), device=cpu, dtype=torch.float32, is_shared=False),
        step_count: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)

如你所见，我们的环境现在多了一个条目，"`step_count`"用于跟踪自上次重置以来的步数。鉴于我们将可选参数传递 `max_steps=10` 给了变换构造函数，我们还在 10 步后截断了轨迹（没有像我们在调用时要求的那样完成 100 步的完整展开rollout）。我们可以通过查看截断的条目来看到轨迹被截断了：

```Python
print(rollout["next", "truncated"])
```

输出如下：

> tensor([[False],
>        [False],
>        [False],
>        [False],
>        [False],
>        [False],
>        [False],
>        [False],
>        [False],
>        [ True]])