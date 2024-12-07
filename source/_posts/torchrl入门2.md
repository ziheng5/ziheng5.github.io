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
与环境与实例交互的方式类似 `TensorDict`，用于表示策略和值函数的模块也执行相同的操作。核心思想很简单：将标准 `Module` （或任何其他函数）封装在一个类中，该类知道需要读取哪些条目并将其传递给模块，然后使用分配的条目记录结果。为了说明这一点，我们将使用最简单的策略：从观察空间到动作空间的确定性映射。为了获得最大的通用性，我们将使用一个带有 `LazyLinear` 的我们在上一个教程中实例化的 `Pendulum` 环境的模块。

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