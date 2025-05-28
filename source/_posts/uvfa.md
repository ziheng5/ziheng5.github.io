---
title: 关于 UAFA 算法的笔记
date: 2025-05-27 21:06:54
tags:
    - 强化学习
categories:
    - 强化学习
description: |
    📚 对 Q 函数的优化
---

> 论文地址：

# 1. 什么是 Universal Value Function Approximator（UVFA）？

Universal Value Function Approximator（通用值函数逼近器）是强化学习中对传统 **Q函数** 的一个扩展。

在传统的强化学习中，**Q函数 Q(s, a)** 表示在状态 $s$ 下采取动作 $a$ 能获得的期望回报。但这个函数只能处理**单一任务/目标**。

而 UVFA 允许我们同时处理 **多个目标（multi-goal）或多个任务（multi-task）** 的问题。

---

## 1.1 核心思想

### 🔁 从 Q(s, a) → Q(s, a, g)

UVFA 把“目标 $g$” 作为一个额外的输入引入 Q 函数中：

$$
Q(s, a, g) = \mathbb{E}[R_t | s_t = s, a_t = a, g]
$$

其中：

* $s$：当前状态；
* $a$：动作；
* $g$：目标（goal）或任务的编码；
* $R_t$：未来累计回报。

> ✅ 本质上是学习一个可以泛化到多个目标的 Q 函数。

---

## 1.2 应用场景

UVFA 非常适合处理 **目标可变的任务**，比如：

| 任务          | 目标 g 表示什么？ |
| ----------- | ---------- |
| 迷宫导航        | 终点位置坐标     |
| 抓取物体        | 抓取目标物体的 ID |
| 控制机器人到达某个位置 | 目标位置坐标     |

在这种情况下，我们可以用一个统一的 Q 函数，训练出能够适应不同目标的策略。

---

## 1.3 UVFA 的训练方式

UVFA 可以和常规的强化学习算法（如 DQN、DDPG）结合使用。其训练方式与标准 Q-learning 类似，只是将目标 $g$ 一并作为输入。

### Bellman 更新形式：

$$
Q(s, a, g) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a', g)
$$

这里的目标 $g$ 在整个 episode 中保持不变。

---

## 1.4 神经网络实现方式

你可以将状态 $s$、目标 $g$、动作 $a$ 作为神经网络的输入之一，例如：

```python
input = concat(state, goal)
Q_value = Q_network(input, action)
```

目标 $g$ 可以是：

* 坐标（如 \[x, y, z]）；
* 离散编码（如 one-hot）；
* 图像（可用 CNN 编码）；
* 语言（可用 BERT/Transformer 编码）等。

---

## 1.5 UVFA 的优势

| 优点                | 说明                    |
| ----------------- | --------------------- |
| 多目标统一建模           | 一个模型适应多个任务或目标         |
| 目标泛化能力强           | 可以学习未见过的新目标           |
| 适用于 sparse reward | 可与 HER 等方法联合使用，提高学习效率 |

---

## 1.6 与 HER 的关系

HER（Hindsight Experience Replay）正是构建在 UVFA 之上的。

* HER 的 replay buffer 中，不仅记录状态和动作，还记录每一步的新目标；
* 借助 UVFA，HER 可以处理目标变化带来的策略变化；
* 因此 HER 能在稀疏奖励中快速学习达成目标的方法。


---

# 2. 示例任务：GridWorld 多目标导航

## 2.1 环境设定

- 环境是一个 5×5 的二维网格
- 状态 $s = (x, y)$，表示智能体当前位置
- 目标 $g = (x_g, y_g)$
- 动作空间：上、下、左、右（共 4 个动作）
- 奖励：

  - 到达目标时：+0（成功）
  - 其他任何状态：-1（稀疏奖励）

---

## 2.2 UVFA 代码结构（基于 PyTorch）

### 1️⃣ 状态 + 目标 合并输入

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UVFA_QNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)  # 拼接 s 和 g
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)
        return q_values
```

✅ 输入维度：状态 + 目标 = 2 + 2 = 4
✅ 输出维度：动作数（4个）

---

### 2️⃣ 示例输入

```python
state = torch.tensor([[1.0, 2.0]])  # 当前在 (1,2)
goal = torch.tensor([[4.0, 4.0]])   # 目标在 (4,4)

q_net = UVFA_QNetwork()
q_values = q_net(state, goal)

print(q_values)
# 输出：tensor([[..., ..., ..., ...]])，代表4个动作的Q值
```

---

### 3️⃣ Q-Learning 更新（伪代码）

```python
# 假设 transition (s, a, r, s', g)
q_values = q_net(s, g)
q_value = q_values.gather(1, a)  # 选中动作a对应的Q值

with torch.no_grad():
    q_next = q_net(s_next, g).max(1)[0].unsqueeze(1)
    q_target = r + gamma * q_next

loss = F.mse_loss(q_value, q_target)
```

---

### 4️⃣ 多目标训练示意

每次 episode：

* 随机采样初始位置 $s$
* 随机采样目标位置 $g$
* 输入网络的是 $s$ 与 $g$，目标是到达 $g$

训练后：

* 给定任意 $g$，都能导出一条通往目标的路径！

---

### 📈 效果与泛化能力

| 训练目标数    | 泛化目标        |
| -------- | ----------- |
| 5 个目标    | ✔ 部分未见目标可成功 |
| 所有位置作为目标 | ✔ 全部目标都可导航  |
| 单目标训练    | ✖ 无法泛化到其他目标 |

> ✅ 使用 UVFA 后，模型具备 **目标泛化能力**。

---

### 🚀 小结

| 模块   | 内容                      |
| ---- | ----------------------- |
| 状态输入 | s = (x, y)              |
| 目标输入 | g = (xg, yg)            |
| 网络输入 | \[x, y, xg, yg]         |
| 输出   | Q(s, a, g)，每个动作的Q值      |
| 好处   | 一套模型，多目标导航、可泛化、支持 HER 等 |

---

## 📚 参考文献

* Schaul et al., 2015 — *Universal Value Function Approximators*, ICML
* Andrychowicz et al., 2017 — *Hindsight Experience Replay*, NeurIPS

