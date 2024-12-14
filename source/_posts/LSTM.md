---
title: LSTM 原理及其 PyTorch 实现
date: 2024-12-13 13:00:36
tags: 
    - PyTorch
    - 深度学习
categories:
    - 深度学习
    - RNN 族算法
description: |
    📖 使用 PyTorch 实现 LSTM
---
> PyTorch 其实内置了 LSTM 模型，直接调用即可，不需要费劲去手搓了
>
> （某个人复现到一半才反应过来 😭）
>
> ✨通俗易懂的 LSTM 原理讲解（力推）：
> https://www.youtube.com/watch?v=YCzL96nL7j0&t=1s
>
> （ 发明 LSTM 的人真 ** 是个天才！）

## 1. ❓ 什么是 LSTM
长短期记忆网络（LSTM，Long-Short-Term Memory）是传统 RNN 网络的 Plus 版本。
### 1.1 发明背景
传统的 RNN 网络在训练的时候，当遇到长序列数据时，很容易出现 **梯度爆炸** 与 **梯度消失** 的情况，导致训练效果不太好。

为了解决这一问题，LSTM 在传统 RNN 的基础上，加入了 **门控机制（Gate）** 来控制信息流动，从而记住长期依赖信息。

### 1.2 原理详解
先看视频：https://www.youtube.com/watch?v=YCzL96nL7j0&t=1s

LSTM 由多个 **LSTM 单元（Cell）** 组成，每个单元包含以下三个门和一个单元状态：  

#### **遗忘门（Forget Gate）**
- **功能：** 决定哪些信息需要 **遗忘**。
- **公式：**  
$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
- **解释：**
  - 输入：前一个隐藏状态 $h_{t-1}$ 和当前输入 $x_t$。
  - 输出：范围在 $[0, 1]$，其中 0 表示完全遗忘，1 表示完全保留。  

#### **输入门（Input Gate）**
- **功能：** 决定哪些新信息需要 **存储**。
- **公式：**
  - 候选信息生成：  
    $$
    \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
    $$
  - 输入门激活：  
    $$
    i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
    $$
  - 更新单元状态：  
    $$
    C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
    $$
- **解释：**
  - 候选信息 $\tilde{C}_t$：当前时间步的新信息。
  - 输入门 $i_t$：控制候选信息的存储程度。  

#### **输出门（Output Gate）**
- **功能：** 决定单元状态中的信息 **公开输出**。
- **公式：**
  - 输出门激活：  
    $$
    o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
    $$
  - 最终隐藏状态：  
    $$
    h_t = o_t \cdot \tanh(C_t)
    $$
- **解释：**
  - 输出门决定当前时间步的隐藏状态 $h_t$，该状态将作为下一个时间步的输入。

#### **单元状态（Cell State）**
- **功能：**  
  - 作为信息的“长期记忆”路径，在整个时间序列中流动。
  - 线性传递，几乎不受激活函数的影响，确保长时间的信息保留。

### 1.3 LSTM 数据流总结：
1. 接收输入 $x_t$ 和上一个隐藏状态 $h_{t-1}$。  
2. **遗忘门** 确定需要遗忘的信息。  
3. **输入门** 确定存储的新信息。  
4. 更新单元状态 $C_t$。  
5. **输出门** 确定当前时间步的输出隐藏状态 $h_t$。  

### 1.4 LSTM 的优势：
- **长期依赖记忆：** 能够有效记住长期信息，解决了传统 RNN 的梯度消失问题。
- **适用场景：** 广泛用于自然语言处理、时间序列预测、语音识别等领域。  
- **灵活性高：** 支持多层堆叠，能够学习高度复杂的数据模式。  

---

## 2. PyTorch 实现 LSTM
### 2.1 导入必要的库
```Python
import torch
import torch.nn as nn
import torch.optim as optim
```

### 2.2 定义 LSTM 模型

```Python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        return out
```

PyTorch 提供的 `nn.LSTM` 含有四个参数：`input_size`、`hidden_size`、`num_layers`、`batch_first`。

其中，`batch_first` 用于指定输入张量的维度顺序，控制输入和输出张量的形状解释方式：

- 默认值：`batch_first=True`
  - 输入张量形状：`(seq_len, batch_size, input_size)`
  - 输出张量形状：`(seq_len, batch_size, hidden_size)`
- 设置为 `batch_first=True`：
  - 输入张量形状：`(batch_size, seq_len, input_size)`
  - 输出张量形状：`(batch_size, seq_len, hidden_size)`

这里为什么我们要用 `batch_first=True` 呢？
- 设置 `batch_first=True` 更符合大多数深度学习库的常用张量形式，使代码更直观。


### 2.3 初始化模型和超参数

```Python
input_size = 10    # 输入特征数
hidden_size = 50   # 隐藏层大小
output_size = 1    # 输出特征数
num_layers = 2     # LSTM 层数

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 2.4 创建数据示例

```Python
# 生成随机输入和标签
x_train = torch.randn(100, 5, input_size)  # (批次大小, 时间步, 输入大小)
y_train = torch.randn(100, output_size)
```

### 2.5 训练模型

```Python
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

### 2.6 评估模型

```Python
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 5, input_size)
    prediction = model(test_input)
    print(f'Prediction: {prediction}')
```
