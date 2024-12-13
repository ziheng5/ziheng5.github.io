---
title: LSTM 原理及其 PyTorch 实现
date: 2024-12-13 13:00:36
tags: 
    - PyTorch
    - 深度学习
categories:
    - 深度学习
description: |
    📖 使用 PyTorch 实现 LSTM
---
> PyTorch 其实内置了 LSTM 模型，直接调用即可，不需要费劲去手搓了
>
> （某个人复现到一半才反应过来 😭）
>
> ✨通俗易懂的 LSTM 原理讲解（力推）：
> https://www.youtube.com/watch?v=YCzL96nL7j0&t=1s

### 1. 导入必要的库
```Python
import torch
import torch.nn as nn
import torch.optim as optim
```

### 2. 定义 LSTM 模型

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


### 3. 初始化模型和超参数

```Python
input_size = 10    # 输入特征数
hidden_size = 50   # 隐藏层大小
output_size = 1    # 输出特征数
num_layers = 2     # LSTM 层数

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 4. 创建数据示例

```Python
# 生成随机输入和标签
x_train = torch.randn(100, 5, input_size)  # (批次大小, 时间步, 输入大小)
y_train = torch.randn(100, output_size)
```

### 5. 训练模型

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

### 6. 评估模型

```Python
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 5, input_size)
    prediction = model(test_input)
    print(f'Prediction: {prediction}')
```
