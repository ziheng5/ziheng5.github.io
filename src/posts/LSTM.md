---
title: LSTM 原理及其 PyTorch 实现
date: 2024-12-13 13:00:36
tags: 
    - PyTorch
    - 深度学习
    - RNN
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
>
> （ 发明 LSTM 的人真 ** 是个天才！）

## 1. ❓ 什么是 LSTM
长短期记忆网络（LSTM，Long-Short-Term Memory）是传统 RNN 网络的 Plus 版本。
### 1.1 发明背景
传统的 RNN 网络在训练的时候，当遇到长序列数据时，很容易出现 **梯度爆炸** 与 **梯度消失** 的情况，导致训练效果不太好。

> 👀 什么？你不知道什么是 **梯度爆炸** 和 **梯度消失**？！快来看看这个视频：
> 
> https://www.youtube.com/watch?v=AsNTP8Kwu80

为了解决这一问题，LSTM 在传统 RNN 的基础上，加入了 **门控机制（Gate）** 来控制信息流动，从而记住长期依赖信息。

### 1.2 原理详解
> 先看视频：https://www.youtube.com/watch?v=YCzL96nL7j0&t=1s

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
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
  - 输入门激活：  
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
  - 更新单元状态：  
    $$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$
- **解释：**
  - 候选信息 $\tilde{C}_t$：当前时间步的新信息。
  - 输入门 $i_t$：控制候选信息的存储程度。  

#### **输出门（Output Gate）**
- **功能：** 决定单元状态中的信息 **公开输出**。
- **公式：**
  - 输出门激活：  
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  - 最终隐藏状态：  
    $$h_t = o_t \cdot \tanh(C_t)$$
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

## 2. PyTorch 手动复现 LSTM（灵活度高）
> 参考教程：https://www.youtube.com/watch?v=RHGiXPuo_pI
### 2.1 📦导入包
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
```

⚠️注意：这里有个叫 `lightning` 的包。

> 什么？你不知道这个包是用来干什么的？！PyTorch Lightning 是一个基于 PyTorch 的深度学习框架，其功能相当强大，可以一键实现很多功能！
>
> 官方文档：https://lightning.ai/docs/pytorch/stable/
>
> 下载方式:
>
> 1. pip 用户：
> ```Terminal
> pip install lightning
> ```
> 2. conda 用户：
> ```Terminal
> conda install lightning
> ```

### 2.2 ✋手搓 LSTM 网络
```python
class LSTMbyHand(L.LightningModule):
    def __init__(self):
        # create and initialize weight and bias tensors
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        ## 1. 遗忘门 
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## 2. 输入门
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## 3. 输出门
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)


    def lstm_unit(self, input_value, long_memory, short_memory):
        # do the lstm math
        ## 1. 遗忘门
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) +
                                              (input_value * self.wlr2) +
                                              self.blr1)
        
        ## 2. 输入门
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) +
                                                   (input_value * self.wpr2) +
                                                   self.bpr1)
        
        potential_memory = torch.tanh((short_memory * self.wp1) +
                                      (input_value * self.wp2) +
                                      self.bp1)
        
        updated_long_memory = ((long_memory * long_remember_percent) +
                              (potential_memory * potential_remember_percent))
        
        ## 3. 输出门
        output_percent = torch.sigmoid((short_memory * self.wo1) +
                                       (input_value * self.wo2) +
                                       self.bo1)
        
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        ## 4. 输出
        return ([updated_long_memory, updated_short_memory])


    def forward(self, input):
        # make a forward pass through unrolled lstm
        long_memory = 0
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory


    def configure_optimizers(self):
        # configure adam optimizer
        return Adam(self.parameters())
    

    def training_step(self, batch, batch_idx):
        # calculate loss and log training progress
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)
        
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
```

### 2.3 🔍检查网络是否正确搭建
```python
model = LSTMbyHand()

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted = ", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("Company B: Observed = 1, Predicted = ", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
```

### 2.4 💪开始训练
```python
inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=2000)
trainer.fit(model, train_dataloaders=dataloader)
```

### 2.5 🔎检查训练效果
```bash
tensorboard --logdir=lightning_logs/
```
![result](/images/lstm/tensorboard1.png)

发现效果一般😢

### 2.6 💪迁移学习
```python
path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path

trainer = L.Trainer(max_epochs=5000)
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)
```
再次查看效果：

![result2](/images/lstm/tensorboard3.png)

效果巨好👌

## 3. PyTorch 内置 LSTM 的使用
### 3.1 📦导入包
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader
```

### 3.2 🧱搭建网络
```python
class LightningLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    
    def forward(self, input):
        input_trans = input.view(len(input), 1)

        lstm_out, temp = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction
    

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if (label_i==0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
```
这里需要注意的是，在 PyTorch 中，LSTM 的输入格式为 `(batch_size, sequence_length, input_size)`，即**样本数量、时间步长、特征数量**。

如果实际中的数据格式与此不同，需要手动处理成对应的格式。


### 3.3 💪开始训练
```python
model = LightningLSTM()

trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
trainer.fit(model, train_dataloaders=dataloader)
```