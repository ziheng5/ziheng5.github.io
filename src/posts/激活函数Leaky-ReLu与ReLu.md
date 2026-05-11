---
title: 激活函数 Leaky ReLU 与 ReLU 的区别
date: 2024-11-29 17:04:48
tags:
    - 深度学习
categories: 
    - 深度学习
description: |
    简单解释 Leaky ReLU 与 ReLU 之间的区别
---
# 1. ReLU（线性修正单元）
ReLU 函数对于正数部分直接输出，对于负数部分输出为零
> 如果你学过模电的话，这玩意儿是不是很像**二极管**？ ReLU 函数的发明其实正是参考了二极管的功能特性。

$$ f(x)=max(0,x) $$
- 优点：非常简单，计算速度快。在很多情况下表现良好

- 缺点：可能存在**神经元死亡**的问题，即某些神经元在训练过程中可能永远不会被激活，导致权重无法更新。

# 2. Leaky ReLU
Leaky ReLU 对于负数部分不再输出零，而是输出一个很小的负数，通常用一个小斜率 $\alpha$ 乘以输入。

$$ f(x)=\left\\{\begin{matrix} x, {if} {x>0} \\\\ \alpha x, if x\le0 \end{matrix}\right. $$

- 优点：解决零 ReLU 的神经元死亡问题，因为负数有一个小的梯度。

- 缺点：对于 $\alpha$ 的选择比较敏感，需要调参。 
