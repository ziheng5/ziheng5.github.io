---
title: Coldrain 的 27 考研数一高数强化阶段拾遗（下）
date: 2026-05-24 16:13:00
tags: 
    - 考研数学
categories: 
    - 考研数学
description: |
    Coldrain 的高数代数备考笔记，涵盖了基础 + 强化的重要内容，以及一些题解拾遗（施工中 🚧）
---

> ✍ 写在前面
>
> 本笔记为 Coldrain 二刷基础时所记，故笔记内容并没有做到全覆盖，而只针对每一章节重要且容易遗忘的知识点，所以本笔记可用于一轮学习结束之后对重难考点进行查漏补缺，但请不要用于替代考研书籍来进行一轮复习
>
> “岂不闻天无绝人之路，只要我想走，路就在脚下。”—— 25 奥本海豚


## 15. 微分方程

1. **一阶线性微分方程**
   - 形如 $y' + p(x) y =q(x)$ 的方程叫做一阶线性微分方程，其中 $p(x)$、$q(x)$ 为连续函数
   - 通解：$y = e^{-\int p(x)dx} [\int e^{\int p(x)dx} \cdot q(x)dx + C]$
   - 其推导过程为：在原微分方程两边同时乘 $e^{\int p(x)dx}$

2. **伯努利方程**
   - 形如 $\dfrac{dy}{dx} + p(x)y = q(x) y^n (n\ne 0, 1)$
   - 解法：
     - （1）先变形为 $y^{-n} \cdot \dfrac{dy}{dx} + p(x)y^{1-n} = q(x)$
     - （2）令 $z = y^{1-n}$ 可得 $\dfrac{dz}{dx} = (1-n) y^{-n} \dfrac{dy}{dx}$，则 $\dfrac{1}{1-n} \dfrac{dz}{dx} + p(x)z = q(x)$
     - （3）解此一阶线性微分方程

3. **二阶常系数齐次线性微分方程**
   - 形如 $y'' + py' + qy = 0$，其中 $p$、$q$ 为常数
   - 通解：先写出对应的特征方程 $r^2 + pr + q = 0$（对原微分方程令 $y = e^{rx}$ 得到的）
     - （1）特征方程有两个不等实根：通解为 $y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$
     - （2）特征方程有两个相等实根：通解为 $y = (C_1 + C_2 x) e^{rx}$
     - （3）特征方程有共轭复根 $\alpha \pm \beta i$：通解为 $y = e^{\alpha x} (C_1 \cos\beta x + C_2 \sin\beta x)$

4. **二阶常系数非齐次线性微分方程**
   - 形如 $y'' + py' + qy = f(x)(f(x) \ne 0)$
   - 解的结构：
     - （1）若 $y_1^*(x)$ 是 $y'' + py' + qy = f_1(x)$ 的解，$y_2^*(x)$ 是 $y'' + py' + qy = f_2(x)$ 的解，那么 $y_1^*(x) + y_2^*(x)$ 是 $y'' + py' + qy = f_1(x) + f_2(x)$ 的解（线性）
     - （2）若 $y_1^*$、$y_2^*$ 都是 $y'' + py' + qy = f(x)$ 的特解，则 $y_1^* - y_2^*$ 对应齐次方程的解
   - 当自由项 $f(x) = P_n(x) e^{\alpha x}$ （$P_n(x)$ 为 $x$ 的 $n$ 次多项式）时，特解为 $y^* = e^{\alpha x}Q_n(x) x^k$，其中：
     - （1）$e^{\alpha x} $ 照抄
     - （2）$Q_n(x)$ 为 $x$ 的 $n$ 次多项式
     - （3）$k = \begin{cases} 0, & \alpha 不是特征根 \\ 1, & \alpha 是单特征根 \\ 2, & \alpha 是二重特征根 \end{cases}$
   - 当自由项 $f(x) = e^{\alpha x} [P_m(x) \cos \beta x + P_n(x) \sin \beta x]$ 时，特解为 $y^* = e^{\alpha x} [Q_l^{(1)}(x) \cos \beta x + Q_l^{(2)}(x) \sin\beta x] x^k$
     - （1）$e^{\alpha x}$ 照抄
     - （2）$l = \max \{m, n\}$，$Q_l^{(1)}(x)$、$Q_l^{(2)}(x)$ 分别为 $x$ 的两个不同的 $l$ 次多项式
     - （3）$k = \begin{cases} 0, & \alpha\pm\beta i 不是特征根 \\ 1, & \alpha\pm\beta i 是特征根 \end{cases}$