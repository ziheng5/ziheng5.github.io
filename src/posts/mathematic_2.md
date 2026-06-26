---
title: Coldrain 的 27 考研数一高数强化阶段拾遗（下）
date: 2026-06-24 16:13:00
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

## 13. 多元函数微分学
1. **证明偏导数存在（可偏导）、连续、可微、偏导数连续的方法**
   - （1）可偏导（一元概念）：为证函数 $z = f(x, y)$ 在点 $(x_0, y_0)$ 处可偏导，即偏导数分别存在，即证两个极限 $\lim\limits_{\Delta x \to 0} \dfrac{f(x_0 + \Delta x, y_0) - f(x_0, y_0)}{\Delta x}$ 和 $\lim\limits_{\Delta y \to 0} \dfrac{f(x_0 , y_0+ \Delta y) - f(x_0, y_0)}{\Delta y}$ 存在
   - （2）连续（二元概念）：为证函数 $z = f(x ,y )$ 在点 $(x_0, y_0)$ 处连续，即证 $\lim\limits_{(x, y) \to (x_0, y_0)} f(x, y)= f(x_0, y_0)$（取两条不同路径看看结果是否一样）
   - （3）可微（二元概念，可以直接推出连续、偏导数存在）：为证函数 $z = f(x)$ 在点 $(x_0, y_0)$ 处可微，即证 $\lim\limits_{\rho \to 0} \dfrac{\Delta z - [f_x '(x_0 , y_0)\Delta x + f_y'(x_0, y_0)\Delta y]}{\rho} = \lim\limits_{\rho \to 0} \dfrac{f(x_0 + \Delta x, y_0 + \Delta y) - f(x_0, y_0) - [f_x '(x_0 , y_0)\Delta x + f_y'(x_0, y_0)\Delta y]}{\rho}$ 是否为 0，若为 0，则可微，若不为 0，则不可微，其中 $\rho = \sqrt{(\Delta x)^2+(\Delta y)^2}$（这个 $\rho$ 就是二元函数自变量的增量，对应一元函数 $\lim\limits_{\Delta x \to 0}\dfrac{f(x_0+\Delta x) - f(x_0)}{\Delta x}$ 中的 $\Delta x$）
   - （4）偏导数连续（二元概念）：为证 $z = f(x, y)$ 在 $(x_0, y_0)$  处偏导数连续，即证 $\lim\limits_{(x, y) \to (x_0, y_0)} f_x'(x, y) = f_x'(x_0, y_0)$、$\lim\limits_{(x, y) \to (x_0, y_0)} f_y'(x, y) = f_y'(x_0, y_0)$ 是否成立，若成立，则 $z = f(x, y)$ 在点 $(x_0, y_0)$ 处的偏导数是连续的

> ⚠️ 易错的误区：
> - $\lim\limits_{x\to x_0} f_x'(x, y_0) = f_x'(x_0, y_0)$ 是一个一元概念，无法用来证明偏导连续（二元概念）！（即前面那个式子和 $\lim\limits_{(x, y) \to (x_0, y_0)} f_x'(x, y) = f_x'(x_0, y_0)$ 并不等价！）


2. **上面四个概念的关系**

![relationship](/images/mathematic/chapter13_1.png)

> 💡 常用反例：
> - （1）$f(x, y) = \begin{cases} \dfrac{xy}{x^2 + y^2}, & (x,y ) \ne (0, 0) \\ 0, & (x, y)=(0, 0) \end{cases}$ 在 $(0 ,0 )$ 点偏导数存在，但不连续
> - （2）$f(x, y) = |x| + |y|$ 在 $(0, 0)$ 点连续，但偏导数不存在，也不可微
> - （3）$f(x, y) = \begin{cases} \dfrac{xy}{\sqrt{x^2 + y^2}}, & (x,y ) \ne (0, 0) \\ 0, & (x, y)=(0, 0) \end{cases}$ 在 $(0, 0)$ 点处连续且偏导数存在，但不可微
> - （4）$f(x, y) = \begin{cases} (x^2 + y^2)\sin \dfrac{1}{x^2 + y^2}, & (x,y ) \ne (0, 0) \\ 0, & (x, y)=(0, 0) \end{cases}$ 在 $(0, 0)$ 点可微，但偏导数不连续

> 💡 上面的概念辨析比较易错，可以看看[没咋了的视频讲解](https://www.bilibili.com/video/BV1MVjSz1Exj/?)

3. **偏导数的一些散装知识点**
   - （1）$\dfrac{\partial ^2 z}{\partial x \partial y} = \dfrac{\partial}{\partial y} (\dfrac{\partial z}{ \partial x})$
   - （2）如果 $z = f(x, y)$ 的两个二阶混合偏导数 $f_{12}''(x, y)$ 和 $f_{21}''(x, y)$ 都在区域 $D$ 内连续，则 $f_{12}''(x, y) = f_{21}''(x, y)$，即二阶混合偏导数在连续的条件下与求导次序无关
   - （3）$z = f(u, v)$，$u = u(x, y)$，$v = v(x, y)$ 且三个函数均有连续偏导数，则 $z = f(u, v)$ 的全微分为 $dz = \dfrac{\partial z}{\partial u} du + \dfrac{\partial z}{ \partial v} dv$

4. **隐函数存在定理**
   - （1）对于由方程 $F(x, y) = 0$ 确定的隐函数 $y = f(x)$，当 $\textcolor{red}{F_y'(x, y) \ne 0}$ 且二元函数 $F(x, y)$ 在给定区间上有连续偏导数时，有 $\dfrac{dy}{dx} = -\dfrac{F_x'(x, y)}{F_y'(x, y)}$（反之，若 $F_y'(x, y) = 0$，则无法确定隐函数 $y = f(x, y)$ 是否存在，所以这是一个充分非必要条件，反例可举 $F(x, y) = (y-x)^2$）
   - （2）对于由方程 $F(x, y, z) = 0$ 确定的隐函数 $z = f(x, y)$，当 $\textcolor{red}{F_z'(x, y, z) \ne 0}$ 且函数 $F(x, y, z)$ 在给定区间上有连续偏导数时时，有 $\dfrac{\partial z}{\partial x} = -\dfrac{F_x'(x, y, z)}{F_z'(x, y, z)}$、$\dfrac{\partial z}{\partial y} = -\dfrac{F_y'(x, y, z)}{F_z'(x, y, z)}$

> ⚠️ 上面的隐函数存在定理中，$F_x'(x, y)$ 是指对变量 $x$ 求导，而不是对位置求导！

5. **隐函数求偏导数与全微分方法总结**
   - 设函数 $F(x, y, z)$ 有连续一阶偏导数，$F_z' \ne 0$，$z = f(x, y)$ 由方程 $F(x, y, z) = 0$ 所确定，求 $\dfrac{\partial z}{ \partial x}$，$\dfrac{\partial z}{\partial y}$
   - （1）方法一：方程 $F(x, y,z)= 0$ 两边同时求偏导，得到 $F_x' + F_z' \dfrac{\partial z}{\partial x} = 0$ 与 $F_y' + F_z' \dfrac{\partial z}{\partial y} = 0$，此时 $x, y$ 独立，将 $z$ 看作 $\textcolor{red}{z = f(x, y)}$
   - （2）方法二（公式法）：即 $\dfrac{\partial z}{\partial x} = -\dfrac{F_x'}{F_z'}$ 与 $\dfrac{\partial z}{\partial y} = -\dfrac{F_y'}{F_z'}$，此时将 $x,y,z$ 视为独立变量
   - （3）方法三（全微分法，无脑计算）：$F_x' dx + F_y' dy + F_z' dz = 0$（将 $x,y,z$ 视为独立变量），整理得 $dz = -\dfrac{F_x'}{F_z'} dx - \dfrac{F_y'}{F_z'} dy$，然后根据微分形式不变性可以得到 $\dfrac{\partial z}{\partial x} = -\dfrac{F_x'}{F_z'}$ 与 $\dfrac{\partial z}{\partial y} = -\dfrac{F_y'}{F_z'}$


6. **由方程组所确定的隐函数**
   - 设 $u = u(x, y), v = v(x, y)$，由 $\begin{cases} F(x, y, u, v) = 0 \\ G(x, y, u, v) = 0 \end{cases}$ 所确定，求 $\dfrac{\partial u}{\partial x}$，$\dfrac{\partial u}{\partial y}$，$\dfrac{\partial v}{\partial x}$，$\dfrac{\partial v}{\partial y}$
   - （1）法一：等式两边对 $x$ 求偏导，即 $\begin{cases} F_x' + F_u' \dfrac{\partial u}{\partial x} + F_v' \dfrac{\partial v}{\partial x} =0 \\ G_x' + G_u' \dfrac{\partial u}{\partial x} + G_v' \dfrac{\partial v}{\partial x} = 0\end{cases}$，然后通过解方程可以得到 $\dfrac{\partial u}{\partial x}, \dfrac{\partial v}{\partial x}$。等式两边对 $y$ 求偏导，类似可求 $\dfrac{\partial u}{\partial y}, \dfrac{\partial v}{\partial y}$
   - （2）法二（无脑全微分）：利用微分形式不变性，即 $\begin{cases} F_x' dx + F_y' dy + F_u'du + F_v'dv = 0 \\ G_x' dx + G_y' dy + G_u'du + G_v'dv = 0 \end{cases}$，然后消去 $dv$，整理成 $du = (...)dx + (...) dy$ 形式，就可以得到 $\dfrac{\partial u}{\partial x}, \dfrac{\partial u}{\partial y}$，类似的可以求得 $\dfrac{\partial v}{\partial x}, \dfrac{\partial v}{\partial y}$

> 💡 其实上面的方法一就是方法二中公式的推导过程 🌚

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