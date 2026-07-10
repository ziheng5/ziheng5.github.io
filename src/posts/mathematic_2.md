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
   - （注意，分段点处的偏导数要用定义求）

> ⚠️ 易错的误区：
> - $\lim\limits_{x\to x_0} f_x'(x, y_0) = f_x'(x_0, y_0)$ 是一个一元概念，无法用来证明偏导连续（二元概念）！（即前面那个式子和 $\lim\limits_{(x, y) \to (x_0, y_0)} f_x'(x, y) = f_x'(x_0, y_0)$ 并不等价！）



1. **上面四个概念的关系**

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

7. **二元函数的极值**
   - （1）必要条件：设 $z = f(x, y)$ 在点 $(x_0, y_0)$ 处 $\begin{cases} 一阶偏导存在 \\ 取极值 \end{cases}$，则 $f_x'(x_0, y_0) = 0, f_y'(x_0, y_0) = 0$
   - （2）充分条件：设 $f(x, y)$ 在 $(x_0, y_0)$ 的某邻域内连续且有一阶或二阶连续偏导数，又 $f_x'(x_0, y_0) = f_y'(x_0, y_0) = 0$，记 $\begin{cases} f_{xx}''(x_0, y_0)=A \\ f_{xy}''(x_0, y_0)=B \\ f_{yy}''(x_0, y_0)=C \end{cases}$，则 $\Delta = AC - B^2 \begin{cases} >0 \Rightarrow 极值 {\begin{cases} A <0 \Rightarrow 极大值 \\ A>0 \Rightarrow 极小值 \end{cases}} \\ <0 \Rightarrow 非极值 \\ =0 \Rightarrow 方法失效，另寻他法 \end{cases}$


8. **拉格朗日数乘法求多元函数条件极值**
   - 求 $u = f(x, y, z)$ 在条件 $\varphi(x, y, z)= 0$ 下的极值
   - （1）令 $F(x, y, z, \lambda) = f(x, y, z) + \lambda \varphi(x, y, z)$
   - （2）解方程组：$\begin{cases} F_x' = f_x'(x, y , z) + \lambda \varphi_x'(x, y, z) = 0 \\ F_y' = f_y'(x, y , z) + \lambda \varphi_y'(x, y, z) = 0 \\ F_z' = f_z'(x, y , z) + \lambda \varphi_z'(x, y, z) = 0 \\ F_{\lambda}' = \varphi(x, y, z) = 0 \end{cases}$ 得到 $x= x_0, y=y_0, z = z_0$
   - （3）经过比较，得出最大值、最小值
   - （思路很固定，难点在解方程步骤）

> 💡 那假如有多个约束呢？
> - 如求 $u = f(x, y, z)$ 在条件 $\varphi(x, y, z)= 0$ 和 $\psi(x, y, z) = 0$ 下的极值
> - 构造拉格朗日方程：$F(x, y, z, \lambda, \mu) = f(x, y, z) + \lambda \varphi(x, y, z) + \mu \psi(x, y, z)$
> - 然后依次求偏导，令偏导为 0，解出方程

> ⚠️ 此外还有一些做题时需要注意的问题：
> - （1）当约束条件中存在$\textcolor{red}{不可微点}$的情况下，须从定义重新考虑，不能因为拉格朗日数乘法失效而妄下结论！！！
> - （2）考试还可能出现多元函数 + 圆锥曲线求几何最值问题，要注意这方面的练习（【1000b-13-31、32】）


9. **特殊路径法判断二重极限是否存在**
   - （1）同阶路径法：分子分母中 $x, y$ 次数相同，考虑同阶路径法，即 $y = kx$
   - （2）变阶路径法：分子分母次数混乱，可尝试变阶+同阶

10. **二重极限与累次极限**
   - （1）二重极限存在，累次极限未必存在；累次极限存在，二重极限未必存在
   - （2）若二重极限存在，且累次极限存在，则二者必相等
   - （3）若 $\lim\limits_{x\to x_0}[\lim\limits_{y\to y_0} f(x, y)]$，$\lim\limits_{y\to y_0}[\lim\limits_{x\to x_0} f(x, y)]$ 均存在但不相等，则二重极限不存在


## 14. 二重积分

> ⚠️ 本章重点习题：
> - 【1000b-14-4、5、14、26、33、35、38、41】
>
> 💡 本章 tips：积分区域要画对，根据被积函数和积分趋于考虑是否使用对称性、交换积分次序、换元

1. **普通对称性和轮换对称性**
   - （1）普通对称性：这个比较简单，通过几何理解即可
   - （2）轮换对称性：在直角坐标系下，若把 $x$ 与 $y$ 对调后，区域 $D$ 不变（或区域 $D$ 关于 $y=x$ 对称），则 $\iint\limits_{D} f(x, y) d\sigma = \iint\limits_{D} f(y, x) d\sigma$（主要应对的问题：$\iint\limits_{D} f(x, y) d\sigma$ 难算但 $f(x, y) + f(y, x)$ 之和式子简单）

2. **极坐标系下的二重积分**：$\iint\limits_{D} f(x, y) d\sigma = \int^{\beta}_{\alpha} d\theta \int^{r_1(\theta)}_{r_2(\theta)} f(r\sin \theta, r\cos \theta) r dr$

3. **换元法的映射**：在直角坐标系下，若令 $\begin{cases} x= x(u, v) \\ y=y(u,v) \end{cases}$，则 $dx dy = \begin{vmatrix}\dfrac{\partial(x, y)}{\partial(u, v)}\end{vmatrix} dudv = \begin{vmatrix}\begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} \\ \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} \end{vmatrix}\end{vmatrix} du dv$

4. **和式极限**：$\iint\limits_{D} f(x, y) d\sigma = \lim\limits_{n\to +\infty} \sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n} f(a + \dfrac{b-a}{n}i, c + \dfrac{d-c}{n} j) \cdot\dfrac{b-a}{n}\cdot\dfrac{d-c}{n}$，其中 $D = \{ (x, y) | a\le x \le b, c\le y \le d\}$


5. **二重积分中值定理**
   - 设 $f(x, y)$ 和 $g(x, y)$ 都在闭区域 $D$ 上连续，且 $g(x, y)$ 在 $D$ 上不变号，则至少存在一点 $(\xi, \eta) \in D$ 使得 $\iint\limits_{D}f(x, y) g(x, y) dxdy = f(\xi, \eta) \iint\limits_{D} g(x, y) dxdy$
   - 那么，根据上面的定理，有 $\iint\limits_{D}f(x, y) dxdy = f(\xi, \eta) \iint\limits_{D} dxdy = S_D f(\xi, \eta)$
   - 在 “二重积分+极限” 或证明题中可能会用到，如`【1000a-14-3】`



## 15. 微分方程

> ⚠️ 本章重点习题：
> - 【🐙 强化-15-例3、8】
> - 【1000b-15-6、9、13、15、29、30、31、33、34、39】

1. **一阶线性微分方程**
   - 形如 $y' + p(x) y =q(x)$ 的方程叫做一阶线性微分方程，其中 $p(x)$、$q(x)$ 为连续函数
   - 通解：$y = e^{-\int p(x)dx} [\int e^{\int p(x)dx} \cdot q(x)dx + C]$
   - 其推导过程为：在原微分方程两边同时乘 $e^{\int p(x)dx}$

> 💡 还可以写成 $y = e^{-\int^x_{x_0} p(x)dx} [\int^x_{x_0} e^{\int^x_{x_0} p(x)dx} \cdot q(x)dx + C]$，其中 $x_0$ 可以按照方便解题的原则来设置

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

5. **一阶齐次微分方程求解**（真的不是 🐙 的小巧思吗...）
   - （1）能写成 $y' = f(\dfrac{y}{x})$：令 $u = \dfrac{y}{x}$，则 $y = ux, \dfrac{dy}{dx} = u + x\dfrac{du}{dx}$
   - （2）能写成 $\dfrac{1}{y'} = f(\dfrac{x}{y})$：令 $u = \dfrac{x}{y}$，则 $x = uy, \dfrac{dx}{dy} = u + y \dfrac{du}{dy}$
   - （3）能写成 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1})$：
     - a. 若 $c = c_1 = 0$，则令 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1})$
     - b. 若 $c \ne 0$ 或 $c_1 \ne 0$，且 $\dfrac{a}{a_1} = \dfrac{b}{b_1}$ 时，令 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1}) = g(ax + by)$
     - c. 若 $c \ne 0$ 或 $c_1 \ne 0$，且 $\dfrac{a}{a_1} \ne \dfrac{b}{b_1}$ 时，由 $\begin{cases} ax +by + c = 0 \\ a_1 x + b_1 y + c = 0 \end{cases}$ 解得 $x_0, y_0$，然后令 $\begin{cases} x = X + x_0 \\ y = Y + y_0 \end{cases}$，则 $y' = f(\dfrac{ax + by + c}{a_1x + b_1 y + c_1}) = f(\dfrac{aX + bY}{ a_1 X + b_1 Y})$，再令 $u = \dfrac{Y}{X}$


6. **二阶可降阶微分方程的求解**
   - （1）能写成 $y'' = f(x, y')$ 或 $y'' = f(y')$：缺 $y$，令 $y' = p$（求解时要小心 $\textcolor{red}{p \equiv 0}$ 的解）
   - （2）能写成 $y'' = f(y, y')$：缺 $x$，令 $y' = p, y'' = \dfrac{dp}{dx} = \dfrac{dp}{dy} \dfrac{dy}{dx} = p\dfrac{dp}{dy}$

7. **欧拉方程**
   - （1）形如 $x^2 y'' + px y' + qy = f(x)$
   - （2）解法
     - a. 当 $x > 0$ 时，令 $x = e^t$，则 $t = \ln x, \dfrac{dt}{dx} = \dfrac{1}{x}$，于是 $\dfrac{dy}{dx} = \dfrac{dy}{dt} \dfrac{dt}{dx} = \dfrac{1}{x} \dfrac{dy}{dt}$、$\dfrac{d^2 y}{dx^2} = -\dfrac{1}{x^2} \dfrac{dy}{dt} + \dfrac{1}{x} \dfrac{d}{dx}(\dfrac{dy}{dt}) = -\dfrac{1}{x^2} \dfrac{dy}{dt} + \dfrac{1}{x^2} \dfrac{d^2 y}{d t^2}$
     - b. 原方程化为 $\dfrac{d^2 y}{d t^2} + (p-1) \dfrac{dy}{dt} + qy = f(e^t)$，即可求解
     - c. 若 $x < 0$ 时，令 $x = -e^t$，同理

> 💡 还有一种进阶版的，参考【1000b-15-39】：求微分方程 $y' = \dfrac{1}{xy(1 + xy^2)}$，满足 $y(1)=0$ 的解
> - 先把方程倒过来：$\dfrac{dx}{dy} = xy(1 + xy^2) = xy + x^2 y^3$
> - 然后把 $x$ 挪走：$\dfrac{1}{x^2} \dfrac{dx}{dy} + (-y) \dfrac{1}{x} = y^3$
> - 令 $z = \dfrac{1}{x}$，则 $\dfrac{dz}{dy} = \dfrac{dz}{dx}  \dfrac{dx}{dy} = -\dfrac{1}{x^2} \dfrac{dx}{dy}$
> - 然后方程就只剩 $z$ 和 $y$ 了，计算也很简单

8. **n 阶常系数齐次线性微分方程求解**
   - （1）若 $\lambda$ 为单实根，则写 $Ce^{\lambda x}$
   - （2）若 $\lambda$ 为 $k$ 重实根，则写 $(C_1 + C_2 x + C_3 x^2 + \cdots + C_k x^{k-1}) e^{\lambda x}$
   - （3）若 $\lambda$ 为单复根 $\alpha \pm \beta$，则写 $e^{\alpha x} (C_1 \cos \beta x + C_2 \sin \beta x)$
   - （4）若 $\lambda$ 为二重复根 $\alpha \pm \beta$，则写 $e^{\alpha x} (C_1 \cos \beta x + C_2 \sin \beta x + C_3 x\cos \beta x + C_4 x\sin \beta x)$

## 16. 无穷级数




1. **傅里叶级数**：设函数 $f(x)$ 为周期为 $2l$ 的周期函数，且在 $[-l, l]$ 上可积，则
   - （1）傅里叶系数（以 $2l$ 为周期）：
     - $a_n = \dfrac{1}{l} \int^{l}_{-l} f(x) \cos \dfrac{n\pi}{l} x dx (n = 0, 1, 2,...)$
     - $b_n = \dfrac{1}{l} \int^{l}_{-l} f(x) \sin \dfrac{n\pi}{l} x dx (n = 0, 1, 2,...)$
   - （2）傅里叶级数（以 $2l$ 为周期）：$f(x) ～ \dfrac{a_0}{2} + \sum\limits_{n=1}^{\infty} (a_n \cos \dfrac{n\pi}{l} x + b_n \sin \dfrac{n\pi}{l} x) = S(x) $
   - （3）狄利克雷收敛定理：设 $f(x)$ 是以 $2l$ 为周期的可积函数，如果在 $[-l, l]$ 上 $f(x)$ 满足“连续或只有有限个第一类间断点”、“至多只有有限个极值点”，则 $f(x)$ 的傅里叶级数在 $[-l, l]$ 上处处收敛，记其和函数为 $S(x)$，则 $S(x) = \begin{cases} f(x), & x 为连续点 \\ \dfrac{f(x-0) + f(x+0)}{2}, & x 为间断点 \\ \dfrac{f(-l+0)+ f(l-0)}{2}, & x = \pm l \end{cases}$（即 $S(x)$ 收敛于点的 $\dfrac{左极限+右极限}{2}$）
   - （4）正弦级数：当 $f(x)$ 为奇函数时，其展开式是正弦级数 $f(x) ～ \sum\limits_{n=1}^{\infty} b_n \sin \dfrac{n\pi x}{l} , b_n = \dfrac{2}{l} \int^l_0 f(x) \sin \dfrac{n\pi x}{l} dx$
   - （5）余弦级数：当 $f(x)$ 为偶函数时，其展开式是余弦级数 $f(x) ～ \dfrac{a_0}{2} + \sum\limits_{n=1}^{\infty} a_n \cos \dfrac{n\pi x}{l} , a_n = \dfrac{2}{l} \int^l_0 f(x) \cos \dfrac{n\pi x}{l} dx$（$a_0 = \dfrac{2}{l} \int^l_0 f(x) dx$）

> 💡 关于傅里叶级数，建议搭配《信号与系统》中的解释来理解：[孟桥老师的信号课](https://www.bilibili.com/video/BV144411D73H?p=48)

2. **周期奇延拓与周期偶延拓**
   - （1）周期奇延拓：设 $f(x)$ 定义在 $[0, l]$ 上，令 $F(x) = \begin{cases} f(x), & 0<x\le l \\ -f(-x), & -l \le x < 0 \\ 0, & x=0 \end{cases}$，再令 $F(x)$ 为以 $2l$ 为周期的周期函数
   - （2）周期偶延拓：设 $f(x)$ 定义在 $[0, l]$ 上，令 $F(x) = \begin{cases} f(x), & 0\le x\le l \\ f(-x), & -l \le x < 0  \end{cases}$，再令 $F(x)$ 为以 $2l$ 为周期的周期函数




## 17. 多元函数积分学（Part1）

## 18. 多元函数积分学（Part2）