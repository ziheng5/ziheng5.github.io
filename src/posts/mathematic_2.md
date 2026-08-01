---
title: Coldrain 的 27 考研数一高数强化笔记（下）
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
> - 【1000b-15-6、9、13、15、29、30、31、33、34、39、40】

1. **一阶线性微分方程**
   - 形如 $y' + p(x) y =q(x)$ 的方程叫做一阶线性微分方程，其中 $p(x)$、$q(x)$ 为连续函数
   - 通解：$y = e^{-\int p(x)dx} [\int e^{\int p(x)dx} \cdot q(x)dx + C]$
   - 其推导过程为：在原微分方程两边同时乘 $e^{\int p(x)dx}$

> 💡 小细节：
> - 还可以写成 $y = e^{-\int^x_{x_0} p(x)dx} [\int^x_{x_0} e^{\int^x_{x_0} p(x)dx} \cdot q(x)dx + C]$，其中 $x_0$ 可以按照方便解题的原则来设置
> - 一般情况下题目中遇到 $p(x) = \dfrac{1}{x}$ 这种形式时，$y = e^{-\int p(x)dx} [\int e^{\int p(x)dx} \cdot q(x)dx + C]$ 中的那个 $\int p(x)dx$ 积分出来$\textcolor{red}{不用加绝对值}$（讲解视频：[传送门](https://www.bilibili.com/video/BV1GtKnzjEQr/?)），但也有$\textcolor{red}{特例}$：$y' - \dfrac{1}{2x} y = x$，如果不加绝对值的话解中包含 $\sqrt{x}$ 限定了 $x>0$ 而原题中并未有此限定（但这种特例不太可能会考，因为太偏了）

2. **伯努利方程**
   - 形如 $\dfrac{dy}{dx} + p(x)y = q(x) y^n (n\ne 0, 1)$
   - 解法：
     - （1）先变形为 $y^{-n} \cdot \dfrac{dy}{dx} + p(x)y^{1-n} = q(x)$
     - （2）令 $z = y^{1-n}$ 可得 $\dfrac{dz}{dx} = (1-n) y^{-n} \dfrac{dy}{dx}$，则 $\dfrac{1}{1-n} \dfrac{dz}{dx} + p(x)z = q(x)$
     - （3）解此一阶线性微分方程


3. **全微分方程**
   - 若函数 $P(x, y), Q(x, y)$ 在单连通区域 $D$ 上具有一阶连续偏导数，且在 $D$ 内满足 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$，则 $Pdx + Qdy$ 是某二元函数 $u(x, y)$ 的全微分
   - 若一阶微分方程写成 $P(x,y ) dx + Q(x, y)dy = 0$ 的形式时，等式左端表达式是 $u(x, y)$ 的全微分，则称该式子为全微分方程
   - 解题思路就是通过 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$ 来求解方程

> 💡 全微分的积分与路径无关（其物理意义可以理解为保守场下变力沿曲线做工，与路径无关），因此由 $P(x,y ) dx + Q(x, y)dy = 0$ 求解二元函数 $u(x, y)$ 时可以采用折线法（或者干脆 $F(A) - F(B)$），选取一条简单的折线来进行积分，参考【1000b-15-40】


4. **二阶常系数齐次线性微分方程**
   - 形如 $y'' + py' + qy = 0$，其中 $p$、$q$ 为常数
   - 通解：先写出对应的特征方程 $r^2 + pr + q = 0$（对原微分方程令 $y = e^{rx}$ 得到的）
     - （1）特征方程有两个不等实根：通解为 $y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$
     - （2）特征方程有两个相等实根：通解为 $y = (C_1 + C_2 x) e^{rx}$
     - （3）特征方程有共轭复根 $\alpha \pm \beta i$：通解为 $y = e^{\alpha x} (C_1 \cos\beta x + C_2 \sin\beta x)$

5. **二阶常系数非齐次线性微分方程**
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

6. **一阶齐次微分方程求解**（真的不是 🐙 的小巧思吗...）
   - （1）能写成 $y' = f(\dfrac{y}{x})$：令 $u = \dfrac{y}{x}$，则 $y = ux, \dfrac{dy}{dx} = u + x\dfrac{du}{dx}$
   - （2）能写成 $\dfrac{1}{y'} = f(\dfrac{x}{y})$：令 $u = \dfrac{x}{y}$，则 $x = uy, \dfrac{dx}{dy} = u + y \dfrac{du}{dy}$
   - （3）能写成 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1})$：
     - a. 若 $c = c_1 = 0$，则令 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1})$
     - b. 若 $c \ne 0$ 或 $c_1 \ne 0$，且 $\dfrac{a}{a_1} = \dfrac{b}{b_1}$ 时，令 $y' = f(\dfrac{ax + by + c}{a_1x + b_1y + c_1}) = g(ax + by)$
     - c. 若 $c \ne 0$ 或 $c_1 \ne 0$，且 $\dfrac{a}{a_1} \ne \dfrac{b}{b_1}$ 时，由 $\begin{cases} ax +by + c = 0 \\ a_1 x + b_1 y + c = 0 \end{cases}$ 解得 $x_0, y_0$，然后令 $\begin{cases} x = X + x_0 \\ y = Y + y_0 \end{cases}$，则 $y' = f(\dfrac{ax + by + c}{a_1x + b_1 y + c_1}) = f(\dfrac{aX + bY}{ a_1 X + b_1 Y})$，再令 $u = \dfrac{Y}{X}$


7. **二阶可降阶微分方程的求解**
   - （1）能写成 $y'' = f(x, y')$ 或 $y'' = f(y')$：缺 $y$，令 $y' = p$（求解时要小心 $\textcolor{red}{p \equiv 0}$ 的解）
   - （2）能写成 $y'' = f(y, y')$：缺 $x$，令 $y' = p, y'' = \dfrac{dp}{dx} = \dfrac{dp}{dy} \dfrac{dy}{dx} = p\dfrac{dp}{dy}$

8. **欧拉方程**
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

9. **n 阶常系数齐次线性微分方程求解**
   - （1）若 $\lambda$ 为单实根，则写 $Ce^{\lambda x}$
   - （2）若 $\lambda$ 为 $k$ 重实根，则写 $(C_1 + C_2 x + C_3 x^2 + \cdots + C_k x^{k-1}) e^{\lambda x}$
   - （3）若 $\lambda$ 为单复根 $\alpha \pm \beta$，则写 $e^{\alpha x} (C_1 \cos \beta x + C_2 \sin \beta x)$
   - （4）若 $\lambda$ 为二重复根 $\alpha \pm \beta$，则写 $e^{\alpha x} (C_1 \cos \beta x + C_2 \sin \beta x + C_3 x\cos \beta x + C_4 x\sin \beta x)$

## 16. 无穷级数

> 👉 本章强化阶段强烈推荐 26 年方浩强化无穷级数章节的课程，质量很高（尤其是小猪配齐）

> ⚠️ 本章重点习题：
> - 【1000b-16-1、7、8、10、14、17、20、24、36、39】
>
> “这一章非常非常难”

1. **级数的性质**
   - （1）$\sum\limits_{n=1}^{\infty} u_n$ 与 $\sum\limits_{n=1}^{\infty} ku_n$ 同敛散性，其中 $k \ne 0$
   - （2）若 $\sum\limits_{n=1}^{\infty} u_n$ 和 $\sum\limits_{n=1}^{\infty} v_n$ 分别收敛于 $s, \sigma$，则 $\sum\limits_{n=1}^{\infty} (u_n \pm v_n)$ 收敛于 $s \pm \sigma$（收敛$\pm$收敛=收敛，收敛+发散=发散，发散+发散=发散）
   - （3）改变级数**有限项**不影响级数的敛散性
   - （4）收敛级数加括号仍收敛且和不变
   - ⭐（5）$\sum\limits_{n=1}^{\infty} u_n$ 收敛 $\Rightarrow \lim\limits_{n \to \infty} u_n = 0$（逆否命题： $\lim\limits_{n \to \infty} u_n \ne 0$ $\Rightarrow \sum\limits_{n=1}^{\infty} u_n$ 不收敛）

2. **正项级数及其敛散性判别**
   - （1）收敛原理：正项级数 $\sum\limits_{n=1}^{\infty} u_n$ 收敛 $\Leftrightarrow S_n$ 有界
   - （2）比较判别法（适合抽象级数，用放缩来证明有界）：给出两个正项级数 $\sum\limits_{n=1}^{\infty} u_n$ 和 $\sum\limits_{n=1}^{\infty} v_n$，如果从某项起（前有限项不影响敛散性）有 $u_n \le v_n$ 成立，则
     - a. 若 $\sum\limits_{n=1}^{\infty} v_n$ 收敛，则 $\sum\limits_{n=1}^{\infty} u_n$ 收敛
     - b. 若 $\sum\limits_{n=1}^{\infty} u_n$ 发散，则 $\sum\limits_{n=1}^{\infty} v_n$ 发散
   - （3）比较判别法的极限形式（无穷小比阶，适合初等函数构成的级数，即 $e,\sin,\ln$）：给出两个正项级数 $\sum\limits_{n=1}^{\infty} u_n$ 和 $\sum\limits_{n=1}^{\infty} v_n$，且 $\lim\limits_{n\to \infty} \dfrac{u_n}{v_n} = A$
     - a. 若 $A = 0$，则当 $\sum\limits_{n=1}^{\infty} v_n$ 收敛时，$\sum\limits_{n=1}^{\infty} u_n$ 也收敛
     - b. 若 $A = + \infty$，则当 $\sum\limits_{n=1}^{\infty} v_n$ 发散时，$\sum\limits_{n=1}^{\infty} u_n$ 也发散
     - c. 若 $0 < A < +\infty$，则 $\sum\limits_{n=1}^{\infty} u_n$ 和 $\sum\limits_{n=1}^{\infty} v_n$ 有相同的敛散性（等价无穷小）
   - （4）比值判别法（达朗贝尔判别法，适用于含阶乘的级数）：给出一正项级数 $\sum\limits_{n=1}^{\infty} u_n$，如果 $\lim\limits_{n\to \infty}\dfrac{u_{n+1}}{u_n} = \rho$，那么
     - a. 若 $\rho < 1$，则 $\sum\limits_{n=1}^{\infty} u_n$ 收敛
     - b. 若 $\rho > 1$，则 $\sum\limits_{n=1}^{\infty} u_n$ 发散
     - c. 若 $\rho = 1$，则无法使用此方法
   - （5）根值判别法（柯西判别法，适合含 $n$ 次幂的级数）：给出一正项级数 $\sum\limits_{n=1}^{\infty} u_n$，如果 $\lim\limits_{n\to \infty} \sqrt[n]{u_n} = p$
     - a. 若 $\rho < 1$，则 $\sum\limits_{n=1}^{\infty} u_n$ 收敛
     - b. 若 $\rho > 1$，则 $\sum\limits_{n=1}^{\infty} u_n$ 发散
     - c. 若 $\rho = 1$，则此方法失效
   - （6）积分判别法（近几年新增考点）：设 $\sum\limits_{n=1}^{\infty} u_n$ 为正项级数，若存在 $[1, + \infty)$ 上单调减少的非负连续函数 $f(x)$，使得 $u_n = f(n)$，则级数 $\sum\limits_{n=1}^{\infty} u_n$ 与反常积分 $\int^{+\infty}_1 f(x) dx$ 的敛散性相同


> 💡 p 级数：
> - $\sum\limits_{n=1}^{\infty} \dfrac{1}{n^p}$ 叫做 $p$ 级数
> - $p$ 级数 $\sum\limits_{n=1}^{\infty} \dfrac{1}{n^p} \begin{cases} 发散, & p\le 1 \\ 收敛, & p>1 \end{cases}$

> 💡 另外，本章放缩可能会用到的重要不等式如下：
> - （1）$\ln n < \ln(n+1) < n$
> - （2）$(\dfrac{n}{e})^n \le n! \le (\dfrac{n+1}{2})^n \le n^n$
> - （3）$\begin{cases} \dfrac{x}{1+x} >\dfrac{1}{2}, & x>1 \\ \dfrac{x}{2} < \dfrac{x}{1+x} <x, &0<x<1 \end{cases}$
> - （4）$\begin{cases} x< \dfrac{x}{1-x} < 2x, & 0<x<\dfrac{1}{2} \\  \dfrac{x}{1-x} >x, &0<x<1 \end{cases}$



3. **交错级数及其敛散性判别**
   - 莱布尼茨判别法（充分不必要）：给出一交错级数 $\sum\limits_{n=1}^{\infty} (-1)^{n-1}u_n, u_n >0$，若 $\{u_n\}$ 单调不增且 $\lim\limits_{n \to \infty} u_n = 0$，则此级数收敛

4. **任意项级数及其敛散性判别（绝对值判别法）**
   - （1）绝对收敛：设 $\sum\limits_{n=1}^{\infty} u_n$ 为任意项级数，若 $\sum\limits_{n=1}^{\infty} |u_n|$ 收敛，则称 $\sum\limits_{n=1}^{\infty} u_n$ 绝对收敛
   - （2）条件收敛：设 $\sum\limits_{n=1}^{\infty} u_n$ 为任意项级数，若 $\sum\limits_{n=1}^{\infty} u_n$ 收敛，但 $\sum\limits_{n=1}^{\infty} |u_n|$ 发散，则称 $\sum\limits_{n=1}^{\infty} u_n$ 条件收敛

> 💡 注意：
> - （1）若 $\sum\limits_{n=1}^{\infty} |u_n|$ 收敛（即绝对收敛），则 $\sum\limits_{n=1}^{\infty} u_n$ 必收敛
> - （2）若 $\sum\limits_{n=1}^{\infty} u_n$、$\sum\limits_{n=1}^{\infty} v_n$ 均绝对收敛，则 $\sum\limits_{n=1}^{\infty} (u_n \pm v_n)$ 绝对收敛
> - （3）若 $\sum\limits_{n=1}^{\infty} u_n$ 绝对收敛，$\sum\limits_{n=1}^{\infty} v_n$ 条件收敛，则 $\sum\limits_{n=1}^{\infty} (u_n \pm v_n)$ 条件收敛
> - （4）若 $\sum\limits_{n=1}^{\infty} u_n$、$\sum\limits_{n=1}^{\infty} v_n$ 均条件收敛，则 $\sum\limits_{n=1}^{\infty} (u_n \pm v_m)$ 收敛
> - （5）如果级数 $\sum\limits_{n=1}^{\infty} |u_n|$ 发散，我们不能断定级数 $\sum\limits_{n=1}^{\infty} u_n$ 也发散
> - （6）交错 $p$ 级数 $\sum\limits_{n=1}^{\infty} (-1)^{n-1} u_n \begin{cases} 绝对收敛, & p > 1 \\ 条件收敛, & 0<p \le 1 \end{cases}$
> - （7）若 $\sum\limits_{n=1}^{\infty} u_n$ 绝对收敛、$\sum\limits_{n=1}^{\infty} v_n$ 条件收敛，那么 $\sum\limits_{n=1}^{\infty} u_n v_n$ 绝对收敛（证明见【张宇强化高数-16-例32】）


5. **幂级数及其收敛域**
   - （1）收敛点和发散点：给定 $x_0 \in I$，有 $\sum\limits_{n=1}^{\infty} u_n(x_0)$ 收敛，则称点 $x_0$ 为函数项级数 $\sum\limits_{n=1}^{\infty} u_n(x)$ 的收敛点；反之，则为发散点
   - （2）收敛域：函数项级数 $\sum\limits_{n=1}^{\infty} u_n(x)$ 的所有收敛点的集合称为它的收敛域
   - （3）阿贝尔定理：当幂级数 $\sum\limits_{n=0}^{\infty} a_n x^n$ 在点 $x = x_1(x_1\ne 0)$ 处收敛时，对于满足 $|x| < |x_1|$ 的一切 $x$，幂级数绝对收敛；当幂级数 $\sum\limits_{n=0}^{\infty} a_n x^n$ 在点 $x = x_2(x_2\ne 0)$ 处发散时，对于满足 $|x| > |x_2|$ 的一切 $x$，幂级数发散
   - （4）收敛半径：若 $R\ge 0$ 满足条件：1️⃣ 当 $|x|< R$ 时，$\sum\limits_{n=0}^{\infty} a_n x^n$ 绝对收敛；2️⃣ 当 $|x| > R$ 时，$\sum\limits_{n=0}^{\infty} a_n x^n$ 发散。则称 $R$ 为幂级数 $\sum\limits_{n=0}^{\infty} a_n x^n$ 的收敛半径，区间 $(-R, R)$ 称为 $\sum\limits_{n=0}^{\infty} a_n x^n$ 的收敛区间


> 💡 结论 1：根据阿贝尔定理，已知 $\sum\limits_{n=0}^{\infty} a_n(x - x_0)^n$ 在某点 $x_1(x_1 \ne x_0)$ 的敛散性，确定该幂级数的收敛半径可分为以下三种情况：
> - （1）若在 $x_1$ 处收敛，则收敛半径 $R \ge |x_1 - x_0|$
> - （2）若在 $x_1$ 处发散，则收敛半径 $R \le |x_1 - x_0|$
> - （3）若在 $x_1$ 处条件收敛，则收敛半径 $R = |x_1 - x_0|$（因为在两个端点处刚好一个收敛一个发散）

> 💡 结论 2：已知 $\sum a_n(x - x_1)^n$ 的敛散性，讨论 $\sum b_n(x - x_2)^m$ 的敛散性
> - （1）$(x - x_1)^n$ 与 $(x - x_2)^m$ 的转化一般通过初等变形来完成，包括：a.“平移”收敛区间；b. 提出或者乘以因式 $(x - x_0)^k$ 等
> - （2）$a_n$ 与 $b_n$ 的转化一般通过微积分变形来完成，包括：a. 对级数逐项求导；b. 对级数逐项求积分等
> - （3）以下三种情况，级数的收敛半径不变，收敛域要具体问题具体分析：
>   - a. 对级数提出或者乘以因式 $(x - x_0)^k$，或者作平移等，收敛半径不变
>   - b. 对级数逐项求导，收敛半径不变，收敛域可能缩小（端点处的敛散性可能发生改变）
>   - c. 对级数逐项积分，收敛半径不变，收敛域可能扩大（端点处的敛散性可能发生改变）

> 💡 做题时遇到的二级结论（【1000b-16-14】）：
> - 对于 $\sum\limits_{n=0}^{\infty}(u_n + v_n)x^n$，设其收敛半径为 $R$，则有 $R = \min\{R_1, R_2\}(\textcolor{red}{R_1 \ne R_2})$，其中 $R_1,R_2$ 分别为 $\sum\limits_{n=0}^{\infty} u_n x^n$、$\sum\limits_{n=0}^{\infty} v_n x^n$ 的收敛半径。当 $R_1 = R_2$ 时，$R \ge R_1 = R_2$
> - 👆 该结论理解起来很简单，但是学的时候容易遗漏，这里重点记忆一下



6. **收敛域的求法**
   - （1）对于不缺项幂级数 $\sum\limits_{n=0}^{\infty} a_n x^n$
     - a. 收敛半径的求法：若 $\lim\limits_{n\to \infty} \begin{vmatrix} \dfrac{a_{n+1}}{a_n} \end{vmatrix} = \rho$ 或 $\lim\limits_{n\to \infty} \sqrt[n]{|a_n|} = \rho$，则 $\sum\limits_{n=0}^{\infty} a_n x^n$ 的收敛半径 $R$ 的表达式为 $R = \begin{cases} \dfrac{1}{\rho},& \rho\ne 0, \rho\ne +\infty \\ +\infty, & \rho=0 \\ 0, & \rho=+\infty \end{cases}$
     - b. 收敛区间与收敛域：区间 $(-R, R)$ 为幂函数 $\sum\limits_{n=0}^{\infty} a_n x^n$ 的收敛区间，单独考查幂级数在 $\pm R$ 处的敛散性就可以确定其收敛域为 $(-R, R)$ 或 $[-R, R)$ 或 $(-R, R]$ 或 $[-R, R]$
   - （2）对于缺项幂级数或一般函数项级数 $\sum\limits_{n=0}^{\infty} u_n(x)$
     - a. 加绝对值，写成 $\sum\limits_{n=0}^{\infty} |u_n(x)|$
     - b. 用正项级数的比值（或根值）判别法，令 $\lim\limits_{n \to \infty} \dfrac{|u_{n+1}(x)|}{|u_n(x)|}$（或 $\lim\limits_{n\to \infty} \sqrt[n]{|u_n(x)|}$）$<1$，求出收敛区间 $(a, b)$
     - c. 单独讨论 $x=a, x=b$ 时 $\sum\limits_{n=0}^{\infty} u_n(x)$ 的敛散性，从而确定收敛域

7. **求和函数常用结论**（和泰勒展开一样，$\textcolor{red}{全部要背}$，不过这些推一遍也差不多记住了 🌚）
   - （1）母型级数（⭐⭐⭐ 考频最高，考试时不要推，直接套）
     - $\sum\limits_{n=1}^{\infty} \dfrac{1}{n} x^n = -\ln (1- x), -1\le x < 1$（无缺项）
     - $\sum\limits_{n=1}^{\infty} \dfrac{x^{2n-1}}{2n-1} = \dfrac{-\ln(1-x) - (-\ln(1 + x))}{2} = \dfrac{1}{2} \ln \dfrac{1+x}{1-x}, -1<x < 1$（缺项，只有奇次幂，即 $\sum\limits_{n=1}^{\infty} \dfrac{x^{2n-1}}{2n-1} = \dfrac{1}{2}\sum\limits_{n=1}^{\infty} \dfrac{x^n}{n} - \dfrac{1}{2}\sum\limits_{n=1}^{\infty} \dfrac{(-x)^n}{n}$，目前最热门的考点！）
     - $\sum\limits_{n=0}^{\infty} \dfrac{x^{2n+1}}{2n+1}(-1)^n = \arctan x, |x| \le 1$（缺项 + 交错）
   - （2）子型级数（⭐⭐ 尽量用巧力，不要蛮力硬算）
     - $\sum\limits_{n=0}^{\infty} (an^2 + bn + c)x^n = ax^2 \sum\limits_{n=0}^{\infty} n(n-1)x^{n-2} + (a + b)x \sum\limits_{n=0}^{\infty} nx^{n-1} + c \sum\limits_{n=0}^{\infty} x^n = ax^2[\dfrac{2}{(1-x)^3}] + (a + b)x[\dfrac{1}{(1-x)^2}] + c[\dfrac{1}{1-x}]$
   - （3）阶乘级数（⭐ 收敛域都是 $(-\infty, +\infty)$）
     - $\sum\limits_{n=0}^{\infty} \dfrac{x^n}{n!} = e^x$（不跳项+不交错）
     - $\sum\limits_{n=0}^{\infty} \dfrac{x^{2n}}{(2n)!} = \dfrac{e^2 + e^{-x}}{2}$（跳项+不交错，从上面那个推出来的）
     - $\sum\limits_{n=0}^{\infty} \dfrac{(-1)^n}{(2n+1)!} x^{2n+1} = \sin x$（跳项+交错）
     - $\sum\limits_{n=0}^{\infty} \dfrac{(-1)^n}{(2n)!} x^{2n} = \cos x$（跳项+交错）


> ⚠️ 求和函数的易错点
> - $n$ 和 $n+1$，不要被这个东西纠结住
> - 阶数和系数一定要配齐！（真题中考过的求和函数的题目，无一例外都可以小猪佩奇）
> - 收敛域内无定义的点，要记得补上对应的值，千万别忘了！
> - $0^0 = 1$，小心 $x = 0、n = 0$ 的情况
>
> “拆配凑补”


8. **函数展开为级数**（其实都是泰勒展开，下面列几个常考的）
   - （1）$e^x = \sum\limits_{n=0}^{\infty} \dfrac{x^n}{n!} (-\infty < x < \infty)$
   - （2）$\sin x = \sum\limits_{n=0}^{\infty} (-1)^n \dfrac{x^{2n+1}}{(2n+1)!} (-\infty < x < \infty)$
   - （3）$\cos x = \sum\limits_{n=0}^{\infty} (-1)^n \dfrac{x^{2n}}{(2n)!} (-\infty < x < \infty)$
   - （4）$\ln (x+1) = \sum\limits_{n=1}^{\infty} \dfrac{x^n}{n} (-1)^{n-1} (-1 < x \le 1)$
   - （5）$\dfrac{1}{1 - x} = \sum\limits_{n=0}^{\infty} x^n (-1 < x < 1)$


9. **傅里叶级数**：设函数 $f(x)$ 为周期为 $2l$ 的周期函数，且在 $[-l, l]$ 上可积，则
   - （1）傅里叶系数（以 $2l$ 为周期）：
     - $a_n = \dfrac{1}{l} \int^{l}_{-l} f(x) \cos \dfrac{n\pi}{l} x dx (n = 0, 1, 2,...)$
     - $b_n = \dfrac{1}{l} \int^{l}_{-l} f(x) \sin \dfrac{n\pi}{l} x dx (n = 0, 1, 2,...)$
   - （2）傅里叶级数（以 $2l$ 为周期）：$f(x) ～ \dfrac{a_0}{2} + \sum\limits_{n=1}^{\infty} (a_n \cos \dfrac{n\pi}{l} x + b_n \sin \dfrac{n\pi}{l} x) = S(x) $
   - （3）狄利克雷收敛定理：设 $f(x)$ 是以 $2l$ 为周期的可积函数，如果在 $[-l, l]$ 上 $f(x)$ 满足“连续或只有有限个第一类间断点”、“至多只有有限个极值点”，则 $f(x)$ 的傅里叶级数在 $[-l, l]$ 上处处收敛，记其和函数为 $S(x)$，则 $S(x) = \begin{cases} f(x), & x 为连续点 \\ \dfrac{f(x-0) + f(x+0)}{2}, & x 为间断点 \\ \dfrac{f(-l+0)+ f(l-0)}{2}, & x = \pm l \end{cases}$（即 $S(x)$ 收敛于点的 $\dfrac{左极限+右极限}{2}$）
   - （4）正弦级数：当 $f(x)$ 为奇函数时，其展开式是正弦级数 $f(x) ～ \sum\limits_{n=1}^{\infty} b_n \sin \dfrac{n\pi x}{l} , b_n = \dfrac{2}{l} \int^l_0 f(x) \sin \dfrac{n\pi x}{l} dx$
   - （5）余弦级数：当 $f(x)$ 为偶函数时，其展开式是余弦级数 $f(x) ～ \dfrac{a_0}{2} + \sum\limits_{n=1}^{\infty} a_n \cos \dfrac{n\pi x}{l} , a_n = \dfrac{2}{l} \int^l_0 f(x) \cos \dfrac{n\pi x}{l} dx$（$a_0 = \dfrac{2}{l} \int^l_0 f(x) dx$）

> 💡 关于傅里叶级数，建议搭配《信号与系统》中的解释来理解：[孟桥老师的信号课](https://www.bilibili.com/video/BV144411D73H?p=48)

10. **周期奇延拓与周期偶延拓**
   - （1）周期奇延拓：设 $f(x)$ 定义在 $[0, l]$ 上，令 $F(x) = \begin{cases} f(x), & 0<x\le l \\ -f(-x), & -l \le x < 0 \\ 0, & x=0 \end{cases}$，再令 $F(x)$ 为以 $2l$ 为周期的周期函数
   - （2）周期偶延拓：设 $f(x)$ 定义在 $[0, l]$ 上，令 $F(x) = \begin{cases} f(x), & 0\le x\le l \\ f(-x), & -l \le x < 0  \end{cases}$，再令 $F(x)$ 为以 $2l$ 为周期的周期函数





## 17. 多元函数积分学（Part1，空间几何基础）

> 👉 本章强化阶段强烈推荐李艳芳老师的课，讲的非常非常详细，而且比 🐙 的要更深刻
> 
> 💡 本章重点习题：
> - 【1000b-17-1、5、10、13、14】
>
> 此外，推荐学完线代二次型之后，回来再看看旋转双曲面、旋转抛物面，会有更深层次的理解

1. **方向余弦**
   - 非零向量 $\vec{r}$ 与三条坐标轴的夹角 $\alpha,\beta,\gamma$ 称为向量 $\vec{r}$ 的方向角，$(\cos\alpha, \cos\beta, \cos\gamma) = (\dfrac{x}{|\vec{r}|}, \dfrac{y}{|\vec{r}|}, \dfrac{y}{|\vec{r}|}) = \dfrac{1}{|\vec{r}|}(x, y, z) = \dfrac{\vec{r}}{|\vec{r}|} = \vec{e_r}$，其中 $\cos\alpha, \cos\beta, \cos\gamma$ 称为 $\vec{r}$ 的方向余弦


2. **向量的基本运算**：
   - 设 $\vec{a} = (a_x, a_y, a_z)$，$\vec{b} = (b_x, b_y, b_z)$，$\vec{c} = (c_x, c_y, c_z)$
   - （1）$\vec{a} \cdot \vec{b} = (a_x, a_y, a_z)\cdot (b_x, b_y, b_z)$
   - （2）$\vec{a} \cdot \vec{b} = |\vec{a}||\vec{b}| \cos\theta$，$\cos\theta = \dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$
   - （3）$\vec{a} \times \vec{b} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{vmatrix}$（$\vec{a}$ 与 $\vec{b}$ 张成的区域面积为 $ |\vec{c}| = |\vec{a} \times \vec{b}| = |\vec{a}||\vec{b}| \sin \theta$）
   - （4）混合积：$[\vec{a} \vec{b} \vec{c}] = (\vec{a} \times \vec{b})\cdot \vec{c} = \begin{vmatrix} a_x & a_y & a_z \\ b_x & b_y & b_z  \\ c_x & c_y & c_z \end{vmatrix}$（三个向量张成的体积）

> 💡 常用性质：
> - （1）$\vec{a} \times \vec{a} = 0$（因为张不成一个面积出来）
> - （2）$\vec{b} \times \vec{a} = - \vec{a} \times \vec{b}$
> - （3）向量积对数乘的结合律：$(\lambda \vec{a}) \times \vec{b} = \lambda(\vec{a} \times \vec{b}) = a \times (\lambda \vec{b})$（因为是行列式运算嘛）
> - （4）混合积计算是满足右手系的（即从左到右字母是顺序的，如 abcd、bcda、cdab），因此有 $(\vec{a} \times \vec{b}) \cdot \vec{c} = (\vec{b} \times \vec{c}) \cdot \vec{a} = (\vec{c} \times \vec{a}) \cdot \vec{b}$，进而有 $(\vec{a} \times \vec{b}) \cdot \vec{c} = \vec{a}\cdot (\vec{b} \times \vec{c}) $
> - （5）$(\vec{a} \times \vec{b}) \cdot \vec{c} \Leftrightarrow \vec{a},\vec{b},\vec{c}$ 共面（张不成一个空间出来）


3. **平面方程常见形式**
   - （1）点法式（点斜式）：$A(x - x_0) + B(y - y_0) + C(z - z_0) = 0$【$\vec{n} = (A, B, C)$ 为平面的法向量，理解方法：$(A, B, C) \cdot (x-x_0, y-y_0, z-z_0) = 0$】
   - （2）一般形式：$Ax + By + Cz + D = 0$【$\vec{n} = (A, B , C)$ 为平面的一个法向量，其实相当于上面的点法式，不过将 $-Ax_0-By_0 -Cz_0$ 写成 $D$】
   - （3）截距式（用的非常少）：设平面和三个坐标轴的交点截距分别为 $a, b, c$，则平面方程为 $\dfrac{x}{a} + \dfrac{y}{b} + \dfrac{z}{c} = 1$

> 💡 平面与平面的夹角（考的不多，特此布防）
>    - 设两个平面的法向量分别为 $\vec{n_1} = (A_1 , B_1, C_1)$、$\vec{n_2} = (A_2 , B_2, C_2)$，则 $\cos\theta = \dfrac{|\vec{n_1}\vec{n_2}|}{|\vec{n_1}||\vec{n_2}|} = > \dfrac{|A_1A_2 + B_1B_2 + C_1 C_2|}{\sqrt{A_1^2 + B_1^2 + C_1^2} \sqrt{A_2^2 + B_2^2 + C_2^2}}$

4. **距离公式**
   - （1）点到平面距离：点 $P_0(x_0, y_0, z_0)$ 到平面 $Ax + By + Cz + D = 0$ 的距离公式为 $d = \dfrac{|Ax_0 + By_0 + Cz_0 + D|}{\sqrt{A^2 + B^2 + C^2}}$
   - （2）两平行平面之间的距离公式：$d = \dfrac{|Ax_0 + By_0 + Cz_0 + D_1|}{\sqrt{A^2 + B^2 + C^2}} = \dfrac{|D_0 - D_1|}{\sqrt{A^2 + B^2 + C^2}}$


5. **空间直线方程的常见形式**（把空间直线看成两个平面的交线）
   - （1）一般形式（两个平面相交）：$\begin{cases} A_1 x + B_1 y + C_1 z + D_1 = 0 \\ A_2 x + B_2 y + C_2 z + D_2 = 0 \end{cases}$
   - （2）点向式（一般会和线性代数一块考）：$M_0(x_0, y_0 ,z_0)$ 为直线上一点，$\vec{s} = (m, n, p)$ 为直线的方向向量，则直线方程为 $\dfrac{x - x_0}{m} = \dfrac{y - y_0}{n} = \dfrac{z - z_0}{p}$【我们约定：在点向式方程中，若分母为零，则分子也为零】
   - （3）参数方程（用的比较多）：令点向式中的 $\dfrac{x - x_0}{m} = \dfrac{y - y_0}{n} = \dfrac{z - z_0}{p} = t$，那么有 $\begin{cases} x = x_0 + mt \\ y = y_0 + nt \\ z = z_0 + pt \end{cases}$
   - （4）两点式（其实完全就是点向式，可以不用记）：$\dfrac{x - x_0}{x_1 - x_0} = \dfrac{y - y_0}{y_1 - y_0} = \dfrac{z - z_0}{z_1 - z_0}$

> 💡 直线与直线的夹角（用方向向量算）：$\cos \varphi = \dfrac{|m_1m_2 + n_1n_2 + p_1p_2|}{\sqrt{m_1^2 + n_1^2 + p_1^2}\sqrt{m_2^2 + n_2^2 + p_2^2}}$

> 💡 平面与直线的夹角：$\sin\varphi = \dfrac{|Am + Bn + Cp|}{\sqrt{A^2 + B^2 + C^2}\sqrt{m^2 + n^2 + p^2}}$（注意这里是正弦，可以从几何的角度理解）



6. **直线相关的距离公式**
   - （1）点到直线的距离：设 $M_0$ 是直线外一点，$M$ 是直线上一点，直线方向向量为 $\vec{s}$，则点 $M_0$ 到直线 $L$ 的距离为 $d = \dfrac{|\overrightarrow{M_0M} \times \vec{s}|}{|\vec{s}|} = \dfrac{|\overrightarrow{M_0M} \times \vec{s}|}{\sqrt{m^2 + n^2 + p^2}}$
   - （2）两条平行直线之间的距离：根据上面的式子，把直线上的点代入即可



7. **方向导数**
   - （1）概念：设 $f(x, y)$ 在点 $(a, b)$ 及其附近有定义，$\vec{l} = (\cos \alpha, \cos\beta)$ 是一单位向量，若极限 $\lim\limits_{t\to 0^+} \dfrac{f(a+t\cos\alpha, b+t\cos\beta) - f(a, b)}{t}$ 存在，则称其值为 $f(x, y)$ 在点 $(a, b)$ 沿方向 $\vec{l} = (\cos\alpha, \cos\beta)$ 的方向导数，记作 $\dfrac{\partial f}{\partial \vec{l}} |_{(a, b)}$
   - （2）计算：若函数 $f(x, y)$ 在点 $(a, b)$ $\textcolor{red}{可微}$，则其在点 $(a, b)$ 沿任意方向 $\vec{l} = (\cos \alpha, \cos\beta)$ 的方向导数都存在，且 $\dfrac{\partial f}{\partial \vec{n}} |_{(a, b)} = \dfrac{\partial f}{\partial x} |_{(a, b)}\cos\alpha + \dfrac{\partial f}{\partial y} |_{(a, b)} \cos\beta = \mathbf{grad} f(a, b) \cdot \vec{l}^o$

> 💡 这里的 $\vec{l}$ 是单位向量，$\vec{l}^o$ 是指归一化后的 $\vec{l}$，计算方向导数的时候千万不要忘记归一化！题目很多时候给的方向向量是没有经过归一化的！


8. **梯度**
   - （1）概念：设函数 $f(x, y)$ 在点 $(a, b)$ 及其附近有定义，若单位向量 $\vec{l_0}$ 满足 $\dfrac{\partial d}{\partial \vec{l_0}} |_{(a, b)} = \max\limits_{|\vec{l}| = 1} \{ \dfrac{\partial d}{\partial \vec{l}} |_{(a, b)} \}$，则称向量 $\dfrac{\partial d}{\partial \vec{l_0}} |_{(a, b)} \vec{l_0}$ 为函数 $f(x, y)$ 在点 $(a,  b)$ 的梯度，记作 $\mathbf{grad}f(a, b) = (f'_x, f'_y)|_{(a, b)} = f'_x(a, b)\vec{i} + f'_y(a, b) \vec{j}$（梯度是个向量）
   - （2）梯度向量的几何意义：方向为取到最大方向导数的方向；长度为方向导数的最大值 
   - （3）在可微条件下的梯度为 $\mathbf{grad}f(a, b) = (\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y})|_{(a, b)}$，此时方向导数最大值 = 梯度的模值 $ = \sqrt{(f'_x)^2 + (f'_y)^2}|_{(a, b)}$

> 💡 关于梯度的一些小结论：
> - （1）$\mathbf{grad}(u \pm v) = \mathbf{grad}$
> - （2）$\mathbf{grad}(uv) = v\mathbf{grad}u + u\mathbf{grad} v$
> - （3）$\mathbf{grad}(\dfrac{u}{v}) = \dfrac{v\mathbf{grad}u - u\mathbf{grad}v}{v^2}$

> ⚠️ 易错题：设 $f(x, y) = e^{-(x^2 + 2y^2)}$，曲线 $y = y(x)$ 上任意一点 $P(x, y)$ 的切线方向始终指向 $f(x, y)$ 变化率最大的方向，且 $y(1) = 2$，求 $y(x)$
> - 这一题求梯度的时候要特别注意，求 $f(x, y)$ 梯度的时候，要把 $y$ 当成独立变量而不是关于 $x$ 的函数（实际上梯度计算时求偏导是对位置求偏导，即 $f'_x = f'_1$、$f'_y = f'_2$）


9. **散度**
   - 设向量场 $\vec{A}(x, y, z) = P(x, y, z)\vec{i} + Q(x, y, z) \vec{j} + R(x, y, z) \vec{k}$，则 $\mathbf{div} \vec{A} = \dfrac{\partial P}{\partial x} + \dfrac{\partial Q }{\partial y} + \dfrac{\partial R}{\partial z}$

10. **旋度**
   - 设向量场 $\vec{A}(x, y, z) = P(x, y, z)\vec{i} + Q(x, y, z) \vec{j} + R(x, y, z) \vec{k}$，则 $\mathbf{rot} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ \dfrac{\partial}{\partial x} & \dfrac{\partial}{\partial y} & \dfrac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$

11. **多元函数的泰勒多项式**（老登说数二之前考过，但数一真的会考这种吗...）
   - 设 $f(x, y)$ 二阶偏导数连续，记 $X_0 (x_0, y_0), \Delta X = (\Delta x, \Delta y) = (x-x_0, y-y_0)$，则 $f(x, y)$ 的二次泰勒多项式为 $f(x_0, y_0) + (f'_x, f'_y)|_{X_0} \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} + \dfrac{1}{2!}(\Delta x, \Delta y) \begin{pmatrix} f''_{xx} & f''_{xy} \\ f''_{yx} & f''_{yy} \end{pmatrix} |_{X_0} \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix}$
   - 其中，$(f'_x, f'_y)|_{X_0}$ 为 $f(x, y)$ 在 $X_0$ 点处的梯度

12. **空间曲线的切线与法平面**
   - a. 用参数方程给出曲线   
      - （1）设曲线 $L$ 的方程为 $\begin{cases} x = x(t) \\ y = y(t) \\ z=z(t) \end{cases}, t\in [\alpha, \beta]$，点 $M_0$ 对应参数 $t = t_0$。又设曲线 $L$ 光滑，即 $x(t), y(t), z(t)$ 在 $[\alpha, \beta]$ 上一阶偏导数连续，且 $[x'(t)]^2 + [y'(t)]^2 + [z'(t)]^2 \ne 0$
      - （2）则曲线 $L$ 在点 $M_0$ 处的切向量为 $\vec{\tau} = (x'(t_0), y'(t_0), z'(t_0))$
      - （3）切线方程：$\dfrac{x - x(t_0)}{x'(t_0)} = \dfrac{y - y(t_0)}{y'(t_0)} = \dfrac{z - z(t_0)}{z'(t_0)}$
      - （4）法平面方程为 $x'(t_0)[x - x(t_0)] + y'(t_0)[y - y(t_0)] + z'(t_0)[z - z(t_0)] = 0$
   - b. 用方程组给出曲线：$\begin{cases} F(x, y, z) = 0 \\ G(x, y, z) = 0 \end{cases}$
     - （1）当在 $\dfrac{\partial(F, G)}{\partial(y, z)}  = \begin{vmatrix} F'_y & F'_z \\ G'_y & G'_z \end{vmatrix} \ne 0$ 时，可以确定 $\begin{cases} x = x \\ y = y(x) \\ z = z(x) \end{cases}$
     - （2）切向量：$\vec{\tau} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ F'_x & F'_y & F'_z \\ G'_x & G'_y & G'_z \end{vmatrix} = (A, B, C)$（相当于两条法向量求向量积）
     - （3）切线：$\dfrac{x - x_0}{A} = \dfrac{y - y_0}{B} = \dfrac{z - z_0}{C}$
     - （4）法平面：$A(x - x_0) + B(y - y_0) + C(z - z_0) = 0$

> 💡 小技巧：
> - （1）当曲线 $L$ 的方程是 $\begin{cases} y = y(x) \\ z = z(x) \end{cases}$ 时，可写成 $\begin{cases} x= x \\ y = y(x) \\ z=z(t) \end{cases}$，在点 $(x_0, y(x_0), z(x_0))$ 处的切向量为 $\vec{\tau} = (1, y'(x_0), z'(x_0))$
> - （2）将曲线的一般方程转化为参数方程的常用技巧：有二元一次方程时直接令其中一元为 t、见到椭圆或圆时使用三角换元等（换元之后别忘了取值范围的确定）

13. **空间曲面的切平面和法线**
   - （1）记光滑曲面 $\Sigma$ 方程为 $F(x, y, z) = 0$，且有 $(F'_x)^2 + (F'_y)^2 + (F'_z)^2 \ne 0$
   - （2）则曲面 $\Sigma$ 在点 $(a, b , c)$ 处的法向量为 $\vec{n} = (F'_x(a, b, c), F'_y(a, b, c), F'_z(a, b, c))$（⭐⭐⭐ 重中之重）
   - （3）切平面方程为 $F'_x(a, b, c)(x-a) + F'_y(a, b, c)(y-b) + F'_z(a, b, c)(z-c) = 0$
   - （4）法线方程为：$\dfrac{x-a}{F'_x(a, b, c)} = \dfrac{y-b}{F'_y(a, b, c)} = \dfrac{z-c}{F'_z(a, b, c)}$

> 💡 注意：
> - （1）$\vec{n} = \mathbf{grad} F$
> - （2）当曲面 $\Sigma$ 由显式方程 $z = f(x, y)$ 表示，且 $f(x, y)$ 具有一阶偏导时，其实可以直接看作 $F(x, y, z) = z - f(x, y) = 0$


14. **旋转曲面**（曲线 $\Gamma$ 绕一条定直线旋转一周所形成的曲面）
    - （1）曲线 $\Gamma \begin{cases} F(x, y, z) = 0 \\ G(x, y, z)=0 \end{cases}$ 绕直线 $L$：$\dfrac{x-x_0}{l} = \dfrac{y-y_0}{m} = \dfrac{z-z_0}{n}$ 旋转一周形成一个旋转曲面，求法如下
    - （2）设直线 $L$ 取上一点 $M_0(x_0, y_0, z_0)$，方向向量为 $\vec{\tau} = (l, m, n)$
    - （3）在母线 $\Gamma$ 上取一点 $M_1(x_1, y_1, z_1)$，则过 $M_1$ 的纬圆（过 $M_1$ 作直线 $L$ 的法线段后旋转一周转出来的圆）上任意一点 $P(x, y, z)$ 满足条件
      - a. $\overrightarrow{M_1P} \bot \vec{\tau}$
      - b. $|\overrightarrow{M_0P}| = |\overrightarrow{M_0M_1}|$
      - 即 $\begin{cases} l(x-x_1)+m(y-y_1)+n(z-z_1) = 0 \\ (x-x_0)^2 + (y-y_0)^2 + (z-z_0)^2 = (x_1-x_0)^2 + (y_1-y_0)^2 + (z_1-z_0)^2 \end{cases}$
    - （4）与方程 $F(x_1, y_1, z_1) = 0$ 和 $G(x_1, y_1, z_1)=0$ 联立消去 $x_1,y_1,z_1$ 即可得到旋转曲面的方程

15. **投影曲线**（后面的线面积分会用到）
    - 设空间曲线 $C$ 的一般方程为 $\begin{cases} F(x, y, z) = 0 \\ G(x, y, z) = 0 \end{cases}$，由该方程组消去变量 $z$ 后得到的方程 $H(x, y) = 0$ 为空间曲线在 $xOy$ 平面上的投影曲线
    - 其他坐标面上的投影曲线求法类似
    - 特殊情况：当出现 $\begin{cases} F(x, y) = 0 \\ G(x, y, z) = 0 \end{cases}$ 这样的空间曲线时，则其在 $xOy$ 平面上的投影曲线为 $\begin{cases} F(x, y) = 0 \\ z = 0 \end{cases}$，其他情况以此类推



## 18. 多元函数积分学（Part2）
> 这一章主要是东西多，难倒不算太难，整个这一章就是一个巨大的麦克斯韦方程组
>
> 本章重点习题：【1000b-18-1、3、6、7、10、15、16、22、23、24、25、26、27】

1. **三重积分的和式积分**：$\iiint\limits_{\Omega} g(x, y, z) dv = \lim\limits_{n\to \infty}\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n}\sum\limits_{k=1}^{n} g(a + \dfrac{b-a}{n}i, c + \dfrac{d-c}{n}j, e + \dfrac{f-e}{n}k) \cdot \dfrac{b-a}{n} \cdot \dfrac{d-c}{n} \cdot \dfrac{f-e}{n}$，其中 $\Omega = \{(x, y, z) | a\le x\le b, c\le y \le d, e\le z\le f\}$


> 💡 三重积分计算技巧：
> - （1）遇到积分到后面很难算等情况时，要记得尝试交换积分次序
> - （2）注意观察被积函数和积分区域，看情况使用三重积分的普通对称性和轮换对称性（这一块的技巧和二重积分部分完全一样）

2. **三重积分的柱面坐标系积分**
   - （1）令 $\begin{cases} x = r\cos \theta \\ y = r\sin\theta \end{cases}$
   - （2）则有 $\iiint\limits_{\Omega} f(x, y, z) dxdydz = \iiint\limits_{\Omega} f(r\cos\theta, r\sin\theta, z) r drd\theta dz$



3. **三重积分的球面坐标系积分**
   - （1）适用场合：
     - a. 被积函数中包含 $\begin{cases} f(x^2+y^2+z^2) \\ f(x^2 + y^2) \end{cases}$
     - b. 积分区域为 $\begin{cases} 球或球的部分 \\ 锥或锥的部分 \end{cases}$
   - （2）计算方法：
     - a. 令 $\begin{cases} x = r\sin\varphi\cos\theta \\ y = r\sin\varphi\sin\theta \\ z = r\cos\varphi \end{cases}$
     - c. $dv = r^2 \sin\varphi d\theta d\varphi dr$

> 💡关于积分换元
> - 在前面二重积分的地方，我们知道换元的时候，比如 $\begin{cases} x = r\cos\theta \\ y = r\sin\theta \end{cases}$，有 $d\sigma = dxdy = \begin{vmatrix} \dfrac{\partial x}{\partial r} & \dfrac{\partial x}{\partial \theta} \\ \dfrac{\partial y}{\partial r} & \dfrac{\partial x}{\partial \theta} \end{vmatrix} drd\theta$
> - 这里同理有 $dv = dxdydz = \begin{vmatrix} \dfrac{\partial x}{\partial r} & \dfrac{\partial x}{\partial \varphi} & \dfrac{\partial x}{\partial \theta} \\ \dfrac{\partial y}{\partial r} & \dfrac{\partial y}{\partial \varphi} & \dfrac{\partial y}{\partial \theta} \\ \dfrac{\partial z}{\partial r} & \dfrac{\partial z}{\partial \varphi} & \dfrac{\partial z}{\partial \theta} \end{vmatrix} d\theta d\varphi dr = r^2\sin\varphi d\theta d\varphi dr$
> - 更一般的，设 $\begin{cases} x = x(u, v, w) \\ y = y(u, v, w) \\ z = z(u, v, w) \end{cases}$，则 $dxdydz =  \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} & \dfrac{\partial x}{\partial w} \\ \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} & \dfrac{\partial y}{\partial w} \\ \dfrac{\partial z}{\partial u} & \dfrac{\partial z}{\partial v} & \dfrac{\partial z}{\partial w} \end{vmatrix} dudvdw$


4. **空间图形的质心坐标公式**
   - （1）$\overline{x} = \dfrac{\iiint\limits_{\Omega} x \rho(x, y, z) dv}{\iiint\limits_{\Omega} \rho(x, y, z) dv}$
   - （2）$\overline{y} = \dfrac{\iiint\limits_{\Omega} y \rho(x, y, z) dv}{\iiint\limits_{\Omega} \rho(x, y, z) dv}$
   - （3）$\overline{z} = \dfrac{\iiint\limits_{\Omega} z \rho(x, y, z) dv}{\iiint\limits_{\Omega} \rho(x, y, z) dv}$

> 💡 当 $\rho(x, y, z)$ 为常数时，质心就是形心

> 💡 做题时还有一种常用代换：$\iiint\limits_{\Omega} y dxdydz = \int^a_b dz \iint\limits_{D} y dxdy = \int^a_b \overline{y} S_D dz $


5. **积分的其他物理应用**（感觉不太可能考到啊...）
   - （1）求引力：对于空间物体，若体密度为 $\rho(x, y, z)$，$\Omega$ 是物体所占的空间区域，则计算该物体对物体外一点 $M_0 (x_0, y_0, z_0)$ 处的质量为 $m$ 的质点的引力 $F_x, F_y, F_z$ 公式为：
     - a. $F_x = Gm \iiint\limits_{\Omega} \dfrac{\rho(x, y, z) (x-x_0)}{[(x-x_0)^2 + (y-y_0)^2 + (z- z_0)^2]^{\frac{3}{2}}} dv$
     - b. $F_y = Gm \iiint\limits_{\Omega} \dfrac{\rho(x, y, z) (y-y_0)}{[(x-x_0)^2 + (y-y_0)^2 + (z- z_0)^2]^{\frac{3}{2}}} dv$
     - c. $F_z = Gm \iiint\limits_{\Omega} \dfrac{\rho(x, y, z) (z-z_0)}{[(x-x_0)^2 + (y-y_0)^2 + (z- z_0)^2]^{\frac{3}{2}}} dv$
   - （2）求转动惯量（遇到再回来补吧）


6. **第一类曲线积分**（不考虑方向，标量计算）
   - （1）定义：$\int_L f(x, y) ds = \lim\limits_{\lambda\to 0} \sum\limits_{i=1}^{n} f(\xi_i, \eta_i) \Delta s_i$，其中 $f(x, y)$ 叫做被积函数，$L$ 叫做被积弧段（相当于在三维坐标系下，某条空间曲线在 $xOy$ 平面上的投影曲线是 $f(x, y)$，沿着这条平面上的投影曲线进行积分计算）
   - （2）性质
     - a. 线性：$\int_L [\alpha f(x, y) + \beta g(x, y)] ds = \alpha \int_L f(x,y)ds + \beta \int_L g(x, y) ds$
     - b. 可加性：若积分弧段 $L$ 可以划分成两段光滑曲线弧 $L_1, L_2$，则 $\int_L f(x,y)ds = \int_{L_1} f(x,y)ds + \int_{L_2} f(x,y)ds$
     - c. 若在区域 $D$ 上， $f(x, y) $ 恒等于 1，$L$ 又是 $D$ 内的一条分段光滑曲线，$l$ 为 $L$ 的长度，则 $l = \int_L ds$
     - d. 积分弧段是可以应用普通对称性和轮换对称性的
   - （3）计算方式（是不是想到定积分那一块，有一个物理应用叫“求弧长”）
     - a. 曲线 $L$ 由参数方程表示：设 $\begin{cases} x = \varphi(t) \\ y = \psi(t) \end{cases}$，$\varphi(t), \psi(t)$ 在 $[\alpha, \beta]$ 上具有一阶连续导数，且 $[\varphi'(t)]^2 + [\psi'(t)]^2 \ne 0$，则曲线积分 $\int_L f(x, y)ds$ 存在且 $\int_L f(x, y) ds = \int^{\beta}_{\alpha} f(\varphi(t), \psi(t)) [\sqrt{\varphi'(t)]^2 + [\psi'(t)]^2} dt$
     - b. 曲线 $L$ 由 $y = \psi(x)(x_0 \le x \le x_1)$ 表示的情形：可以把这种情况看成特殊的参数方程 $\begin{cases} x = x \\ y = \psi(x) \end{cases} (x_0 \le x \le x_1)$，从而有 $\int_L f(x, y) ds = \int^{x_1}_{x_0} f(x, \psi(x)) \sqrt{1 + [\psi'(x)]^2} dx$
     - c. 曲线 $L$ 由极坐标 $r = r(\theta)(\alpha \le \theta \le \beta)$ 表示的情形：$\int_L f(x, y) ds = \int^{\beta}_{\alpha} f(r(\theta) \cos \theta, r(\theta)\sin\theta) \sqrt{[r(\theta)]^2 + [r'(\theta)]^2} d\theta$
   - （3）物理意义：空间质量不均匀的曲线的质量

> ⚠️ 计算曲线积分的时候是可以把曲线表达式（比如 $x^2 + y^2 = 1$）代入到被积函数 $f(x, y)$ 中的，但学完这一部分后，定积分、二重积分、三重积分的计算千万不要犯糊涂！


> ❓ 假如说第一类曲线积分是沿着线的方向积分，那么下面第二类曲线积分就是把这个方向上的 $ds$ 变成向量然后拆分成 $dx$ 和 $dy$ 两个分量

7. **第二类曲线积分**
   - （1）原理：令向量值函数 $\vec{F}(x,y ) = P(x, y) \vec{i} + Q(x y) \vec{j}$，积分弧段为 $L$，注意到 $d\vec{r} = dx \vec{i} + dy \vec{j}$（可以将 $\vec{F}$ 看成变力，$L$ 是做功路径），那么 $\int_L \vec{F}(x, y) d\vec{r} = \int_L [P(x, y) \vec{i} + Q(x y) \vec{j}] \cdot (dx \vec{i} + dy \vec{j}) = \int_L P(x, y)dx + Q(x, y)dy$
   - （2）性质：线性、可加性、有向性
   - （3）计算（参数方程）：
      - $\int_L P(x, y) dx + Q(x, y) dy  = \int^{\beta}_{\alpha} P(\varphi(t), \psi(t)) d[\varphi(t)] + Q(\varphi(t), \psi(t)) d[\psi(t)] = \int^{\beta}_{\alpha} [P(\varphi(t), \psi(t)) \varphi'(t) + Q(\varphi(t), \psi(t)) \psi'(t)] dt$（⚠️ 注意：下限 $\alpha$ 对应 $L$ 起点，$\beta$ 对应 $L$ 终点，$\alpha$ 不一定小于 $\beta$）




8. **两类曲线积分之间的联系**
   - （1）平面曲线弧 $L$ 上的两类曲线积分之间有这样一个联系：$\int_L Pdx + Qdy = \int_L (P\cos \alpha + Q\cos\beta) ds$，其中 $\alpha(x, y),\beta(x, y)$ 为有向曲线弧 $L$ 在 $(x,y )$ 处的切向量的方向角（其实就是 $\begin{cases} dx = \cos\alpha ds \\ dy =  \cos\beta ds \end{cases}$）
   - （2）类似的，空间曲线弧 $\Gamma$ 上的两类曲线积分之间有这样一个联系：$\int_{\Gamma} Pdx + Qdy + Rdz = \int_{\Gamma} (P\cos \alpha + Q\cos\beta + R\cos\gamma)  ds$，其中 $\alpha(x, y, z),\beta(x, y, z),\gamma(x, y, z)$ 为有向曲线弧 $\Gamma$ 在 $(x,y )$ 处的切向量的方向角


9. **格林公式**（积分与路径无关）
    - 设闭区域 $D$ 由分段光滑的封闭曲线 $L$ 围成，若函数 $P(x, y)$ 及 $Q(x, y)$ 在 $D$ 上具有一阶连续偏导数，则有 $\iint\limits_D (\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y}) dxdy = \oint_{L} Pdx + Qdy$（$\oint$ 代表闭合曲线），其中 $L$ 是 $D$ 的取正向（人在线上走，左手在 D 内）的边界曲线

> ⚠️ 在使用格林公式的时候一定要注意使用条件 “$\textcolor{red}{P(x, y), Q(x, y) 在 D 上具有一阶连续偏导数}$”，出题老头可能会在这里挖坑骗你用格林公式。遇到这种情况，可以尝试使用 “挖洞法” 或 “补线法”


10. **第二类曲线积分与路径无关**（参考保守场或无旋场）
    - 设 $D$ 为平面中的单连通区域，函数 $P(x,y ), Q(x, y)$ 在 $D$ 上具有一阶连续偏导数，则下列条件相互等价
    - （1）对于 $D$ 中任意分段光滑闭曲线 $C$ 都有 $\oint_C Pdx + Qdy  =0$
    - （2）对于 $D$ 中从点 $M_1$ 到点 $M_2$ 的任意两条分段光滑曲线 $L_1, L_2$，都有 $\int_{L_1} Pdx + Qdy = \int_{L_2} Pdx + Qdy$
    - （3）存在在 $D$ 上具有一阶连续偏导数的函数 $u(x, y)$，使得 $d[u(x, y)] = Pdx + Qdy$
    - （4）在 $D$ 中恒有 $\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}$

> 💡 单连通就是整个闭合曲线可以慢慢缩成一个点的区域，复连通就是内部有一个或多个洞的区域

> 💡 对于空间中第二类曲线积分计算技巧：
> - （1）直接计算：参数方程（见上面）或斯托克斯公式（见下面）
> - （2）对于无旋场，即旋度 $\textbf{rot} F = 0$ 时，可以更换积分路径（积分与路径无关）

11. **曲线积分的基本定理**（这不就是保守场嘛...）
    - 设 $\overrightarrow{F}(x, y) = P(x, y) \vec{i} + Q(x, y) \vec{j}$ 是平面区域 $D$ 内的一个向量，若 $P(x, y)$ 与 $Q(x, y)$ 都在 $D$ 内连续，且存在一个数量函数 $f(x , y)$ 使得 $\overrightarrow{F} = \nabla f = \dfrac{\partial f }{\partial x} \vec{i} + \dfrac{\partial f}{\partial y} \vec{j}$，则曲线积分 $\int_L \overrightarrow{F} \cdot d\vec{r}$ 在 $D$ 内与路径无关，且 $\int_L \overrightarrow{F} \cdot d\vec{r} = f(B) - f(A)$，其中 $L$ 是位于 $D$ 内起点为 $A$ 终点为 $B$ 的任意光滑曲线

> 💡 看到这里有没有想到 15 讲微分方程部分的那个全微分方程 $P(x, y) dx + Q(x, y) dy$？这里再补充几点：
>
> - （1）$P(x, y)dx + Q(x, y)dy$ 为全微分的充分必要条件：设区域 $D$ 是一个$\textcolor{red}{单连通}$区域，若函数 $P(x, y)$ 与 $Q(x, y)$ 在 $D$ 内具有一阶连续偏导数，则 $P(x, y) dx + Q(x, y)dy$ 在 $D$ 内为某一函数 $u(x, y)$ 的全微分的充分必要条件是 $\textcolor{red}{\dfrac{\partial Q}{\partial x} = \dfrac{\partial P}{\partial y}}$ 在 $D$ 内恒成立
> - （2）若一个微分方程能写成 $P(x, y) dx + Q(x, y) dy = 0$ 的形式，而 $P(x, y) dx + Q(x, y) dy$ 为某一个函数 $u(x, y) $ 的全微分，则上述方程称为全微分方程，$u(x, y) = C$ 是它的隐式通解，其中 $C$ 为任意常数
> - （3）已知某微分式，我们求其原函数 $u(x, y)$ 的方法一般有三种：折线法、积分法、凑微分法（要求经验丰富）
>     - a. 折线法：$u(x, y) = \int^{(x, y)}_{(a, b)} Pdx + Qdy$
>     - b. 积分法：$u(x, y) = \int \dfrac{\partial u}{\partial x} dx = ... + \varphi(y)$
>     - c. 凑微分法（瞪眼法）：对经验要求比较高


12. **第一类曲面积分**（不考虑方向）
    - （1）定义：$\iint\limits_{\Sigma} f(x, y, z) dS = \lim\limits_{\lambda \to 0} \sum\limits_{i=1}^{n} f(\xi_i, \eta_i, \varsigma_i) \Delta S_i$
    - （2）性质：和第一类曲线积分完全类似
    - （3）计算：若积分曲面 $\Sigma$ 由方程 $z = z(x, y)$ 给出，$\Sigma$ 在 $xOy$ 面上的投影区域为 $D_{xy}$，函数 $z = z(x, y)$ 在 $D_{xy}$ 上具有一阶连续偏导数，被积函数 $f(x, y, z)$ 在 $\Sigma$ 上连续，则 $\iint\limits_{\Sigma} f(x, y, z) dS = \iint\limits_{D_{xy}} f(x, y, z(x, y)) \sqrt{[z'_x(x,y)]^2 + [z'_y(x, y)]^2 + 1} dx dy$（和第一类曲线积分也很类似）
    - （4）物理意义：空间质量不规则曲面的质量

> 💡 算曲面积分的时候，曲面方程也是可以直接代入的

13. **第二类曲面积分**（本质是通量，要考虑方向）
    - （1）定义：$\iint\limits_{\Sigma} R(x, y, z) dxdy = \lim\limits_{\lambda \to 0} \sum\limits_{i=1}^{n} R(\xi_i, \eta_i, \varsigma_i) (\Delta S_i)_{x, y}$
    - （2）右手直角坐标系下有向曲面、法向量以及方向余弦的关系：
    - ![p18-1](/images/mathematic/18-1.png) 
    - （3）计算：$\iint\limits_{\Sigma} R(x, y, z) dxdy = \pm \iint\limits_{D_{xy}} R(x, y, z(x, y)) dxdy$（注意，投影区域是没有方向的，而是对投影谈方向），若 $\Sigma$ 取上侧，即 $\cos \gamma > 0$，则上式右端取正号，反之，若 $\Sigma$ 取下侧，即 $\cos \gamma < 0$，则上式右端取负号。
    - （4）第二类曲面积分为零的三种特殊情况：
      - a. 当曲面 $\Sigma$ 垂直于 $xOy$ 面时，$\iint\limits_{\Sigma} R(x, y, z) dxdy = 0$;
      - b. 当曲面 $\Sigma$ 垂直于 $yOz$ 面时，$\iint\limits_{\Sigma} R(x, y, z) dydz = 0$;
      - c. 当曲面 $\Sigma$ 垂直于 $zOx$ 面时，$\iint\limits_{\Sigma} R(x, y, z) dzdx = 0$;
      - d. 对称点的值相等但方向相反时通量为零（🐙 书上的 “类对称”）
    - （5）还有一种特殊情况，就是当散度 $\textbf{div} F = 0$ 时，所给的是无源场，通过任何封闭曲面（且无奇点在内部）的通量为 0，此时可以换个面积分；即使是非封闭曲面，散度为 0 时也可以换个面积分
    - （6）合一投影法：$\iint\limits_{\Sigma} Pdydz + Qdxdz + Rdxdy = \iint\limits_{\Sigma} (P, Q, R) \cdot (dydz, dxdz, dxdy) = \iint\limits_{\Sigma} (P, Q, R) \cdot \vec{n} \cdot dxdy$（其中 $\vec{n}$ 取决于曲面方向，和上表中相符，这里以投影到 $xOy$ 面为例）





14. **两类曲面积分之间的联系**（2020 年考了一道当年很难的题）
    - $\iint\limits_{\Sigma} Pdydz + Qdxdz + Rdxdy = \iint\limits_{\Sigma} (P\cos \alpha + Q\cos\beta + R\cos\gamma) dS$，其中 $\cos \alpha$、$\cos\beta$、$\cos\gamma$ 为有向曲面 $\Sigma$ 在点 $(x, y, z)$ 处的法向量的方向余弦（其实就是向量点积）
      - $dS = \sqrt{(z'_x)^2 + (z'_y)^2 + 1} dx dy$
      - $dS = \sqrt{1 + (x'_y)^2 + (x'_z)^2 } dy dz$
      - $dS = \sqrt{(y'_x)^2 + 1 + (y'_z)^2 } dx dz$


15. **高斯公式**（三重积分与两类曲面积分的转换）
    - 设空间闭区域 $\Omega$ 由分片光滑的闭曲线 $\Sigma$ 所围成，若函数 $P(x, y, z), Q(x, y, z), R(x, y, z)$ 在 $\Omega$ 上$\textcolor{red}{具有一阶连续偏导数}$，则有 $\iiint\limits_{\Omega} (\dfrac{\partial P}{\partial x} + \dfrac{\partial Q}{\partial y} + \dfrac{\partial R}{\partial z}) dv = \oiint\limits_{\Sigma} Pdydz + Qdxdz + Rdxdy = \oiint\limits_{\Sigma} P\cos\alpha + Q\cos\beta + R\cos\gamma dS$
    - 其中 $\Sigma$ 是 $\Omega$ 的整个边界曲面的外侧，$\cos\alpha, \cos\beta, \cos\gamma$ 是 $\Sigma$ 在点 $(x, y, z)$ 处的法向量的方向余弦

> 💡 高斯公式一般考挖洞、补线/补面，或者两种同时出现（26 年考到了）

16. **斯托克斯公式**（出现频率会稍微低一点）
    - 设 $\Gamma$ 为分段光滑的空间有向闭曲线，$\Sigma$ 为以 $\Gamma$ 为边界的分片光滑的有向曲面，$\Gamma$ 的正向与 $\Sigma$ 的侧符合右手法则。若函数 $P(x, y, z), Q(x, y, z), R(x, y, z)$ 在曲面 $\Sigma$（连同边界 $\Gamma$）上$\textcolor{red}{具有一阶连续偏导数}$，则有 $\iint\limits_{\Sigma} (\dfrac{\partial R}{\partial y} - \dfrac{\partial Q}{\partial z} ) dydz + (\dfrac{\partial P}{\partial z} - \dfrac{\partial R}{\partial x} ) dxdz + (\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y} ) dxdy = \oint_{\Gamma} Pdx + Qdy + Rdz$
    - 上述写法不太好记，可以写成下面的形式：$\oint_{\Gamma} Pdx + Qdy + Rdz = \iint\limits_{\Sigma} \begin{vmatrix} dydz & dxdz & dxdy \\ \dfrac{\partial}{\partial x} & \dfrac{\partial}{\partial y} & \dfrac{\partial}{\partial z} \\ P & Q & R \end{vmatrix} = \iint\limits_{\Sigma} \begin{vmatrix} \cos\alpha & \cos\beta & \cos\gamma \\ \dfrac{\partial}{\partial x} & \dfrac{\partial}{\partial y} & \dfrac{\partial}{\partial z} \\ P & Q & R \end{vmatrix} dS$，其中 $\vec{n} = (\cos\alpha, \cos\beta, \cos\gamma)$ 是有向曲面 $\Sigma$ 在点 $(x, y, z)$ 处的单位法向量

> 💡 Tips：
> - 斯托克斯的旋度转化为高斯散度，$\mathbf{rot} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ \dfrac{\partial}{\partial x} & \dfrac{\partial}{\partial y} & \dfrac{\partial}{\partial z} \\ P & Q & R \end{vmatrix} = (\dfrac{\partial R}{\partial y} - \dfrac{\partial Q}{\partial z} ) \vec{i} + (\dfrac{\partial P}{\partial z} - \dfrac{\partial R}{\partial x} ) \vec{j} + (\dfrac{\partial Q}{\partial x} - \dfrac{\partial P}{\partial y} ) \vec{k}$
> - 格林公式就是这玩意的特殊情况