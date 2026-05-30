---
title: Coldrain 的 27 考研数一概统强化阶段拾遗
date: 2026-05-14 17:30:00
tags: 
    - 考研数学
categories: 
    - 考研数学
description: |
    Coldrain 的概率论备考笔记，涵盖了基础 + 强化的重要内容，以及一些题解拾遗
---

> ✍ 写在前面
>
> Coldrain 在第一轮基础复习的时候并没有留下任何笔记，直到概率论刷题的时候，发现知识点比较零散容易忘记，做题卡住了，遂准备从第二轮基础复习开始好好留下笔记，便于知识点检索与记忆
>
> 至于为什么选择以 blog 的形式记录，因为 Coldrain 平时没有手写笔记的习惯，而且 blog 形式的笔记可以在任何设备上随时随地打开💦
>
> 本笔记内容并没有全覆盖，过于基础的公式与结论未在本笔记中记录，故本笔记可用于一轮学习结束之后对重难考点进行查漏补缺，但请不要用于替代考研书籍来进行一轮复习
>
> “岂不闻天无绝人之路，只要我想走，路就在脚下。”—— 25 奥本海豚

## 1. 随机事件及其概率

> ✍ 本部分内容过于简单，所以这里只记录重点公式与结论


1. 对立运算
   - $P(\overline{A}) = 1- P(A)$

2. 加法运算（并集）
   - $P(A\cup B) = P(A+B) = P(A) + P(B) - P(AB)$
   - $P(A\cup B \cup C) = P(A+B+C) = P(A) + P(B) + P(C)- P(AB) - P(AC) - P(BC) + P(ABC)$
 
> 💡 加法运算这里 Coldrain 刻意将 “$\cup$” 与 “+” 都写了上去，因为在有些题目中将“并”运算换成加法运算可以**快速化简**冗长的随机事件表达式

3. 互不相容
   - $P(AB) = 0$

4. 两事件独立
   - $P(AB) = P(A) P(B)$

> 💡 两事件独立，进而有：
> - $P(A\cup B) = P(A) + P(B) - P(A)P(B) = 1 - P(\overline{A})P(\overline{B})$

5. 减法运算
   - $P(A-B) = P(A\overline{B}) = P(A) - P(AB)$

6. 乘法运算（交集）
   - $P(AB) = P(B)P(A|B) = P(A) P(B|A)$

7. 条件概率
   - $P(B|A) = \dfrac{P(AB)}{P(A)}$

> 💡 条件概率常用性质
> - $P(B|A) + P(\overline{B}|A) = 1$
> - $P(A_1 \cup A_2 | B) = P(A_1 | B) + P(A_2 |B) - P(A_1 A_2 |B)$
> - $P(A_1 - A_2 | B) = P(A_1 | B) - P(A_1 A_2 |B)$
> - $A$ 与 $B$ 独立时，有 $P(AB|C) = P(A|C) P(B|C)$

8. 全概率公式
   - $P(B) = \sum\limits_{i=1}^{n} P(A_i)P(B|A_i)$

9. 贝叶斯公式
   - $P(A_i|B) = \dfrac{P(A_i B)}{P(B)} = \dfrac{P(A_i)P(B|A_i)}{P(B)}$


> ⚠️ 本章易错细节
> - $P(A) = 0$，不代表 $A$ 为空集！



## 2. 一维随机变量及其分布


### 2.1 分布函数

1. 分布函数的定义：$F(x) = P\{X \le x\}, -\infty < x< \infty$

2. 分布函数的性质（一般出选择题）
   - （1）单调不减性：$F(x)$ 是单调非减函数
   - （2）有界性：$0\le F(x) \le 1$
   - （3）右连续性：对任意 $x_0$ 有 $\lim\limits_{x\to x_0^+} F(x) = F(x_0)$ 即 $F(x_0 + 0) = F(x_0)$

3. 设 $F_1(x), F_2(x)$ 均是分布函数，则
   - （1）当 $a_i \ge 0, a_1 + a_2 = 1$ 时，$a_1 F_1(x) + a_2 F_2(x)$ 仍为分布函数
   - （2）$F_1(x)F_2(x)$ 仍为分布函数（这也是 $X, Y$ 独立时 $\max\{X, Y\}$ 的分布函数）
   - （3）$1 - [1 - F_1(x)][1 - F_2(x)]$ 仍未分布函数（这也是 $X, Y$ 独立时 $\min\{X, Y\}$ 的分布函数）

4. 当 $a_1\ge 0, a_2 \ge 0$ 且 $a_1 + a_2 = 1$ 时，$a_1 f_1(x) + a_2 f_2(x)$ 必为某随机变量的概率密度 

> ✍ 做题小结论：
> - 如果 $X$ 的分布函数 $F(x)$ 是连续函数，则有 $Y=F(X) ～ U(0, 1)$（🇷🇺 套娃，坐标余丙森强化`例 2.14`）
> - 如果遇到 $Y = F(X)$ 这种分布套分布的问题，可以尝试将 $F(X)$ 图像画出来分类讨论


### 2.2 常见离散型随机变量及其分布律

1. 0-1 分布
   - $X～B(1, p)$

2. 二项分布
   - $X～B(n, p)$
   - $P\{X = k\} = C_n^k p^k (1-p)^{n-k}$

3. 泊松分布
   - $X～P(\lambda)$，其中 $\lambda \ge 0$
   - $P\{X = k\} = \dfrac{\lambda^{k}}{k!} e^{-\lambda}$

> 💡 **泊松定理（用于近似计算）**
>
> 设随机变量 $X～B(n, p)$，若 $\lim\limits_{n\to +\infty} np = \lambda$
>
> 则 $\lim\limits_{n\to +\infty} C_n^k p^k (1-p)^{n-k} = \dfrac{\lambda^{k}}{k!} e^{-\lambda} = \dfrac{(np)^{k}}{k!} e^{-(np)}$

> 💡 **关于泊松分布的公式**
>
> 回忆一下，在高数里面我们学过一个泰勒展开式：$e^x = 1+x+\dfrac{x^2}{2!} + \dfrac{x^3}{3!} + ... = \sum\limits_{k=0}^{\infty} \dfrac{x^k}{k!}$
>
> 我们将其中的 $x$ 换成 $\lambda$，就有 $e^{\lambda} = \sum\limits_{k=0}^{\infty} \dfrac{\lambda^k}{k!}$
>
> 那么接下来就有 $\sum\limits_{k=0}^{\infty} \dfrac{\lambda^k}{k!}e^{-\lambda} = e^{-\lambda} \sum\limits_{k=0}^{\infty} \dfrac{\lambda^k}{k!} = e^{-\lambda} \cdot e^{\lambda} = 1$
>
> 诶？！所有离散样本点的概率和为 1，这就是泊松分布的分布律了！
>
> 上面泊松分布的来历可以记一下，也许有的题目会出喵 🐱

4. 几何分布
   - $P\{ X = k\} = (1-p)^{k-1} p$
   - 几何分布具有**无记忆性**，即 $P\{ X > m+n | X > m\} = P\{ X > n\}$

5. 超几何分布
   - $P\{ X = k\} = \dfrac{C_M^k C_{N-M}^{n-k}}{C_N^n}$

### 2.3 常见连续型随机变量及其概率密度

1. 均匀分布
   - $X～U(a, b)$
   - 概率密度：$f(x) = \begin{cases} \dfrac{1}{b-a}, & a<x<b \\ 0,& otherwise \end{cases}$
   - 分布函数：$F(X) = \begin{cases} 0, & x<a \\ \dfrac{x-a}{b-a}, & a\le x < b \\ 1, & x\ge b \end{cases}$

2. 指数分布
   - $X～E(\lambda)$，其中 $\lambda \ge 0$
   - 概率密度：$f(x) = \begin{cases} \lambda e^{-\lambda x}, & x>0 \\ 0, & otherwise\end{cases}$
   - 分布函数：$F(X) = \begin{cases} 0, & x\le 0 \\ 1-e^{-\lambda x}, & x>0\end{cases}$
   - 指数分布也具有**无记忆性**


3. 正态分布
   - $X～N(\mu, \sigma^2)$
   - 概率密度：$f(x) = \dfrac{1}{\sqrt{2\pi} \sigma} e^{- \frac{(x-\mu)^2}{2\sigma^2}}, -\infty < x < +\infty$
   - 分布函数：$F(X) = \dfrac{1}{\sqrt{2\pi}\sigma} \int^{x}_{-\infty} e^{-\frac{(t-\mu)^2}{2\sigma^2}} dt$

> 💡 正态分布的性质：
> - 当 $\mu =0 , \sigma^2=1$ 时，$X～N(0, 1)$ 称为标准正态分布，密度函数为 $\varphi(x)$，分布函数为 $\Phi(x)$
>
> - 对任意 $X～N(\mu, \sigma^2)$，有 $\dfrac{X - \mu}{\sigma}～N(0, 1)$
> - $\Phi(-a) = 1 - \Phi(a)$
> - $aX+b ～N(a\mu + b, a^2\sigma^2)$

> ⚠️ 正态分布常考难点
> - 做题的时候，给定类似于 $f(x) = A e^{x(B - x)}$ 要能看出来是正态分布
> - 计算 $E(e^X) = \int^{+\infty}_{-\infty} e^x \dfrac{1}{\sqrt{2\pi} \sigma} e^{\frac{(x-\mu)^2}{2\sigma^2}} $，思路是将这个积分里的式子转换成另一个正态分布的的概率密度函数（27 张宇 1000a P70 第 9 题）


## 3. 二维随机变量及其分布


### 3.1 二维随机变量及其分布
1. 二维随机变量 $(X, Y)$ 落在矩形区域 $D = \{ (X, Y) | x_1 < X \le x_2, y_1 < Y \le y_2\}$ 上的概率为 $P\{x_1 < X \le x_2, y_1 < Y \le y_2\} = F(x_2, y_2) - F(x_1, y_2) - F(x_2, y_1) + F(x_1, y_1)$

2. 二维随机变量的边缘分布
   - （1）$F_X(x) = F(X, +\infty) = \lim\limits_{y\to +\infty}F(X, Y)$
   - （2）$F_Y(y) = F(+\infty, Y) = \lim\limits_{x\to +\infty}F(X, Y)$

3. 卷积公式（应对特殊分布）：
   - （1）$Z = X + Y$：$f_Z(z) = \int^{+\infty}_{-\infty} f(x, z-x) dx = \int^{+\infty}_{-\infty} f(z-y, y) dy$
   - （2）$Z = X - Y$：$f_Z(z) = \int^{+\infty}_{-\infty} f(x, x-z) dx = \int^{+\infty}_{-\infty} f(z+y, y) dy$

> 💡 当 $X$ 与 $Y$ 相互独立时，有：
> - （1）$Z = X + Y$：$f_Z(z) = \int^{+\infty}_{-\infty} f_X(x)f_Y(z-x) dx = \int^{+\infty}_{-\infty} f_X(z-y)f_Y(y) dy$
> - （2）$Z = X - Y$：$f_Z(z) = \int^{+\infty}_{-\infty} f_X(x)f_Y(x-z) dx = \int^{+\infty}_{-\infty} f_X(z+y)f_Y(y) dy$

4. 二维连续随机变量：$F(X, Y) = \int^{x}_{-\infty} \int^{y}_{-\infty}f(x, y) dxdy$

> 💡 做题的时候经常使用：$F(X, Y) = \int^{x}_{-\infty} \int^{y}_{-\infty}f(u, v) dudv$，这样方便区分 $x, y$

5. 二维连续随机变量的边缘概率密度：
   - （1）$f_X(x) \int^{+\infty}_{-\infty} f(x, y)dy$
   - （2）$f_Y(y) \int^{+\infty}_{-\infty} f(x, y)dx$


### 3.2 二维随机变量的独立性
1. 二维随机变量独立，则有 $F(x, y)  =F_X(x) F_Y(y)$

2. 二维离散型随机变量独立的充要条件：$P\{X=x_i, Y=y_j\} = P\{X=x_i\}P\{Y=y_j\}$
3. 二维连续型随机变量独立的充要条件：$f(x, y) = f_X(x) f_Y(y)$

4. 若 $X$ 与 $Y$ 相互独立，则其函数 $f(X)$ 与 $g(Y)$ 也相互独立

5. 若 $X$ 与 $Y$ 相互独立，则
   - （1）若 $X～P(\lambda_1)$，$Y～P(\lambda_2)$，则 $X+Y ～P(\lambda_1+\lambda_2)$
   - （2）若 $X～B(m, p)$，$Y～B(n, p)$，则 $X+Y ～B(m+n, p)$
   - （3）若 $X～E(\lambda_1)$，$Y～E(\lambda_2)$，则 $\min\{X, Y\} ～E(\lambda_1 + \lambda_2)$

> ⚠️ 可加性结论考试时最好先证明一遍！

### 3.3 二维均匀分布
1. 定义：设 $G$ 为平面上面积为 $A$ 的有界区域，$(X, Y)$ 服从区域 $G$ 上的均匀分布，则有 $f(x, y) = \begin{cases} \dfrac{1}{A}, & (x, y) \in G \\  0, & otherwise \end{cases}$

2. 性质：
   - （1）若 $(X, Y)$ 服从**矩形区域** $G = \{(x, y)| a<x<b, c<y<d\}$ 上的均匀分布，则 $X～U(a, b)$，$Y～U(c, d)$，且 $X$ 与 $Y$ 相互独立，两个条件分布也是均匀分布
   - （2）若 $(X, Y)$ 服从**圆形区域** $G = \{(x, y)|x^2 + y^2 \le r^2\}$ 上的均匀分布，则两个边缘分布都不是均匀分布，且 $X$ 与 $Y$ 不独立，但其两个条件分布都是均匀分布

### 3.4 二维正态分布
1. 定义：
   - $(X, Y)～N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$
   - $f(x ,y) = \dfrac{1}{2\pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}} \exp \{-\dfrac{1}{(1-\rho^2)}[\dfrac{(x-\mu_1)^2}{2\sigma_1^2} - \rho \dfrac{(x-\mu_1)(y-\mu_2)}{\sigma_1 \sigma_2} + \dfrac{(y-\mu_2)^2]}{2\sigma_2^2}] \}$

> 💡 公式里的 $\rho$ 是 $X$ 与 $Y$ 的相关系数（Pearson 相关系数），在后面的章节里面会学到，计算方式为：$ \rho = \dfrac{Cov(X, Y)}{\sigma_1 \sigma_2} = \dfrac{E(XY) - E(X)E(Y)}{\sigma_1 \sigma_2}$

2. 性质：
   - （1）两个边缘分布都是正态分布，即 $X～N(\mu_1, \sigma_1^2)$、$Y～N(\mu_2, \sigma_2^2)$
   - （2）$X$ 与 $Y$ 相互独立 $\Leftrightarrow \rho_{XY} = \rho = 0 \Leftrightarrow X$ 与 $Y$ 不相关
   - （3）$X$ 与 $Y$ 的非零线性组合 $(aX + bY, cX + dY)$ 也遵从二维正态分布
   - （4）$X$ 与 $Y$ 的线性组合 $aX + bY$ 仍为正态分布，即 $aX + bY ～N(a\mu_1 + b\mu_2, a^2\sigma_1^2 + b^2\sigma_2^2 + 2ab\rho\sigma_1\sigma_2)$
   - （5）令 $\begin{cases} U = a_1 X + b_1 Y \\ V = a_2 X + b_2 Y\end{cases}$，当 $\begin{vmatrix}  a_1 & b_1 \\ a_2 & b_2 \end{vmatrix} \ne 0$ 时，$(U, V)$ 服从二维正态分布


### 3.5 连续型随机变量 (X, Y) 的分布函数 F(x, y)
1. 求解方法：设 $F(x, y) = \int^{x}_{-\infty} \int^{y}_{-\infty} f(u, v) dudv$，然后分类讨论，画图求解


## 4. 数字特征
### 4.1 随机变量的数学期望和方差
1. 离散型随机变量的数学期望：$E(X) = \sum\limits_{k=1}^{\infty} x_k p_k$
2. 连续型随机变量的数学期望：$E(X) = \int^{+\infty}_{-\infty} xf(x)dx$
3. 一维随机变量函数的数学期望：
   - 设 $Y = g(X)$
   - （1）离散型：$E(Y) = E(g(X)) = \sum\limits_{k=1}^{\infty} g(x_k) p_k$
   - （2）连续型：$E(Y) = E(g(X)) = \int^{+\infty}_{-\infty} g(x)f(x)dx$

4. 二维随机变量函数的数学期望：
   - 设 $Z = g(X, Y)$
   - （1）离散型：$E(Z) = E(g(X, Y)) = \sum\limits_{i=1}^{+\infty} \sum\limits_{j=1}^{+\infty} g(x_i, y_j) p_{ij}$
   - （2）连续型：$E(Z) = E(g(X, Y)) = \int^{+\infty}_{-\infty} \int^{+\infty}_{-\infty} g(x, y) f(x, y) dx dy$

5. 数学期望的性质：
   - （1）$E(c) = c$
   - （2）$E(cX) = cE(X)$
   - （3）$E(aX + bY) = aE(X) + bE(Y)$
   - （4）若 $X$ 与 $Y$ 相互独立，则 $E(XY) = E(X) E(Y)$

6. 方差的计算：$D(X) = E[X - E(X)]^2 = E(X^2) - [E(X)]^2$

7. 方差的性质：
   - （1）$D(c) = 0$
   - （2）$D(cX) = c^2 D(X)$，$D(aX + b) = a^2 D(X)$
   - （3）$D(X \pm Y) = D(X) + D(Y) \pm 2Cov(X, Y)$
   - （4）若 $X$ 与 $Y$ 是相互独立的随机变量 $\Rightarrow D(X\pm Y) = D(X) + D(Y)$
   - （5）$D(X\pm Y) = D(X) + D(Y) \Leftrightarrow X$ 与 $Y$ 不相关
   - （6）$X$ 与 $Y$ 相互独立，且 $E(X) = E(Y) = 0 \Rightarrow D(XY) = D(X) D(Y)$ 

8. 常见随机变量分布的数学期望与方差

|分布名称|符号|分布列或概率密度|数学期望|方差|
|:---:|:---:|:---:|:---:|:---:|
|0-1 分布|$B(1, p)$|$P\{X=k\}=p^k(1-p)^{1-k}$|$p$|$p(1-p)$|
|二项分布|$B(n, p)$|$P\{X=k\}=C_n^kp^k(1-p)^{1-k}$|$np$|$np(1-p)$|
|泊松分布|$P(\lambda)$|$P\{X=k\}=\dfrac{\lambda^k}{k!} e^{-\lambda}$|$\lambda$|$\lambda$|
|几何分布|$G(p)$|$P\{X=k\}=(1-p)^{k-1}p$|$\dfrac{1}{p}$|$\dfrac{1-p}{p^2}$|
|超几何分布|$H(N, M, n)$| 待续 |$\dfrac{nM}{N}$|$\dfrac{nM}{N}(1-\dfrac{M}{N})(\dfrac{N-n}{N-1})$|
|均匀分布|$U(a, b)$|$f(x) = \dfrac{1}{b-a}$|$\dfrac{a+b}{2}$|$\dfrac{(b-a)^2}{12}$|
|指数分布|$E(\lambda)$|$f(x) = \lambda e^{-\lambda}$|$\dfrac{1}{\lambda}$|$\dfrac{1}{\lambda^2}$|
|正态分布|$N(\mu, \sigma^2)$|$f(x) = \dfrac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$|$\mu$|$\sigma^2$|
|卡方分布|$\chi^2(n)$| 待续 |$n$|$2n$|

9. $\max\{...\}$ 与 $\min\{...\}$
    - （1）$U = \max\{X, Y\} = \dfrac{X + Y + | X - Y |}{2}$
    - （2）$V = \min\{X, Y\} = \dfrac{X + Y - |X - Y|}{2}$
    - 由上面两个式子，有 $U + V = X + Y$，$U - V = |X - Y|$、$UV = XY$
    - 进而，$E(U+V) = E(X+Y)$，$E(U-V) = E(|X-Y|)$，$E(UV) = E(XY)$


### 4.2 协方差与相关系数
1. 协方差的计算：$Cov(X, Y) = E\{[X - E(X)][Y - E(Y)]\} = E(XY) - E(X)E(Y)$

2. 协方差的性质
   - （1）$Cov(X, Y) = Cov(Y, X)$
   - （2）$Cov(X, Y) = D(X)$
   - （3）$Cov(X, c) = 0$
   - （4）$Cov(aX, bY) = abCov(X, Y)$
   - （5）$Cov(X_1+X_2, Y) = Cov(X_1, Y) + Cov(X_2, Y)$
   - （6）如果 $X$ 与 $Y$ 相互独立 $\Rightarrow$ $X$ 与 $Y$ 不相关 $\Leftrightarrow$ $Cov(X, Y) = 0$

3. 相关系数 $\rho_{XY} = \dfrac{Cov(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}$

4. 相关系数的性质：
   - （1）$|\rho_{XY}| = 1$ 的充要条件是：存在常数 $a, b(a\neq 0)$ 使 $P\{Y = aX + b\} = 1$。且 $a>0$ 时 $\rho_{XY} = 1$，$a<0$ 时 $\rho_{XY} = -1$

5. 随机变量不相关：$\rho_{XY} = 0$（或 $Cov(X, Y) = 0$），则 $X$ 与 $Y$ 不相关

6. 不相关与独立
   - （1）用韦恩图来表示的话，`独立` 是被包含在 `不相关` 里的，即独立一定不相关，但不相关不代表独立
   - （2）当 $X$ 和 $Y$ 的联合分布为二维正态分布时，独立等价于相关


### 4.3 随机变量的矩
1. $k$ 阶原点矩：$E(X^k)$
2. $k$ 阶中心矩：$E\{[X - E(X)]^k\}$
3. $k+l$ 阶混合原点矩：$E(X^k Y^l)$
4. $k+l$ 阶混合中心矩：$E\{[X - E(X)]^k [Y - E(Y)]^l\}$

## 5. 大数定律和中心极限定理
### 5.1 大数定律
1. **切比雪夫不等式**：设随机变量 $X$ 的数学期望 $E(X)$ 和方差 $D(X)$ 均存在，则对任意 $\epsilon >0 $，有
   - $P\{|X - E(X)| \ge \epsilon\} \le \dfrac{D(X)}{\epsilon^2}$
   - 或 $P\{|X - E(X)| < \epsilon\} > 1 - \dfrac{D(X)}{\epsilon^2}$

> 💡 Coldrain 是这样记的：$\ge \epsilon \le$

2. **切比雪夫大数定律**：设随机变量 $X_1, X_2, ..., X_n$ 相互独立，且数学期望和方差都存在，且存在常数 $c$，使 $D(X_i)\le c, i=1, 2, ...$，则对任意正数 $\epsilon$，有
   - $\lim\limits_{n\to \infty}P\{|\dfrac{1}{n} \sum\limits_{k=1}^{n} X_k - \dfrac{1}{n} \sum\limits_{k=1}^{n} E(X_k)|<\epsilon \} = 1$
   - 该定理表明：当 $n$ 很大时，$\dfrac{1}{n}\sum\limits_{k=1}^{n} X_k \overset{P}{\longrightarrow} \dfrac{1}{n}\sum\limits_{k=1}^{n}E(X_k)$

> 下面两个大数定律都是**切比雪夫大数定律**的特殊形式

3. **伯努利大数定律**：设 $X_1, X_2, ..., X_n$ 独立且同分布于 0-1 分布 $B(1, p)$，则对任意正数 $\epsilon$，有
   - $\lim\limits_{n\to \infty} P\{|\dfrac{1}{n}\sum\limits_{i=1}^{n}X_i - p|<\epsilon\} = 1$

> 💡 **伯努利大数定律的等价形式**
>
> 设 $n$ 次独立重复事件 A 发生的次数为 $n_A ～B(n, p)$，则对任意正数 $\epsilon$ 有：
> - $\lim\limits_{n\to \infty} P\{|\dfrac{n_A}{n} - p|<\epsilon\} = 1$

4. **辛钦大数定律**：设 $X_1, X_2, ..., X_n$ 独立且同分布，且有相同数学期望 $E(X_i) = \mu$，则对任意正数 $\epsilon$，有
   - $\lim\limits_{n\to \infty} P\{|\dfrac{1}{n}\sum\limits_{i=1}^{n}X_i - \mu|<\epsilon\} = 1$

### 5.2 中心极限定理
1. **列维-林德伯格中心极限定理（独立、同分布、相同期望方差）**：设 $X_1, X_2, ..., X_n$ 独立且同分布，且有相同数学期望 $E(X_i) = \mu$ 和方差 $D(X_i) = \sigma^2$，则对任意实数 $x$，有
   - $\lim\limits_{n\to\infty} P\{\dfrac{\frac{1}{n}\sum_{i=1}^{n}X_i - \mu}{\sigma / \sqrt{n}} \le x\} = \int^{x}_{-\infty} \dfrac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \Phi(x)$
   - 定理表明：当 $n$ 充分大时，$\overline{X} = \dfrac{1}{n}\sum\limits_{i=1}^{n}X_i\overset{近似}{～}N(\mu, \sigma^2)$，其标准化 $\dfrac{\overline{X} - \mu}{\sigma / \sqrt{n}}$ 近似服从标准正态分布 $N(0, 1)$

2. **棣莫佛-拉普拉斯中心极限定理（二项分布的极限是正态）**：设随机变量 $Y_n ～B(n, p)$，则对任意实数 $x$，有
   - $\lim\limits_{n\to\infty} P\{\dfrac{Y_n - np}{\sqrt{np(1-p)}} \le x\} = \int^{x}_{-\infty} \dfrac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt = \Phi(x)$
   - 定理表明：当 $n$ 充分大时，$Y_n\overset{近似}{～}N(np, np(1-p))$，其标准化随机变量 $\dfrac{Y_n - np}{\sqrt{np(1-p)}}$ 近似服从标准正态分布 $N(0, 1)$，即正态分布是二项分布的极限分布



## 6. 数理统计的基本概念

### 6.1 总体与样本

1. **总体**是指与所研究的问题有关的个体的全体所构成的集合，在数理统计中，总体就是一个服从某概率分布的随机变量 $X$，其概率分布称为**总体分布**，其数字特征称为**总体数字特征**

2. 样本的性质：
   - **独立性**：$X_1, X_2, ..., X_n$ 相互独立
   - **代表性**：$X_i$ 与 $X$ 同分布
   - 无变量重叠的连续函数 $U(X_1, X_2, ..., X_k)$ 和 $V(X_{k+1}, X_{k+2}, ..., X_{n})$ 相互独立

### 6.2 统计量

1. **统计量**：样本 $X_1, X_2, ..., X_n$ 的**不含总体任何未知参数**的函数 $g(X_1, X_2, ..., X_n)$

2. **常见统计量**
    - （1）样本均值：$\overline{X} = \dfrac{1}{n} \sum\limits_{i=1}^{n} X_i$
    - （2）样本方差：$S^2 = \dfrac{1}{n-1}\sum\limits^{n}_{i=1}(X_i - \overline{X})^2 = \dfrac{1}{n-1} (\sum\limits_{i=1}^{n} X_i^2 - n \overline{X})$
    - （3）样本标准差 $S = \sqrt{S^2}$
    - （4）样本 $k$ 阶原点矩 $A_k = \dfrac{1}{n} \sum\limits_{i=1}^{n} X_i^k$
    - （5）样本 $k$ 阶中心矩 $B_k = \dfrac{1}{n} \sum\limits_{i=1}^{n} (X_i - \overline{X})^k$
    - （6）顺序统计量：$X_1^* = \min\{X_1, X_2, ..., X_n\}$，$X_2^* = \max\{X_1, X_2, ..., X_n\}$，其分布函数分别为
      - $F_{\min}(x) = P\{\min(X_1, X_2, ..., X_n) \le x\} = 1 - [1-F(x)]^n$
      - $F_{\max}(x) = P\{\max(X_1, X_2, ..., X_n) \le x\} = [F(x)]^n$

> **重要结论**
> - $E(\overline{X}) = E(X) = \mu$
> - $D(\overline{X}) = \dfrac{D(X)}{n} = \dfrac{\sigma^2}{n}$
> - $E(S^2) = D(X) = \sigma^2$

### 6.3 卡方分布

1. 定义：设$(X_1, X_2, ..., X_n)$为来自总体 $X～N(0, 1)$的一个简单随机样本，那么统计量 $\chi^2 = X_1^2 + X_2^2 + ... + X_n^2$ 为服从自由度为 $n$ 的 $\chi^2$ 分布，记作 $\chi^2～\chi^2(n)$

2. $\chi^2$ 分布的性质：
   - （1）设 $X～N(0, 1)$，则 $X^2 ～ \chi^2(1)$，$E(X^2) = 1$，$D(X^2) = 2$
   - （2）设 $\chi^2～\chi^2(n)$，则 $E(\chi^2) = n$，$D(\chi^2) = 2n$
   - （3）设 $\chi_i^2～\chi^2(n_i)$，且 $\chi_1^2$，$\chi_2^2$ 相互独立，则 $\chi_1^2 + \chi_2^2 ～ \chi^2(n_1 + n_2)$


### 6.4 t 分布
1. 定义：设 $X～N(0, 1)$，$Y～\chi^2(n)$，且 $X$ 与 $Y$ 相互独立，则称 $T = \dfrac{X}{\sqrt{Y/n}}$ 为服从自由度为 $n$ 的 $t$ 分布，记作 $T～t(n)$

2. 性质：
   - （1）$t$ 分布概率密度 $f(x)$ 为偶函数，函数图像关于 $y$ 轴对称，则 $t_{1-\alpha}(n) = -t_{\alpha}(n)$
   - （2）当 $n\to \infty$ 时，$t$ 分布 $T\overset{近似}{～}N(0, 1)$
   - （3）$T^2 ～ F(1, n)$

> ⚠️ 易错题：余丙森强化`例 6.5、6.7`

### 6.5 F 分布
1. 定义：设 $X～\chi^2(n_1)$，$Y～\chi^2(n_2)$，且 $X$ 与 $Y$ 相互独立，则称 $F = \dfrac{X/n_1}{Y/n_2}$ 为服从第一自由度为 $n_1$，第二自由度为 $n_2$ 的 $F$ 分布，记作 $F～F(n_1, n_2)$

2. F 分布的性质
   - （1）若 $F～F(n_1, n_2)$，则 $\dfrac{1}{F} ～ F(n_2, n_1)$
   - （2）若 $T～t(n)$，则 $T^2 = \dfrac{X^2}{Y/n} ～ F(1, n)$
   - （3）$F_{1-\alpha}(n_1, n_2) = \dfrac{1}{F_{\alpha}(n_2, n_1)}$

### 6.6 上侧 alpha 分位点

原本是正态分布里面的那个标准正态分布查表法，这里也可以推广到 $\chi^2$、$t$、$F$ 分布中

1. $\chi^2$ 分布的上侧 $\alpha$ 分位点
   - $P\{\chi^2 > \chi^2_{\alpha}(n)\} = \int^{+\infty}_{\chi^2_{\alpha}(n)}f(x)dx = \alpha$

2. $t$ 分布的上侧 $\alpha$ 分位点
   - $P\{T > t_{\alpha}(n)\} = \int^{+\infty}_{t_{\alpha}(n)}f(x)dx = \alpha$

3. $F$ 分布的上侧 $\alpha$ 分位点
   - $P\{F > F_{\alpha}(n_1, n_2)\} = \int^{+\infty}_{F_{\alpha}(n_1, n_2)}f(x)dx = \alpha$

### 6.7 单正态总体下常用统计量的分布
设 $X ～ N(\mu, \sigma^2)$，$(X_1, X_2, ..., X_n)$ 为来自总体 $X$ 的简单随机样本，则

1. 关于 $\overline{X}$
   - （1）$\overline{X} = \dfrac{1}{n}\sum\limits_{i=1}^{n} X_i ～ N(\mu, \dfrac{\sigma^2}{n})$
   - （2）$\dfrac{\overline{X} - \mu}{\frac{\sigma}{\sqrt{n}}}～N(0, 1)$
   - （3）$\overline{X}$ 与 $S^2$ 相互独立，且有 $\dfrac{\overline{X} - \mu}{\frac{S}{\sqrt{n}}}～t(n - 1)$

2. 关于 $S^2$
   - （1）$\overline{X}$ 与 $S^2$ 相互独立，且 $\dfrac{(n-1)S^2}{\sigma^2} = \dfrac{\sum\limits_{i=1}^{n} (X_i - \overline{X})^2}{\sigma^2} ～ \chi^2(n - 1)$
   - （2）$\dfrac{\sum\limits_{i=1}^{n} (X_i - \mu)^2}{\sigma^2} ～ \chi^2(n)$（提示：$\dfrac{X_i - \mu}{\sigma} ～ N(0, 1)$）

> ⚠️ 东西有点多，但一定要熟练掌握！

### 6.8 双正态总体
设 $X_1, X_2, ..., X_{n_1}$ 和 $Y_1, Y_2, ..., Y_{n_2}$ 分别为来自正态分布 $N(\mu_1, \sigma_1^2)$ 和 $N(\mu_2, \sigma_2^2)$ 的简单随机样本，且两组样本相互独立，令两个样本的均值和方差分别为
   - $\overline{X} = \dfrac{1}{n_1}\sum\limits_{i=1}^{n_1}X_i$
   - $S_1^2 = \dfrac{1}{n_1 - 1} \sum\limits_{i=1}^{n_1} (X_i - \overline{X})^2$
   - $\overline{Y} = \dfrac{1}{n_2}\sum\limits_{i=1}^{n_2}Y_i$
   - $S_2^2 = \dfrac{1}{n_2 - 1} \sum\limits_{i=1}^{n_2} (Y_i - \overline{Y})^2$

则有

1. 关于均值
   - $\overline{X} \pm \overline{Y} ～ N(\mu_1 \pm \mu_2, \dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2})$
   - $\dfrac{(\overline{X} \pm \overline{Y}) - (\mu_1 \pm \mu_2) }{\sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}}～N(0, 1)$

2. 关于方差
   - $\dfrac{(n_1 - 1)S_1^2}{\sigma_1^2} + \dfrac{(n_2 - 1)S_2 ^2}{\sigma_2^2} ～\chi^2 (n_1 + n_2 - 2)$
   - $\dfrac{\dfrac{1}{\sigma_1^2} \sum\limits_{i=1}^{n_1}(X_i -\mu_1)^2 / n_1}{\dfrac{1}{\sigma_2^2} \sum\limits_{i=1}^{n_2}(Y_i -\mu_2)^2 / n_2} = \dfrac{ \sum\limits_{i=1}^{n_1}(X_i- \mu_1)^2 / (n_1\sigma_1^2)}{ \sum\limits_{i=1}^{n_2}(Y_i -\mu_2)^2 / (n_2 \sigma_2^2)} ～F(n_1, n_2)$
   - $\dfrac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2}～F(n_1-1, n_2 - 1)$

3. 设 $\sigma_1^2 = \sigma_2^2 = \sigma^2$，则有
   - $T = \dfrac{(\overline{X} - \overline{Y}) - (\mu_1 - \mu_2)}{S_W^2 \sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}}}～t(n_1 + n_2 - 2)$
   - $S_W^2 = \dfrac{(n_1-1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}$

> 💡 这里可以看看 27 余丙森强化`例 6.4`

## 7. 参数估计
### 7.1 参数的点估计、估计量、估计值

1. **估计量** $\hat{\theta}(X_1, X_2, ..., X_n)$ 是一个随机变量
2. **估计值** $\hat{\theta}(x_1, x_2, ..., x_n)$ 为估计量所取的具体值
3. **点估计** 就是用估计量的值估计位置参数的值

### 7.2 矩估计法
1. 用**样本矩**估计相应的总体矩从而得到参数估计的方法称为**矩估计法**

2. 总体矩：
   - $\mu_k = E(X^k)$
   - $\gamma_k = E\{[X - E(X)]^k\}$

> 小结论：
> - $\mu_1 = E(X)$
> - $\mu_2 = E(X^2)$
> - $\gamma_2 = D(X)$

3. 样本矩：
   - $A_k = \dfrac{1}{n} \sum\limits_{i=1}^{n}X_i^k$
   - $B_k = \dfrac{1}{n} \sum\limits_{i=1}^{n} (X_i - \overline{X})^k$

> 小结论：
> - $A_1 = \overline{X}$
> - $A_2 = \dfrac{1}{n}\sum\limits_{i=1}^{n} X_i^2$
> - $B_2 = \dfrac{1}{n}(\sum\limits_{i=1}^{n}X_i^2 - n\overline{X}^2)$

4. 关系：
   - $E(A_k) = \mu_k$
   - 由大数定律，$A_k \overset{P}{\longrightarrow} \mu_k$，$A_1 \overset{P}{\longrightarrow} \mu_1$、$A_2 \overset{P}{\longrightarrow} \mu_2$、$B_k \overset{P}{\longrightarrow} D(X)$


5. 计算方法：设总体 $X$ 的分布函数为 $F(x;\theta_1, ..., \theta_k)$，其中 $\theta_i (i = 1,2,..., k)$ 为待估参数，$X_1, X_2, ..., X_n$ 为总体 $X$ 的一个样本，则求矩估计的步骤为
   - （1）求出总体矩（原点矩或中心矩）$E(X^i)$ 或 $E[X - E(X)]^i$
   - （2）令总体矩等于相应的样本矩，得方程组，即 
     - 原点矩 $\dfrac{1}{n} \sum\limits_{j=1}^{n}X_j^i = E(X^i)$
     - 或样本矩 $\dfrac{1}{n} \sum\limits_{j=1}^{n}(X_j - \overline{X})^i = E[X - E(X)]^i$
   - （3）解上面的方程组，得 $\theta_i$ 的矩估计值为 $\hat{\theta}_i(x_1, x_2, ..., x_n)$，$\theta_i$ 的矩估计量为 $\hat{\theta}_i(X_1, X_2, ..., X_n)$

> ⚠️ 有 $k$ 个未知参数就求到 $k$ 阶原点矩或中心矩，为方便计算，一般取原点矩

### 7.3 最大似然估计法
1. 似然函数：样本 $X_1, X_2, ..., X_n$ 取到观察值 $x_1, x_2, ..., x_n$ 的概率 $L(\theta)$
   - （1）**离散型**：$X$ 分布律为 $P\{X=x_i\} = p(x_i;\theta)$，则似然函数 $L(\theta) = P\{X_1 = x_1, X_2=x_2, ..., X_n=x_n\} = \prod\limits^{n}_{i=1} P\{X_i=x_i\} = \prod\limits^{n}_{i=1} p\{x_i; \theta\}$
   - （2）**连续型**：$X$ 概率密度为 $f(x) = f(x; \theta)$，则似然函数 $L(\theta) = \prod\limits_{i=1}^{n} f(x_i; \theta)$

2. 思想：在 $\theta$ 的取值范围内求 $\hat{\theta}$ 使 $L(\hat{\theta}) = \max L(\theta)$

3. 解题步骤：
   - （1）写出似然函数 $L(\theta) = \begin{cases} \prod\limits^{n}_{i=1} p\{x_i; \theta\} & Discrete \\ \prod\limits_{i=1}^{n} f(x_i; \theta) & Continuous \end{cases}$
   - （2）求似然函数 $L(\theta)$ 的最大值点，若 $L(\theta)$ 或 $\ln L(\theta)$ 可微且易于计算，则可令 $\dfrac{dL(\theta)}{d\theta} = 0$ 或 $\dfrac{ d \ln L(\theta)}{d\theta} = 0$，从而解得 $\theta$（若 $X$ 的分布中包含多个未知量，即 $\theta = (\theta_1, \theta_2, ..., \theta_n)$，则可以分别令偏导数等于 0 解出对应的 $\theta_i$）
   - （3）解出来的 $\theta$ 就是最大似然估计值 $\hat{\theta}_i(x_1, x_2, ..., x_n)$，最大似然估计量为 $\hat{\theta}_i(X_1, X_2, ..., X_n)$

> 💡 若题目给出“样本观察值为 $x_1$、$x_2$、...、$x_n$”，那么
> - 矩估计中样本均值就是这几个样本的平均值
> - 最大似然函数就是把这几个样本观察值代入概率密度函数后相乘

4. 最大似然估计的不变性：
   - 设 $\hat{\theta}$ 是未知参数 $\theta$ 的最大似然估计，对于 $\theta$ 的函数 $g(\theta)$，如果 $g(\theta)$ 具有单值反函数，则 $g(\hat{\theta})$ 为 $g(\theta)$ 的最大似然估计



### 7.4 估计量的评选标准

1. **无偏性**：
   - 设 $\hat{\theta}$ 为 $\theta$ 的估计量，若 $E(\hat{\theta}) = \theta$，则 $\hat{\theta}$ 为 $\theta$ 的**无偏估计量**，否则为**有偏估计量**
   - 若 $\lim\limits_{n \to 0} E(\hat{\theta}) = \theta$，则称 $\hat{\theta}$ 为 $\theta$ 的**渐近无偏估计**

> 💡 常用结论：
> - （1）$\overline{X}$ 是 $E(X) = \mu$ 的无偏估计，即 $E(\overline{X}) = E(X) = \mu$
> - （2）$S^2$ 是 $D(X) = \sigma^2$ 的无偏估计，即 $E(S^2) = D(X) = \sigma^2$
> - （3）设 $\hat{\theta_1}, \hat{\theta_2}, ..., \hat{\theta_n}$ 均为 $\theta$ 的无偏估计，$c_1, c_2, ..., c_n$ 为常数且 $\sum\limits_{i=1}^{n}c_i = 1$，则 $c_1\hat{\theta_1} + c_2\hat{\theta_2} + ... + c_n\hat{\theta_n}$ 仍是 $\theta$ 的无偏估计


2. **有效性**：
   - 设 $\hat{\theta_1}, \hat{\theta_2}$ 均为 $\theta$ 的无偏估计，若 $D(\hat{\theta_1}) < D(\hat{\theta_2})$，则称 $\hat{\theta_1}$ 比 $\hat{\theta_2}$ 更有效

3. **一致性（相合性）**：
   - 若对 $\forall \epsilon > 0$，有 $\lim\limits_{n\to \infty} P\{|\hat{\theta} - \theta| < \epsilon\} = 1$，则称 $\hat{\theta}$ 为 $\theta$ 的一致估计量或相合估计量

> 💡 看到一致性，有没有想到切比雪夫不等式？


### 7.5 区间估计
1. 置信区间：$P\{\hat{\theta_1} < \theta < \hat{\theta_2}\} \ge 1 - \alpha$，则称 $(\hat{\theta_1}, \hat{\theta_2})$ 为未知参数 $\theta$ 的置信水平（置信度）为 $1- \alpha$ 的置信区间。

2. **单正态总体**下参数 $\mu, \sigma^2$ 的置信区间（设总体 $X～N(\mu, \sigma^2)$，求取置信度为 $1-\alpha$）


| 题意 | 枢轴量 | 双侧置信区间 | 单侧置信限 |
|:---: | :---: | :---: | :---: |
|$\sigma^2$ 已知，估 $\mu$|$$Z = \dfrac{\overline{X} - \mu}{\sigma / \sqrt{n}} \sim N\left(0, 1\right)$$|$$\left(\overline{X} - \dfrac{\sigma}{\sqrt{n}}z_{\alpha/2}, \overline{X} + \dfrac{\sigma}{\sqrt{n}}z_{\alpha/2}\right)$$|$$\overline{\mu} = \overline{X} + \dfrac{\sigma}{\sqrt{n}}z_{\green\alpha} \\ \underline{\mu} = \overline{X} - \dfrac{\sigma}{\sqrt{n}}z_{\green\alpha}$$|
|$\sigma^2$ 未知，估 $\mu$|$$T = \dfrac{\overline{X} - \mu}{S / \sqrt{n}} \sim t\left(\red{n - 1}\right)$$|$$\left(\overline{X} - \dfrac{S}{\sqrt{n}}t_{\alpha/2}\left(\red{n - 1}\right), \overline{X} + \dfrac{S}{\sqrt{n}}t_{\alpha/2}\left(\red{n - 1}\right)\right)$$|$$\overline{\mu} = \overline{X} + \dfrac{S}{\sqrt{n}}t_{\green\alpha}\left(\red{n - 1}\right) \\ \underline{\mu} = \overline{X} - \dfrac{S}{\sqrt{n}}t_{\green\alpha}\left(\red{n - 1}\right)$$|
|$\mu$ 已知，估 $\sigma^2$|$$\chi^2 = \dfrac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\sigma^2} \sim \chi^2\left(n\right)$$|$$\left(\dfrac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{\alpha/2}\left(n\right)}, \dfrac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{1-\alpha/2}\left(n\right)}\right)$$|$$\overline{\sigma^2} = \dfrac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{1-\green\alpha}\left(n\right)} \\ \underline{\sigma^2} = \dfrac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{\green\alpha}\left(n\right)}$$|
|$\mu$ 未知，估 $\sigma^2$|$$\chi^2 = \dfrac{\left(n - 1\right) S^2}{\sigma^2} \sim \chi^2\left(n - 1\right)$$|$$\left(\dfrac{\left(n - 1\right) S^2}{\chi^2_{\alpha/2}\left(n - 1\right)}, \dfrac{\left(n - 1\right) S^2}{\chi^2_{1 - \alpha/2}\left(n - 1\right)}\right)$$|$$\overline{\sigma^2} = \dfrac{\left(n - 1\right) S^2}{\chi^2_{1 - \green\alpha}\left(n - 1\right)} \\ \underline{\sigma^2} = \dfrac{\left(n - 1\right) S^2}{\chi^2_{\green\alpha}\left(n - 1\right)}$$|

3. **双正态总体**均值和方差的置信水平为 $1-\alpha$ 的区间估计

| 待估参数 | 其他参数 | 双侧置信区间 |
|:---:|:---:|:---:|
|$\mu_1 - \mu_2$|$\sigma_1^2,\sigma_2^2$ 已知|$\left( \overline{X} - \overline{Y} - u_{\alpha/2}\sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}, \overline{X} - \overline{Y} + u_{\alpha/2}\sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}  \right)$|
|$\mu_1 - \mu_2$|$\sigma_1^2,\sigma_2^2$ 未知但 $\sigma_1^2 = \sigma_2^2$|$\left( \overline{X} - \overline{Y} - t_{\alpha/2}(n_1+n_2-2) S_w \sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}}, \overline{X} - \overline{Y} + t_{\alpha/2}(n_1+n_2-2) S_w \sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}} \right)$|
|$\dfrac{\sigma_1^2}{\sigma_2^2}$|$\mu_1, \mu_2$ 已知|$\left( \dfrac{\frac{1}{n_1} \sum\limits_{i=1}^{n}(X_i - \mu_1) / \frac{1}{n_2} \sum\limits_{i=1}^{n}(Y_i - \mu_2)}{F_{\alpha/2}(n_1, n_2)}, \dfrac{\frac{1}{n_1} \sum\limits_{i=1}^{n}(X_i - \mu_1) / \frac{1}{n_2} \sum\limits_{i=1}^{n}(Y_i - \mu_2)}{F_{1- \alpha/2}(n_1, n_2)} \right)$|
|$\dfrac{\sigma_1^2}{\sigma_2^2}$|$\mu_1, \mu_2$ 未知|$\left(\dfrac{S_1^2}{S_2^2} \cdot \dfrac{1}{F_{\alpha/2}(n_1 - 1, n_2 - 1)}, \dfrac{S_1^2}{S_2^2} \cdot \dfrac{1}{F_{1-\alpha/2}(n_1 - 1, n_2 - 1)}\right)$|

- 其中 $S_w^2 = \dfrac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 -2 }$

> 💡 **置信水平的意义**：
>
> 用同一种抽样方法反复抽样、反复构造置信区间时，其中大约有 $1- \alpha$ 的比例能够包含真实参数
>
> 比如：$1 - \alpha = 95%$
>
> 上面这个置信度的含义是：如果我们反复抽样 100 次，并且每次都按同样的方法算一个置信区间，那么大约有 95 个区间会包含真实参数总体，大约 5 个区间不包含真实参数
>
> 那么，置信水平越高，区间通常越宽；置信水平越低，区间通常越窄


## 8. 假设检验

### 8.1 假设检验的定义与常用概念
1. **假设检验的具体做法：**
   - （1）根据问题的需要对所研究的总体作某种假设，记作 $H_0$
   - （2）选取合适的统计量，这个统计量的选取要使得在假设 $H_0$ 成立时，其分布为已知
   - （3）由实测的样本，计算出统计量的值，并根据预先给定的显著性水平进行检验，作出拒绝或接受假设 $H_0$ 的判断

2. **备择假设**：与原假设 $H_0$ 相对的假设 $H_1$

3. **检验统计量**：用于假设检验问题的统计量称为检验统计量

4. **拒绝域与临界点**
   - 当检验统计量的观测值落在某一个区域时就拒绝 $H_0$，这一区域称为拒绝域
   - 拒绝域的边界称为临界点

5. **假设检验的两类错误**
   - （1）第一类错误（弃真错误）：原假设 $H_0$ 为真时，但检验结果为**拒绝**原假设 $H_0$
   - （2）第二类错误（取伪错误）：原假设 $H_0$ 不真时，但检验结果为**接受**原假设 $H_0$

### 8.2 显著性检验

1. **显著性检验的定义**
   - （1）显著性水平：在假设检验中允许犯第一类错误的概率记为 $\alpha(0<\alpha<1)$，则 $\alpha$ 称为显著性水平，它体现了对弃真错误的控制程度
   - （2）显著性检验：只控制第一类错误概率 $\alpha$ 的统计检验，称为显著性检验

2. **显著性检验的一般步骤（大题套路）**
   - （1）根据实际问题提出原假设 $H_0$
   - （2）如果为单侧检验（$\ge$ 或 $\le$），那么可以将原假设 $H_0$ 或备择假设 $H_1$ 转化到边界处（例如原假设 $H_0:\mu \ge 10$ 可以转化为 $H_0: \mu = 10$）
   - （3）选择合适的检验统计量 $T$ 并写出拒绝域 $W$ 的形式
   - （3）给出显著性水平 $\alpha(0<\alpha<1)$，并依据第一类错误的概率等于 $\alpha$ 求出拒绝域
   - （4）根据题目所给样本值计算检验统计量 $T$ 的观测值，当观测值落在拒绝域内则拒绝原假设 $H_0$，否则接受原假设 $H_0$

> ⚠️ 题目如果说要检验什么，那么原假设 $H_0$ 就应该设为要检验的命题的逆命题

3. **显著性水平为 $\alpha$ 的单正态总体均值和方差的假设检验**

| $H_0 \leftrightarrow H_1$ | $H_0$ 为真时检验统计量及其分布 | $H_0$ 的拒绝域 $W$ |
|:---:|:---:|:---:|
|$\mu = \mu_0 \leftrightarrow \mu \ne \mu_0$|（$\sigma^2$ 已知） $\\$ $U = \dfrac{\overline{X} - \mu_0}{\sigma / \sqrt{n}} ～N(0, 1)$|$\begin{vmatrix} U \end{vmatrix} \ge u_{\frac{\alpha}{2}}$|
|$\mu \le \mu_0 \leftrightarrow \mu > \mu_0$|同上|$U\ge u_{\alpha}$|
|$\mu \ge \mu_0 \leftrightarrow \mu < \mu_0$|同上|$U \le -u_{\alpha}$|
|$\mu = \mu_0 \leftrightarrow \mu \ne \mu_0$|（$\sigma^2$ 未知） $\\$ $T = \dfrac{\overline{X} - \mu_0}{S / \sqrt{n}} ～t(n-1)$|$\begin{vmatrix} T \end{vmatrix} \ge t_{\frac{\alpha}{2}}(n-1)$|
|$\mu \le \mu_0 \leftrightarrow \mu > \mu_0$|同上|$T\ge t_{\alpha}(n-1)$|
|$\mu \ge \mu_0 \leftrightarrow \mu < \mu_0$|同上|$T\le -t_{\alpha}(n-1)$|
|$\sigma^2 = \sigma_0^2 \leftrightarrow \sigma^2 \ne \sigma_0^2$|（$\mu$ 已知）$\\$ $\chi^2 = \dfrac{\sum\limits_{i=1}^{n}(X_i - \mu)^2}{\sigma_0^2} ～\chi^2(n)$|$\chi^2 \ge \chi^2_{\frac{\alpha}{2}}(n)$ 或 $\chi^2 \le \chi^2_{1-\frac{\alpha}{2}}(n)$|
|$\sigma^2 \le \sigma_0^2 \leftrightarrow \sigma^2 > \sigma_0^2$|同上|$\chi^2 \ge \chi^2_{\alpha}(n)$|
|$\sigma^2 \ge \sigma_0^2 \leftrightarrow \sigma^2 < \sigma_0^2$|同上|$\chi^2 \le \chi^2_{1-\alpha}(n)$|
|$\sigma^2 = \sigma_0^2 \leftrightarrow \sigma^2 \ne \sigma_0^2$|（$\mu$ 未知）$\\$ $\chi^2 = \dfrac{(n-1)S^2}{\sigma_0^2}～\chi^2(n-1)$|$\chi^2 \ge \chi^2_{\frac{\alpha}{2}}(n-1)$ 或 $\chi^2 \le \chi^2_{1-\frac{\alpha}{2}}(n-1)$|
|$\sigma^2 \le \sigma_0^2 \leftrightarrow \sigma^2 > \sigma_0^2$|同上|$\chi^2 \ge \chi^2_{\alpha}(n-1)$|
|$\sigma^2 \ge \sigma_0^2 \leftrightarrow \sigma^2 < \sigma_0^2$|同上|$\chi^2 \le \chi^2_{1-\alpha}(n-1)$|


> 🐱 有没有感觉拒绝域就是上面 7.5 置信区间取反喵？


4. **双正态总体**

| $H_0 \leftrightarrow H_1$ | $H_0$ 为真时检验统计量及其分布 | $H_0$ 的拒绝域 $W$ |
|:---:|:---:|:---:|
|$\mu_1 = \mu_2 \leftrightarrow \mu_1 \ne \mu_2$|（$\sigma_1^2, \sigma_2^2$ 均未知，但 $\sigma_1^2 = \sigma_2^2$） $\\$ $T = \dfrac{\overline{X} - \overline{Y}}{S / \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} ～t(n_1 + n_2 -2)$|$\begin{vmatrix} T \end{vmatrix} \ge t_{\frac{\alpha}{2}}(n_1 + n_2 -2)$|
|$\mu_1 \le \mu_2 \leftrightarrow \mu_1 > \mu_2$|同上|$T\ge t_{\alpha}(n_1 + n_2 -2)$|
|$\mu_1 \ge \mu_2 \leftrightarrow \mu_1 < \mu_2$|同上|$T\le -t_{\alpha}(n_1 + n_2 -2)$|
|$\sigma_1^2 = \sigma_2^2 \leftrightarrow \sigma_1^2 \ne \sigma_2^2$|（$\mu_1, \mu_2$ 未知）$\\$ $F = \dfrac{S_1^2}{S_2^2}～F(n_1-1, n_2-1)$|$F \ge F_{\frac{\alpha}{2}}(n_1-1, n_2-1)$ 或 $F \le F_{1-\frac{\alpha}{2}}(n_1-1, n_2-1)$|
|$\sigma_1^2 \le \sigma_2^2 \leftrightarrow \sigma_1^2 > \sigma_2^2$|同上|$F \ge F_{\alpha}(n_1-1, n_2-1)$|
|$\sigma_1^2 \ge \sigma_2^2 \leftrightarrow \sigma_1^2 < \sigma_2^2$|同上|$F \le F_{1-\alpha}(n_1-1, n_2-1)$|


5. **显著性水平的意义**：原假设 $H_0$ 成立，经检验 $H_0$ 被拒绝的概率（可以理解为犯错的概率）

> 💡 来看个典型例题
>
> ![problem81](/images/theory_of_possibility/problem81.png)
> 
> ![answer81](/images/theory_of_possibility/answer81.png)
> 



## 参考文献

[1] 《2026 方浩概率论基础课程讲义》，方浩

[2] 《考研数学概率论与数理统计辅导讲义》2027 版，余丙森，国家开放大学出版社