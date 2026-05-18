---
title: Coldrain 的 27 考研数一概统学习笔记
date: 2026-05-14 17:30:00
tags: 
    - 考研数学
categories: 
    - 数学
description: |
    Coldrain 的概率论备考笔记，涵盖了基础 + 强化的所有内容，以及一些题解拾遗（施工中 🚧）
---

> ### 写在前面
>
> Coldrain 在第一轮基础复习的时候并没有留下任何笔记，直到概率论刷题的时候，发现知识点比较零散容易忘记，做题卡住了，遂准备从第二轮基础复习开始好好留下笔记，便于知识点检索与记忆
>
> 至于为什么选择以 blog 的形式记录，因为 Coldrain 平时没有手写笔记的习惯，而且 blog 形式的笔记可以在任何设备上随时随地打开💦
>
> 本笔记内容并没有全覆盖，过于基础的公式与结论未在本笔记中记录，故本笔记可用于一轮学习结束之后对重难考点进行查漏补缺，但请不要用于替代考研书籍来进行一轮复习
>
> 本笔记参考书目包括：
> - 27 版的**余丙森**概率论（基础与强化在同一本书中）
> - 26 版的**方皓**概率论基础与强化两本书
> - 27 版的**张宇**概率论基础 9 讲
> - 本科概率论留下的笔记（和社团的朋友们共同整理的）

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
   - $X～P(\lambda)$
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
   - $X～E(\lambda)$
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


## 3. 二维随机变量及其分布


### 3.1 二维随机变量及其分布
1. 二维随机变量 $(X, Y)$ 落在矩形区域 $D = \{ (X, Y) | x_1 < X \le x_2, y_1 < Y \le y_2\}$ 上的概率为 $P\{x_1 < X \le x_2, y_1 < Y \le y_2\} = F(x_2, y_2) - F(x_1, y_2) - F(x_2, y_1) + F(x_1, y_1)$


### 3.2 二维随机变量的独立性
1. 二维随机变量独立，则有 $F(x, y)  =F_X(x) F_Y(y)$

2. 二维离散型随机变量独立的充要条件：$P\{X=x_i, Y=y_j\} = P\{X=x_i\}P\{Y=y_j\}$
3. 二维连续型随机变量独立的充要条件：$f(x, y) = f_X(x) f_Y(y)$

4. 若 $X$ 与 $Y$ 相互独立，则其函数 $f(X)$ 与 $g(Y)$ 也相互独立

### 3.3 二维均匀分布
1. 定义：设 $G$ 为平面上面积为 $A$ 的有界区域，$(X, Y)$ 服从区域 $G$ 上的均匀分布，则有 $f(x, y) = \begin{cases} \dfrac{1}{A}, & (x, y) \in G \\  0, & otherwise \end{cases}$

2. 性质：
   - （1）若 $(X, Y)$ 服从**矩形区域** $G = \{(x, y)| a<x<b, c<y<d\}$ 上的均匀分布，则 $X～U(a, b)$，$Y～U(c, d)$，且 $X$ 与 $Y$ 相互独立，两个条件分布也是均匀分布
   - （2）若 $(X, Y)$ 服从**圆形区域** $G = \{(x, y)|x^2 + y^2 \le r^2\}$ 上的均匀分布，则两个边缘分布都不是均匀分布，且 $X$ 与 $Y$ 不独立，但其两个条件分布都是均匀分布

### 3.4 二维正态分布
1. 定义：
   - $(X, Y)～N(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$
   - $f(x ,y) = \dfrac{1}{2\pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}} \exp \{-\dfrac{1}{2(1-\rho^2)}[\dfrac{(x-\mu_1)^2}{\sigma_1^2} - 2\rho \dfrac{(x-\mu_1)(y-\mu_2)}{\sigma_1 \sigma_2} + \dfrac{(y-\mu_2)^2]}{\sigma_2^2}] \}$

> 💡 公式里的 $\rho$ 是 $X$ 与 $Y$ 的相关系数（Pearson 相关系数），在后面的章节里面会学到，计算方式为：$ \rho = \dfrac{Cov(X, Y)}{\sigma_1 \sigma_2} = \dfrac{E(XY) - E(X)E(Y)}{\sigma_1 \sigma_2}$

2. 性质：
   - （1）两个边缘分布都是正态分布，即 $X～N(\mu_1, \sigma_1^2)$、$Y～N(\mu_2, \sigma_2^2)$
   - （2）$X$ 与 $Y$ 相互独立的充要条件是 $\rho = 0$
   - （3）$X$ 与 $Y$ 的非零线性组合 $(aX + bY, cX + dY)$ 也遵从二维正态分布
   - （4）$X$ 与 $Y$ 的线性组合 $aX + bY$ 仍为正态分布，即 $aX + bY ～N(a\mu_1 + b\mu_2, a^2\sigma_1^2 + b^2\sigma_2^2 + 2ab\rho\sigma_1\sigma_2)$


## 4. 数字特征

## 5. 大数定律和中心极限定理

## 6. 数理统计的基本概念

### 6.1 总体与样本

1. **总体**是指与所研究的问题有关的个体的全体所构成的集合，在数理统计中，总体就是一个服从某概率分布的随机变量 $X$，其概率分布称为**总体分布**，其数字特征称为**总体数字特征**

2. 样本的性质：
   - **独立性**：$X_1, X_2, ..., X_n$ 相互独立
   - **代表性**：$X_i$ 与 $X$ 同分布

### 6.2 统计量

1. 统计量：样本 $X_1, X_2, ..., X_n$ 的**不含总体任何未知参数**的函数 $g(X_1, X_2, ..., X_n)$

2. 常见统计量
    - （1）样本均值：$\overline{X} = \dfrac{1}{n} \sum\limits_{i=1}^{n} X_i$
    - （2）样本方差：$S^2 = \dfrac{1}{n-1}\sum\limits^{n}_{i=1}(X_i - \overline{X})^2 = \dfrac{1}{n-1} (\sum\limits_{i=1}^{n} X_i^2 - n \overline{X})$
    - （3）样本标准差 $S = \sqrt{S^2}$
    - （4）样本 $k$ 阶原点矩 $A_k = \dfrac{1}{n} \sum\limits_{i=1}^{n} X_i^k$
    - （5）样本 $k$ 阶中心矩 $B_k = \dfrac{1}{n} \sum\limits_{i=1}^{n} (X_i - \overline{X})^k$
    - （6）顺序统计量：$X_1^* = \min\{X_1, X_2, ..., X_n\}$，$X_2^* = \max\{X_1, X_2, ..., X_n\}$

> 重要结论：
> - $E(\overline{X}) = E(X)$
> - $D(\overline{X}) = \dfrac{D(X)}{n}$
> - $E(S^2) = D(X)$

### 6.3 卡方分布

1. 定义：设$(X_1, X_2, ..., X_n)$为来自总体 $X～N(0, 1)$的一个简单随机样本，那么统计量 $\chi^2 = X_1^2 + X_2^2 + ... + X_n^2$ 为服从自由度为 $n$ 的 $\chi^2$ 分布，记作 $\chi^2～\chi^2(n)$

2. $\chi^2$ 分布的性质：
   - （1）设 $X～N(0, 1)$，则 $X^2 ～ \chi^2(1)$，$E(X^2) = 1$，$D(X^2) = 2$
   - （2）设 $\chi^2～\chi^2(n)$，则 $E(\chi^2) = n$，$D(\chi^2) = 2n$
   - （3）设 $\chi_i^2～\chi^2(n_i)$，且 $\chi_1^2$，$\chi_2^2$ 相互独立，则 $\chi_1^2 + \chi_2^2 ～ \chi^2(n_1 + n_2)$


### 6.4 t 分布
1. 定义：设 $X～N(0, 1)$，$Y～\chi^2(n)$，且 $X$ 与 $Y$ 相互独立，则称 $T = \dfrac{X}{\sqrt{Y/n}}$ 为服从自由度为 $n$ 的 $t$ 分布，记作 $T～t(n)$

### 6.5 F 分布
1. 定义：设 $X～\chi^2(n_1)$，$Y～\chi^2(n_2)$，且 $X$ 与 $Y$ 相互独立，则称 $F = \dfrac{X/n_1}{Y/n_2}$ 为服从第一自由度为 $n_1$，第二自由度为 $n_2$ 的 $F$ 分布，记作 $F～F(n_1, n_2)$

2. F 分布的性质
   - （1）若 $F～F(n_1, n_2)$，则 $\dfrac{1}{F} ～ F(n_2, n_1)$
   - （2）若 $T～t(n)$，则 $T^2 = \dfrac{X^2}{Y/n} ～ F(1, n)$

### 6.6 上侧 alpha 分位点

原本是正态分布里面的那个标准正态分布查表法，这里也可以推广到 $\chi^2$、$t$、$F$ 分布中，此处暂时没什么好写的，可以去看一下参考书上的图

### 6.7 单正态总体下常用统计量的分布
1. 设 $X ～ N(\mu, \sigma^2)$，$(X_1, X_2, ..., X_n)$ 为来自总体 $X$ 的简单随机样本，则：
   - （1）$\overline{X} = \dfrac{1}{n}\sum\limits_{i=1}^{n} X_i ～ N(\mu, \dfrac{\sigma^2}{n})$
   - （2）$U = \dfrac{\overline{X} - \mu}{\sigma/\sqrt{n}} ～ N(0, 1)$
   - （3）$\overline{X}$ 与 $S^2$ 相互独立，且 $\dfrac{(n-1)S^2}{\sigma^2} = \dfrac{\sum\limits_{i=1}^{n} (X_i - \overline{X})^2}{\sigma^2} ～ \chi^2(n - 1)$
   - （4）$\dfrac{\sum\limits_{i=1}^{n} (X_i - \mu)^2}{\sigma^2} ～ \chi^2(n)$（提示：$\dfrac{X_i - \mu}{\sigma} ～ N(0, 1)$）
   - （5）$T = \dfrac{\overline{X} - \mu}{S / \sqrt{n}} ～ t(n-1)$

### 6.8 双正态总体
（这个真的会考吗...💦）


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


5. 计算方法：按照低阶矩优先原理，建立方程（组），从中解出未知参数
   - （1）当 $k=1$ 时，建立方程：若 $E(X)$ 含 $\theta$，令 $\overline{X} = E(X)$，解出 $\hat{\theta}$
   - （2）当 $k=2$ 时，最常用的两个方程为 $\begin{cases} \overline{X} = E(X) \\ \dfrac{1}{n}\sum\limits_{i=1}^{n}X_i^2 = E(X^2) \end{cases}$ 或 $\begin{cases} \overline{X} = E(X) \\ \dfrac{1}{n}(\sum\limits_{i=1}^{n}X_i^2 - n\overline{X}^2) = \dfrac{1}{n} \sum\limits_{i=1}^{n}(X_i - \overline{X})^2 = D(X) \end{cases}$

### 7.3 最大似然估计法
1. 似然函数：样本 $X_1, X_2, ..., X_n$ 取到观察值 $x_1, x_2, ..., x_n$ 的概率 $L(\theta)$
   - （1）**离散型**：$X$ 分布律为 $P\{X=x\} = p(x;\theta)$，则似然函数 $L(\theta) = P\{X_1 = x_1, X_2=x_2, ..., X_n=x_n\} = \prod\limits^{n}_{i=1} P\{X_i=x_i\} = \prod\limits^{n}_{i=1} p\{x_i; \theta\}$
   - （2）**连续型**：$X$ 概率密度为 $f(x) = f(x; \theta)$，则似然函数 $L(\theta) = \prod\limits_{i=1}^{n} f(x_i; \theta)$

2. 思想：在 $\theta$ 的取值范围内求 $\hat{\theta}$ 使 $L(\hat{\theta}) = \max L(\theta)$

3. 解题步骤：
   - （1）写出似然函数 $L(\theta)$，取对数 $\ln{L(\theta)}$
   - （2）对 $\theta$ 求导，令导函数为 0，计算得到驻点
   - （3）再针对具体情况分析

4. 最大似然估计的不变性：
   - 设 $\hat{\theta}$ 是未知参数 $\theta$ 的最大似然估计，对于 $\theta$ 的函数 $g(\theta)$，如果 $g(\theta)$ 具有单值反函数，则 $g(\hat{\theta})$ 为 $g(\theta)$ 的最大似然估计

### 7.4 估计量的评选标准

1. 无偏性：
   - 设 $\hat{\theta}$ 为 $\theta$ 的估计量，若 $E(\hat{\theta}) = \theta$，则 $\hat{\theta}$ 为**无偏估计量**，否则为**有偏估计量**
   - 若 $\lim\limits_{n \to 0} E(\hat{\theta}) = \theta$，则称 $\hat{\theta}$ 为 $\theta$ 的**渐近无偏估计**

> 💡 常用结论：
> - （1）$\overline{X}$ 是 $E(X) = \mu$ 的无偏估计，即 $E(\overline{X}) = E(X) = \mu$
> - （2）$S^2$ 是 $D(X) = \sigma^2$ 的无偏估计，即 $E(S^2) = D(X) = \sigma^2$
> - （3）设 $\hat{\theta_1}, \hat{\theta_2}, ..., \hat{\theta_n}$ 均为 $\theta$ 的无偏估计，$c_1, c_2, ..., c_n$ 为常数且 $\sum\limits_{i=1}^{n}c_i = 1$，则 $c_1\hat{\theta_1} + c_2\hat{\theta_2} + ... + c_n\hat{\theta_n}$ 仍是 $\theta$ 的无偏估计


2. 有效性：
   - 设 $\hat{\theta_1}, \hat{\theta_2}$ 均为 $\theta$ 的无偏估计，若 $D(\hat{\theta_1}) < D(\hat{\theta_2})$，则称 $\hat{\theta_1}$ 比 $\hat{\theta_2}$ 更有效

3. 一致性（相合性）：
   - 若对 $\forall \epsilon > 0$，有 $\lim\limits_{n\to \infty} P\{|\hat{\theta} - \theta| < \epsilon\} = 1$，则称 $\hat{\theta}$ 为 $\theta$ 的一致估计量或相合估计量

4. 置信区间：$P\{\hat{\theta_1} < \theta < \hat{\theta_2}\} = 1 - \alpha$，则称 $(\hat{\theta_1}, \hat{\theta_2})$ 为未知参数 $\theta$ 的置信度为 $1- \alpha$ 的置信区间。

5. 正态总体下参数 $\mu, \sigma^2$ 的置信区间（设总体 $X～N(\mu, \sigma^2)$，求取置信度为 $1-\alpha$）


| 题意 | 枢轴量 | 双侧置信区间 | 单侧置信限 |
|:---: | :---: | :---: | :---: |
|$\sigma^2$ 已知，估$\mu$|$$Z = \frac{\overline{X} - \mu}{\sigma / \sqrt{n}} \sim N\left(0, 1\right)$$|$$\left(\overline{X} - \frac{\sigma}{\sqrt{n}}z_{\alpha/2}, \overline{X} + \frac{\sigma}{\sqrt{n}}z_{\alpha/2}\right)$$|$$\overline{\mu} = \overline{X} + \frac{\sigma}{\sqrt{n}}z_{\green\alpha} \\ \underline{\mu} = \overline{X} - \frac{\sigma}{\sqrt{n}}z_{\green\alpha}$$|
|$\sigma^2$ 未知，估$\mu$|$$T = \frac{\overline{X} - \mu}{S / \sqrt{n}} \sim t\left(\red{n - 1}\right)$$|$$\left(\overline{X} - \frac{S}{\sqrt{n}}t_{\alpha/2}\left(\red{n - 1}\right), \overline{X} + \frac{S}{\sqrt{n}}t_{\alpha/2}\left(\red{n - 1}\right)\right)$$|$$\overline{\mu} = \overline{X} + \frac{S}{\sqrt{n}}t_{\green\alpha}\left(\red{n - 1}\right) \\ \underline{\mu} = \overline{X} - \frac{S}{\sqrt{n}}t_{\green\alpha}\left(\red{n - 1}\right)$$|
|$\mu$ 已知，估$\sigma^2$|$$\chi^2 = \frac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\sigma^2} \sim \chi^2\left(n\right)$$|$$\left(\frac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{\alpha/2}\left(n\right)}, \frac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{1-\alpha/2}\left(n\right)}\right)$$|$$\overline{\sigma^2} = \frac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{1-\green\alpha}\left(n\right)} \\ \underline{\sigma^2} = \frac{\sum\limits^n_{i=1}\left(X_i - \mu\right)^2}{\chi^2_{\green\alpha}\left(n\right)}$$|
|$\mu$ 未知，估$\sigma^2$|$$\chi^2 = \frac{\left(n - 1\right) S^2}{\sigma^2} \sim \chi^2\left(n - 1\right)$$|$$\left(\frac{\left(n - 1\right) S^2}{\chi^2_{\alpha/2}\left(n - 1\right)}, \frac{\left(n - 1\right) S^2}{\chi^2_{1 - \alpha/2}\left(n - 1\right)}\right)$$|$$\overline{\sigma^2} = \frac{\left(n - 1\right) S^2}{\chi^2_{1 - \green\alpha}\left(n - 1\right)} \\ \underline{\sigma^2} = \frac{\left(n - 1\right) S^2}{\chi^2_{\green\alpha}\left(n - 1\right)}$$|



## 8. 假设检验