---
title: Coldrain 的 27 考研数一概率论笔记
date: 2026-05-14 17:30:00
tags: 
    - 考研数学
categories: 
    - 数学
description: |
    Coldrain 的概率论备考笔记（施工中 🚧）
---

> ### 写在前面
>
> Coldrain 在第一轮基础复习的时候并没有留下任何笔记，直到概率论刷题的时候，发现知识点比较零散容易忘记，做题卡住了，遂准备从第二轮基础复习开始好好留下笔记，便于知识点检索与记忆
>
> 至于为什么选择以 blog 的形式记录，因为 Coldrain 平时没有手写笔记的习惯💦
>
> 本笔记参考书目包括：
> - 27 版的**余丙森**概率论（基础与强化在同一本书中）
> - 26 版的**方皓**概率论基础与强化两本书
> - 27 版的**张宇**概率论基础 9 讲
> - 本科概率论留下的笔记（和社团的朋友们共同编辑的）

## 1. 随机事件及其概率

## 2. 一维随机变量及其分布

## 3. 二维随机变量及其分布

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

> 常用结论：
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