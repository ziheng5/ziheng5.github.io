---
title: Coldrain 的 27 考研数一线代强化阶段拾遗
date: 2026-05-24 16:11:00
tags: 
    - 考研数学
categories: 
    - 考研数学
description: |
    Coldrain 的线性代数备考笔记，涵盖了基础 + 强化的重要内容，以及一些题解拾遗（施工中 🚧）
---

> ✍ 写在前面
>
> 本笔记为 Coldrain 二刷基础时所记，故笔记内容并没有做到全覆盖，而只针对每一章节重要且容易遗忘的知识点，所以本笔记可用于一轮学习结束之后对重难考点进行查漏补缺，但请不要用于替代考研书籍来进行一轮复习
>
> “岂不闻天无绝人之路，只要我想走，路就在脚下。”—— 25 奥本海豚


## 1. 行列式

1. **n 阶行列式**
   - $\begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix} = \sum\limits_{j_1 j_2 \cdots j_n} (-1)^{\tau(j_1 j_2 ... j_n)} a_{1j_1} a_{2j_2} a_{nj_n}$ 
   - 逆序数：一个排列中，如果一个大的数排在小的数之前，就称这两个数构成一个**逆序**，而一个排列中逆序的总数称为这个排列的逆序数，记作 $\tau(j_1 j_2 ... j_n)$

2. **行列式的性质**
   - （1）$|A^T| = |A|$
   - （2）两行（或两列）互换，行列式变号
   - （3）两行（或两列）相同或对应成比例，行列式的值为 0
   - （4）用数 $k$ 乘行列式 $|A|$ 等于用 $k$ 乘它的某行或某列
   - （5）如果行列式某行（某列）是两个元素之和，则可以把行列式拆成两个行列式之和
   - （6）把某行（某列）的 $k$ 倍加到另一行（或列），行列式值不变

> 💡 注意区分 $k|A|$ 和 $kA$：$k \begin{pmatrix} a & b \\ c & d \end{pmatrix} = \begin{pmatrix} ka & kb \\ kc & kd \end{pmatrix}$

3. **行列式展开公式**
   - （1）余子式：在 $n$ 阶行列式 $\begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix}$ 中划去 $a_{ij}$ 所在的第 $i$ 行和第 $j$ 列的元素，由剩下的元素构成的一个 $n-1$ 阶行列式称为 $a_{ij}$ 的余子式，记为 $M_{ij}$
   - （2）代数余子式：$A_{ij} = (-1)^{i+j} M_{ij}$
   - （3）行列式按行展开：$|A| = a_{i1}A_{i1} + a_{i2}A_{i2} + \cdots + a_{in}A_{in} = \sum\limits_{k=1}^n a_{ik}A_{ik}$
   - （4）行列式按列展开：$|A| = a_{1j}A_{1j} + a_{2j}A_{2j} + \cdots + a_{nj}A_{nj} = \sum\limits_{k=1}^n a_{kj}A_{kj}$
   - （5）行列式任一行（列）元素与另一行（列）元素的代数余子式乘积之和为 0，即 $\sum\limits_{k=1}^n a_{ik}A_{jk} = a_{i1}A_{j1} + a_{i2}A_{j2} + \cdots + a_{in}A_{jn} = 0, i\ne j$（$\sum\limits_{k=1}^n a_{ki}A_{kj} = a_{1i}A_{1j} + a_{2i}A_{2j} + \cdots + a_{ni}A_{nj} = 0, i\ne j$）

4. **特殊行列式**
   - （1）上（下）三角形行列式的值等于主对角线元素的乘积：$\begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ 0 & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_{nn} \end{vmatrix} = \begin{vmatrix} a_{11} & 0 & \cdots & 0 \\ a_{21} & a_{22} & \cdots & 0 \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix} = a_{11} a_{22} \cdots a_{nn}$
   - （2）关于副对角线的行列式：$\begin{vmatrix} a_{11} & \cdots & a_{1, n-1} & a_{1,n} \\ a_{21} & \cdots & a_{2, n-1} & a_{2n} \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & 0 & 0 \end{vmatrix} =        \begin{vmatrix} 0 & \cdots & 0 & a_{1,n} \\ 0 & \cdots & a_{2, n-1} & a_{2n} \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & a_{n, n-1} & a_{n, n} \end{vmatrix} = \begin{vmatrix} 0 & \cdots & 0 & a_{1,n} \\ 0 & \cdots & a_{2, n-1} & 0 \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & 0 & 0 \end{vmatrix} = (-1)^{\frac{n(n-1)}{2}} a_{1n}a_{2,n-1}\cdots a_{n,1}$
   - （3）拉普拉斯展开式：设 $A$ 为 $m$ 阶矩阵，$B$ 为 $n$ 阶矩阵，则 $\begin{vmatrix} A & O \\ O & B \end{vmatrix} = \begin{vmatrix} A & O \\ C & B \end{vmatrix} = \begin{vmatrix} A & C \\ O & B \end{vmatrix} = |A||B|$、$\begin{vmatrix} O & A \\ B & O \end{vmatrix} = \begin{vmatrix} O & A \\ B & C \end{vmatrix} = \begin{vmatrix} C & A \\ B & O \end{vmatrix} = (-1)^{mn}|A||B|$
   - （4）范德蒙德行列式：$\begin{vmatrix} 1 & 1 & \cdots & 1 \\ x_1 & x_2 & \cdots & x_n \\ x_1^2 & x_2^2 & \cdots & x_n^2 \\ \cdots & \cdots & \cdots & \cdots \\ x_1^{n-1} & x_2^{n-1} & \cdots & x_n^{n-1} \end{vmatrix} = \prod\limits_{1\le i < j \le n} (x_j - x_i)$

5. **余子式与代数余子式的线性组合运算**
   - $k_1 A_{i1} + k_2 A_{i2} + \cdots + k_n A_{in} = \begin{vmatrix} \vdots & \vdots & \vdots & \vdots \\ k_1 & k_2 & \cdots & k_n \\ \vdots & \vdots & \vdots & \vdots \end{vmatrix}$

6. **克莱姆法则**
   - （1）对 $n$ 个方程 $n$ 个未知数的非齐次线性方程组 $\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1  \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \cdots \\ a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n = b_n \end{cases}$，若系数行列式 $D = \begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix} \ne 0$，则方程组有唯一解，且解为 $x_i = \dfrac{D_i}{D}$，其中 $D_i$ 是由常数项 $b_1, b_2, \cdots, b_n$ 替换掉 $D$ 中的第 $i$ 列元素得到的行列式（反之，$D = 0$ 时方程组有无穷多的解）
   - （2）对 $n$ 个方程 $n$ 个未知数的齐次线性方程组 $\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = 0  \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = 0 \\ \cdots \\ a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n = 0 \end{cases}$，若 $D \ne 0$ 则齐次方程组只有 0 解；若 $D = 0$ 则齐次方程组有非零解


## 2. 矩阵


## 3. 向量组

## 4. 线性方程组

## 5. 特征值与特征向量

## 6. 二次型