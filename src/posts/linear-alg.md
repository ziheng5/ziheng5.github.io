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

> 💡 本章重点习题
> - 【880-10-基础-选择-3】
> - 【880-10-基础-填空-8】
> - 【880-10-基础-解答-2】
> - 【1000b-1-2、7】

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

> 💡 行列式与矩阵的区别
> - 注意区分 $k|A|$ 和 $kA$：$k \begin{pmatrix} a & b \\ c & d \end{pmatrix} = \begin{pmatrix} ka & kb \\ kc & kd \end{pmatrix}$（$|kA| = k^n |A|$）
> - 注意区分 $A + B$ 和 $|A| + |B|$：$\begin{pmatrix} a_1 & b_1 \\ c_1 & d_1 \end{pmatrix} + \begin{pmatrix} a_2 & b_2 \\ c_2 & d_2 \end{pmatrix} = \begin{pmatrix} a_1 + a_2 & b_1 + b_2 \\ c_1+c_2 & d_1+d_2 \end{pmatrix}$，而 $\begin{vmatrix} a_1 & b_1 \\ c_1 & d_1 \end{vmatrix} + \begin{vmatrix} a_2 & b_2 \\ c_1 & d_1 \end{vmatrix} = \begin{vmatrix} a_1 + a_2 & b_1 + b_2 \\ c_1 & d_1 \end{vmatrix}$

3. **行列式展开公式**
   - （1）余子式：在 $n$ 阶行列式 $\begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix}$ 中划去 $a_{ij}$ 所在的第 $i$ 行和第 $j$ 列的元素，由剩下的元素构成的一个 $n-1$ 阶行列式称为 $a_{ij}$ 的余子式，记为 $M_{ij}$
   - （2）代数余子式：$A_{ij} = (-1)^{i+j} M_{ij}$
   - （3）行列式按行展开：$|A| = a_{i1}A_{i1} + a_{i2}A_{i2} + \cdots + a_{in}A_{in} = \sum\limits_{k=1}^n a_{ik}A_{ik}$
   - （4）行列式按列展开：$|A| = a_{1j}A_{1j} + a_{2j}A_{2j} + \cdots + a_{nj}A_{nj} = \sum\limits_{k=1}^n a_{kj}A_{kj}$
   - （5）行列式任一行（列）元素与另一行（列）元素的代数余子式乘积之和为 0，即 $\sum\limits_{k=1}^n a_{ik}A_{jk} = a_{i1}A_{j1} + a_{i2}A_{j2} + \cdots + a_{in}A_{jn} = 0, i\ne j$（$\sum\limits_{k=1}^n a_{ki}A_{kj} = a_{1i}A_{1j} + a_{2i}A_{2j} + \cdots + a_{ni}A_{nj} = 0, i\ne j$）

> 💡 学会展开之后，做题时有一种比较巧妙的操作：加边
> - （1）设 $x_1 x_2 x_3 x_4 \ne 0$，则 $\begin{vmatrix} x_1 + a_1^2 & a_1a_2 & a_1a_3 & a_1a_4 \\ a_2a_1 & x_2 + a_2^2 & a_2a_3 & a_2a_4 \\ a_3a_1 & a_3a_2 & x_3 + a_3^2 & a_3a_4 \\ a_4a_1 & a_4a_2 & a_4a_3 & x_4 + a_4^2 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 & a_4 \\ 0 & x_1 + a_1^2 & a_1a_2 & a_1a_3 & a_1a_4 \\ 0 & a_2a_1 & x_2 + a_2^2 & a_2a_3 & a_2a_4 \\ 0 & a_3a_1 & a_3a_2 & x_3 + a_3^2 & a_3a_4 \\ 0 & a_4a_1 & a_4a_2 & a_4a_3 & x_4 + a_4^2 \end{vmatrix} = \begin{vmatrix} 1 & a_1 & a_2 & a_3 & a_4 \\ -a_1 & x_1 & 0 & 0 & 0 \\ -a_2 & 0 & x_2 & 0 & 0 \\ -a_3 & 0 & 0 & x_3 & 0 \\ -a_4 & 0 & 0 & 0 & x_4 \end{vmatrix} = \begin{vmatrix} 1 + \sum\limits_{i=1}^{4} \frac{a_i^2}{x_i^2} & 0 & 0 & 0 & 0 \\ -a_1 & x_1 & 0 & 0 & 0 \\ -a_2 & 0 & x_2 & 0 & 0 \\ -a_3 & 0 & 0 & x_3 & 0 \\ -a_4 & 0 & 0 & 0 & x_4 \end{vmatrix} = x_1 x_2 x_3 x_4(1 + \sum\limits_{i=1}^{4}\frac{a_i^2}{x_i^2})$

> 💡 余子式与代数余子式的计算问题一般有三种方法
> - （1）余子式的定义（计算 1～2 个余子式时适用）
> - （2）展开定理（计算一行或一列余子式适用）：$k_1 A_{i1} + k_2 A_{i2} + \cdots + k_n A_{in} = \begin{vmatrix} \vdots & \vdots & \vdots & \vdots \\ k_1 & k_2 & \cdots & k_n \\ \vdots & \vdots & \vdots & \vdots \end{vmatrix}$
> - （3）伴随矩阵（题目需要计算所有代数余子式时适用）


4. **特殊行列式**
   - （1）上（下）三角形行列式的值等于主对角线元素的乘积：$\begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ 0 & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & a_{nn} \end{vmatrix} = \begin{vmatrix} a_{11} & 0 & \cdots & 0 \\ a_{21} & a_{22} & \cdots & 0 \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix} = a_{11} a_{22} \cdots a_{nn}$
   - （2）关于副对角线的行列式：$\begin{vmatrix} a_{11} & \cdots & a_{1, n-1} & a_{1,n} \\ a_{21} & \cdots & a_{2, n-1} & a_{2n} \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & 0 & 0 \end{vmatrix} =        \begin{vmatrix} 0 & \cdots & 0 & a_{1,n} \\ 0 & \cdots & a_{2, n-1} & a_{2n} \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & a_{n, n-1} & a_{n, n} \end{vmatrix} = \begin{vmatrix} 0 & \cdots & 0 & a_{1,n} \\ 0 & \cdots & a_{2, n-1} & 0 \\ \vdots &  & \vdots & \vdots \\ a_{n1} & \cdots & 0 & 0 \end{vmatrix} = (-1)^{\textcolor{red}{\frac{n(n-1)}{2}}} a_{1n}a_{2,n-1}\cdots a_{n,1}$
   - （3）$n$ 阶 $ab$ 型行列式：$\begin{vmatrix} a & b & \cdots & b \\ b & a & \cdots & b \\ \vdots & \vdots & \ddots & \vdots \\ b & b & \cdots & a \end{vmatrix} = [a + (n - 1)b](a - b)^{n-1}$
   - （4）拉普拉斯展开式：设 $A$ 为 $m$ 阶矩阵，$B$ 为 $n$ 阶矩阵，则 $\begin{vmatrix} A & O \\ O & B \end{vmatrix} = \begin{vmatrix} A & O \\ C & B \end{vmatrix} = \begin{vmatrix} A & C \\ O & B \end{vmatrix} = |A||B|$、$\begin{vmatrix} O & A \\ B & O \end{vmatrix} = \begin{vmatrix} O & A \\ B & C \end{vmatrix} = \begin{vmatrix} C & A \\ B & O \end{vmatrix} = (-1)^{\textcolor{red}{mn}}|A||B|$
   - （5）范德蒙德行列式：$\begin{vmatrix} 1 & 1 & \cdots & 1 \\ x_1 & x_2 & \cdots & x_n \\ x_1^2 & x_2^2 & \cdots & x_n^2 \\ \cdots & \cdots & \cdots & \cdots \\ x_1^{n-1} & x_2^{n-1} & \cdots & x_{\textcolor{red}{n}}^{\textcolor{red}{n-1}} \end{vmatrix} = \prod\limits_{1\le i < j \le n} (x_j - x_i)$
   - ⭐⭐⭐（6）三对角线行列式：$D_n$ 按第一行展开得到 $aD_{n-1} + bD_{n-2}$，然后向后递推（可能会用到二阶差分方程：[传送门](https://coldrain.top/2026/06/24/mathematic_2/#16-%E6%97%A0%E7%A9%B7%E7%BA%A7%E6%95%B0)）

> 💡 $\begin{vmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \\ 4 & 0 & 0 & 0 \end{vmatrix}$ 也是拉普拉斯的经典形式！


> 💡 2 道关于范德蒙行列式的难题：
> - （1）$\begin{vmatrix} a & a^2 & bc \\ b & b^2 & ac \\ c & c^2 & ab \end{vmatrix} = \begin{vmatrix} a & a^2 & a^2 + ab + bc + ac \\ b & b^2 & b^2 + ab + bc + ac \\ c & c^2 & c^2 + ab + bc + ac \end{vmatrix} = \begin{vmatrix} a & a^2 & ab + bc + ac \\ b & b^2 & ab + bc + ac \\ c & c^2 & ab + bc + ac \end{vmatrix} = (ab + bc + ac)\begin{vmatrix} 1 & a & a^2\\ 1 & b & b^2 \\ 1 & c & c^2 \end{vmatrix}$
> - （2）计算 $D_n = \begin{vmatrix} 1 & 1 & \cdots & 1 \\ x_1 & x_2 & \cdots & x_n \\ x_1^2 & x_2^2 & \cdots & x_n^2 \\ \vdots & \vdots & \ddots & \vdots \\ x_1^{n-2} & x_2^{n-2} & \cdots & x_n^{n-2} \\ x_1^{n} & x_2^{n} & \cdots & x_n^{n} \end{vmatrix} \Rightarrow D_{n+1} \begin{vmatrix} 1 & 1 & \cdots & 1 & 1 \\ x_1 & x_2 & \cdots & x_n & y \\ x_1^2 & x_2^2 & \cdots & x_n^2 & y^2 \\ \vdots & \vdots & \ddots & \vdots &\vdots \\ x_1^{n-2} & x_2^{n-2} & \cdots & x_n^{n-2} & y^{n-2} \\ x_1^{n-1} & x_2^{n-1} & \cdots & x_n^{n-1} & \textcolor{red}{y^{n-1}} \\ x_1^{n} & x_2^{n} & \cdots & x_n^{n} & y^n \end{vmatrix} = (y-x_1)(y-x_2)\cdots (y-x_n) \prod\limits_{1\le i < j \le n}(x_j - x_i)$；$D_{n+1}$ 中 $y^{n-1}$ 的系数为 $-(x_1 + x_2 + \cdots + x_n)\prod\limits_{1\le i < j \le n} (x_j - x_i)$；$D_{n+1}$ 按 $n+1$ 列展开得 $D_{n+1} = A_{1,n+1} + \cdots + y^{n-1} A_{n, n+1} + y^nA_{n+1, n+1} \Rightarrow A_{n, n+1} = -(x_1 + x_2 + \cdots + x_n)\prod\limits_{1\le i < j \le n} (x_j - x_i) \Rightarrow D_n = M_{n, n+1} = -A_{n, n+1} = (x_1 + x_2 + \cdots + x_n)\prod\limits_{1\le i < j \le n} (x_j - x_i)$ 
> - 【（2）的补充】（3）$\begin{vmatrix} 1 & 1 & 1 & 1 \\ a & b & c & d \\ a^2 & b^2 & c^2 & d^2 \\ a^4 & b^4 & c^4 & d^4 \end{vmatrix} = (a + b + c + d)(d-a)(d-b)(d-c)(c-a)(c-b)(b-a)$



5. **克莱姆法则**（有时候有奇效）
   - （1）对 $n$ 个方程 $n$ 个未知数的非齐次线性方程组 $\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1  \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \cdots \\ a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n = b_n \end{cases}$，若系数行列式 $D = \begin{vmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots &  & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{vmatrix} \ne 0$（$\Leftrightarrow$ 系数矩阵 $A$ 可逆 $\Leftrightarrow$ 矩阵 $A$ 满秩），则方程组有唯一解，且解为 $x_i = \dfrac{D_i}{D}$，其中 $D_i$ 是由常数项 $b_1, b_2, \cdots, b_n$ 替换掉 $D$ 中的第 $i$ 列元素得到的行列式（反之，$D = 0$ 时方程组有无穷多的解）
   - （2）对 $n$ 个方程 $n$ 个未知数的齐次线性方程组 $\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = 0  \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = 0 \\ \cdots \\ a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n = 0 \end{cases}$，若 $D \ne 0$ 则齐次方程组只有 0 解；若 $D = 0$ 则齐次方程组有非零解

6. **抽象行列式的计算**（综合提高）
   - （1）设 $\alpha_1$、$\alpha_2$、$\alpha_3$ 均为三维列向量，矩阵 $A = (\alpha_1, \alpha_2, \alpha_3)$，已知 $|A|$ 求 $B = (\alpha_1 + \alpha_2 + \alpha_3, \alpha_1 + 2\alpha_2 + 4\alpha_3, \alpha_1 + 3\alpha_2 + 9\alpha_3)$（思路：$B = (\alpha_1, \alpha_2, \alpha_3)\cdot \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{pmatrix}$）
   - （2）三阶矩阵 $A, B$ 满足 $A^2 B - A - B = E$，其中 $E$ 为三阶单位矩阵，已知 $A$ 求 $|B|$（提示：$AB = 0$，则有 $r(A) + r(B) \le n$；行列式不为 0，则矩阵满秩）
   - （3）$B$ 是 3 阶正交矩阵，且 $|B| < 0$，$A$ 是三阶矩阵，且 $|A - B| = 6$，求 $|E - BA^T|$（提示：正交矩阵的行列式要么等于 1，要么等于 -1，且 $E = BB^T = B^TB$）

## 2. 矩阵

> 💡 本章重点习题
> - 【880-11-基础-选择-6】
> - 【880-11-基础-填空-5、7、11】
> - 【880-11-基础-解答-2、6、7】
> - 【880-11-综合-选择-5】
> - 【880-11-综合-解答-2】


> 📢 抽象小题如果用常规方法做不出来，或者哪个结论忘记了，一定要多试试特值法（万一遇到超纲的就只能用特值法了，比如 25 考研的一道选择题考了 Sylvester 不等式）

1. **初等矩阵的定义**：单位矩阵经过一次初等变换（行列交换、行列数乘、数乘后加至其他行列）所得的矩阵，称为初等矩阵
   - （1）$E_{ij}$ 表示单位矩阵 $E$ 交换第 $i$ 行与第 $j$ 行（或交换第 $i$ 列与第 $j$ 列）所得的初等矩阵
   - （2）$E_i(k)(k\ne 0)$ 表示单位矩阵 $E$ 的第 $i$ 行（或列）乘以非零常数 $k$ 所得的初等矩阵
   - （3）$E_{ij}(c)$ 表示单位矩阵 $E$ 的第 $j$ 行乘以 $c$ 加到第 $i$ 行（或第 $i$ 列乘以 $c$ 加到第 $j$ 列）所得的初等矩阵（不建议记，写法有争议，老头不太会考）

2. **左行右列定理**：
   - （1）矩阵 $A$ 左乘初等矩阵 $P$ 得到 $PA$，相当于对 $A$ 作了一次与 $P$ 完全相同的初等行变换
   - （2）矩阵 $A$ 右乘初等矩阵 $P$ 得到 $AP$，相当于对 $A$ 作了一次与 $P$ 完全相同的初等列变换
   - 例如 $\begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix} = \begin{pmatrix} 1 & 2 & 3 \\ 6 & 9 & 12 \\ 7 & 8 & 9 \end{pmatrix}$
   - 提示：利用初等矩阵将初等变换过程转化为用初等矩阵左乘、右乘矩阵，从而建立等式关系

3. **初等矩阵的行列式、逆矩阵、转置矩阵**
   - （1）$|E_{ij}| = -1$，$|E_i(k)| = k(k\ne 0)$，$|E_{ij}(c)| = 1$
   - （2）$E_{ij}^{-1} = E_{ij}$，$E_i^{-1}(k) = E_i(\dfrac{1}{k})$，$E_{ij}^{-1}(c) = E_{ij}(-c)$
   - （3）$E_{ij}^T = E_{ij}$，$E_i^T(k) = E_i(k)$，$E_{ij}^T(k) = E_{ji}(k)$
   - （4）此外，根据 $|A|A^{-1} = A^*$，还可以得到伴随矩阵的关系


> 💡 上述三种初等矩阵都是可逆矩阵（矩阵 $A$ 可逆 $\Leftrightarrow |A| \ne 0$）

> 💡 设 $A$ 为 $n$ 阶矩阵，$P, Q$ 为 $n$ 阶可逆矩阵，则 
> - （1）$P(A, E) = (PA, P)$ 相当于对 $A$ 作初等行变换的同时将变换过程体现在 $E \rightarrow P$ 上
> - （2）$\begin{pmatrix} A \\ E \end{pmatrix} Q = \begin{pmatrix} AQ \\ Q \end{pmatrix}$ 相当于对 $A$ 作初等列变换的同时将变换过程体现在 $E \rightarrow Q$ 上
> - 该思想可以用于初等变换和矩阵求逆等各类题型上【21 年选择题】


4. **矩阵的逆**
   - （1）定义：对 $n$ 阶方阵 $A, B$，$AB = E$ 则 $A$ 可逆且 $A^{-1} = B$
   - （2）性质：
     - a. $AB = E = BA$（$A, B$ 可交换，这个性质可以有奇效）

> 💡 设 $A, B$ 为 $n$ 阶矩阵，则 $AB = BA$（可交换）的充分条件包括：
> - （1）$B = f(A)$ 或 $A^{-1}$ 或 $A^*$
> - （2）$A^2 + aAB = E$（$a\ne 0$）
> - （3）$AB = aA + bB$（$ab \ne 0$）

> 💡 $A$ 为 $n$ 阶矩阵，若对任意非零向量 $\alpha$ 有 $A\alpha \ne 0$，则 $A$ 可逆（翻译：$A\alpha = 0$ 只有零解，即只有 $\alpha$ 为零向量时才能 $=0$）

5. **伴随矩阵**
   - （1）定义：$A^* = \begin{vmatrix} A_{11} & A_{21} & \cdots & A_{n1} \\ A_{12} & A_{22} & \cdots & A_{n2} \\ \vdots & \vdots & \ddots & \vdots \\ A_{1n} & A_{2n} & \cdots & A_{nn} \end{vmatrix}$ 且有 $AA^* = A^* A = |A|E$
   - （2）性质：
     - a. $AA^* = A^* A = |A|E$
     - b. $|A^*| = |A|^{n-1}$
     - c. $A^* = |A|A^{-1}$
     - d. $(AB)^* = B^* A^*$（穿透性质）
     - e. $(A^T)^* = (A^*)^T$，$(A^{-1})^* = (A^*)^{-1}$，$(A^*)^* = |A|^{n-2} A$

> ⚠️ 注意伴随矩阵的下标！这里也可能会出题 🌚【880-十-基础题-选择-3】

> 💡 设 $A$ 为 $n(n\ge 3)$ 阶非零矩阵，两个 $A^T$ 与 $A^*$ 的小结论（后面二次型、正交矩阵部分还会用到）：
> - （1）$a_{ij} = A_{ij} \Leftrightarrow A^* = A^T \Leftrightarrow AA^T = E$ 且 $|A| = 1 \Rightarrow$ 立即推：正交矩阵
> - （2）$a_{ij} = -A_{ij} \Leftrightarrow A^* = -A^T \Leftrightarrow AA^T = E$ 且 $|A| = -1 \Rightarrow$ 立即推：正交矩阵
> - （证明可以使用展开定理：$|A| = a_{11} A_{11} + a_{12} A_{12} + \cdots + a_{1n}A_{1n} = a_{11}^2 + a_{12}^2 + \cdots + a_{1n}^2 > 0 $）

> 💡 若 $A^*$ 为非零矩阵，则 $A$ 至少存在一个代数余子式不为 0，从而 $r(A) \ge n-1$

6. **矩阵的秩相关结论**（全部要牢记）
   - （1）$0 \le r(A_{m\times n}) \le \min\{m, n\}$
   - （2）$r(kA) = r(A)(k\ne 0)$
   - （3）设 $A_{m\times n}$，$P$ 为 $m$ 阶可逆矩阵，$Q$ 为 $n$ 阶可逆矩阵，则 $r(A) = r(PA) = r(AQ) = r(PAQ)$（即 $A$ 做初等变换之后秩不变，进一步的结论：$A$ 左乘列满秩，或右乘行满秩，秩不变）
   - （4）$r(AB) \le \min \{r(A), r(B)\}$（$\textcolor{red}{秩越乘越小}$）
   - （5）若 $A_{m\times n}B_{n\times s} = O$
     - a. $r(A) + r(B) \le n$（$\textcolor{red}{重中之重}$，考研最喜欢考这个了）
     - b. $B$ 的列向量均为线性方程组 $Ax=0$ 的解
     - c. 若 $A$ 为 $n$ 阶矩阵，则 $B$ 的非零列向量均为 $A$ 属于特征值 0 的特征向量（当然，必须是方阵，只有方阵才谈特征值和特征向量）
   - （6）$r(A \pm B) \le r([A, B]) \le r(A) + r(B)$（$\textcolor{red}{重中之重}$）
   - （7）$r(A^*) = \begin{cases} n, r(A) = n \\ 1, r(A) = n-1 \\ 0, r(A) < n-1 \end{cases}$（$\textcolor{red}{一定要记住}$）
   - （8）对任意 $A_{m\times n}$，都有 $r(A) = r(A^T) = r(AA^T) = r(A^TA)$（后面在方程组那里会用到）
   - （9）若 $A^2 = A$ 则 $r(A) + r(A - E) = n$
   - （10）若 $A^2 = E$ 则 $r(A+E) + r(A - E) = n$
   - （11）$Ax = 0$ 的基础解系所含向量的个数 $s = n - r(A)$
   - （12）若 $A～\Lambda$，则 $n_i = n - r(\lambda_i E -A)$，其中 $\lambda_i$ 是 $n_i$ 重特征根（$\textcolor{red}{重中之重}$）
   - （13）若 $A～\Lambda$，则 $r(A)$ 等于非零特征值的个数，重根按重数算
   - （14）$r(\begin{pmatrix} A & O \\ O & B \end{pmatrix}) = r(A) + r(B)$
   - （15）$r(A) + r(B) \le r\begin{pmatrix} A & O \\ C & B \end{pmatrix} \le r(A) + r(B) + r(C)$（$A$ 可逆时左边可以取等号）
   - （16）若 $A$ 满秩，则 $|A| \ne 0$，进而矩阵 $A$ 可逆（可逆的充要条件是 $|A|\ne 0$），所以 $AB$ 相当于对矩阵 $B$ 做初等行变换，从而 $r(AB) = r(B)$ 
   - （17）$\alpha$ 为 $n$ 维列向量，则 $r(\alpha \alpha^T) = 1$，且 $(\alpha\alpha^T)\alpha = (\alpha^T\alpha) \alpha \Leftrightarrow \alpha\alpha^T$ 的特征值为 $\lambda_1 = \alpha^T\alpha, \lambda_2 = \cdots = \lambda_n = 0$，$\lambda_1$ 对应特征向量为 $\alpha$
   - （18）Sylvester 不等式：设 $A_{m\times n}$，$B_{n\times s}$，则 $r(AB) \ge r(A) + r(B) - n$（口诀：$\textcolor{red}{乘积的秩 \ge 秩的和减 n}$）
   - （19）设 $A_{m\times n}$，若 $r(A) = n$（列满秩），则 $r(AB) = r(B)$；若 $r(A) = m$（行满秩），则 $r(CA) = r(C)$

> 💡 上述结论的证明：
> - （4）![linear7](/images/mathematic/linear7.png)
> - （6）![linear6](/images/mathematic/linear6.png)
> - （9）$A^2 = A \Rightarrow A(E-A) = 0 \Rightarrow r(A) + r(E-A) \le n$（用结论 5），又根据结论 6 有 $r(A) + r(E - A) \ge r(E) = n$，两边一夹故有 $r(A) + r(A - E) = n$
> - （10）同（9）
> - （18）对 $\begin{pmatrix} A & O \\ E & B \end{pmatrix}$ 作广义初等变换：$\begin{pmatrix} O & -AB \\ E & O \end{pmatrix}$（列倍加），得到 $r(A) + r(B) \le r\begin{pmatrix} A & O \\ E & B \end{pmatrix} = r\begin{pmatrix} O & -AB \\ E & O \end{pmatrix} = r(AB) + r(E) = r(AB) + n$，故 $r(AB) \ge r(A) + r(B)- n$

> 💡 关于秩为 1 的矩阵 $A$
> - （1）$A$ 可以表示成 $\alpha \beta^T$（即由列向量乘以行向量得到的矩阵 $A$，其极大线性无关组里面只有 $\alpha$ 这一列，秩一定为 1）【推广：$A = \alpha\beta^T + \beta \alpha^T \Rightarrow r(A) = r(\alpha\beta^T + \beta \alpha^T) \le r(\alpha\beta^T) + r(\beta \alpha^T) = 2$】
> - （2）$A$ 的特征值为 $\beta^T\alpha, 0, 0, \cdots, 0$（矩阵可相似对角化，则秩等于非零特征值个数），其对应的特征向量分别为 $\beta^T \alpha$ 和 $\beta^Tx = 0$ 的基础解系
> - （3）$tr(A) = \beta^T \alpha = \alpha^T \beta$（等于内积，相关证明：$A\alpha = (\alpha\beta^T) \alpha = \alpha(\beta^T\alpha) = (\beta^T\alpha)\alpha = tr(A)\alpha$）
> - （4）$\textcolor{red}{A^n} = (\alpha\beta^T)(\alpha\beta^T)\cdots (\alpha\beta^T) = \alpha(\beta^T\alpha)(\beta^T\alpha)\cdots\beta^T = \textcolor{red}{[tr(A)]^{n-1} \cdot A}$（看到 $A^2 = aA$ 要能够想到 $tr(A) = a$，或者已知 $tr(A)$ 要知道 $A^2$ 等于什么）
> - （5）$A$ 的特征值为 $tr(A)$ 和 $n-1$ 个 0（可以从方程组的角度来理解）$\begin{cases} 一定可以相似对角化（A～\Lambda）,  & tr(A) \ne 0 \\ 不能相似对角化, & tr(A) = 0 \end{cases}$

> 💡 行满秩矩阵：设 $A$ 为 $m\times n$ 矩阵，且 $r(A) = m$，则
> - （1）若 $BA = CA$，则 $B = C$
> - （2）$r(BA) = r(B)$
> - （3）$A$ 的行向量组线性无关
> - （4）非齐次线性方程组 $Ax = \beta$ 有解（$r(A) = r(A, \beta)$）

> 💡 列满秩矩阵：设 $A$ 为 $m\times n$ 矩阵，且 $r(A) = n$，则
> - （1）若 $AB = AC$，则 $B = C$
> - （2）$r(AB) = r(B)$
> - （3）$A$ 的列向量组线性无关
> - （4）线性方程组 $Ax = 0$ 只有零解
> - （5）线性方程组 $ABx = 0$ 和 $Bx = 0$ 同解

> 💡 如果 $r(A) = r$，则意味着存在一个 $r$ 阶子式不为 0，且任意的 $r + 1$ 阶子式为 0




7. **分块矩阵**
   - （1）先回忆一下上一章的拉普拉斯展开式
   - （2）分块对角矩阵的幂：$\begin{pmatrix} A_1 & & & \\ & A_2 & & \\ & & \ddots & \\ & & & A_n \end{pmatrix}^n = \begin{pmatrix} A_1^n & & & \\ & A_2^n & & \\ & & \ddots & \\ & & & A_n^n \end{pmatrix}$（分块副对角矩阵的幂没有这个规律）
   - （3）分块矩阵的逆：若 $A, B, C$ 均为可逆矩阵，则
     - a. $\begin{pmatrix} A & O \\ O & B \end{pmatrix}^{-1} = \begin{pmatrix} A^{-1} & O \\ O & B^{-1} \end{pmatrix}$
     - b. $\begin{pmatrix} O & A \\ B & O \end{pmatrix}^{-1} = \begin{pmatrix} O & B^{-1} \\ A^{-1} & O \end{pmatrix}$
     - c. $\begin{pmatrix} A & C \\ O & B \end{pmatrix}^{-1} = \begin{pmatrix} A^{-1} & -A^{-1}CB^{-1} \\ O & B^{-1} \end{pmatrix}$
     - d. $\begin{pmatrix} A & O \\ C & B \end{pmatrix}^{-1} = \begin{pmatrix} A^{-1} & O \\ -B^{-1}CA^{-1} & B^{-1} \end{pmatrix}$
     - e. $\begin{pmatrix} C & A \\ B & O \end{pmatrix}^{-1} = \begin{pmatrix} O & B^{-1} \\ A^{-1} & -A^{-1}CB^{-1} \end{pmatrix}$
     - f. $\begin{pmatrix} O & A \\ B & C \end{pmatrix}^{-1} = \begin{pmatrix} -B^{-1}CA^{-1} & B^{-1} \\ A^{-1} & O \end{pmatrix}$
   - （4）分块矩阵的转置：$\begin{pmatrix} A & B \end{pmatrix} ^T = \begin{pmatrix} A^T \\ B^T \end{pmatrix}$，$\begin{pmatrix} A \\ B \end{pmatrix}^T = \begin{pmatrix} A^T & B^T \end{pmatrix}$
   - （5）分块矩阵的高次幂：若 $A = \begin{pmatrix} B & O \\ O & C \end{pmatrix}$，则 $A^n = \begin{pmatrix} B & O \\ O & C \end{pmatrix}^n = \begin{pmatrix} B^n & O \\ O & C^n \end{pmatrix}$




8. **矩阵乘法的进一步解读**（$\textcolor{red}{非常重要}$）
   - 若 $A$ 是 $m\times n$ 矩阵，$B$ 是 $n\times s$ 矩阵且 $AB = O$，对 $B$ 和 $O$ 矩阵按列分块有 $AB = A [b_1, b_2, \cdots, b_s] = [Ab_1, Ab_2, ..., Ab_s] = [0,0,..., 0]$，$Ab_i = 0$，即 $B$ 的列向量是齐次方程组 $Ax = 0$ 的解
   - 若 $AB = C$，则对 $B, C$ 按行分块有 $\begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} \begin{pmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n \end{pmatrix} = \begin{pmatrix} \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_m \end{pmatrix} \Leftrightarrow \begin{cases} a_{11}\beta_1 + a_{12}\beta_2 + \cdots + a_{1n}\beta_n = \alpha_1 \\ a_{21}\beta_1 + a_{22}\beta_2 + \cdots + a_{2n}\beta_n = \alpha_2 \\ \vdots \\ a_{m1}\beta_1 + a_{m2}\beta_2 + \cdots + a_{mn}\beta_n = \alpha_m \end{cases}$
   - 【前列线定理】可见矩阵 $AB$ 的 $\textcolor{red}{行向量}$ $\alpha_1, \alpha_2, \cdots, \alpha_m$ 可由 $B$ 的 $\textcolor{red}{行向量}$ 线性表出，矩阵 $AB$ 的 $\textcolor{red}{列向量}$ 可由 $A$ 的 $\textcolor{red}{列向量}$ 线性表出（例题参考 18 年真题：$r(A , AB) = r(A), r(A , BA)$ 不一定等于 $r(A)$）



9. **二项式展开定理**（用来求高次幂）
   - 若 $A = E + B$，则 $A^n = (E+B)^n = E + C^1_nB + C^2_nB^2 + \cdots + B^n$
   - 特别的，若 $B = \begin{pmatrix} 0 & a & b \\ 0 & 0 & c \\ 0 & 0 & 0 \end{pmatrix}$，则 $B^2 = \begin{pmatrix} 0 & 0 & ac \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$，$B^3 = O$，从而 $A^n = (E + B)^n = E + nB + \dfrac{n(n-1)}{2}B^2$


10. **分块矩阵的广义初等变换**（左行右列）
   - （1）行互换：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} C & D \\ A & B \end{pmatrix}$，$\begin{pmatrix} O & E \\ E & O \end{pmatrix} \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix} C & D \\ A & B \end{pmatrix}$
   - （2）行倍加：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} A +MC & B+MD \\ C & D \end{pmatrix}$，$\begin{pmatrix} E & M \\ O & E \end{pmatrix} \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix} A +MC & B+MD \\ C & D \end{pmatrix}$
   - （3）行倍乘：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} MA & MB \\ C & D \end{pmatrix}$，$\begin{pmatrix} M & O \\ O & E \end{pmatrix} \begin{pmatrix} A & B \\ C & D \end{pmatrix} = \begin{pmatrix} MA & MB \\ C & D \end{pmatrix}$
   - （4）列互换：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} B & A \\ D & C \end{pmatrix}$，$ \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} O & E \\ E & O \end{pmatrix} = \begin{pmatrix} B & A \\ D & C \end{pmatrix}$
   - （5）列倍加：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} A +BM & B \\ C + DM & D \end{pmatrix}$，$ \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} E & O \\ M & E \end{pmatrix} = \begin{pmatrix} A +BM & B \\ C + DM & D \end{pmatrix}$
   - （6）列倍乘：$\begin{pmatrix} A & B \\ C & D \end{pmatrix} \rightarrow \begin{pmatrix} AM & B \\ CM & D \end{pmatrix}$，$ \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} M & O \\ O & E \end{pmatrix}= \begin{pmatrix} AM & B \\ CM & D \end{pmatrix}$

> 💡 广义初等变换不改变分块矩阵的秩



## 3. 向量组

> ⚠️ 本章考抽象小题偏多，注意性质的运用

> 💡 本章重点习题
> - 【880-12-基础-选择-5】
> - 【880-12-基础-解答-3、7】
> - 【880-12-综合-选择-1、3、4】
> - 【880-12-综合-解答-6】

1. **向量内积**
   - （1）定义：设 $\alpha = (a_1, a_2, ..., a_n)^T, \beta = (b_1, b_2, ..., b_n)^T$，则内积 $(\alpha, \beta) = (a_1, a_2, ..., a_n) \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix} = \alpha^T \beta$
   - （2）性质
     - a. $(\alpha, \alpha) = \alpha^T \alpha = a_1^2 + a_2^2 + ... + a_n^2$
     - b. 向量的模（或长度）$||\alpha|| = \sqrt{(\alpha, \alpha)} = \sqrt{a_1^2 + a_2^2 + ... + a_n^2}$
     - c. $(\alpha, \alpha) = \alpha^T \alpha = 0 \Leftrightarrow \alpha = 0$
     - d. $(\alpha, \beta) = (\beta, \alpha) = \alpha^T\beta = \beta^T \alpha$

2. **判断向量组线性相关性的主要方法**
   - （1）线性相关的定义：存在一组不全为 0 的 $k$，使得 $k_1\alpha_1 + k_2\alpha_2 + \cdots + k_s\alpha_s = 0$（也就是说存在一个向量是其余向量的线性组合）
   - （2）齐次线性方程组 $(\alpha_1, \alpha_2, \cdots, \alpha_s) \begin{pmatrix} k_1 \\ k_2 \\ \vdots \\ k_n \end{pmatrix} = 0$ 有非零解（线性无关的话是只有零解）
   - （3）$r(\alpha_1, \alpha_2, \cdots, \alpha_s) < s(线性无关的话=s)$

> 💡 向量组线性无关等价于该向量组中任意一个向量均不能由其他向量线性表示

3. **判断向量组线性表示的主要方法**
   - （1）线性表出的定义：$\beta = k_1 \alpha_1 + k_2 \alpha_2 + \cdots + k_s \alpha_s$
   - （2）非齐次线性方程组 $(\alpha_1, \alpha_2, \cdots, \alpha_s) \begin{pmatrix} k_1 \\ k_2 \\ \vdots \\ k_s \end{pmatrix} = \beta$ 有解
   - （3）$r(\alpha_1, \alpha_2, \cdots, \alpha_s) = r(\alpha_1, \alpha_2, \cdots, \alpha_s, \beta)$

> 💡 判定线性相关性的常用定理
> - （1）整体与部分：若 $\alpha, \beta$ 线性相关，则 $\alpha, \beta, \gamma$ 必线性相关；若 $\alpha, \beta, \gamma$ 线性无关，则 $\alpha , \beta$ 必线性无关
> - （2）接长于局部：若 $\alpha_1, \alpha_2$ 线性无关，$\beta_1, \beta_2$ 维数相同，则 $\begin{pmatrix} \alpha_1 \\ \beta_1 \end{pmatrix},\begin{pmatrix} \alpha_2 \\ \beta_2 \end{pmatrix}$ 必线性无关；若 $\begin{pmatrix} \alpha_1 \\ \beta_1 \end{pmatrix},\begin{pmatrix} \alpha_2 \\ \beta_2 \end{pmatrix}$ 线性相关，则原向量组 $\alpha_1, \alpha_2$ 必线性相关
> - （3）以少表多，多的相关：如果向量组 $\beta_1, \beta_2, \cdots, \beta_t$ 可以由向量组 $\alpha_1, \alpha_2, \cdots, \alpha_s$ 线性表示，且 $t > s$，则 $\beta_1, \beta_2, \cdots, \beta_t$ 线性相关
> - （4）非零向量之间正交的话，那么必然线性无关；零向量天生与任意向量正交，与任意向量线性相关

> ⚠️ 易错题：设 $n$ 维向量组 $\alpha_1, \alpha_2, \cdots, \alpha_{n-1}$ 线性无关，$\beta_1, \beta_2$ 均与 $\alpha_1,\alpha_2, \cdots, \alpha_{n-1}$ 正交，则（）
> - A. $\alpha_1,\alpha_2, \cdots, \alpha_{n-1}, \beta_1$ 线性无关（错，当 $\beta_1$ 为零向量时线性相关）
> - B. $\beta_1, \beta_2$ 线性相关（对，从方程组的角度理解，设 $\beta_1, \beta_2$ 为齐次方程组 $Ax = 0$ 的解，那么由于 $n - r(A) = n - (n-1) = 1$，所以解系中自由向量只有 1 个，即 $\beta_1, \beta_2$ 成比例，故线性相关）
> - C. $\beta_1, \beta_2$ 线性无关
> - D. $\beta_1, \beta_2$ 均可由 $\alpha_1, \alpha_2, \cdots, \alpha_{n-1}$ 线性表示


4. **极大线性无关组**（这一块的内容比较死板）
   - （1）定义：若向量组 $\alpha_1, \alpha_2, \cdots, \alpha_s$ 中存在 $r$ 个（这个 $r$ 就是秩）向量 $\alpha_{i1}, \alpha_{i2}, \cdots, \alpha_{ir}$ 线性无关，且再添加任一个 $\alpha_j(j=1,2,\cdots, s)$ 就有 $\alpha_{i1}, \alpha_{i2} , \cdots, \alpha_{ir}, \alpha_{j}$ 线性相关，则称 $\alpha_{i1}, \alpha_{i2}, \cdots, \alpha_{ir}$ 是向量组 $\alpha_1, \alpha_2, \cdots, \alpha_s$ 的一个极大线性无关组，向量组 $\alpha_1, \alpha_2, \cdots, \alpha_s$ 的极大线性无关组 $\alpha_{i1}, \alpha_{i2}, \cdots, \alpha_{ir}$ 中所含向量个数 $r$，称为向量组的秩，记为 $r(\alpha_1, \alpha_2, \cdots, \alpha_s)$
   - （2）若向量组的秩为 $r$，则：
     - a. 其中任意 $r$ 个向量都可为其最大无关组？（❌）
     - b. 其中任意 $r$ 个线性无关的向量都可以为其最大无关组？（✅）
     - c. 其中多于 $r$ 个向量组成的向量组一定线性相关？（✅）
     - d. 其中 $r$ 个向量组成的向量组一定线性无关？（❌）
   - （3）求极大线性无关组的方法（利用初等行变换不改变列向量组的线性相关性，也因此求解的时候只能都作初等行变换或都作列变换）
     - a. 构造 $A = [\alpha_1, \alpha_2, \cdots, \alpha_s]$
     - b. $A$ 行变换为阶梯形矩阵
     - c. 算出台阶数，按列找出一个秩为 $r$ 的子矩阵即可
     - ![problem1](/images/mathematic/linear1.png)

5. **向量组等价**
   - （1）若 $\alpha_i, \beta_i$ 同维，则：
     - $\{\alpha_1, \alpha_2, \cdots, \alpha_s\} \cong \{\beta_1, \beta_2, \cdots, \beta_t\}$
     - $\Leftrightarrow \{\alpha_1, \alpha_2, \cdots, \alpha_s\}$ 与 $\{\beta_1, \beta_2, \cdots, \beta_t\}$ 可以互相线性表出
     - $\Leftrightarrow r\{\alpha_1, \alpha_2, \cdots, \alpha_s\} = r\{\beta_1, \beta_2, \cdots, \beta_t\}$，且可单方向表出，即只需要知道 $\alpha_1, \alpha_2, \cdots, \alpha_s$ 与 $\beta_1, \beta_2, \cdots, \beta_t$ 这两个向量组中的某一个向量组可由另一个向量组线性表出
     - $\Leftrightarrow r\{\alpha_1, \alpha_2, \cdots, \alpha_s\} = r\{\beta_1, \beta_2, \cdots, \beta_t\} = r(\alpha_1, \alpha_2, \cdots, \alpha_s, \beta_1, \beta_2, \cdots, \beta_t)$（三秩相同）
   - （2）提醒：若向量组的秩为 $r$，则
     - a. 此向量组与自己的极大线性无关组等价
     - b. 此向量组的任意两个极大线性无关组等价
   - （3）易错点：
     - a. 两个向量组等价，一个线性无关，另一个也必线性无关吗？（❌）
     - b. 两个向量组等价，向量的个数一定相同吗？（❌）


> ⚠️ 注意向量组等价于矩阵等价的区别：
> - （1）向量组（I）与（II）等价 $\Rightarrow r(I) = r(II)$
> - （2）同型矩阵 $A$ 与 $B$ 等价 $\Leftrightarrow r(A) = r(B)$

> ⚠️ 易错题：设 $n$ 维向量组（I）$\alpha_1, \alpha_2, \cdots, \alpha_k(k<n)$ 线性无关，则 $n$ 维向量组（II）$\beta_1, \beta_2, \cdots, \beta_k$ 也线性无关的充要条件是（）
> - A. $\beta_1, \beta_2, \cdots, \beta_k$ 可由 $\alpha_1, \alpha_2, \cdots, \alpha_k$ 线性表示（❌）
> - B. $\alpha_1, \alpha_2, \cdots, \alpha_k$ 可由 $\beta_1, \beta_2, \cdots, \beta_k$ 线性表示（❌）
> - C. 向量组（I）和向量组（II）等价（❌）
> - D. 矩阵 $(\alpha_1, \alpha_2, \cdots, \alpha_k)$ 与 $(\beta_1, \beta_2, \cdots, \beta_k)$ 等价（✅）

> 💡 $A$ 和 $B$ 为 $n$ 阶矩阵，$P, Q$ 为 $n$ 阶可逆矩阵，则
> - （1）若 $A = BQ$，则 $A, B$ 列向量组等价
> - （2）若 $A = PB$，则 $A, B$ 行向量组等价
> - （3）若 $A = PBQ$，则 $A, B$ 等价（注意，不是行（列）向量组等价！）

> 📖 总结梳理：
> - （1）本章重点是 $AB = C$ 问题
> - （2）对于 $AB  =C$，我们知道 $AB$ 的列可以由 $A$（左边的）的列线性表出（前列腺定理...）
> - （3）那么，$AB$ 的列可以被 $A$ 的列表出，可以得到 $r(AB) \le r(A)$
> - （4）做题的时候还遇到了 $r(A, AB) = r(A)$（如果 $A$ 和 $AB$ 能拼在一起的话）
> - （5）此外，还有 $B$ 的每一列 $b_i$ 都是非齐次方程组 $Ax = C_i$ 的解
> - （6）对于 $AB = C$，我们还有 $AB$ 的行可以由 $B$（右边的）的行线性表出
> - （7）进而 $r(AB) \le r(B)$
> - （8）根据 3 和 7，我们可以得到结论：秩越乘越小，即 $r(AB) \le \min\{r(A), r(B)\}$
> - （9）同 4，我们依然可以得到 $r\begin{pmatrix} B \\ AB \end{pmatrix} = r(B)$
> - （10）特殊的，若 $AB = 0$，那么就由非齐次的问题变成了一个齐次的问题，既然是齐次问题那么它对应的就是线性相关和线性无关的问题
> - （11）在此基础上，若 $B\ne 0$ 的话，$A$ 的列向量不满秩（列相关）
> - （12）$B$ 的每一列 $b_i$ 都是 $Ax = 0$ 的解，根据方程组解的结构，$n - r(A) \ge r(B)$，进而有 $r(A) + r(B) \le n$
> - （13）若 $A \ne 0$，$B$ 的行也是不满秩的，后续结论同 12
> - （14）对于向量组等价问题，要牢牢记住那三个充要条件
> - （15）对于前面提到的 $AB = C$ 题型，有进一步补充：若 $B$ 可逆，则 $AB$ 的列等价于 $A$ 的列（证明要会）；若 $A$ 可逆，则 $AB$ 的行等价于 $B$ 的行（乘以可逆矩阵，相当于做行变换和列变换）
> - （16）此外，初等行列变换也是相当重要的：初等行变换对应行等价，且不改变列的相关性；初等列变换对于列等价，且不改变行的线性相关性

6. **向量空间**
   - （1）基本概念：若 $\xi_1, \xi_2, \cdots, \xi_n$ 是 $n$ 维向量空间 $R^n$ 中的线性无关的有序向量组，则任一向量 $\alpha \in R$ 均可由 $\xi_1, \xi_2, \cdots, \xi_n$ 线性表示为 $\textcolor{red}{\alpha = a_1\xi_1 + a_2\xi_2 + \cdots, a_n\xi_n}$，称有序向量组 $\xi_1, \xi_2, \cdots, \xi_n$ 是 $R^n$ 的一个基，基向量的个数 $n$ 称为空间的维度，而 $[a_1, a_2, \cdots, a_n]$ 称为向量 $\alpha$ 在基 $\xi_1, \xi_2, \cdots, \xi_n$ 下的坐标
   - （2）基变换公式与过度矩阵：$[\eta_1, \eta_2, \cdots, \eta_n] = [\xi_1, \xi_2, \cdots, \xi_n]\begin{pmatrix} c_{11} & c_{12} & \cdots & c_{1n} \\ c_{21} & c_{22} & \cdots & c_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ c_{n1} & c_{n2} & \cdots & c_{nn} \end{pmatrix} = [\xi_1, \xi_2, \cdots, \xi_n]C$，其中 $C$ 为由基 $\xi_1, \xi_2, \cdots, \xi_n$ 到基 $\eta_1, \eta_2, \cdots, \eta_n$ 的**过渡矩阵**（$C$ 的第 $i$ 列是 $\eta_i$ 在基 $\xi_1, \xi_2, \cdots, \xi_n$ 下的坐标，且过渡矩阵为可逆矩阵）
   - （3）坐标变换公式：设 $\alpha$ 在基 $\xi_1, \xi_2, \cdots, \xi_n$ 和 $\eta_1, \eta_2, \cdots, \eta_n$ 下的坐标分别为 $x = [x_1, x_2, \cdots, x_n]^T$ 和 $y = [y_1, y_2, \cdots, y_n]^T$，那么 $ \alpha = [\xi_1, \xi_2, \cdots, \xi_n]x = [\eta_1, \eta_2, \cdots, \eta_n]y \Leftrightarrow x = Cy$（或 $y = C^{-1}x$）

> 💡 若 $\alpha_1, \alpha_2, \cdots, \alpha_n$ 线性无关，则称 $\alpha_1, \alpha_2, \cdots, \alpha_n$ 为 $R^n$ 的一个基


7. **施密特正交化**（用来求正交基的，考试最多考到 $\beta_3$）
   - $\beta_1 = \alpha_1$
   - $\beta_2 = \alpha_2 - \dfrac{(\alpha_2, \beta_1)}{(\beta_1, \beta_1)} \beta_1$
   - $\beta_3 = \alpha_3 - \dfrac{(\alpha_3, \beta_1)}{(\beta_1, \beta_1)} \beta_1 - \dfrac{(\alpha_3, \beta_2)}{(\beta_2, \beta_2)} \beta_2$

> ⚠️ 求完正交基后，别忘了标准化为标准正交基

## 4. 线性方程组

> 💡 本章重点习题
> - 【880-13-基础-选择-3、7、8、9、11】
> - 【880-13-基础-填空-1、】
> - 【880-13-基础-解答-6】
> - 【880-13-综合-选择-6、7、9】
> - 【880-13-综合-解答-1、3、5】
> - 【880-13-拓展-2、3】


1. **齐次线性方程组**
   - （1）有解的条件（$m$ 个方程，$n$ 个未知量）（$\textcolor{red}{重点}$）
     - a. $r(A) = n$（即 $\alpha_1, \alpha_2, \cdots, \alpha_n$ 线性无关时），方程组只有零解
     - b. $r(A) = r < n$ 时（即线性相关时），方程组有非零解，且有 $n - r$ 个线性无关解
   - （2）解的性质：
     - a. 若 $A \xi_1 = 0, A\xi_2  = 0 $，则 $A(k_1 \xi_1 + k_2 \xi_2)= 0$，其中 $k_1, k_2$ 为任意常数
     - b. 类消去律：若 $A_{m\times n}$，$r(A) = n$（满秩），$AB = AC$，则 $B = C$
     - c. 齐次方程组，系数矩阵的行向量与解向量是 $\textcolor{red}{正交}$ 的（非常隐蔽的条件，以后看到正交要想到齐次方程组）
   - （3）基础解系和解的结构
     - a. 基础解系满足：是方程组 $Ax = 0$ 的解、线性无关、方程组 $Ax = 0$ 的任意解均可由 $\xi_1, \xi_2, \cdots, \xi_n$ 线性表示
     - b. 通解：$k_1 \xi_1 + k_2 \xi_2 + \cdots + k_n \xi_n$
     - c. 基础解系中向量个数（自由变量 $k$ 的个数） $s = n - r(A)$（如果是 $A_{m\times n}$，则是 $s = n - r(A)$）
   - （4）求解方式：
     - a. 先将系数矩阵初等行变换为行阶梯形矩阵，阶梯数位 $r = r(A)$
     - b. 按列找出一个秩为 $r$ 的子矩阵，剩余列位置的未知数设为自由变量 $k_i$
     - c. 按基础解系的定义求处 $\xi_1, \xi_2, \cdots, \xi_n$ 并写出通解


2. **非齐次线性方程组**
   - （1）有解的条件（$\textcolor{red}{重点}$）
     - a. 若 $r(A) \ne r([A, b])$（即 $b$ 不能用 $\alpha_1, \alpha_2, \cdots, \alpha_n$ 线性表示），则方程组无解
     - b. 若 $r(A) = r([A, b]) = n$（即 $\alpha_1, \alpha_2, \cdots, \alpha_n$ 线性无关，$\alpha_1, \alpha_2, \cdots, \alpha_n, b$ 线性相关），则方程组有唯一解
     - c. 若 $r(A) = r([A, b]) = r < n$，则方程组有无穷多解（也就是说只要 $r(A) = r([A, b])$，方程组一定是有解的）
   - （2）解的性质（⭐⭐⭐ 非常重要）
     - a. 设 $\eta_1, \eta_2$ 都是 $Ax = b$ 的解，则 $\eta_1 - \eta_2$ 是它的导出组 $Ax = 0$ 的解
     - b. 设 $\eta_1, \eta_2, \cdots, \eta_s$ 都是 $Ax = b$ 的解，则 $k_1 \eta_1 + k_2\eta_2 + \cdots + k_s \eta_s = \begin{cases} 是 Ax = b 的解,& 当 k_1 + k_2 + \cdots + k_s = 1 \\ 是 Ax = 0 的解,& 当 k_1 + k_2 + \cdots + k_s = 0 \end{cases}$（证明方法就是左乘一个 $A$）
     - c. 设 $\eta$ 是 $Ax = b$ 的一个解，$\xi$ 是它的导出组 $Ax = 0$ 的解，则 $\xi + \eta$ 是 $Ax = b$ 的解
   - （3）求解方法：先求齐次解，再求特解，合并得到通解


> 💡 小结论：$Ax = 0$ 有 $n -r(A)$ 个线性无关的解，则 $Ax = \beta$ 至多有 $n - r(A) + 1$ 个线性无关的解；若 $Ax = \beta$ 有 $n - r(A) + 1$ 个线性无关的解，则 $Ax = 0$ 至少有 $n -r(A)$ 个线性无关的解（其中那个 “+1” 是特解）

> 💡 用定义证明线性相关性时，常用的一个操作：设 $k_1 \xi_1 + k_2 \xi_2 + \cdots + k_n \xi_n = 0$，然后左乘系数矩阵 $A$


3. **矩阵方程**
   - （1）矩阵方程的判定：设 $A$ 为 $m\times n$ 阶矩阵，则
     - a. $AX = B$ 无解 $\Leftrightarrow r(A) < r(A, B)$
     - b. $AX = B$ 有唯一解 $\Leftrightarrow r(A) = r(A, B) = n$
     - c. $AX = B$ 有无穷多解 $\Leftrightarrow r(A) = r(A, B) < n$
   - （2）矩阵方程的求解：对 $(A, B)$ 作初等行变换，化为行最简矩阵，得矩阵 $X$
   - （3）推广：$XA = B \Leftrightarrow A^TX^T = B^T$

4. **克拉默法则**（这里重新回顾一下）
   - （1）内容：对 $n$ 个方程 $n$ 个未知数（$n\times n$ 的方阵）的非齐次线性方程组 $A_{n\times n}x = b$，若 $|A| \ne 0$（满秩），则方程组有唯一解，且解为 $x_i = \dfrac{|A_i|}{|A|}$（其中 $A_i$ 是指将 $A$ 的第 $i$ 列用 $b$ 替换）
   - （2）例子：设 $A = \begin{pmatrix} a_{11} & a_{12} & 0 \\ a_{21} & a_{22} & 0 \\ 0 & 0 & 1 \end{pmatrix}$，$A\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} =\begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$，则有 $x_1 = \dfrac{|A_1|}{|A|} = \dfrac{\begin{vmatrix} \textcolor{red}{0} & a_{12} & 0 \\ \textcolor{red}{0} & a_{22} & 0 \\ \textcolor{red}{1} & 0 & 1 \end{vmatrix}}{|A|} = 0$


5. **求两个方程组的公共解的三个方法**（这块想要学得好，一定要结合向量组等价）
   - （1）联立 $Ax = 0$ 和 $Bx = 0$ 求解 $\begin{pmatrix} A \\ B \end{pmatrix} x = 0$
   - （2）求出 $A_{m\times n}x = 0$ 的通解 $k_1 \xi_1 + k_2 \xi_2 + \cdots + k_s\xi_s$，代入 $B_{m\times n}x = 0$，求出 $k_i$ 之间的关系，代回 $A_{m\times n}x = 0$ 的通解
   - （3）给出 $A_{m\times n} x = 0$ 的基础解系 $\xi_1, \xi_2, \cdots, \xi_s$ 与 $B_{m\times n}x = 0$ 的基础解系 $\eta_1, \eta_2, \cdots, \eta_t$，则公共解 $\gamma = k_1 \xi_1 + k_2 \xi_2 + \cdots + k_s\xi_s = l_1\eta_1 + l_2\eta_2 + \cdots + l_t\eta_t$

6. **同解方程组**
   - （1）$Ax = 0, Bx = 0$ 是同解方程组的充要条件：
     - a. $Ax = 0$ 的解满足 $Bx = 0$ 且 $Bx = 0$ 的解满足 $Ax = 0$
     - b. $r(A) = r(B)$ 且 $Ax = 0$ 的解满足 $Bx = 0$
     - c. $r(A) = r(B) = r(\begin{pmatrix} A \\ B \end{pmatrix})$（三秩相同）
     - d. $A, B$ 行向量组等价
   - （2）综上，同解方程组有一个更本质的表达：$\textcolor{red}{行向量组等价}$（用这个做题更快）
   - （3）同解方程组的常用结论：
     - ⭐⭐⭐ a. $Ax = 0$ 和 $Bx = 0$ 同解 $\Leftrightarrow A$ 与 $B$ 行向量组等价 $\Leftrightarrow r(A) = r(B) = r(\begin{pmatrix} A \\ B \end{pmatrix}) \Rightarrow r(A) = r(B)$
     - b. $Ax = \alpha, Bx = \beta$ 同解 $\Leftrightarrow (A, \alpha)$ 与 $(B, \beta)$ 行向量组等价 $\Leftrightarrow r(A, \alpha) = r(B, \beta) = r(\begin{pmatrix} A & \alpha \\ B  & \beta \end{pmatrix}) \Rightarrow r(A, \alpha) = r(B, \beta)$
     - （易错）c. 设 $m\times n$ 矩阵 $A = \begin{pmatrix} \alpha_1^T \\ \alpha_2^T \\ \vdots \\ \alpha_n^T \end{pmatrix}$，$B = \begin{pmatrix} \beta_1^T \\ \beta_2^T \\ \vdots \\ \beta_n^T \end{pmatrix}$，$A$ 的行向量组可以由 $B$ 的行向量组线性表示 $\Leftrightarrow Bx = 0$ 的解都是 $Ax = 0$ 的解（$Bx = 0$ 的约束更多，集合范围小一点）$\Leftrightarrow$ 存在矩阵 $C$ 使 $A = CB \Rightarrow r(A) \le r(B)$
     - d. $A^T Ax = 0$ 与 $Ax = 0$ 同解（四秩相等，都是同解）
     - e. 设矩阵 $P$ 列满秩，则 $PAx = 0$ 与 $Ax = 0$ 同解
     - f. 若 $Ax = 0$ 的解均是 $Bx = 0$ 的解，则 $Ax = 0$ 与 $\begin{pmatrix} A \\ B \end{pmatrix} x = 0$ 同解

> 💡 推广：非齐次方程组 $Ax = \alpha$ 与 $Bx = \beta$ 同解 $\Leftrightarrow r(A, \alpha) = r\begin{pmatrix} A, \alpha \\ B, \beta \end{pmatrix} = r(B, \beta)$

> 💡 例题 1：设有齐次线性方程组 $Ax = 0$ 和 $Bx = 0$，其中 $A, B$ 均为 $m\times n$ 矩阵，现有四个命题
> - （1）若 $Ax = 0$ 的解均是 $Bx = 0$ 的解，则 $r(A) \ge r(B)$（✅，因为解的集合越小，说明约束越多，方程数越多，故秩越大）
> - （2）若 $r(A) \ge r(B)$，则 $Ax = 0$ 的解均是 $Bx = 0$ 的解（❌，不一定是 $B$ 的解，解集不一定包含于 $B$）
> - （3）若 $Ax = 0$ 与 $Bx = 0$ 同解，则 $r(A) = r(B)$（✅）
> - （4）若 $r(A) = r(B)$，则 $Ax = 0$ 与 $Bx = 0$ 同解（❌）

> 💡 例题 2：设 $A$ 是 $n$ 阶矩阵，对方程组（I）$Ax = 0$ 和（II）$A^TAx = 0$，则有：
> - （1）$A^TA x= 0 \Rightarrow x^TA^TAx = 0 \Rightarrow (Ax)^T(Ax) = 0 \Rightarrow ||Ax||^2 = 0 \Rightarrow Ax = 0$
> - （2）$Ax = 0 \Rightarrow A^T Ax = 0$

> 💡 此外，当遇到分块矩阵作为系数矩阵的方程组时，千万$\textcolor{red}{不要做列变换}$，因为会改变解系


7. **线性方程组的几何意义**
   - 空间三平面 $\begin{cases} \pi_1: a_1x + b_1 y + c_1 z = d_1 \\ \pi_2: a_2x + b_2 y + c_2 z = d_2 \\ \pi_3: a_3x + b_3 y + c_3 z = d_3 \end{cases}$ 的位置关系：令 $A = \begin{pmatrix} a_1 & b_1 & c_1 \\ a_2 & b_2 & c_2 \\ a_3 & b_3 & c_3 \end{pmatrix} = \begin{pmatrix} \alpha_1 \\ \alpha_2 \\ \alpha_3 \end{pmatrix}$，$(A, d) = \begin{pmatrix} a_1 & b_1 & c_1 & d_1 \\ a_2 & b_2 & c_2 & d_2 \\ a_3 & b_3 & c_3 & d_3 \end{pmatrix} = \begin{pmatrix} \alpha_1 \\ \alpha_2 \\ \alpha_3 \end{pmatrix}$，则三平面的位置关系完全由 $r(A)$ 和 $r(A, d)$ 决定
   - （1）线性方程组有解
     - a. 若 $r(A) = r(A, d) = 3$，则线性方程组有唯一解，即三平面相交于一点
     - ![liear8_1](/images/mathematic/linear8_1.png)
     - b. 若 $r(A) = r(A ,d) = 2$，则线性方程组有无穷多解，且解中含有一个参数（自由变量），即三平面相交于一条直线。其中又分成两种小情况：当 $\alpha_1, \alpha_2, \alpha_3$ 两两线性无关时（比如 $(1, 0), (0, 1), (1, 1)$），三平面互异；当 $\alpha_1, \alpha_2, \alpha_3$ 存在两个线性相关（成比例）时，三平面中有两个平面重合（$\alpha_1, \alpha_2, \alpha_3$ 是三个平面的法向量，法向量成比例 + 两平面相交 = 两平面重合）
     - ![liear8_2](/images/mathematic/linear8_2.png)
     - c. 若 $r(A) = r(A, d) = 1$，则线性方程组有无穷多解，且解中含有两个参数（自由变量），故解集合构成一个平面，即 $\textcolor{red}{三平面重合}$
     - ![liear8_3](/images/mathematic/linear8_3.png)
   - （2）线性方程组无解
     - a. $r(A)= 2,  r(A, d) = 3$，则线性方程组无解，即三平面无公共交点。其中又分成两种小情况：当 $\alpha_1, \alpha_2, \alpha_3$ 两两线性无关时，三平面两两相交形成一个三棱柱；当 $\alpha_1, \alpha_2, \alpha_3$ 存在两个线性相关（成比例）时，三平面中有两个面平行，另一个平面和这两个面相交
     - ![liear8_4](/images/mathematic/linear8_4.png)
     - b. 若 $r(A) = 1, r(A, d) = 2$，则线性方程组无解，即三平面无公共交点。由于 $r(A) = 1$，故三平面平行或重合。又 $r(A, d) = 2$，故三平面中至少有两个互异
     - ![liear8_5](/images/mathematic/linear8_5.png)



## 5. 特征值与特征向量（相似理论）

> ⚠️ 线代的核心内容，考试重难点

1. **特征值与特征向量的定义**
   - 设 $A$ 为 $n$ 阶矩阵，若 $A\alpha = \lambda \alpha(\alpha \ne 0)$，则称 $\lambda$ 是 $A$ 的特征值，$\alpha$ 是 $A$ 的属于 $\lambda$ 的特征向量

> ⚠️ 易错点：特征向量不能为 0


2. **求特征值与特征向量**
   - （1）方法一（具体型）
     - a. 解方程 $|\lambda E - A| = 0$，求得特征值 $\lambda$（这里计算的时候千万别把正负号写错写漏了，所以要养成解题好习惯：先写对角，再添 $-A$）
     - b. 解方程组 $(\lambda E - A) \alpha = 0$，得特征向量 $\alpha$
   - （2）方法二（抽象型）：利用定义 $A \alpha = \lambda \alpha(\alpha \ne 0)$


> 💡 分块矩阵的特征方程：对于分块矩阵 $\begin{pmatrix} A & O \\ O & B \end{pmatrix}$，其中 $A$ 为 $m$ 阶矩阵，$B$ 为 $n$ 阶矩阵，则其特征方程为 $\begin{vmatrix} A - \lambda_m E & O \\ O & B - \lambda_n E \end{vmatrix} = |A - \lambda_m E| |B - \lambda_n E|$

3. **用特征值命题**
   - （1）$\lambda_0$ 是 $A$ 的特征值 $\Leftrightarrow |\lambda_0 E - A| = 0$（建立方程求参数或证明行列式 $|\lambda_0 E - A| = 0$）；$\lambda_0$ 不是 $A$ 的特征值 $\Leftrightarrow |\lambda_0 E -A| \ne 0$（矩阵可逆；满秩）
   - ⭐⭐（2）若 $\lambda_1, \lambda_2, \cdots, \lambda_n$ 是 $A$ 的 $n$ 个特征值，则 $\begin{cases} |A| = \lambda_1 \lambda_2 \cdots \lambda_n \\ tr(A) = \lambda_1 + \lambda_2 + \cdots + \lambda_n \end{cases}$（矩阵的迹 $tr(A)$ 是矩阵对角线元素之和）
   - ⭐⭐⭐⭐⭐（3）重要结论：
     - a. 记住下表（表中 $\lambda$ 在分母上的，设 $\lambda \ne 0$）
     - ![problem2](/images/mathematic/linear2.png)
     - b. $f(x)$ 为 $\textcolor{red}{加法、数乘、高次幂、逆、伴随} $ 组成的多项式，若矩阵 $A$ 满足 $f(A) = O$，$\lambda$ 是 $A$ 的任一特征值，则 $\lambda$ 满足 $f(\lambda) = 0$
     - c. 虽然 $A^T$ 的特征值与 $A$ 相同，但特征向量不再是 $\xi$，要单独计算才能得出
     - d. $A^*$ 的特征值其一：$\dfrac{|A|}{\lambda_1} = \dfrac{\lambda_1 \lambda_2 \cdots \lambda_n}{\lambda_1} = \lambda_2 \lambda_3 \cdots \lambda_n$（即剩余特征值之积）
     - e. $A^2 = A \Leftrightarrow A^2 - A = 0 \Leftrightarrow\lambda^2 - \lambda = 0 \Leftrightarrow$ 特征值只能为 0 或 1
     - f. 上（下）三角矩阵、主对角矩阵的特征值为主对角线元素
     - g. 设 $A$ 为 $n$ 阶矩阵，则 $aA + bE(a\ne 0)$ 不可逆 $\Leftrightarrow |aA+bE| = 0 \Leftrightarrow \lambda = - \dfrac{b}{a}$ 为 $A$ 的特征值

> 💡 小结论：设 $A, B$ 为 $n$ 阶矩阵，则 $AB$ 与 $BA$ 有相同的行列式、$\textcolor{red}{特征值}$、迹【24 考研选择】
>
> 证明：
> - （1）设 $\alpha$ 为 $AB$ 的特征值为 $\lambda$ 的特征向量，则 $(AB)\alpha = \lambda\alpha$，得 $(BA)(B\alpha) = \lambda(BA)$
> - （2）若 $B\alpha \ne 0$，则 $\lambda$ 为 $BA$ 的特征值
> - （3）若 $B\alpha = 0$，则 $(AB)\alpha = \lambda \alpha = 0$，得 $\lambda = 0$，故 $|AB| = |A||B| = |BA| = 0$，从而 $\lambda = 0$ 为 $BA$ 的特征值
> - （4）综上所述，$AB$ 的特征值均为 $BA$ 的特征值
> - （5）同理，$BA$ 的特征值均为 $AB$ 的特征值，故 $AB$ 与 $BA$ 有相同的特征值

> 💡 小结论：设 $n$ 阶矩阵 $A, B$ 均可相似对角化，则 $AB = BA \Leftrightarrow A, B$ 有相同的特征向量（证明过程和上面的小结论类似）

> 💡 小结论：只要 $A$ 满足的二次方程 $f(A) = 0$ 存在两个互异的根，则 $A$ 一定可以相似对角化
>
> 证明：
> - （1）例如 $f(A) = A^2 - \lambda_1 A - \lambda_2 A + \lambda_1 \lambda_2 E = (A - \lambda_1E)(A - \lambda_2E) = O$（假设 $\lambda_1 \ne \lambda_2$）
> - （2）则 $ f(\lambda) = \lambda^2 - \lambda_1 \lambda - \lambda_2 \lambda + \lambda_1\lambda_2= O$，解得 $\lambda = \lambda_1$ 或 $\lambda_2$
> - （3）$r(A - \lambda_1 E) + r(A - \lambda_2 E) = r(A - \lambda_1 E) + r(\lambda_2 E - A) \ge r(r - \lambda_1 E + \lambda_2 E - A) = r((\lambda_2 - \lambda_1)E) = n$
> - （4）$(A - \lambda_1E)(A - \lambda_2E) = O \Rightarrow r(A - \lambda_1 E) + r(A - \lambda_2 E) \le n$
> - （5）所以 $r(A - \lambda_1 E) + r(A - \lambda_2 E) = n$
> - （6）所以 $A$ 线性无关的特征向量个数为 $n - r(A - \lambda_1 E) + n - r(A - \lambda_2 E) = n$
> - （7）所以 $A$ 可以相似对角化

4. **用特征向量命题**
   - （1）$\xi(\ne 0)$ 是 $A$ 的属于 $\lambda_0$ 的特征向量 $\Leftrightarrow \xi$ 是 $(\lambda_0 E - A) x = 0$ 的非零解
   - （2）重要结论
     - a. $k$ 重特征值 $\lambda$ 至多只有 $k$ 个线性无关的特征向量（直接用，不用证明）
     - b. 不同特征值的特征向量线性无关
     - c. 不同特征值的特征向量之和不是特征向量
     - d. 若 $\xi_1, \xi_2$ 是 $A$ 的属于同一特征值 $\lambda$ 的特征向量，则 $k_1 \xi_1 + k_2 \xi_2$（$k_1,k_2$ 不同时为 0）仍是 $A$ 的属于特征值 $\lambda$ 的特征向量
     - e. 若 $\xi_1, \xi_2$ 是 $A$ 的属于不同特征值 $\lambda_1, \lambda_2$ 的特征向量，则当 $k_1\ne 0, k_2\ne 0$ 时，$k_1\xi_1 + k_2\xi_2$ 不是 $A$ 的特征向量（常考 $k_1 = k_2 = 1$ 的情形）


> 💡 这些条件在告诉你特征值或特征向量
> - （1）$AB = O \Rightarrow A[\beta_1, \beta_2, \cdots, \beta_n] = [0, 0, \cdots, 0]$，即 $A\beta_i = 0 \beta_i(i = 1, 2, \cdots, n)$，若其中 $\beta_i$ 均为非零列向量，则 $\beta_i$ 为 $A$ 的属于 $\lambda = 0$ 的特征向量（这条结论还可以翻译成：齐次方程组 $Ax = 0$ 的自由解向量对应了 $A$ 特征值为 0 时的特征向量）
> - （2）$AB = C \Rightarrow A[\beta_1, \beta_2, \cdots, \beta_n] = [\gamma_1, \gamma_2, \cdots, \gamma_n] $，若 $[\gamma_1, \gamma_2, \cdots, \gamma_n] = [\lambda_1\beta_1, \lambda_2\beta_2, \cdots, \lambda_n\beta_n]$，$A\beta_i = \lambda_i\beta_i$，其中 $\gamma_i = \lambda_i \beta_i$，$\beta_i$ 为非零列向量，则 $\beta_i$ 为 $A$ 的属于 $\lambda_i$ 的特征向量
> - （3）若 $A$ 的各行元素之和均为 $k$，则 $A \begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{pmatrix} = k\begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{pmatrix} \Rightarrow k $ 是特征值，$\begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{pmatrix}$ 是 $A$ 的属于 $k$ 的特征向量（万一题目里说 $\textcolor{red}{“各列元素之和”}$，有两种方案：转置一下变成 “各行元素之和” ；直接左边乘行向量 $(1, 1, \cdots, 1)$）（再来个万一，题目里出现 $A_11 + A_21 + A_31$，要能想到这是 $A^*$ 第一行之和）


5. **相似矩阵**
   - （1）定义：设 $A, B$ 是两个 $n$ 阶方阵，若存在 $n$ 阶可逆矩阵 $P$ 使得 $P^{-1}AP = B$，则称 $A$ 相似于 $B$，记为 $A～B$
   - （2）相似矩阵的性质
     - a. $A～A$
     - b. 若 $A～B$，$B～C$，则 $A～C$
     - c. 若 $A～B$，则 $A, B$ 有相同的行列式、秩、特征方程、特征值、迹（或 $|\lambda E - A| = |\lambda E - B|$）、$r(\lambda E - A) = r(\lambda E - B)$，且 $A, B$ 的各阶主子式之和相等
   - （3）重要结论：
     - a. $A～B \Rightarrow A^T ～B^T, A^{-1} ～B^{-1}, A^*～B^*$（后面两个要求 $A$ 可逆）
     - b. $A～B \Rightarrow A^m ～B^m, f(A)～f(B)$（这里同样包括加法、数乘、高次幂、逆、伴随，但多项式中 $\textcolor{red}{不包含转置}$）
     - c. $A～B, B～\Lambda \Rightarrow A～\Lambda$
     - d. $A～\Lambda, B～\Lambda \Rightarrow A～B$
     - e. $A～C, B～D \Rightarrow \begin{pmatrix} A & O \\ O & B \end{pmatrix} ～ \begin{pmatrix} C & O \\ O & D \end{pmatrix}$

> ⚠️ 矩阵相似和等价的区别
> - （1）矩阵等价只要求两个矩阵同型即可，而矩阵相似要求两个矩阵为同型$\textcolor{red}{方阵}$
> - （2）相似必等价，等价不一定相似
> - （3）等价只要求 $PAQ = B$（$P, Q$ 均可逆），但相似要求 $P^{-1}AP = B$


6. **A 的相似对角化**
   - （1）定义：$P^{-1} A P = \Lambda$（这里 $P$ 的每一个列向量都是其所在列的特征值一一对应的特征向量，即 $P = [\xi_1, \xi_2, \cdots, \xi_n], \Lambda = \begin{pmatrix} \lambda_1 & & & \\ & \lambda_2 & & \\ & & \ddots & \\ & & & \lambda_n \end{pmatrix}$）
   - （2）充要条件
     - a. $A$ 有 $n$ 个线性无关的特征向量 $\Leftrightarrow A～\Lambda$
     - b. $n_i = n - r(\lambda_i E - A) \Leftrightarrow A～\Lambda$（$n_i$ 重根下有 $n_i$ 个线性无关的特征向量，就可以相似对角化）
   - （3）充分条件：
     - a. $A$ 是实对称矩阵 $\Rightarrow A～\Lambda$
     - b. $A$ 有 $n$ 个互异特征值 $\Rightarrow A～\Lambda$
     - c. $A^2 = A \Rightarrow A～\Lambda$
     - d. $A^2 = E \Rightarrow A～\Lambda$
     - e. $r(A) = 1$ 且 $tr(A) \ne 0 \Rightarrow A～\Lambda$
   - （4）必要条件：
     - a. $A～\Lambda \Rightarrow r(A)$ 等于非零特征值的个数（重根按重数算）

> 💡 相似对角化的高次幂
> - 若 $P^{-1} A P = \Lambda$，则 $A = P\Lambda P^{-1}$，从而 $A^n = (P\Lambda P^{-1})(P\Lambda P^{-1}) \cdots (P\Lambda P^{-1}) = P\Lambda^n P^{-1}$

7. **实对称矩阵**
   - （1）若 $A$ 为实对称矩阵，则
     - a. 特征值均为实数，特征向量均为实向量
     - b. 不同特征值对应的特征向量正交（即 $\lambda_1 \ne \lambda_2 \Rightarrow \xi_1 \perp \xi_2 \Rightarrow (\xi_1, \xi_2) = 0$，建方程）
     - c. 可 $\textcolor{red}{正交}$ 相似对角化（即存在正交矩阵 $P$ 使 $P^{-1} A P = P^T AP = \Lambda$）
   - （2）实对称矩阵相似对角化的基本步骤：
     - a. 求特征值
     - b. 求特征值对应的特征向量
     - c. 将特征向量正交化、单位化
     - d. 令 $Q$ 为单位正交化之后的特征向量组成的矩阵，则 $Q$ 为正交矩阵，且 $Q^{-1}AQ = Q^T AQ = \Lambda$
   - （3）实对称矩阵的分解定理：设 $A$ 为 $n$ 阶实对称矩阵，$\gamma_1, \gamma_2, \cdots, \gamma_n$ 为矩阵 $A$ 分别属于特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$ 的单位正交的特征向量，则 $A = \lambda_1 \gamma_1 \gamma_1^T + \lambda_2 \gamma_2 \gamma_2^T + \cdots + \lambda_n \gamma_n \gamma_n^T$。特别的，若 $r(A) = 1$，则 $A = tr(A) \gamma_1 \gamma_1^T$



> 💡 正交比线性无关更强一些

8. **正交矩阵**
   - （1）定义：设 $A$ 是 $n$ 阶方阵，若 $A^TA = E$ 则 $A$ 为正交矩阵
   - （2）性质：
     - a. $A^TA = E$
     - b. $A^T = A^{-1}$
     - c. $A$ 的行向量和列向量都是标准正交基（也就是每一行每一列的模都为 1，且相互正交）
     - d. 若 $P, Q$ 均为正交矩阵，则 $PQ$ 还是正交矩阵【由此性质引出一类题型：已知两个二次型 $f(x_1, x_2), g(y_1, y_2)$，判断 $f$ 能否正交变换到 $g$，若能则求出正交矩阵 $Q$，思路是先相似对角化，然后求出两个正交矩阵 $Q_1, Q_2$，最后传递一下得到 $Q$】
     - e. 若 $Q$ 为正交矩阵，则 $Q^{-1}, Q^T, Q^*$ 还是正交矩阵
   - （3）正交矩阵的特征值 $\lambda \in \{1, -1\}$
   - （4）若 $A$ 为正交矩阵，则 $A^T A= E$
     - $\Leftrightarrow A^{-1} = A^T$
     - $\Leftrightarrow A$ 由规范正交基组成
     - $\Leftrightarrow A^T$ 是正交矩阵
     - $\Leftrightarrow A^{-1}$ 是正交矩阵
     - $\Leftrightarrow A^*$ 是正交矩阵
   - （5）若 $A, B$ 为同阶正交矩阵，则 $AB$ 为正交矩阵（$A+B$ 不一定）
   - （6）正交矩阵 $Q$ 的计算方法：
     - a. 求 $A$ 的 $n$ 个特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$
     - b. 求 $A$ 的 $n$ 个线性无关的特征向量 $\alpha_1, \alpha_2, \cdots, \alpha_n$
     - c. 将不同特征值的特征向量分别施密特正交化，得 $\gamma_1, \gamma_2, \cdots, \gamma_n$，得正交矩阵 $Q = (\gamma_1, \gamma_2, \cdots, \gamma_n)$

> 💡 小技巧：对于 3 阶矩阵 $A$，已经得到正交矩阵中的两个列向量 $\alpha_1, \alpha_2$，则 $\alpha_3 = \alpha_1 \times \alpha_2 = \begin{vmatrix} i & j & k \\ & \alpha_1^T & \\ & \alpha_2^T & \end{vmatrix}$

> 💡 例题一道（注意 $A$ 右乘 $\alpha$ 和 $\beta$ 的小技巧）
>
> ![problem3](/images/mathematic/linear3.png)

## 6. 二次型
1. **二次型及其表示**
   - （1）含有 $n$ 个变量：$x_1, x_2, \cdots, x_n$ 的二次齐次函数 $f(x_1, x_2, \cdots, x_n) = a_{11}x_1^2 + a_{22}x_2^2 + \cdots + a_{nn}x_n^2 + 2a_{12}x_1x_2 + 2a_{13}x_1x_3 + \cdots + 2a_{n-1, n}x_{n-1}x_n$，称为 $n$ 元二次型
   - （2）也可以写成 $f(x_1, x_2, \cdots, x_n) = \sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n}a_{ij} x_i x_j$ 的形式。出现该形式时，不要犹豫，立即推：记 $x = (x_1, x_2, \cdots, x_n)^T$，$A = (a_{ij})_{n\times n}$，则 $f(x_1, x_2, \cdots, x_n) = x^T A x$（需要注意的是，此时 $A$ 矩阵还不能称为二次型矩阵，若增加条件 $a_{ij} = a_{ji}$，也即 $A^T = A$，此时实对称矩阵 $A$ 被称为二次型矩阵，且 $r(A)$ 是二次型的秩）

> 💡 若上述 2 中得到的 $A$ 不是实对称的，那么可以通过 $\dfrac{A+A^T}{2}$ 调整成二次型矩阵，从而二次型矩阵的秩也应当是调整后的矩阵的秩

> 💡 遇到 $f(x_1, x_2, x_3) = (x_1 - x_2 + x_3)^2 + (x_2 + x_3)^2 + (x_1 + ax_3)^2$ 这种比较麻烦的形式，直接令 $y_1 = x_1 - x_2 + x_3 , y_2 = x_2 + x_3 , y_3 = x_1 + ax_3$，那么有 $\begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} 1 & -1 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & a \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = B \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$，那么根据 $f = y_1^2 + y_2^2 + y_3^2 = y^T y = (Bx)^TBx = x^TB^TBx$，即二次型矩阵等于 $B^TB$

2. **矩阵合同**
   - （1）定义：设 $A, B$ 为 $n$ 阶方阵，若存在可逆矩阵 $C$ 使得 $C^T AC = B$，则称 $A$ 与 $B$ 合同，记作 $A\simeq B$
   - （2）合同的性质：
     - a. $A\simeq A$
     - b. 若 $A\simeq B$，则 $B\simeq A$
     - c. 若 $A \simeq B, B\simeq C$，则 $A \simeq C$
     - d. $r(A) = r(B)$
   - （3）合同的充要条件：
     - a. 二次型 $x^TAx$ 与 $x^TBx$ 有相同的正、负惯性指数
     - b. $A, B$ 有相同的正、负特征值个数
   - （4）合同的充分条件：
     - a. 若 $n$ 阶实对称矩阵 $A$ 与 $B$ 相似，则 $A$ 与 $B$ 合同（但并不意味着所有既相似又合同的矩阵都是实对称矩阵）
   - （5）合同的必要条件
     - a. 若 $n$ 阶实对称矩阵 $A$ 与 $B$ 合同，则 $A$ 与 $B$ 等价
   - ![linear5](/images/mathematic/linear5.png)
   

> 💡 上面图中
> - （1）等价只要求 $r(A) = r(B)$，条件比较弱
> - （2）相似，意味着存在可逆矩阵 $P$ 使得 $B = P^{-1} AP$
> - （3）合同，意味着存在矩阵 $Q$ 使得 $B = Q^T AQ$
> - （4）既相似又合同，意味着同时存在矩阵 $Q$ 和可逆矩阵 $P$，使 $B = Q^TAQ = P^{-1}AP$
> - （5）既相似又合同里面，有一种特殊情况 “正交变换”，即存在正交矩阵 $Q$，使得 $B = Q^TAQ = Q^{-1}AQ$（即正交变换 $\Rightarrow$ 既合同又相似）


3. **二次型的标准形、规范形**
   - （1）定义：
     - a. 若二次型中只含有平方项，没有交叉项（即所有交叉项的系数均为 0），即形如 $d_1x_1^2 + d_2x_2^2 + \cdots + d_nx_n^2$ 的二次型称为标准形（系数不唯一）
     - b. 若标准型中，系数 $d_i(i=1, 2, \cdots, n)$ 仅为 $1, -1, 0$，即形如 $x_1^2 + \cdots + x_p^2 - x_{p-1}^2 - \cdots x_{p+q}^2$ 的二次型称为规范形（系数唯一）
   - （2）常用定理
     - a. 任何二次型均可以通过配方法（作可逆线性变换）化为标准形及规范形，用矩阵语言表述：任何实对称矩阵 $A$ 必存在可逆矩阵 $C$ 使得 $C^TAC = \Lambda$（这个对角矩阵是不唯一的）
     - b. 任何二次型也可以通过正交变换化成标准形，用矩阵语言表述即是任何实对称矩阵 $A$ 一定存在正交矩阵 $Q$ 使得 $Q^{-1}AQ = Q^TAQ = \Lambda$（正交矩阵不唯一，但这个对角矩阵是唯一的，表征的是特征值的排布，是沟通矩阵相似与二次型的桥梁，高级双料特工）
   - （3）坐标变换和惯性定理
     - a. 坐标变换：$\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} c_{11} & c_{12} & c_{13} \\ c_{21} & c_{22} & c_{23} \\ c_{31} & c_{32} & c_{33} \end{pmatrix}  \begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix}$（其中 $C$ 是可逆矩阵）
     - b. 无论选取什么样的可逆线性变换，将二次型化成标准形或规范形，其正项个数 $p$，负项个数 $q$ 都是不变的，$p$ 称为正惯性指数，$q$ 称为负惯性指数
     - c. 两个二次型（或者叫实对称矩阵）合同的充要条件是有相同的正、负惯性指数，或有相同的秩及正（或负）惯性指数，或有相同的正、负特征值个数




> 💡 $C^TAC$ 变换称为合同变换，而矩阵相似那个地方的变换是 $P^{-1}AP$，不要搞混，一定要区分开来

4. **配方法化标准形**（拉格朗日配方法）
   - （1）必备工具
     - a. $(a + b)^2 = a^2 + 2ab + b^2$
     - b. $(a + b + c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc$
   - （2）标准配方过程：第一步先把 $x_1$ 全部配完，然后再配 $x_2, x_3$
   - （3）例题：
     - ![problem4](/images/mathematic/linear4.png)
   - （4）其他常用技巧：
     - a. 遇到大量交叉项或交叉项不好处理的时候，可以通过 $\begin{cases} x_1 = y_1 + y_2 \\ x_2 = y_1 - y_2 \\ x_3 = y_3 \end{cases}$ （根据实际情况灵活构造）来构造平方项，然后再用 $z = By$ 来构造出规范形和 $x = Cz$

5. **正交变换法**（和配方法的地位不相上下，但没法像配方法一样一步配到规范形）
   - 基本步骤对于 $f = x^T Ax$
   - （1）求 $A$ 的特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$
   - （2）求 $A$ 的对应于特征值 $\lambda_1, \lambda_2, \cdots, \lambda_n$ 的特征向量 $\xi_1, \xi_2, \cdots, \xi_n$
   - （3）将 $\xi_1, \xi_2, \cdots, \xi_n$ 正交化（如果需要的话）、单位化为 $\eta_1, \eta_2, \cdots, \eta_n$
   - （4）令 $Q = [\eta_1, \eta_2, \cdots, \eta_n]$ 则 $Q$ 为正交矩阵且 $Q^{-1}AQ = Q^TAQ = \Lambda$
   - 于是 $f = x^TAx \rightarrow (Qy)^T A(Qy) = y^TQ^T AQ y = y^T \Lambda y$
   - 注：
     - a. 正交变换 $x = Qy$ 下得到的 $\Lambda$ 的主对角线元素是 $A$ 的特征值
     - b. 应用正交变换法只能将二次型化为标准形，而不能化为规范形。一般来说，化二次型为规范形还需要进行一步线性变换 $y = Cz$

> 💡 这里回想一下之前学过的性质：
> - （1）正交矩阵的特征值只能是 1 或 -1（从而可以一步化到规范形）
> - （2）如果 $A,B$ 均是实对称矩阵，且 $A～B$，则一定存在一个正交矩阵 $Q$ 使得 $Q^TAQ = B$


> 💡 平方和形式（即 $f = (...)^2 + (...)^2 + \cdots$）的二次型具有哪些性质（设 $y = Bx$）
> - （1）$f \ge 0$，那么特征值一定 $\ge 0$，也就无负惯性指数（因为总能找到一个 $x = Qy $ 这样一个正交变换使 $f = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \lambda_3 y_3^2$）
> - （2）正惯性指数 $p = r(B)$（因为秩 $r = p + q$，无负惯性指数，即 $q = 0$，$r(B^TB) = p = r(B)$）
> - （3）当 $r(B)$ 满秩时，$p = n$，正惯性指数拉满了，那么 $f$ 正定
>
> ⚠️ 不过要注意，$(...)^2 - (...)^2$ 就不行了

6. **合同变换法**（一般用在实对称矩阵上）
   - $\begin{pmatrix} A \\ E \end{pmatrix}$ 通过行列成对（每一次行变换之后，立即进行一次相同的列变换）的初等变换变为 $C^T\begin{pmatrix} A \\ E \end{pmatrix}C = \begin{pmatrix} \Lambda \\ C \end{pmatrix}$，其中 $\Lambda = \begin{pmatrix} d_1 & & & \\ & d_2 & &  \\ & & \ddots & \\ & & & d_n \end{pmatrix}$
   - 得到标准形 $f = d_1 y_1^2 + d_2y_2^2 + \cdots + d_ny_n^2$ 与所作的可逆线性变换 $x = Cy$
   - 该方法除了可以变换为对角矩阵，还可以变换为其他和 $A$ 相似的实对称矩阵



7. **正定二次型及其判别**
   - （1）定义：$n$ 元二次型 $f(x_1, x_2, \cdots, x_n) = x^TAx$，若对任意 $x = [x_1, x_2, \cdots, x_n]^T \ne 0$，均有 $x^TAx  > 0$，则称 $f$ 为正定二次型，称二次型对应的矩阵 $A$ 为正定矩阵（翻译：$x\ne 0$ 时 $f$ 一定是正的，$f=0$ 只有零解）
   - （2）二次型正定的充要条件
     - a. 对任意 $x\ne 0$ 有 $x^TAx > 0$
     - b. $f$ 的正惯性指数 $p = n$
     - c. 存在可逆矩阵 $D$ 使得 $A=  D^TD = ||D||^2$
     - d. $A\simeq E$（即 $A$ 与 $E$ 合同）
     - e. $A$ 的特征值都大于 0
     - f. $A$ 的顺序主子式均大于 0
   - （3）二次型正定的必要条件
     - a. $a_{ii} > 0$
     - b. $|A| > 0$

> 💡 小结论：设 $A$ 为 $m$ 阶实对称矩阵，$B$ 为 $n$ 阶实对称矩阵，则 $\begin{pmatrix} A & O \\ O & B \end{pmatrix}$ 为正定矩阵 $\Leftrightarrow A, B$ 均为正定矩阵【2005，数三】

> 💡 正定矩阵的分解：设 $A$ 为 $n$ 阶正定矩阵，则对任意正整数 $m$，存在 $n$ 阶正定矩阵 $B$，使得 $A = B^m$【2021，数一】
>
> 具体做法：
> - （1）正交相似对角化 $Q^TAQ = \Lambda \Leftrightarrow A = Q\Lambda Q^T = Q \begin{pmatrix} \lambda_1 & & & \\ & \lambda_2 & & \\ & & \ddots & \\ & & & \lambda_n \end{pmatrix}Q^T$
> - （2）取 $B = Q \begin{pmatrix} \sqrt[m]{\lambda_1} & & & \\ & \sqrt[m]{\lambda_2} & & \\ & & \ddots & \\ & & & \sqrt[m]{\lambda_n} \end{pmatrix}Q^T$

8. **二次型的最值**（一种比较特殊的题型）
    - （1）设 $f(x_1, x_2, \cdots, x_n) = x^TAx$，二次型矩阵 $A$ 的特征值为 $\lambda_1, \lambda_2, \cdots, \lambda_n$，且 $\lambda_1 \le \lambda_2 \le \cdots \le \lambda_n$，则 $\lambda_1 x^Tx \le x^TAx \le \lambda_n x^Tx$
    - （2）证明：
      - a. 存在正交变换 $x = Qy$ 使得 $x^TAx = \lambda_1y_1^2 + \lambda_2y_2^2 + \cdots + \lambda_ny_n^2$
      - b. $x^Tx = (Qy)^T(Qy) = y^TQ^TQy = y^Ty = y_1^2 + y_2^2 + \cdots + y_n^2$
      - c. 又 $\lambda_1 \le \lambda_2 \le \cdots \le \lambda_n$，故 ...



> 💡 二次型奇技淫巧大观（❓）
> - （1）$f(x_1, x_2, x_3) = 2(a_1x_1 + a_2x_2 + a_3 x_3)^2 = 2 (x_1, x_2, x_3) \begin{pmatrix} a_1 \\ a_2 \\ a_3 \end{pmatrix}(a_1, a_2, a_3) \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$【2013，数一】
> - （2）设 $x = (x_1, x_2, \cdots, x_n)^T$ 与 $y = (y_1, y_2, \cdots, y_n)^T$，则 $(x^T, y^T) \begin{pmatrix} A & O \\ O & B \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = x^TAx + y^TBy$【2005，数三】


> 💡 $AA^T$ 矩阵结论大观（可以从头到尾推一遍）
> - 设 $A_{m\times n}$，则 $AA^T_{m\times m}$，若 $r(A) = m$
> - $\Leftrightarrow r(AA^T) = r(A) = m$ 满秩
> - $\Leftrightarrow |AA^T| \ne 0$
> - $\Leftrightarrow AA^T$ 可逆
> - $\Leftrightarrow AA^T$ 与 $E_{m\times m}$ 等价（等价矩阵的充要条件是秩相等）
> - $\Leftrightarrow AA^T$ 的行（列）向量组线性无关
> - $\Leftrightarrow$ 齐次方程组 $AA^T x= 0$ 只有零解
> - $\Leftrightarrow$ 非齐次方程组 $AA^T x = b$ 有唯一解
> - $\Leftrightarrow AA^T$ 特征值均不为零（因为 $|AA^T| = \lambda_1\lambda_2 \cdots \lambda_n\ne 0$）
> - $\Rightarrow AA^T$ 为实对称矩阵
> - $\Rightarrow AA^T$ 可（正交）相似对角化
> - $\Leftrightarrow AA^T$ $\textcolor{red}{正定}$（证明：$x^TAA^Tx = (A^Tx)^T(A^Tx) \ge 0$；由 $r(A) = r(A^T) = m$ 知 $A^Tx = 0$ 只有零解；$x \ne 0$ 时 $A^Tx \ne 0$，从而 $(A^Tx)^T(A^Tx) > 0$，$AA^T$ 正定）
> - $\Leftrightarrow AA^T$ 与 $E_{m\times m}$ 合同（正、负惯性指数相等）


9. **二次型的几何意义**（2026 刚考过）
    - （1）方法：经过 $\textcolor{red}{正交变换}$（刚体旋转，不改变二次型的几何形状），将二次型化为标准型（⚠️ 只能用正交变换！）
    - （2）常见的二次曲面（选择题选项来了）
      - a. （三正）椭球面：$\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} + \dfrac{z^2}{c^2} = 1$
      - b. （两正）柱面：$\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1$
      - c. （两正一负）单叶双曲面：$\dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} - \dfrac{z^2}{c^2} = 1$
      - d. （一正两负）双叶双曲面：$\dfrac{x^2}{a^2} - \dfrac{y^2}{b^2} - \dfrac{z^2}{c^2} = 1$