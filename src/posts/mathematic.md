---
title: Coldrain 的 27 考研数一高数强化阶段拾遗
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


## 1. 函数极限与连续

1. **函数奇偶性相关结论**
   - （1）$f(x) + f(-x)$ 必是偶函数
   - （2）$f(x) - f(-x)$ 必是奇函数
   - （3）$f(\varphi(x))$ 内偶则偶，内奇同外
   - （4）求导一次，奇偶性互换
   - （5）$f(x)$ 奇（偶）$\Rightarrow \int_{0}^{x} f(t)dt$ 偶（奇）
   - （6）对任意 $x$、$y$，都有 $f(x+y) = f(x) + f(y)$，则 $f(x)$ 为奇函数

2. **函数极限的定义**：$\lim\limits_{x \to x_0} f(x) =A \Leftrightarrow \forall \epsilon > 0, \exist \delta > 0$，当 $0<|x-x_0|<\delta$ 时，有 $|f(x) -A| < \epsilon$


3. **无穷小的比阶**
   - （1）高阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的高阶无穷小
   - （2）低阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = \infty$，则 $\alpha(x)$ 为 $\beta(x)$ 的低阶无穷小
   - （3）同阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = c \ne 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的同阶无穷小
   - （4）等价无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = 1$，则 $\alpha(x)$ 为 $\beta(x)$ 的等价无穷小
   - （5）$k$ 阶无穷小：$\lim \dfrac{\alpha(x)}{[\beta(x)]^k} = c \ne 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的 $k$ 阶无穷小

4. $x\to 0$ **时常用等价无穷小**
   - $\sin x ～ x$
   - $\tan x ～ x$
   - $\arcsin x ～ x$
   - $\arctan x ～ x$
   - $\ln (1+x) ～ x$
   - $e^x -1 ～ x$
   - $\alpha^x - 1 ～ x \ln(\alpha)$
   - $1 - \cos x ～ \dfrac{1}{2} x^2$
   - $(1+x)^{\alpha} - 1 ～ \alpha x$

> ⚠️ **差值型表达式不能随便用等价无穷小替换！**（1000a 第一讲 19 题）
>
> 等价无穷小一般适用于**乘除结构**，比如 $\lim\limits_{x \to 0} \dfrac{\sin x}{x} = 1$，在此式子中可以使用 $\sin x ～ x$
>
> 但对于 $\dfrac{e^x + xe^x}{e^x - 1} - \dfrac{1}{x}$，该式子为 “两个无穷大量相减”，最后结果依赖于 $e^x - 1$ 的二阶项，故必须先通分才能使用等价无穷小


5. **泰勒公式**：设 $f(x)$ 在 $x=0$ 处 $n$ 阶可导，则有 $f(x) = f(0) + f'(0)x + \dfrac{f''(0)}{2!} x^2 + ... + \dfrac{f^{(n)}(0)}{n!} x^n = \sum\limits_{k=0}^{n} \dfrac{f^{(k)}(0)}{k!} (x-0)^k$

> 💡 常用泰勒展开式
> - $\sin x = x - \dfrac{x^3}{3!} + o(x^3)$
> - $\cos x = 1 - \dfrac{x^2}{2!} + \dfrac{x^4}{4!} + o(x^4)$
> - $\arcsin x = x + \dfrac{x^3}{3!} + o(x^3)$
> - $\tan x = x + \dfrac{x^3}{3} + o(x^3)$
> - $\arctan x = x - \dfrac{x^3}{3} + o(x^3)$
> - $\ln(1+x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} + o(x^3)$
> - $e^x = 1 + x + \dfrac{x^2}{2!} + \dfrac{x^3}{3!} + o(x^3)$
> - $(1+x)^{\alpha} = 1 + \alpha x + \dfrac{\alpha(\alpha - 1)}{2!}x^2 + o(x^2)$

> 💡 两函数乘积的泰勒展开，等于各自泰勒展开相乘，即 $f(x)g(x)$ 的泰勒展开等于 $f(x)$ 的泰勒展开乘 $g(x)$

6. **间断点**
   - （1）可去间断点：$\lim\limits_{x\to x_0} f(x)= A \ne f(x_0) $（$f(x_0)$ 甚至可以无定义）
   - （2）跳跃间断点：$\lim\limits_{x\to x_0^-} f(x) \ne \lim\limits_{x\to x_0^+} f(x)$
   - （3）无穷间断点：$\lim\limits_{x \to x_0} f(x) = \infty$ 或 $\lim\limits_{x \to x_0^+} f(x) = \infty$ 或 $\lim\limits_{x \to x_0^-} f(x) = \infty$
   - （4）震荡间断点：$\lim\limits_{x \to x_0} f(x)$ 震荡不存在

> 💡 前两个为**第一类间断点**，后两个为**第二类间断点**


7. **两个重要极限**
   - （1）$\lim\limits_{x\to 0} \dfrac{\sin x}{x} = 1$
   - （2）$\lim\limits_{x\to \infty} (1 + \dfrac{1}{x})^{x} = e$

## 2. 数列极限

1. **数列极限**
   - 设 $\{x_n\}$ 为一数列，若存在常数 $a$，对于任意 $\epsilon > 0$（不论它多小），总存在正整数 $N$，使得当 $n> N$ 时，$|x_n - a| < \epsilon$ 恒成立，则称常数 $a$ 是数列 $\{x_n\}$ 的极限，或者称数列 $\{x_n\}$ **收敛**于 $a$，记为：$\lim\limits_{n\to \infty} x_n=a$
   - 如果不存在这样的常数 $a$，则称数列 $\{x_n\}$ 是**发散**的

2. **数列收敛与其子列收敛的关系**
   - 若数列 $\{a_n\}$ 收敛，则其任何子列 $\{a_{n_k}\}$ 也收敛，且 $\lim\limits_{k\to \infty} a_{n_k} = \lim\limits_{n \to \infty} a_n$

3. **海涅定理**
   - 设 $f(x)$ 在去心邻域 $\dot{U}(x_0,\delta)$ 内有定义，则 $\lim\limits_{x\to x_0}f(x) = A$ 存在 $\Leftrightarrow$ 对任意 $\dot{U}(x_0,\delta)$ 内以 $x_0$ 为极限的数列 $\{x_n\}(x_n \ne x_0)$，极限 $\lim\limits_{n \to \infty} f(x) = A$ 存在
   - 如 $f(x) = \dfrac{1}{x} \sin \dfrac{1}{x}$，$x \to 0$ 时
     - （1）若取 $x_n = \dfrac{1}{n\pi} \to 0$，则 $f(x_n) = n\pi \cdot \sin(n\pi)$，故 $\lim\limits_{n\to \infty} f(x_n) = 0$
     - （2）若取 $x_n = \dfrac{1}{(2n + \frac{1}{2})\pi} \to 0, n\to\infty$，则 $f(x_n) = (2n + \dfrac{1}{2}) \to +\infty, n \to \infty$
     - （3）根据海涅定理，极限 $\lim\limits_{x\to 0} \dfrac{1}{x} \sin \dfrac{1}{x}$ 不存在且 $x\to 0$ 时 $\dfrac{1}{x} \sin \dfrac{1}{x}$ 为无界量


