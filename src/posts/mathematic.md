---
title: Coldrain 的 27 考研数一高数强化阶段拾遗（上）
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

3. **函数极限的局部保号性**
   - （1）$\lim\limits_{x\to x_0} f(x) = A < 0 \Rightarrow$ 存在 $x_0$ 的去心领域使 $f(x) < 0$
   - （2）$\lim\limits_{x\to x_0} f(x) = A > 0 \Rightarrow$ 存在 $x_0$ 的去心领域使 $f(x) > 0$

4. **无穷小的比阶**
   - （1）高阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的高阶无穷小
   - （2）低阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = \infty$，则 $\alpha(x)$ 为 $\beta(x)$ 的低阶无穷小
   - （3）同阶无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = c \ne 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的同阶无穷小
   - （4）等价无穷小：$\lim \dfrac{\alpha(x)}{\beta(x)} = 1$，则 $\alpha(x)$ 为 $\beta(x)$ 的等价无穷小
   - （5）$k$ 阶无穷小：$\lim \dfrac{\alpha(x)}{[\beta(x)]^k} = c \ne 0$，则 $\alpha(x)$ 为 $\beta(x)$ 的 $k$ 阶无穷小

5. $x\to 0$ **时常用等价无穷小**
   - $\sin x ～ x$
   - $\tan x ～ x$
   - $\arcsin x ～ x$
   - $\arctan x ～ x$
   - $\ln (1+x) ～ x$
   - $e^x -1 ～ x$
   - $\alpha^x - 1 ～ x \ln(\alpha)$
   - $1 - \cos x ～ \dfrac{1}{2} x^2$
   - $1 - \cos^a x ～ \dfrac{a}{2} x^2$
   - $(1+x)^{\alpha} - 1 ～ \alpha x$
   - $\ln (x + \sqrt{1 + x^2}) ～ x$

> ⚠️ **差值型表达式不能随便用等价无穷小替换！**（1000a 第一讲 19 题）
>
> - 等价无穷小一般适用于**乘除结构**，比如 $\lim\limits_{x \to 0} \dfrac{\sin x}{x} = 1$，在此式子中可以使用 $\sin x ～ x$
>
> - 但对于 $\dfrac{e^x + xe^x}{e^x - 1} - \dfrac{1}{x}$，该式子为 “两个无穷大量相减”，最后结果依赖于 $e^x - 1$ 的二阶项，故必须先通分才能使用等价无穷小
>
> - 更严格的条件是：对于 $\lim\limits_{x\to 0} \dfrac{f(x) \pm g(x)}{x^k}$，若想要使用无穷小等价代换，必须要求分子多项式中代换出 $x^l$，其中 $l \ge k$。如果代换不出来，请老老实实使用泰勒公式展开到至少 $k$ 阶

6. **泰勒公式**：设 $f(x)$ 在 $x=0$ 处 $n$ 阶可导，则有 $f(x) = f(0) + f'(0)x + \dfrac{f''(0)}{2!} x^2 + ... + \dfrac{f^{(n)}(0)}{n!} x^n = \sum\limits_{k=0}^{n} \dfrac{f^{(k)}(0)}{k!} (x-0)^k$

> 💡 等价无穷小是泰勒公式展开的一种特殊情况

> 💡 常用泰勒展开式
> - $\sin x = x - \dfrac{x^3}{3!} + o(x^3)$
> - $\cos x = 1 - \dfrac{x^2}{2!} + \dfrac{x^4}{4!} + o(x^4)$
> - $\arcsin x = x + \dfrac{x^3}{3!} + o(x^3)$
> - $\tan x = x + \dfrac{x^3}{3} + \dfrac{2x^5}{15} + o(x^3)$
> - $\arctan x = x - \dfrac{x^3}{3} + o(x^3)$
> - $\ln(1+x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} + o(x^3)$
> - $e^x = 1 + x + \dfrac{x^2}{2!} + \dfrac{x^3}{3!} + o(x^3) \Rightarrow a^x = e^{x\ln a} = 1 + x\ln 2 +  \dfrac{(x\ln 2)^2}{2!} + ...$
> - $(1+x)^{\alpha} = 1 + \alpha x + \dfrac{\alpha(\alpha - 1)}{2!}x^2 + o(x^2)$
> - $\dfrac{1}{1+x} = 1 - x + x^2 - x^3 + o(x^3)$ （其实就是上面 $\ln (1+x)$ 的泰勒展开求一次导，下面的也同理）
> - $\dfrac{1}{1-x} = 1 + x + x^2 + x^3 + o(x^3)$

> 💡 关于泰勒展开的余项：
> - （1）佩亚诺余项：比如 $\sin x = x - \dfrac{x^3}{3!} + o(x^3)$ 中的 $o(x^3)$ 就是佩亚诺余项，表示 $x^3$ 的高阶无穷小
> - （2）拉格朗日余项：比如 $e^x = 1 + x + \dfrac{x^2}{2!} + \dfrac{e^{\xi}}{3!} x^3$，其形如 $R_n(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!} (x - x_0)^{n+1}$，其中 $\xi$ 是 $x$ 和 $x_0$ 之间的某一个数

> 💡 两函数乘积的泰勒展开，等于各自泰勒展开相乘，即 $f(x)g(x)$ 的泰勒展开等于 $f(x)$ 的泰勒展开乘 $g(x)$

7. **间断点**
   - （1）可去间断点：$\lim\limits_{x\to x_0} f(x)= A \ne f(x_0) $（$f(x_0)$ 甚至可以无定义）
   - （2）跳跃间断点：$\lim\limits_{x\to x_0^-} f(x) \ne \lim\limits_{x\to x_0^+} f(x)$
   - （3）无穷间断点：$\lim\limits_{x \to x_0} f(x) = \infty$ 或 $\lim\limits_{x \to x_0^+} f(x) = \infty$ 或 $\lim\limits_{x \to x_0^-} f(x) = \infty$
   - （4）振荡间断点：$\lim\limits_{x \to x_0} f(x)$ 震荡不存在

> 💡 前两个为**第一类间断点**，后两个为**第二类间断点**

> ⚠️ 求间断点时注意事项：
> - （1）如果发现分子分母可以通分，千万不要消，因为消去的那一项是个可去间断点 🌚
> 

8. **两个重要极限**
   - （1）$\lim\limits_{x\to 0} \dfrac{\sin x}{x} = 1$（在信号与系统中，$\dfrac{\sin x}{x}$ 被称为 Sa 函数，即 $Sa(x) = \dfrac{\sin x}{x}$）
   - （2）$\lim\limits_{x\to \infty} (1 + \dfrac{1}{x})^{x} = e$


9. **求渐近线**
   - （1）垂直渐近线：找函数分母为 0 的点，若分母为 0 的点为 $x_0$，接下来求其极限，若 $\lim\limits_{x\to x_0} f(x) = \infty$，则直线 $x=x_0$ 为曲线 $y = f(x)$ 的垂直渐近线
   - （2）水平渐近线：若 $\lim\limits_{x\to\infty}f(x) = A$，则直线 $y = A$ 为曲线 $y = f(x)$ 的水平渐近线（若 $\lim\limits_{x\to + \infty}f(x) = A$，则曲线右侧有一水平渐近线；若 $\lim\limits_{x\to-\infty}f(x) = A$，则曲线左侧有一水平渐近线）
   - （3）斜渐近线：$k = \lim\limits_{x\to \infty} \dfrac{f(x)}{x} = \lim\limits_{x\to \infty}f'(x)$，$b = \lim\limits_{x\to\infty} [f(x) - kx]$

> ⚠️ 渐近线注意事项：
> - （1）求取渐近线的时候，要小心 $x\to +\infty$ 和 $x\to -\infty$ 两个位置渐近线不同的情况！
> - （2）斜渐近线的求取还有一种快捷方法：利用泰勒展开后略去高阶无穷小直接得到斜渐近线

10. **局部保号性（必考点）**：
    - （1）如果 $f(x) \to A(x \to x_0)$ 且 $A > 0$（或 $A<0$），那么存在常数 $\delta > 0$，使得当 $0 < |x - x_0| < \delta$ 时，有 $f(x) > 0$（或 $f(x)<0$）
    - （2）如果在 $x_0$ 的某去心邻域内 $f(x) \ge 0$（或 $f(x) \le 0$）且 $\lim\limits_{x\to x_0}f(x) = A $，则 $A\ge 0$（或 $A \le 0$）

11. **变上限积分型极限**（张宇强化）
    - （1）一型（$f\to 0$）：当 $x\to 0$ 时，$f(x) ～ ax^m$，$a\ne 0$ 且 $m$ 为正整数，则有 $\int_{0}^{x} f(t) dt ～ \int_{0}^{x} at^m dt$
    - （2）二型（$f \not{\to} 0$）：若 $\lim\limits_{x\to 0}f(x) =A \ne 0$，$\lim\limits_{x\to 0} h(x) = 0$，且在 $x\to 0$ 时，$h(x) \ne 0$，则 $\int_{0}^{h(x)} f(t) dt～ Ah(x)$
    - （3）复合型：当 $x \to 0$ 时，若 $f(x) ～ ax^m$，$g(x) ～ bx^n$，则 $\int_{0}^{g(x)} f(t) dt ～ \int_{0}^{bx^n} at^m dt$

> ⚠️ 上面这种积分类的题型，如果遇到被积函数里面出现 $x$ 该怎么办？
>
> - 不要慌，想办法通过**换元**或**三角恒等变形**将 $x$ 提取到积分外面【1000b.1.4】


## 2. 数列极限

> 可以看看没咋了的数列极限概念题梳理：[传送门](https://www.bilibili.com/video/BV117uBzoE6g/?vd_source=ff8e405dabd5438e702cff8ed19ed966)

1. **数列极限**
   - 设 $\{x_n\}$ 为一数列，若存在常数 $a$，对于任意 $\epsilon > 0$（不论它多小），总存在正整数 $N$，使得当 $n> N$ 时，$|x_n - a| < \epsilon$ 恒成立，则称常数 $a$ 是数列 $\{x_n\}$ 的极限，或者称数列 $\{x_n\}$ **收敛**于 $a$，记为：$\lim\limits_{n\to \infty} x_n=a$
   - 如果不存在这样的常数 $a$，则称数列 $\{x_n\}$ 是**发散**的

2. **收敛数列的基本性质**
   - （1）唯一性：若 $\lim\limits_{n\to \infty} x_n = a$，则 $a$ 必唯一
   - （2）有界性：$\lim\limits_{n\to\infty} x_n = a $，则 $\{x_n\}$ 必有界
   - （3）保号性：$\lim\limits_{n\to\infty}x_n = a>0(<0)$，则 $\exist$ 正整数 $N$，当 $n>N$ 时，$x_n > 0(<0)$
   - （4）保序性（易错）：设 $x_n < y_n(x_n > y_n)$，且 $\lim\limits_{n\to\infty}x_n$，$\lim\limits_{n\to\infty}y_n$ 均存在，则 $\lim\limits_{n\to\infty}x_n \le \lim\limits_{n\to\infty}y_n (\lim\limits_{n\to\infty}x_n \ge \lim\limits_{n\to\infty}y_n)$

> ⚠️ 关于**保序性**的易错点
>
> - 上面关于保序性的说法，正着说是对的，但是反过来就错了，即 $\lim\limits_{n\to\infty}x_n \le \lim\limits_{n\to\infty}y_n (\lim\limits_{n\to\infty}x_n \ge \lim\limits_{n\to\infty}y_n)$ 不能倒推 $x_n < y_n(x_n > y_n)$，因为取等号的时候 $x_n, y_n$ 上下振荡，无法判断大小！
>
> - 只有当 $\lim\limits_{n\to\infty}x_n < \lim\limits_{n\to\infty}y_n (\lim\limits_{n\to\infty}x_n > \lim\limits_{n\to\infty}y_n)$ 时才能说明 $n\to N$ 时有 $x_n < y_n (x_n > y_n)$
>
> - 由此，要注意当数列极限相关的题目中出现 $\textcolor{red}{=, \le, \ge}$ 时，说明 $\textcolor{red}{出题老头要使阴招了！}$
>
> - 比如下面这题：
>
> ![problem1](/images/mathematic/problem1.png)

3. **数列收敛与其子列收敛的关系**
   - 若数列 $\{a_n\}$ 收敛，则其任何子列 $\{a_{n_k}\}$ 也收敛，且 $\lim\limits_{k\to \infty} a_{n_k} = \lim\limits_{n \to \infty} a_n$
   - 特别的，$\lim\limits_{n\to \infty} x_n = a \Leftrightarrow \lim\limits_{k\to \infty} x_{2k} = \lim\limits_{k\to \infty} x_{2k+1} = a$

> ⚠️ 数列子列易错点
>
> - 假如题目告诉你 $\lim\limits_{k\to \infty} x_{3k} = \lim\limits_{k\to \infty} x_{3k+1} = a$，无法说明 $\{x_n\}$ 极限存在！因为缺少了 $\lim\limits_{k\to \infty} x_{3k+2} = a$

4. **海涅定理**
   - 设 $f(x)$ 在去心邻域 $\dot{U}(x_0,\delta)$ 内有定义，则 $\lim\limits_{x\to x_0}f(x) = A$ 存在 $\Leftrightarrow$ 对任意 $\dot{U}(x_0,\delta)$ 内以 $x_0$ 为极限的数列 $\{x_n\}(x_n \ne x_0)$，极限 $\lim\limits_{n \to \infty} f(x) = A$ 存在
   - 如 $f(x) = \dfrac{1}{x} \sin \dfrac{1}{x}$，$x \to 0$ 时
     - （1）若取 $x_n = \dfrac{1}{n\pi} \to 0$，则 $f(x_n) = n\pi \cdot \sin(n\pi)$，故 $\lim\limits_{n\to \infty} f(x_n) = 0$
     - （2）若取 $x_n = \dfrac{1}{(2n + \frac{1}{2})\pi} \to 0, n\to\infty$，则 $f(x_n) = (2n + \dfrac{1}{2}) \to +\infty, n \to \infty$
     - （3）根据海涅定理，极限 $\lim\limits_{x\to 0} \dfrac{1}{x} \sin \dfrac{1}{x}$ 不存在且 $x\to 0$ 时 $\dfrac{1}{x} \sin \dfrac{1}{x}$ 为无界量


5. **数列极限存在准则**
   - （1）夹逼准则：$\begin{cases} z_n \le x_n \le y_n  \\ \lim\limits_{n\to \infty} y_n = \lim\limits_{n\to \infty} z_n= a \end{cases} \Rightarrow \lim\limits_{n\to\infty} = a$
   - （2）单调有界收敛准则：单调有界数列必有极限
     - 设数列 $\{a_n\}$ 单调递增，若数列 $\{a_n\}$ 无上界（极限不存在），则 $\lim\limits_{n\to\infty}a_n = + \infty$
     - 设数列 $\{a_n\}$ 单调递减，若数列 $\{a_n\}$ 无下界（极限不存在），则 $\lim\limits_{n\to\infty}a_n = - \infty$


> ⚠️ $\{x_n\}$ 与 $\{f(x_n)\}$
> - $\{x_n\}$ 收敛 $\overset{f(x) 连续（或者弱一点极限存在）}{\underset{f(x) 在区间上严格连续单调，且\textcolor{red}{f(x) 与 x 必须一一映射，不能有重根}}{\rightleftharpoons}}$ $\{f(x_n)\}$ 收敛
> - 考试常考：“给定 $\{x_n\}$ 收敛判断 $\{f(x_n)\}$ 是否收敛”、“给定 $\{x_n\}$ 发散判断 $\{f(x_n)\}$ 是否发散”、“给定 $\{f(x_n)\}$ 收敛判断 $\{x_n\}$ 是否收敛”、“给定 $\{f(x_n)\}$ 发散判断 $\{x_n\}$ 是否发散”


> 💡 关于单调有界收敛准则相关证明题
> - 单调性：设 $x_{n+1} = f(x_n)$，则当 $f(x)$ 单调递增时，有 $\begin{cases} 若 x_1<x_2，则 \{x_n\} 单调递增 \\ 若 x_1>x_2，则 \{x_n\} 单调递减 \end{cases}$；而当 $f(x)$ 单调递减时，$\{x_n\}$ 一定不单调
> - 有界性：根据 $x_{n+1} = f(x_n)$ 先斩后奏算出极限值 $A$，然后利用数学归纳法证明 $A$ 为一个上界
>
> - 例题（除了下面这道还有 27 张宇基础例 2.14）：
>
> ![problem2](/images/mathematic/problem2.png)
>
> - “师爷真是装糊涂的天才！”

6. **压缩映射定理**
   - （1）方法一：对数列 $\{x_n\}$，若存在常数 $k(0<k<1)$，使得 $ 0 \le \textcolor{red}{|x_{n+1} -a| \le k|x_n -a|} \le k^2 |x_{n-1} - a| \le ... \le k^n |x_1 -a|$，那么根据夹逼准则，有 $\lim\limits_{n \to \infty} |x_{n+1} -a| = 0$ 即 $\{x_n\}$ 收敛于 $a$
   - （2）方法二：对数列 $\{x_n\}$，若 $x_{n+1} = f(x_n)$，$f(x)$ 可导，$a$ 为 $f(x) = x$ 的唯一解，且对任意 $x \in R$，有 $|f'(x)| \le k <1$，则 $\{x_n\}$ 收敛于 $a$
   - ![problem2](/images/mathematic/problem2_2.png)


7. **无界与无穷大的区别**
   - （1）无界数列只需要存在一个无穷大的子列即可
   - （2）无穷大的数列需要所有子列均为无穷大
   - （3）看看例题：
   - ![problem3](/images/mathematic/problem3.png)


## 3. 一元函数微分学（概念）


1. **导数的定义式（增量式）**：
   - $f'(x_0) = \lim\limits_{\Delta x \to 0} \dfrac{\Delta y}{\Delta x} = \lim\limits_{\Delta x \to 0} \dfrac{f(x_0 + \Delta x) - f(x_0)}{ \Delta x}$
   - 可导的定义：函数 $f(x)$ 在 $x_0$ 处存在上述极限（**左右极限均存在且相等**），则在 $x_0$ 处可导
   - 注意，上式中的 $\Delta x$ 可以被广义化为趋于 0 的 “🐶”

> 💡 高阶前向差分公式（经常用于处理离散信号）
>
> - （0）引例：$f''(x) = \lim\limits_{h\to 0} \dfrac{f'(x+h) - f'(x)}{h} = \lim\limits_{h\to 0} \dfrac{\frac{f(x+h+h) - f(x + h)}{h} - \frac{f(x+h) - f(x)}{h}}{h} = \lim\limits_{h\to 0} \dfrac{f(x+2h) - 2f(x+h) + f(x)}{h^2}$
>
> - （1）设 $\Delta_h f(x) = f(x+h) - f(x)$，那么一阶导数可以写成 $f'(x) = \lim\limits_{h\to 0} \dfrac{f(x+h) - f(x)}{h} = \lim\limits_{h\to 0}\dfrac{\Delta_h f(x)}{h}$
> - （2）设 $\Delta_h^2 f(x) = f(x+2h) - 2f(x+h) +f(x)$，那么二阶导数可以写成 $f''(x) = \lim\limits_{h\to 0} \dfrac{f(x+2h) - 2f(x+h) + f(x)}{h^2} = \lim\limits_{h\to 0}\dfrac{\Delta_h^2 f(x)}{h^2}$
> - （3）以此类推到 $n$ 阶，$\Delta_h^nf(x) = \sum\limits_{k=0}^{n} (-1)^{n-k} C_{n}^{k} f(x + kh)$，则有 $f^{(n)}(x) = \lim\limits_{h\to 0} \dfrac{\Delta^n_h f(x)}{h^n}$


2. **导数的函数式**：$f'(x_0) = \lim\limits_{x\to x_0} \dfrac{f(x) - f(x_0)}{x - x_0}$

> ❓ 易错概念：连续、可导，傻傻分不清？（$\textcolor{red}{重难易错}$）
> - （1）连续：对于任意函数 $f(x)$，$f(x)$ 在 $x_0$ 处有定义且**左右极限均等于** $f(x_0)$，那么称 $f(x)$ 在 $x_0$ 点连续（注意，这里只是一点连续，而不是邻域连续！很容易犯错喵 🐱）
> - （2）导数：导数的条件比连续要苛刻一点，不仅要求在 $x_0$ 处连续，还要求 $f'(x_0) = \lim\limits_{\Delta x \to 0} \dfrac{\Delta y}{\Delta x} = \lim\limits_{\Delta x \to 0} \dfrac{f(x_0 + \Delta x) - f(x_0)}{ \Delta x}$ 存在（即左导数等于右导数），才能说明在 $x_0$ 点可导！
>
> - （3）$\lim\limits_{x\to x_0}f'(x)$ 和 $f'(x_0)$ 的关系：若 $f'(x_0)$ 存在，则 $f(x)$ 在 $x = x_0$ 处可导且 $f'(x_0) = \lim\limits_{x\to x_0} \dfrac{f(x) - f(x_0)}{x - x_0}$（也就是说这个极限存在可得 $f(x)$ 在 $x=x_0$ 处可导）。而 $\lim\limits_{x\to x_0} f'(x)$ 存不存在和 $f'(x_0)$ 是否存在无关，其用途为 $\lim\limits_{x\to x_0}f'(x) = f'(x_0)$ 时可以得到 $f'(x)$ 在 $x=x_0$ 处连续，反之则不连续
>
> - 例题：【1000a.3.10】、【1000a.4.8】、【1000b.3】
>
> 这一部分可以去看没咋了、吃尽天下面的[解析](https://www.bilibili.com/video/BV1EedRYeEuH/?vd_source=ff8e405dabd5438e702cff8ed19ed966)


3. $f(x)$ **与** $|f(x)|$（必考，哪个难就考哪个 🌚）
   - （1）设 $f(x)$ 在 $x_0$ 处连续 $ \Rightarrow|f(x)|$ 在 $x_0$ 处连续
   - （2）设 $f(x)$ 在 $x_0$ 处可导，则
      - a. $f(x_0) \ne 0 \Rightarrow |f(x)|$ 在 $x_0$ 处可导且 $[|f(x_0)|]' = \begin{cases} f'(x_0), & f(x_0) > 0 \\ -f'(x_0), & f(x_0) < 0 \end{cases}$
      - b. $f(x_0) = 0$ 且 $\begin{cases} f'(x_0) = 0 \Rightarrow |f(x)| 在 x_0 处可导且 [|f(x_0)|]' = 0\\ f'(x_0) \ne 0 \Rightarrow |f(x)| 在 x_0 处不可导 \end{cases}$

4. **可微的判别**
   - （1）写增量 $\Delta y = f(x_0 + \Delta x) - f(x_0)$
   - （2）写线性增量 $A\Delta x = f'(x_0) \Delta x$
   - （3）作极限 $\lim\limits_{\Delta x \to 0} \dfrac{\Delta y - A \Delta x}{\Delta x}$
   - （4）若上述极限等于 0，则 $f(x)$ 在 $x_0$ 处可微

> 💡 可微 $\Leftrightarrow$ 可导

> 💡 设 $f(x)$ 二阶可导，$f''(x) \ne 0 \Leftrightarrow f''(x)$ 要么恒大于 0，要么恒小于 0

## 4. 一元函数微分学（计算）

1. **基本求导公式（只记了后面几个）**
   - $(\arcsin \dfrac{x}{a})' = \dfrac{1}{\sqrt{a^2 - x^2}}$
   - $(\arccos \dfrac{x}{a})' = -\dfrac{1}{\sqrt{a^2 - x^2}}$
   - $(\cot x)' = -\csc^2 x = -\dfrac{1}{\sin^2 x}$
   - $(\arctan \dfrac{x}{a})' = \dfrac{a}{a^2 + x^2}$
   - $(arccot \dfrac{x}{a})' = -\dfrac{a}{a^2 + x^2}$
   - $(\sec x)' = \sec x \tan x = \dfrac{\sin x}{\cos^2 x}$
   - $(\csc x)' = -\csc x \cot x = -\dfrac{\cos x}{\sin^2 x}$
   - $[\ln(x + \sqrt{x^2 + a^2})]' = \dfrac{1}{\sqrt{x^2 + a^2}} $
   - $[\ln(x + \sqrt{x^2 - a^2})]' = \dfrac{1}{\sqrt{x^2 - a^2yi}} $

> 💡 还有一些比较重要的，后面积分部分会经常碰到的奇妙形式：
> - $[ \ln(\sec x  + \tan x) ]' = \sec x$
> - $[\ln(\csc x - \cot x)]' = \csc x$



2. **分段函数的导数**
   - （1）在分段点 $x_0$ 处用导数定义来求导数，即用 $\lim\limits_{x\to x_0} \dfrac{f(x) - f(x_0)}{x - x_0}$ 分别求出左右导数后，看是否相等来判断分段点导数
   - （2）在非分段点用求导公式

3. **反函数的导数**：设 $y = f(x)$ 为单调、可导函数，且 $f'(x) \ne 0$，则存在反函数 $x = \varphi(y)$，且 $\dfrac{dx}{dy} = \dfrac{1}{\frac{dy}{dx}}$，即 $\varphi'(y) = \dfrac{1}{f'(x)}$

> **💡 反函数的二阶导数**
>
> - 在 $y = f(x)$ 单调且二阶可导的情况下，若 $f'(x) \ne 0$，则存在反函数 $x = \varphi(y)$，记 $f'(x) = y'_x$、$\varphi '(y) = x'_y$，那么：
> - （1）$y'_x = \dfrac{1}{x'_y}$
> - （2）$y''_{xx} = - \dfrac{x''_{yy}}{(x'_y)^3}$
>
> 证明如下：
> - （1）$y = f(x)$，$x = g(x)$，则 $f'(x) = \dfrac{dy}{dx}$，$g'(y) = \dfrac{dx}{dy} = \dfrac{1}{\frac{dy}{dx}} = f'(x)$
> - （2）$g''(x) = \dfrac{d^2 x}{dy^2} = \dfrac{d(\frac{dx}{dy})}{ dy} = \dfrac{d(\frac{1}{f'(x)})}{dx} \cdot \dfrac{dx}{dy} = -\dfrac{f''(x)}{[f'(x)]^2} \cdot \dfrac{1}{f'(x)} = -\dfrac{f''(x)}{[f'(x)]^3}$

4. $n$ **阶导数**
   - （1）归纳法：逐次求导，找到规律，得出通式
   - （2）莱布尼茨公式：设 $u = u(x)$、$v = v(x)$ 均 $n$ 阶可导，则 $(u \pm v)^{(n)} = u^{(n)} + v^{(n)}$、$(uv)^{(n)} = \sum\limits_{k=0}^{n} C_n^k u^{(n-k)} v^{(k)}$
   - （3）泰勒展开法（一般用于求 $f^{(n)}(0)$）



## 5. 一元函数微分学（几何应用）
1. **极值点**：其定义为 $x=x_0$ 点在其邻域内取得最大值或最小值（可以是导数不存在的点）
2. **拐点**：**连续曲线**的凹弧和凸弧的分界点称为该曲线的拐点（拐点的判断标准是在该点处左右两侧二阶导数符号相反，也就是说拐点处可能是 $f''(x_0) = 0$，也可能是 $f''(x_0^+) = \pm \infty = -f''(x_0^-)$）

> ⚠️ 关于凹凸性的易错点：
> - ❌ 当 $f''(x_0) > 0$ 时，曲线 $f(x)$ 在 $x_0$ 的某邻域内是凹的：错，一个点的二阶导数怎么能确定邻域的凹凸性呢？【880-二-综合题-选择-15】

> 💡 凹函数的定义
> - 设 $f(x)$ 在区间 $I$ 上连续，对任意 $x_1 \ne x_2$，$x_1, x_2 \in I$，$0 < \lambda < 1$，若 $f(\lambda x_1 + (1-\lambda)x_2) < \lambda f(x_1) + (1 - \lambda) f(x_2)$，则称 $y = f(x)$ 在 $I$ 上是凹的


3. **极值点与拐点的重要结论**：
   - （1）曲线的$\textcolor{red}{可导}$点不可同时为极值点和拐点；曲线的$\textcolor{red}{不可导}$点可以同时为极值点和拐点
   - （2）设多项式函数 $f(x) = (x - a)^n g(x)(n>1)$ 且 $g(a) \ne 0$，则当 $n$ 为偶数时，$x=a$ 是 $f(x)$ 的极值点，$n$ 为奇数时，$(a, 0)$ 是曲线的拐点
   - （3）【张宇基础 30 讲 P149，感觉很难用到，先插个眼，等考到了再回来补】

4. **曲率**：$k = \dfrac{|y''|}{[1 + (y')^2]^{\frac{3}{2}}}$
5. **曲率半径**：$R = \dfrac{1}{k} = \dfrac{[1 + (y')^2]^{\frac{3}{2}}}{|y''|} (y''\ne 0)$

## 6. 一元函数微分学（中值定理）

> 下面这些定理的前提是 $f(x)$ 在 $[a, b]$ 上连续

1. **有界与最值定理**：$m \le f(x) \le M$，其中 $m, M$ 分别为 $f(x)$ 在 $[a, b]$ 上的最小值qq和最大值
2. **介值定理**：$m\le\mu\le M$ 时，存在 $\xi \in [a, b]$ 使得 $f(\xi) = \mu$
3. **平均值定理**：当 $a < x_1 < x_2 < ... < x_n < b$ 时，在 $[x_1, x_n]$ 内至少存在一点 $\xi$ 使得 $f(\xi) = \dfrac{f(x_1) + f(x_2) + ... + f(x_n)}{n}$

4. **零点定理**：当 $f(a) \cdot f(b) < 0$ 时，存在 $\xi \in (a, b)$ 使得 $f(\xi) = 0$

> 下面这些定理涉及到导数（微分）

5. **费马定理**：设 $f(x)$ 在点 $x_0$ 处可导且取得极值，那么 $f'(x_0) = 0$
6. **罗尔定理**：设 $f(x)$ 在 $[a, b]$ 上连续、在 $(a, b)$ 上可导、$f(a) = f(b)$，则存在 $\xi \in (a, b)$ 使得 $f'(\xi) = 0$

> 💡 如果题目出现 $f''(x)$，可以尝试在三个点构成的区间内用三次罗尔定理得到（罗尔三点图），且往往配合零点定理使用

7. **拉格朗日中值定理**：设 $f(x)$ 在 $[a, b]$ 上连续、$(a, b)$ 上可导，则存在 $\xi \in (a, b)$ 使得 $f(b) - f(a) = f'(\xi)(b-a)$，或 $f'(\xi) = \dfrac{f(b) - f(a)}{ b-a}$

> ⚠️ 拉格朗日中值定理的证明要会！
> - （1）设 $F(x) = f(x) - [\dfrac{f(b) - f(a)}{b-a} (x-a) + f(a)]$
> - （2）易知 $F(a) = F(b) = 0$
> - （3）用罗尔定理得 $\exist \xi \in (a, b)$ 使得 $F'(\xi) = f'(\xi) - \dfrac{f(b) - f(a)}{b - a} = 0$
> - （4）得证

> 💡 拉格朗日中值定理的妙用：
> - （1）做题时如果遇到 $\int^{x+1}_{x} f(t)dt$（或者 $x+1$ 和 $x$ 同时出现的情况）且给定了 $f(x)$ 相关条件的话，要能够联想到拉格朗日中值定理（比如【880-1-综合题-二-2、6】）
> - （2）遇到复杂算式，若遇到 $f(x) - g(x)$ 的形式，且 $f(x)$ 和 $g(x)$ 在特定点处的极限值相等、算式形式相近，要想到拉格朗日中值定理（比如 【880-1-综合题-三-6、7】）

8. **柯西中值定理**：设 $f(x), g(x)$ 满足在 $[a, b]$ 上连续、在 $(a, b)$ 上可导、$g'(x) \ne 0$，则存在 $\xi \in (a, b)$ 使得 $\dfrac{f(b) - f(a)}{g(b) - g(a)} = \dfrac{f'(\xi)}{g'(\xi)}$

> ⚠️ 柯西的证明也要会！
> - （1）设 $F(x) = f(x) - \{\dfrac{f(b) - f(a)}{g(b)-g(a)} [g(x)-g(a)] - f(a)\}$
> - （2）剩下的过程和上面拉格朗日证明一样

> 💡 常用的一阶化归逆用形式（没事多推一推，看到具体形式能想到如何逆用即可，也可以去看看[没咋了的专题视频](https://www.bilibili.com/video/BV1BHdqYUEgu/?)），例题：【880-2-解答题-10】
>
> - （1）$[f(x) \cdot x^n]' = x^{n-1} [xf'(x) + nf(x)]$
> - （2）$[f(x) \cdot e^{nx}]'  = e^{nx}[f'(x) + n f(x)]$
> - （3）$[f(x) \cdot e^{x^n}]' = e^{x^n} [f'(x) + nx^{n-1}f(x)]$
> - （4）$[f(x) \cdot e^{\varphi(x)}]' = e^{\varphi(x)} [f'(x) + f(x) \varphi '(x)]$
> - （5）$\{f(x)\cdot e^{\int^x_0 [f(t)]^{n-1} dt}\}' = e^{\int^x_0 [f(t)]^{n-1} dt} \{f'(x) +[f(x)]^n\}$
> - （6）$[f(x) \cdot f'(x)]' = [f'(x)]^2 + f(x) \cdot f''(x)$
> - （7）$[f(x) \cdot g(x)]' = f'(x)g(x) + f(x)g'(x)$
> - （8）$[f(x) \cdot \arctan x]' = f'(x) \arctan x + \dfrac{f(x)}{1+x^2}$
> - （9）$[f(x) \cdot \sin x]' = f'(x) \sin x + f(x) \cos x = [f'(x) \tan x + f(x)]\cos x$
> - （10）$[\dfrac{f(x)}{x}]' = \dfrac{f'(x)x - f(x)}{x^2}$
> - （11）$[\dfrac{f'(x)}{f(x)}]' = \dfrac{f''(x) f(x) - [f'(x)]^2}{f^2(x)}$
> - （12）$[\ln f(x)]' = \dfrac{f'(x)}{f(x)}$（再导一次就变成（11）了）
> - （本质其实是微分方程？）

> 💡 常用二阶化归逆用（还有第二关？）
>
> - $[f(x) \cdot e^x]'' = e^x [f''(x) + 2f'(x) + f(x)]$


## 7. 一元函数微分学（物理应用）

> 💡 这一章似乎没有什么好说的，做题时要注意题目里有些变量不会直接告诉你，而是隐含在题目背景里的，比如时间 $t$


## 8. 一元函数积分学的概念和性质

1. **连和与连积的极限**
   - （1）基本型（能凑成 $\dfrac{i}{n}$）：用 $\lim\limits_{n\to \infty} \sum\limits_{i=1 or 0}^{n} f(0 + \dfrac{1-0}{n}i) \dfrac{1-0}{n} = \int^1_0 f(x) dx$
   - （2）放缩型（凑不成 $\dfrac{i}{n}$）：夹逼准则或放缩
   - （3）变量型（$\dfrac{x}{n} i$）：用 $\lim\limits_{n\to \infty} \sum\limits_{i=1 or 0}^{n} f(0 + \dfrac{x-0}{n}i) \dfrac{x-0}{n} = \int^x_0 f(t) dt$

> ⚠️ 上面的式子 $\lim\limits_{n\to \infty} \sum\limits_{i=1 or 0}^{n} f(0 + \dfrac{x-0}{n}i) \dfrac{x-0}{n} = \int^x_0 f(t) dt$ 很重要，其中的 $x$ 可以替换成其他常数或变量

2. **连续、原函数存在、可积的易错点**
   - （1）若 $f(x)$ 在 $[a, b]$ 上连续，则 $f(x)$ 在 $(a, b)$ 内存在原函数，且为 $\int_a^x f(t)dt$
   - （2）但若 $f(x)$ 在 $[a, b]$ 上不连续，则 $f(x)$ 在 $(a, b)$ 内不一定存在原函数！（比如符号函数）
   - （3）连续一定可积、可积一定有界，但有界不一定可积、可积不一定连续（即使函数有间断点，也可能可积，比如符号函数）
   - （4）对于含有第一类间断点、无穷间断点的函数在包含该间断点的区间内**必无**原函数
   - （5）对于含有振荡间断点的函数在包含该间断点的区间内**可能**有原函数
   - （6）原函数存在与可积没有必然联系，原函数存在不一定可积，可积不一定存在原函数

> 💡 只要函数 $f(x)$ 在某区间上可导，则其导函数 $f'(x)$ 在该区间上一定没有第一类间断点和无穷间断点

3. **反常积分敛散性判别（⭐ 重难点）**
   - （1）基本结论（要背下来）
     - $\int^1_0 \dfrac{1}{x^p} dx \begin{cases} 收敛，& 0<p<1 \\ 发散，& p\ge 1 \end{cases}$，$\int^1_0 \dfrac{\ln x}{x^p} dx \begin{cases} 收敛，& 0 \le p<1 \\ 发散，& p\ge 1 \end{cases}$
     - $\int^{+\infty}_1 \dfrac{1}{x^p} dx \begin{cases} 收敛，& p>1 \\ 发散，& p\le 1 \end{cases}$，$\int^{+\infty}_1 \dfrac{\ln x}{x^p} dx \begin{cases} 收敛，& p>1  \\ 发散，& p\le 1 \end{cases}$
   - （2）二级结论（做题时遇到的）
     -  若 $\int^{+\infty}_{0}f(x)dx$ 收敛，则 $\int^{+\infty}_{0}f^2(x)dx$ 收敛，反之错误（证明：$\int^{+\infty}_{0}f(x)dx$ 收敛 $\Rightarrow \lim\limits_{x\to +\infty} f(x) = 0 \Rightarrow \lim\limits_{x\to +\infty} \dfrac{f^2(x)}{f(x)} = 0 \Rightarrow$ 由比较判别法可知 $\int^{+\infty}_{0}f^2(x)dx$ 收敛）

> ⚠️ 注意：
> - （1）第一行基本结论中 $0<p$ 是因为该函数为“反常积分”，如果题目没有明确说这是个反常积分，则取值范围可以 $\le 0$
> - （2）在上述基本结论中，$\ln x$ 形式上不起作用（一般只有 $x\to 1$ 时才起作用，要注意区分。比如例题 $\int^1_0 \dfrac{\ln x}{x^p (1-x)^q}$ 中，拆成 $(0, \dfrac{1}{2})$ 和 $(\dfrac{1}{2}, 1)$ 两个部分后，在第二个部分中 $\ln x$ 是起作用的，而在第一个部分中不起作用，可以直接抹去）
> - （3）题目中经常需要用到比较判别法，要多训练

> ⚠️ 关于反常积分的易错点：对于 $\int^4_0 \dfrac{1}{x-2}dx$，其柯西主值为 0。但是！因为按定义必须将其分开成 $\int^2_0 \dfrac{1}{x-2} dx$ 和 $\int^4_2 \dfrac{1}{x-2} dx$，这两个式子都发散，所以原积分发散！


## 9. 一元函数积分学的计算

> ⚠️ 本章极其容易犯错的细节：
> - （1）当定积分上下限区间内$\textcolor{red}{包含瑕点}$时，必须将定积分拆开来算！
> - （2）换元、变限积分求导等问题中，要小心 $-\sqrt{x^2}$ 等需要考虑$\textcolor{red}{正负号变换}$的细节！比如【1000a.9.17】

1. **容易遗忘的重要积分公式**
   - （1）$\int \dfrac{dx}{\cos x} = \int \sec x dx = \ln|\sec x + \tan x| + C$
   - （2）$\int \dfrac{dx}{\sin x} = \int \csc x dx = \ln|\csc x - \cot x| + C$
   - （3）$\int \sec ^2 x dx = \tan x + C$
   - （4）$\int \csc ^2 x dx = -\cot x + C$
   - （5）$\int \dfrac{1}{\sqrt{x^2 \pm a^2}} dx = \ln|x + \sqrt{x^2 \pm a^2}| + C$
   - （6）$\int \dfrac{1}{\sqrt{a^2 - x^2}} dx = \arcsin \dfrac{x}{a} + C$
   - （7）$\int - \dfrac{1}{\sqrt{a^2 - x^2}} dx = \arccos \dfrac{x}{a} + C$
   - （8）$\int \sqrt{x^2 \pm a^2} dx = \dfrac{x}{2} \sqrt{x^2 \pm a^2} \pm \dfrac{a^2}{2} \ln |x + \sqrt{x^2 \pm a^2}| + C$
   - （9）$\int \sqrt{a^2 - x^2} dx = \dfrac{x}{2} \sqrt{a^2 - x^2} + \dfrac{a^2}{2}\arcsin \dfrac{x}{a} + C$
   - （10）$\int \tan x dx = -\ln |\cos x| + C$
   - （11）$\int \tan^2 x dx = \tan x - x + C$

> 💡 换元积分时的三角换元法出现的三角函数嵌套问题
>
> - 在求解积分的时候经常需要用到三角函数进行换元，比如 $x = a\tan t / a\cot t / a\cos t / a\sin t / a\sec t / a\csc t$，当出现三角函数套反三角函数时，要记得使用几何方法（画个直角三角形）将其拆分成 $x$ 非三角函数，比如 $\sin(\arctan \dfrac{x}{a}) = \dfrac{x}{\sqrt{x^2 + a^2}}$


2. **万能公式**：
   - 令 $t = \tan \dfrac{x}{2}$，则 $\sin x = \dfrac{2t}{1+t^2}$，$\cos x = \dfrac{1-t^2}{1+t^2}$
   - 一般用于求解 $\int \dfrac{1}{a+b\sin x} dx$ 或 $\int \dfrac{1}{a+b\cos x} dx$
   - 【1000a.9.1.(23)(24)】

> 💡 要是看到 $\int \dfrac{1}{a + b\sin^2 x}$、$\int \dfrac{1}{a + b\cos^2 x}$，可以设 $t \tan x$，一定要敏感！

3. **第二类换元法**：
   - 如果被积函数非常复杂，可以尝试令 $x = g(u)$，引入新的自变量 $u$，使 $f(x) = f[g(u)] = h(u)$，若 $h(u)$ 更简单，则换元成功
   - 常见的换元包括（看到下面的形式要敏感）：
     - （1）三角换元：$x = a\tan t / a\cot t / a\sin t / a\cos t / a\sec t / a\csc t$
     - （2）根式换元：$\sqrt[n]{\dfrac{ax + b}{cx + d}} = u$ 或 $\sqrt{e^x + 1} = u$
     - （3）倒数代换：$x = \dfrac{1}{t}$
     - （4）区间转换：$x = at + c$（$a$ 一般为负数，该代换用途主要为：代换完之后不仅将原函数转换为特殊形式，还能改变积分上下限至其他积分上，比如 $\int^a_b f(x)dx = \int^a_b f(a+b -x)dx$）


> 💡 还有一种原函数不易求解的非奇非偶函数在**关于定义域对称区间**上的积分（或者对折定义域之后可以相消的积分），比较特立独行的题目，非常重要（可以看看第 11 章的区间再现部分）：
>
> - 【1000b.9.12】$\int^1_{-1} x\ln(1+e^x) dx \\ = \int^1_0 x\ln(1+e^x)dx + \int^0_{-1} x\ln(1+e^x)dx \\ = \int^1_0 x\ln(1+e^x)dx + \int^0_{-1} x\ln(1+e^x)dx \\ =\int^1_0 x\ln(1+e^x)dx + \int^0_{1} (-x)\ln(1+e^{-x})d(-x) \\ =\int^1_0 x\ln(1+e^x)dx - \int^1_{0} x\ln(1+e^{-x})dx \\ =\int^1_0 x\ln(\dfrac{1+e^x}{1+e^{-x}})dx \\ =\int^1_0 x\ln e^xdx \\ = \int^1_0 x^2 dx$
> - 【1000b.9.26】$\int^5_{-1} \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx \\ = \int^5_2 \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx + \int^2_{-1} \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx \\ = \int^5_2 \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx + \int^2_5 \dfrac{1}{1 + 2^{\sqrt[3]{2-t}}} d(4-t) \\ = \int^5_2 \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx + \int^5_2 \dfrac{1}{1 + 2^{-\sqrt[3]{t-2}}} dt \\ = \int^5_2 \dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} dx + \int^5_2 \dfrac{1}{1 + 2^{-\sqrt[3]{x-2}}} dx \\ = \int^5_2 (\dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} + \dfrac{1}{1 + 2^{-\sqrt[3]{x-2}}} )dx \\ = \int^5_2 (\dfrac{1}{1 + 2^{\sqrt[3]{x-2}}} + \dfrac{2^{\sqrt[3]{x-2}}}{1 + 2^{\sqrt[3]{x-2}}} )dx \\ = \int^5_2 1 dx$
> - 【1000b.9.28】$\int^{+\infty}_0 \dfrac{x\ln x}{1+x^4} dx \\ = \int^{+\infty}_1 \dfrac{x\ln x}{1+x^4} dx + \int^{1}_0 \dfrac{x\ln x}{1+x^4} dx \\ = \int^{+\infty}_1 \dfrac{x\ln x}{1+x^4} dx + \int^{1}_{+\infty} \dfrac{\frac{1}{t}\ln (\frac{1}{t})}{1+(\frac{1}{t})^4} d(\frac{1}{t}) \\ = \int^{+\infty}_1 \dfrac{x\ln x}{1+x^4} dx + \int^{1}_{+\infty} \dfrac{\frac{1}{t} (-\ln t)}{1+(\frac{1}{t})^4} (-\dfrac{1}{t^2})dt \\ = \int^{+\infty}_1 \dfrac{x\ln x}{1+x^4} dx - \int^{+\infty}_{1} \dfrac{t \ln t}{1+t^4} dt \\ =0$


4. **Gamma 函数**
   - 定义：$\Gamma(\alpha) = \int^{+\infty}_0 x^{\alpha - 1} e^{-x} dx$
   - $\Gamma(\alpha + 1) = \alpha \Gamma(\alpha)$
   - $\Gamma(\dfrac{1}{2}) = \int^{\infty}_{0}x^{-\frac{1}{2}} e^{-x} dx  = 2\int^{\infty}_{0} e^{-x} d\sqrt{x} =   2\int^{+\infty}_{0} e^{-x^2} dx = \sqrt{\pi}$

## 10. 一元函数积分学的几何应用

> ⚠️ 本章重点习题：
> - 【1000a-10-11】
> - 【1000b-10-10、12、17、21、22、24】

1. **计算面积**
   - （1）直角坐标系下：$S = \int^a_b |f_1(x) - f_2(x)| dx$
   - （2）极坐标系下（扇形区域）：$S = \int^{\beta}_{\alpha} \dfrac{1}{2} |r_2^2(\theta) - r_1^2(\theta)| d\theta$

2. **平面曲线绕定直线旋转体体积**
   - （1）设平面曲线 $L$：$y=f(x)$，$a\le x\le b$ 且 $f(x)$ 可导
   - （2）定直线 $L_0$：$Ax + By + C = 0$ 且 $L_0$ 的任一条垂线与 $L$ 至多有一个交点
   - （3）则 $L$ 绕 $L_0$ 旋转一周的旋转体体积：$V = \dfrac{\pi}{(A^2+B^2)^{\frac{3}{2}}} \int^b_a [Ax +Bf(x) +C]^2 |Af'(x) -B|dx$
   - （4）设平面图形 $D = \{(r, \theta) | 0 \le r \le r(\theta), \theta\in[\alpha, \beta]\subset [0, \pi] \}$，则 $D$ 绕极轴（一般是 $x$ 轴）旋转一周所得旋转体的体积为 $V = \dfrac{2}{3} \int^{\beta}_{\alpha} r^3(\theta) \sin\theta d\theta$

3. **平均值**：$\overline{f} = \dfrac{1}{b-a} \int^b_a f(x) dx$

4. **平面曲线的弧长**
   - （1）$y = y(x)(a\le x\le b)$：$s = \int^b_a ds = \int^b_a \sqrt{1 + [y'(x)]^2} dx$
   - （2）$r = r(\theta)(\alpha\le \theta\le\beta)$：$s = \int^{\beta}_{\alpha} ds = \int^{\beta}_{\alpha} \sqrt{[r(\theta)]^2 + [r'(\theta)]^2} d\theta$
   - （3）$\begin{cases} x = x(t) \\ y=y(t) \end{cases}(\alpha\le t\le\beta)$：$s = \int^{\beta}_{\alpha} ds = \int^{\beta}_{\alpha} \sqrt{[x'(t)]^2 + [y'(t)]^2} dt$

> 💡 这里记录的是 $ds = \sqrt{1 + [y'(x)]^2} dx$ 或另外两个形式，实际上也可以旋转一下坐标系，即 $ds = \sqrt{1 + [y'(x)]^2} dx = \sqrt{1 + [x'(y)]^2} dy$

> 💡 关于弧长的计算一定要熟悉，后面线面积分还会用到

5. **旋转曲面面积**
   - （1）$y = y(x)(a\le x\le b)$ 绕 $x$ 轴旋转：$S = 2\pi \int^b_a |y(x)| \sqrt{1 + [y'(x)]^2} dx$
   - （2）$r = r(\theta)(\alpha\le \theta\le\beta)$ 绕 $x$ 轴旋转：$S = 2\pi \int^{\beta}_{\alpha} |r(\theta) \sin \theta| \sqrt{[r(\theta)]^2 + [r'(\theta)]^2} d\theta$
   - （3）$\begin{cases} x = x(t) \\ y=y(t) \end{cases}(\alpha\le t\le\beta)$ 绕 $x$ 轴旋转：$S = 2\pi \int^{\beta}_{\alpha} |y(t)| \sqrt{[x'(t)]^2 + [y'(t)]^2} dt$

> 💡 做题需要注意：
> - 连续函数 $f(x)$ 的一个原函数表达形式常写为 $F(x) = \int^x_0 f(u) du$，考试时见到 $F(x) \int^x_0 f(u)du$，一般令 $F(x) = \int^x_0 f(u)du$，这样就有 $F'(x) = f(x)$，即得 $F'(x) F(x)$，于是 $\int F'(x) F(x) dx = \int F(x) d[F(x)] = \dfrac{1}{2} F^2(x) + C$，这个思路非常重要
> - 上面这个思路主要出现在求平均值里（？）


6. **形心坐标**：
   - （1）$\overline{x} = \dfrac{\iint\limits_{D} x d\sigma}{\iint\limits_{D}  d\sigma} = \dfrac{\int^b_a dx \int^{f(x)}_0 x dy}{\int^b_a dx \int^{f(x)}_0  dy} = \dfrac{\int^b_a xf(x) dx}{\int^b_a f(x) dx}$
   - （2）$\overline{y} = \dfrac{\iint\limits_{D} y d\sigma}{\iint\limits_{D}  d\sigma} = \dfrac{\int^b_a dx \int^{f(x)}_0 y dy}{\int^b_a dx \int^{f(x)}_0  dy} = \dfrac{\frac{1}{2} \int^b_a f^2(x) dx}{\int^b_a f(x) dx}$


7. **重心坐标**
   - （1）$\overline{x} = \dfrac{\iint\limits_{D} x \rho d\sigma}{\iint\limits_{D} \rho d\sigma} $
   - （2）$\overline{y} = \dfrac{\iint\limits_{D} y \rho d\sigma}{\iint\limits_{D} \rho d\sigma} $


## 11. 一元函数积分学的应用：积分等式与积分不等式

> ⚠️ 本章重点习题：
> - 【张宇强化.11.例1】

1. **祖孙三代奇偶性**
   - （1）$f(x)$ 为可导的奇函数 $\Rightarrow f'(x)$ 为偶函数
   - （2）$f(x)$ 为可导的偶函数 $\Rightarrow f'(x)$ 为奇函数
   - （3）$f(x)$ 为可积的奇函数 $\Rightarrow \int^x_0 f(t) dt$ 为偶函数（注意下限为 0）
   - （4）$f(x)$ 为可积的偶函数 $\Rightarrow \int^x_0 f(t) dt$ 为奇函数（注意下限为 0）
   - （5）$\int^a_{-a} f(x) dx = \dfrac{1}{2} \int^a_{-a} [f(x) + f(-x)]dx = \int^a_0 [f(x) + f(-x)] dx$（区间再现公式，比如 $f(x) = \dfrac{1}{1+e^x}, \dfrac{1}{1+e^{\frac{1}{x}}}, \dfrac{1}{1 + e^{-x}}$ 等，本身没有奇偶性，但可以借助上式处理成易积分的函数）

2. **祖孙三代周期性**
   - （1）若 $f(x)$ 是可导且以 $T$ 为周期的周期函数，则 $f'(x)$ 是以 $T$ 为周期的周期函数
   - （2）若 $f(x)$ 是可积且以 $T$ 为周期的周期函数，则 $\int^x_0 f(t) dt$ 是以 $T$ 为周期的周期函数 $\Leftrightarrow \int^T_0 f(x) dx = 0$
   - （3）若 $f(x)$ 是可积且以 $T$ 为周期的周期函数，则 $\int^T_0 f(x) dx = \int^{a+T}_{a} f(x) dx \Leftrightarrow \int^{a+nT}_{a} f(x) dx = n\int^{T}_0 f(x) dx$

3. **区间再现大观 ⭐**
   - （1）$\int^b_a f(x) dx = \int^b_a f(a+b - x) dx$
   - （2）$\int^b_a f(x) dx = \dfrac{1}{2} \int^b_a [f(x) + f(a+b - x)] dx$
   - （3）$\int^b_a f(x) dx = \int^{\frac{a+b}{2}}_a [f(x) + f(a+b - x)] dx$（$F(x) = f(x) +f(a+b-x)$ 关于 $x = \dfrac{a+b}{2}$ 对称）
   - （4）$\int^{\pi}_0 xf(\sin x) dx = \dfrac{\pi}{2} \int^{\pi}_0 f(\sin x)dx$
   - （5）$\int^{\pi}_0 xf(\sin x) dx = \pi \int^{\frac{\pi}{2}}_0 f(\sin x)dx$
   - （6）$\int^{\frac{\pi}{2}}_{0} f(\sin x)dx = \int^{\frac{\pi}{2}}_{0} f(\cos x) dx$
   - （7）$\int^{\frac{\pi}{2}}_{0} f(\sin x, \cos x)dx = \int^{\frac{\pi}{2}}_{0} f(\cos x, \sin x) dx$

4. **华里士公式（点火公式）**
   - （1）$\int^{\frac{\pi}{2}}_0 \sin^n x dx = \int^{\frac{\pi}{2}}_0 \cos^n x dx = \begin{cases} \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{2}{3} \times 1,& n 为大于 1 的奇数 \\ \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{1}{2} \times \dfrac{\pi}{2},& n 为正偶数 \end{cases}$
   - （2）$\int^{\pi}_0 \sin^n x dx = \begin{cases} 2\times \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{2}{3} \times 1,& n 为大于 1 的奇数 \\ 2\times \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{1}{2} \times \dfrac{\pi}{2},& n 为正偶数 \end{cases}$
   - （3）$\int^{\pi}_0 \cos^n x dx = \begin{cases} 0 \\ 2\times \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{1}{2} \times \dfrac{\pi}{2},& n 为正偶数 \end{cases}$
   - （4）$\int^{2\pi}_0 \cos^n x dx = \int^{2\pi}_0 \sin^n x dx = \begin{cases} 0 \\ 4\times \dfrac{n-1}{n} \times \dfrac{n-3}{n-2} \times ... \times \dfrac{1}{2} \times \dfrac{\pi}{2},& n 为正偶数 \end{cases}$

5. **积分中值定理**
   - （1）若函数 $f(x)$ 在 $[a, b]$ 上连续，则存在 $\xi \in [a, b]$，使得 $\int^b_a f(x) dx = f(\xi) (b-a)$
   - （2）需要把积分值表示成函数值的时候要想到这个中值定理，反之，某些特殊函数值也可以用定积分表示（比如 $\dfrac{\pi}{4} = \int^{+\infty}_1 \dfrac{1}{1+x^2} dx$）
   - （3）推广：若 $f(x)$ 在 $[a, b]$ 上连续，$g(x)$ 在 $[a, b]$ 上可积且不变号，则存在 $\xi \in (a, b)$，使得 $\int^b_a f(x) g(x) dx = f(\xi) \int^b_a g(x) dx$

> 💡 积分中值定理（3）推广的证明（其实是基础 30 讲里的原题）：
>
> - 设 $F(x) = \int^x_0 f(t) g(t) dt$，$G(x) = \int^x_0 g(t) dt$
> - 根据柯西中值定理，存在 $\xi \in (a, b)$ 使得 $\dfrac{F(b) - F(a)}{G(b) - G(a)} = \dfrac{F'(\xi)}{G'(\xi)}$
> - 即 $\dfrac{\int^b_a f(t)g(t) dt}{\int^b_a g(t)} = \dfrac{f(\xi)g(\xi)}{g(\xi)}$
> - 化简得到 $\int^b_a f(x) g(x) dx = f(\xi) \int^b_a g(x) dx$


6. **定积分不等式问题**
   - （1）比较定理：设 $f(x)$，$g(x)$ 连续，则 $f(x) \le g(x) \Rightarrow \int^b_a f(x) dx \le \int^b_a g(x) dx, b>a$
   - （2）估值定理：设 $f(x)$ 连续，则 $m\le f(x) \le M \Rightarrow m(b-a) \le \int^b_a f(x) dx \le M(b-a), b>a$
   - （3）绝对值不等式：设 $f(x)$ 连续，则 $|\int^b_a f(x) dx| \le \int^b_a |f(x)| dx, b>a$


> 💡 协方差恒等式【1000a-11-8】：
> - （1）在 $[0, 1]$ 上：$\int^1_0 f(x)g(x) dx - \int^1_0 f(x) dx \int^1_0 g(x) dx = \dfrac{1}{2} \int^1_0 \int^1_0 [f(x) - f(y)][g(x) - g(y)] dx dy$
> - （2）在 $[a, b]$ 上：设 $L = b-a$，则 $\dfrac{1}{L}\int^b_a f(x)g(x) dx - (\dfrac{1}{L}\int^b_a f(x) dx)(\dfrac{1}{L} \int^b_a g(x) dx )= \dfrac{1}{2L^2} \dfrac{1}{2} \int^b_a \int^b_a [f(x) - f(y)][g(x) - g(y)] dx dy$
> - （3）离散形式：$\dfrac{1}{n} \sum\limits^{n}_{i=1} a_i b_i - (\dfrac{1}{n} \sum\limits_{i=1}^{n} a_i)(\dfrac{1}{n} \sum\limits_{i=1}^{n} b_i) = \dfrac{1}{2n^2} \sum\limits^n_{i=1} \sum\limits^n_{j=1} (a_i - a_j)(b_i - b_j)$
> - （4）如果离散形式里的两个数列都是递增的，右边 $\ge 0$，则有：$ \dfrac{1}{n} \sum\limits^{n}_{i=1} a_i b_i \ge (\dfrac{1}{n} \sum\limits^n_{i=1} a_i)(\dfrac{1}{n} \sum\limits^n_{i=1} b_i) $（离散版切比雪夫不等式）
> - （5）把 $x$ 看成一个随机变量，那么就有：$E[fg] - E[f]E[g] = \dfrac{1}{2} E\{[f(X)-f(Y)][g(X) - g(Y)]\}$，其中 $X, Y$ 是两个独立同分布的随机变量

> 💡 一类数形结合的创新题型【1000b-11-3】
> 
> 设 $f(x)$ 是 $[0, 1]$ 上的可导函数，$f(0) = f(1) = 1$，$\max\limits_{0\le x\le 1} \{|f'(x)|\} = 1$，则 $\int^1_0 f(x) dx$ 的范围为：
> - （1）设 $F(x) = f(x) - x - 1$，则 $F'(x) = f'(x) - 1 \le 0 \Rightarrow F(x)$ 单调递减，所以：
>     -  $F(x) < F(0) = 0$ 即 $f(x) \le x+1$
>     -  $F(x) > F(1) = -1$ 即 $f(x) > x$
> - （2）设 $F(x) = f(x) + x$，则 $F'(x) = f'(x) + 1 \ge 0 \Rightarrow F(x)$ 单调递增，所以：
>     -  $F(x) < F(1) = 2$ 即 $f(x) < 2 - x$
>     -  $F(x) > F(0) = 1$ 即 $f(x) > 1 - x$
> - （3）所以 $f(x)$ 在 $y=x$、$y=x+1$、$y=2-x$、$y=1-x$ 四条线围城的区域内
> - （4）计算面积可以得到 $\dfrac{3}{4} < \int^1_0 f(x) dx < \dfrac{5}{4}$


7. **柯西积分不等式**（柯西-施瓦茨不等式，不过🥟神说没用到过...）
   - 若函数 $\varphi(x)$ 与 $\psi(x)$ 在 $[a, b]$ 上连续，则有 $[\int^b_a \varphi(x) \psi(x) dx]^2 \le \int^b_a \varphi^2(x) dx \int^b_a \psi^2(x) dx $


## 12. 一元函数积分学的物理应用
1. **物理应用**
   - （1）变力 $F(x)$ 沿直线做功：$W = \int^b_a F(x)dx$
   - （2）抽水做功：$W = \rho g \int^b_a xA(x) dx$（$A(x)$ 为水平截面面积，$x$ 为水深）
   - （3）静水压力：$P = \rho g \int^b_a x[f(x) - h(x)] dx$（$f(x) - h(x)$ 表示矩形条宽度，$x$ 表示水深）
   - （4）变速直线运动的位移：$s = \int^{t_2}_{t_1} v(t) dt$
   - （5）引力：$F = \int^0_{-l} \dfrac{Gm\mu}{(a-x)^2}dx$

> 💡 剩下的似乎没什么了，遇到创新的题目要记得用微元法试一试


