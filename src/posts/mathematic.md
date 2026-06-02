---
title: Coldrain 的 27 考研数一高数强化阶段拾遗（1-12讲）
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
   - $(1+x)^{\alpha} - 1 ～ \alpha x$

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
> - $\tan x = x + \dfrac{x^3}{3} + o(x^3)$
> - $\arctan x = x - \dfrac{x^3}{3} + o(x^3)$
> - $\ln(1+x) = x - \dfrac{x^2}{2} + \dfrac{x^3}{3} + o(x^3)$
> - $e^x = 1 + x + \dfrac{x^2}{2!} + \dfrac{x^3}{3!} + o(x^3)$
> - $(1+x)^{\alpha} = 1 + \alpha x + \dfrac{\alpha(\alpha - 1)}{2!}x^2 + o(x^2)$
> - $\dfrac{1}{1+x} = 1 - x + x^2 - x^3 + o(x^3)$
> - $\dfrac{1}{1-x} = 1 + x + x^2 + x^3 + o(x^3)$

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
   - （1）$\lim\limits_{x\to 0} \dfrac{\sin x}{x} = 1$
   - （2）$\lim\limits_{x\to \infty} (1 + \dfrac{1}{x})^{x} = e$


9. **求渐近线**
   - （1）垂直渐近线：找函数分母为 0 的点，若分母为 0 的点为 $x_0$，接下来求其极限，若 $\lim\limits_{x\to x_0} f(x) = \infty$，则直线 $x=x_0$ 为曲线 $y = f(x)$ 的垂直渐近线
   - （2）水平渐近线：若 $\lim\limits_{x\to\infty}f(x) = A$，则直线 $y = A$ 为曲线 $y = f(x)$ 的水平渐近线（若 $\lim\limits_{x\to + \infty}f(x) = A$，则曲线右侧有一水平渐近线；若 $\lim\limits_{x\to-\infty}f(x) = A$，则曲线左侧有一水平渐近线）
   - （3）斜渐近线：$k = \lim\limits_{x\to \infty} \dfrac{f(x)}{x} = \lim\limits_{x\to \infty}f'(x)$，$b = \lim\limits_{x\to\infty} [f(x) - kx]$

> ⚠️ 渐近线注意事项：
> - （1）求取渐近线的时候，要小心 $x\to +\infty$ 和 $x\to -\infty$ 两个位置渐近线不同的情况！
> - （2）斜渐近线的求取还有一种快捷方法：利用泰勒展开后略去高阶无穷小直接得到斜渐近线


## 2. 数列极限

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


> 💡 $\{x_n\}$ 与 $\{f(x_n)\}$
> - $\{x_n\}$ 收敛 $\overset{f(x_n) 连续}{\underset{f(x_n) 与 x_n 的映射一一对应 or 具有反函数（连续单调），且\textcolor{red}{极限存在于 f(x_n) 的值域内}}{\rightleftharpoons}}$ $\{f(x_n)\}$ 收敛
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
   - ![problem2](/images/mathematic/problem2.png)


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

> ❓ 易错概念：连续、可导，傻傻分不清？
> - （1）连续：对于任意函数 $f(x)$，$f(x)$ 在 $x_0$ 处有定义且**左右极限均等于** $f(x_0)$，那么称 $f(x)$ 在 $x_0$ 点连续（注意，这里只是一点连续，而不是邻域连续！很容易犯错喵 🐱）
> - （2）导数：导数的条件比连续要苛刻一点，不仅要求在 $x_0$ 处连续，还要求 $f'(x_0) = \lim\limits_{\Delta x \to 0} \dfrac{\Delta y}{\Delta x} = \lim\limits_{\Delta x \to 0} \dfrac{f(x_0 + \Delta x) - f(x_0)}{ \Delta x}$ 存在（即左导数等于右导数），才能说明在 $x_0$ 点可导！
>
> - 例题：【1000a.3.10】、【1000a.4.8】
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

4. $n$ **阶导数**
   - （1）归纳法：逐次求导，找到规律，得出通式
   - （2）莱布尼茨公式：设 $u = u(x)$、$v = v(x)$ 均 $n$ 阶可导，则 $(u \pm v)^{(n)} = u^{(n)} + v^{(n)}$、$(uv)^{(n)} = \sum\limits_{k=0}^{n} C_n^k u^{(n-k)} v^{(k)}$
   - （3）泰勒展开法（一般用于求 $f^{(n)}(0)$）



## 5. 一元函数微分学（应用）