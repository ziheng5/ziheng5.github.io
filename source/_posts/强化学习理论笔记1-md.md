---
title: 深度学习与 PyTorch 入门指南
date: 2024-11-27 00:37:44
tags:
    - PyTorch
    - 深度学习
categories:
    - 杂谈
description: |
    简单介绍深度学习，并简单讲解PyTorch的基本语法
---
> 本文是我为了授课而编写的一篇文档，可以作为一门引导课程供尚未了解深度学习的新人学习
> 考虑到时间问题，本文档内容较为简略，只适用于了解深度学习与 PyTorch，正常学习请参考视频网站📽或网络教程💻与书籍📖

# 1. 深度学习部分
## 1.1 深度学习是什么？
![deep-learning](./images/class/deep-learning.png)

深度学习是机器学习下的一个基于深度神经网络的子领域。

![nn](./images/class/nn.png)

深度学习是一种结构启发的机器学习，人脑在深度学习方面的结构，被称为人工神经网络


# 2. PyTorch 简介
## 2.1 什么是 PyTorch❓
PyTorch 是当下最热门的开源机器学习与深度学习框架。
## 2.2 PyTorch 可以做什么呢❓
PyTorch 允许你使用 Python 代码来操作和处理数据，并编写机器学习算法。
## 2.3 有哪些人在使用它呢❓
许多世界上最大的科技公司，比如[Meta（Facebook）](https://ai.facebook.com/blog/pytorch-builds-the-future-of-ai-and-machine-learning-at-facebook/)、特斯拉和微软，以及人工智能研究公司如[OpenAI](https://openai.com/blog/openai-pytorch/)，都使用PyTorch来推动研究并将机器学习应用到他们的产品中。

例如，Andrej Karpathy（特斯拉的人工智能负责人）在多个演讲中（[PyTorch DevCon 2019](https://youtu.be/oBklltKXtDE)，[特斯拉AI日2021](https://youtu.be/j0z4FweCy4M?t=2904)）讨论了特斯拉如何使用PyTorch来驱动他们的自动驾驶计算机视觉模型。

此外，PyTorch也被用于其他各个行业，比如农业，用于[在拖拉机上实现计算机视觉](https://medium.com/pytorch/ai-for-ag-production-machine-learning-for-agriculture-e8cfdb9849a1)。
## 2.4 为什么选择 PyTorch❓
相比于过去热门的 TensorFlow 框架，如今的机器学习研究者门更倾向于使用 PyTorch 。截至2022年2月， PyTorch 是 [Papers With Code](https://paperswithcode.com/trends) 上使用最多的深度学习框架，这是一个跟踪机器学习研究论文及其附带代码库的网站。

PyTorch 框架还帮助处理许多事情，比如调用 GPU 加速（让您的代码运行得更快），这些都是在后台自动完成的。

因此，在 PyTorch 框架下，你可以专注于操作数据和编写算法，其他繁琐的操作都可以由 PyTorch 自动完成（如梯度计算、权重更新等）

如果像特斯拉和 Meta（Facebook） 这样的公司使用它来构建他们部署的模型，以驱动数百个应用、驱动数千辆汽车和向数十亿人提供内容，那么它在开发方面显然也是能力出众的。

## 2.5 我们将在本节课中讲解哪些内容
| **主题** | **内容** |
| ----- | ----- |
| **张量简介** | 张量是所有机器学习和深度学习的基本构建块。 |
| **创建张量** | 张量可以表示几乎所有类型的数据（图像、文字、数字表格）。 |
| **从张量中获取信息** | 如果你可以将信息放入张量，你也会想要将其取出。 |
| **操作张量** | 机器学习算法（如神经网络）涉及以多种不同方式操作张量，例如添加、乘法、组合。 |
| **处理张量形状** | 机器学习中最常见问题之一是处理形状不匹配（尝试将形状错误的张量与其他张量混合）。 |
| **张量索引** | 如果你曾在Python列表或NumPy数组上进行过索引，那么在张量上的操作非常相似，只是张量可以有更多的维度。 |
| **混合PyTorch张量和NumPy** | PyTorch使用张量（[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)），NumPy喜欢数组（[`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)），有时你可能想要混合使用这些。 |
| **可复现性** | 机器学习非常实验性，因为它使用大量的*随机性*来工作，有时你可能希望这种*随机性*不要太随机。 |
| **在GPU上运行张量** | GPU（图形处理单元）可以使你的代码更快，PyTorch使得在GPU上运行你的代码变得简单。 |

# 3. PyTorch 基本用法
## 3.1 引入 PyTorch
> **注意：** 在运行这个笔记本中的任何代码之前，你应该已经完成了[PyTorch安装步骤](https://pytorch.org/get-started/locally/)。
>
> 然而，**如果你在Google Colab上运行**，一切都应该可以正常工作（Google Colab自带PyTorch和其他库的安装）。

```python
import torch

# 检查能否调用 cuda
print(torch.cuda.is_available())

# 输出 PyTorch 版本
print(torch.__version__)
```

如果调用 `is_available()` 输出的结果为 `True` ，则说明 PyTorch 安装成功且可以正常调用 cuda

这里 `torch.__version__` 输出了我们安装的 PyTorch 的版本。我这里输出的版本为 `2.5.0` 。一般来说，PyTorch 高版本对低版本具有向下兼容性，但注意：PyTorch `2.X.X` 系列版本与 `1.X.X` 系列版本更能差别较大。

## 3.2 什么是张量❓
现在我们已经导入了 PyTorch,是时候学习关于张量的知识了📚

张量是机器学习的基本构建块，它们的任务是以数值的方式表示数据。

例如，你可以将一张图像表示为形状为 `[3, 224, 224]` 的张量，这意味着 `[颜色通道，高度，宽度]` ，即图像有3个颜色通道（红色、绿色、蓝色），高度为224像素，宽度为224像素。

在张量术语（用于描述张量的定义）中，该张量有三个维度，分别对应颜色通道、高度和宽度。

> 如果想要了解更多关于张量的理论知识，请自行查找**线性代数**相关资料学习📚
> 
>这里我们不具体讲解理论知识，各位理解即可。

下面我们通过代码来理解张量。
## 3.3 在 PyTorch 中创建张量
PyTorch非常喜欢张量。以至于有一个完整的文档页面专门介绍[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)类。

你的第一项家庭作业是[阅读`torch.Tensor`的文档](https://pytorch.org/docs/stable/tensors.html)10分钟。但你可以在稍后进行。

让我们开始编码。
### 3.3.1 标量
我们首先将要创建的是一个**标量**。

标量是一个单独的数字，在张量术语中，它是一个零维张量。

> **注意：** 这是本课程的一个趋势。我们将专注于编写特定的代码。但通常我会设置一些练习，涉及阅读和熟悉PyTorch文档。毕竟，一旦你完成了这个课程，你无疑会想要学习更多。而文档是你经常会去查找的地方。

```Python
# Scalar
scalar = torch.tensor(7)
print(scalar)
```
看到上面打印出的`tensor(7)`了吗？

这意味着尽管 `scalar` 是一个单独的数字，但它的类型是 `torch.Tensor`。

我们可以使用 `ndim` 属性来检查张量的维度。

```Python
print(scalar.ndim)
```
如果我们想要从张量中检索数字呢？

比如，将其从 `torch.Tensor` 转换为Python整数？

为此，我们可以使用 `item()` 方法。

```Python
print(scalar.item())
```
### 3.3.2 向量
好的，现在我们来看一个**向量**。

向量是一种单维度的张量，但它可以包含许多数字。

比如，你可以有一个向量 `[3, 2]` 来描述你房子里的 `[卧室，浴室]` 数量。或者你可以有一个 `[3, 2, 2]` 来描述你房子里的 `[卧室，浴室，车位]` 数量。

这里的重要趋势是向量在它所能代表的内容上是灵活的（和张量一样）。

```Python
# vector
vector = torch.tensor([7, 7])
print(vector)
```
太好了，`vector` 现在包含了两个7。

你认为它将有多少个维度？

```Python
print(vector.ndim)
```
嗯，这有点奇怪，`vector` 包含了两个数字，但只有一个维度。

我来告诉你一个窍门。

你可以通过外部的方括号数量（`[`）来判断PyTorch中张量的维度，你只需要数一边的方括号数量。

`vector` 有多少个方括号？

另一个对张量来说重要的概念是它们的 `shape` 属性。`shape` 告诉你它们内部的元素是如何排列的。

让我们来检查一下 `vector` 的形状。

```Python
print(vector,shape)
```

上述返回 `torch.Size([2])` ，这意味着我们的向量的形状是 `[2]` 。这是因为我们把两个元素放在了方括号内部（`[7, 7]`）。

### 3.3.3 矩阵
现在我们来看一个**矩阵**。
```Python
matrix = torch.tensor([[7, 8],
                       [9, 10]])
print(matrix)
```
矩阵和向量一样灵活，只是它们多了一个维度。
```Python
print(matrix.ndim)
```
`matrix` 有两个维度（你数了一边外部的方括号数量了吗？）。

你认为它将具有什么样的`shape`？
```Python
print(matrix.shape)
```
我们得到的输出是 `torch.Size([2, 2])`，因为 `matrix` 深度为两个元素，宽度也为两个元素。

那么我们来创建一个**张量**怎么样？

```Python
tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(tensor)
```
这个张量看起来真不错。

>张量几乎可以代表任何东西。

你认为它有多少个维度？（提示：使用方括号计数技巧）
```Python
print(tensor.ndim)
```
那么它有着什么样的 `shape` 呢？
```Python
print(tensor.shape)
```
好的，输出是 `torch.Size([1, 3, 3])`。

维度是从外向内排列的。

这意味着有一个3乘3的维度。

让我们总结一下。

| 名称 | 它是什么？ | 维度数量 | 通常/示例是小写还是大写 |
| ----- | ----- | ----- | ----- |
| **标量（scalar）** | 一个单独的数字 | 0 | 小写（`a`） | 
| **向量（vector）** | 有方向的数字（例如风速和方向）但也可能有其他许多数字 | 1 | 小写（`y`） |
| **矩阵（matrix）** | 一个二维数组 | 2 | 大写（`Q`） |
| **张量（tensor）** | 一个n维数组 | 可以是任何数量，0维张量是标量，1维张量是向量 | 大写（`X`） |
## 3.4 随机张量

我们已经确定张量代表了某种形式的数据。

机器学习模型，比如神经网络，会操作张量并在其中寻找模式。

但在使用PyTorch构建机器学习模型时，你很少会手工创建张量（就像我们之前所做的那样）。

相反，机器学习模型通常从大量的随机数张量开始，并在处理数据时调整这些随机数，以便更好地表示数据。

本质上是：

`从随机数开始 -> 查看数据 -> 更新随机数 -> 查看数据 -> 更新随机数...`

作为数据科学家，你可以定义机器学习模型如何开始（初始化）、查看数据（表示）以及更新（优化）其随机数。

我们稍后将亲身实践这些步骤。

现在，让我们看看如何创建一个随机数张量。

我们可以使用 [`torch.rand()`](https://pytorch.org/docs/stable/generated/torch.rand.html) 并传入 `size` 参数来实现。

```Python
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.dtype)
```
`torch.rand()` 的灵活性在于我们可以调整 `size` 参数为任何我们想要的值。

例如，假设你想要一个常见图像形状的随机张量 `[224, 224, 3]`（`[高度，宽度，颜色通道]`）。

```Python
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)
```
## 3.5 零一张量

有时候你可能只想用零或一来填充张量。

这种情况在掩码操作中很常见（比如用零掩码一个张量中的一些值，以让模型知道不要学习这些值）。

让我们使用 [`torch.zeros()`](https://pytorch.org/docs/stable/generated/torch.zeros.html) 创建一个全零的张量。

同样，`size` 参数在这里起作用。

``` Python
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros.dtype)
```

我们可以做同样的事情来创建一个全一的张量，只是使用 [`torch.ones()`](https://pytorch.org/docs/stable/generated/torch.ones.html) 代替。

使用 `torch.ones()` 函数时，你可以通过指定 `size` 参数来创建一个指定形状的张量，其中所有元素都是1。
```Python
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)
```
## 3.6 创建一个范围和类似的张量

有时候你可能想要一系列数字，比如从1到10或者从0到100。

你可以使用 `torch.arange(start, end, step)` 来实现这一点。

其中：
* `start` = 范围的开始（例如 0）
* `end` = 范围的结束（例如 10）
* `step` = 每个值之间的步长（例如 1）

> **注意：** 在Python中，你可以使用 `range()` 来创建一个范围。然而在PyTorch中，`torch.range()` 已经被弃用，可能会显示错误。

```Python
# 下面这一行代码可能会报错
zero_to_ten_deprecated = torch.range(0, 10)

# 建议用下面这一行
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
```
有时候你可能想要一个与另一个张量形状相同的特定类型的张量。

例如，一个与之前张量形状相同的全零张量。

为此，你可以使用 [`torch.zeros_like(input)`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html) 或者 [`torch.ones_like(input)`](https://pytorch.org/docs/1.9.1/generated/torch.ones_like.html)，它们分别返回一个与 `input` 形状相同但填充了零或一的张量。

使用 `torch.zeros_like(input)` 和 `torch.ones_like(input)` 函数时，你只需要提供一个已有的张量作为 `input` 参数，这两个函数就会返回一个新的张量，其形状与输入张量相同，但所有元素分别是0或1。

```Python
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeors)
```
## 3.7 张量数据类型

在PyTorch中有许多不同的[张量数据类型](https://pytorch.org/docs/stable/tensors.html#data-types)。

有些是特定于CPU的，有些则更适合GPU。

了解它们可能需要一些时间。

通常，如果你在任何地方看到 `torch.cuda`，那么这个张量是用于GPU的（因为Nvidia GPU使用一个名为CUDA的计算工具包）。

最常见的类型（通常也是默认类型）是`torch.float32`或`torch.float`。

这被称为“32位浮点数”。

但也有16位浮点数（`torch.float16` 或 `torch.half`）和64位浮点数（`torch.float64` 或 `torch.double`）。

还有8位、16位、32位和64位整数。

还有更多！

> **注意：** 整数是一个平坦的圆形数字，如 `7`，而浮点数有一个小数 `7.0`。

所有这些的原因与**计算中的精度**有关。

精度是用来描述一个数字的细节量。

精度值越高（8、16、32），用来表达一个数字的细节和数据就越多。

这在深度学习和数值计算中很重要，因为你正在进行如此多的操作，你拥有的细节越多，你就需要使用更多的计算资源。

因此，低精度数据类型通常计算速度更快，但在评估指标如准确性上会牺牲一些性能（计算速度更快但准确性较低）。

> **资源：**
  * 请参阅[PyTorch文档](https://pytorch.org/docs/stable/tensors.html#data-types)，了解所有可用张量数据类型的列表。
  * 阅读[维基百科页面](https://en.wikipedia.org/wiki/Precision_(computer_science))，了解计算中精度的概述。

让我们看看如何创建具有特定数据类型的张量。我们可以使用`dtype`参数来实现。

```Python
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, 
                               device=None, 
                               requires_grad=False)
print(float_32_tensor.shape)
print(float_32_tensor.dtype)
print(float_32_tensor.device)
```
除了形状问题（张量形状不匹配）之外，你在PyTorch中遇到的另外两个最常见的问题就是数据类型和设备问题。

例如，一个张量是`torch.float32`，而另一个是`torch.float16`（PyTorch通常希望张量是相同的格式）。

或者你的一个张量在CPU上，而另一个在GPU上（PyTorch希望张量之间的计算在同一设备上进行）。

我们稍后会更多地讨论设备问题。

现在，让我们创建一个`dtype=torch.float16`的数据类型的张量。

``` Python
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)

float_16_tensor.dtype
```
## 3.8 从张量中获取信息
一旦你创建了张量（或者别人为你创建了张量，或者PyTorch模块为你创建了它们），你可能想要从它们那里获取一些信息。

我们已经见过这些，但是关于张量，你最想了解的三个最常见的属性是：
* `shape` - 张量的形状是什么？（一些操作需要特定的形状规则）
* `dtype` - 张量内部的元素存储的数据类型是什么？
* `device` - 张量存储在哪个设备上？（通常是GPU或CPU）

让我们创建一个随机张量，并找出有关它的详细信息。

```Python
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"张量的 shape : {some_tensor.shape}")
print(f"张量的类型 : {some_tensor.dtype}")
print(f"张量存储的位置 : {some_tensor.device}")
```
> **注意：** 当你在PyTorch中遇到问题时，这通常与上述三个属性中的一个有关。

## 3.9 张量运算
在深度学习中，数据（图像、文本、视频、音频、蛋白质结构等）被表示为张量。

模型通过研究这些张量并在张量上执行一系列操作（可能是100万次以上）来创建输入数据中模式的表示。

这些操作通常是在以下几项之间进行的：
* 加法
* 减法
* 乘法（逐元素）
* 除法
* 矩阵乘法

仅此而已。当然，这里和那里还有一些其他的操作，但这些是神经网络的基本构建块。

以正确的方式堆叠这些构建块，你可以创建最复杂的神经网络（就像乐高积木一样！）。

### 3.9.1 基本操作

让我们从几个基本操作开始，加法（`+`）、减法（`-`）、乘法（`*`）。

它们的工作方式正如你所想象的那样。
```Python
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
```

注意到上面的张量值最终并没有变成 `tensor([110, 120, 130])`，这是因为张量内部的值除非被重新赋值，否则不会改变。

```Python
print(tensor)
```
好的，让我们从一个数字中减去，并这次我们将重新赋值给 tensor 变量。
```Python
tensor = tensor - 10
print(tensor)
```
```Python
tensor = tensor + 10
print(tensor)
```

PyTorch还提供了许多内置函数，比如 [`torch.mul()`](https://pytorch.org/docs/stable/generated/torch.mul.html#torch.mul)（乘法的简称）和 [`torch.add()`](https://pytorch.org/docs/stable/generated/torch.add.html)，用于执行基本操作。

这些内置函数提供了一种不使用Python运算符（如 `*` 和 `+` ）而是使用PyTorch函数的方式来执行张量运算，这在某些情况下非常有用，比如当需要明确指定操作的维度或者当需要保持操作的梯度信息时。

```Python
print(torch.multiply(tensor, 10))

print(tensor)
```

然而，更常见的是使用操作符符号，如 `*`，而不是 `torch.mul()`。

```Python
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)
```
### 3.9.2 矩阵乘法

在机器学习和深度学习算法（如神经网络）中最常用的操作之一是[矩阵乘法](https://www.mathsisfun.com/algebra/matrix-multiplying.html)。

PyTorch在[`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html)方法中实现了矩阵乘法功能。

需要记住的矩阵乘法的两个主要规则是：

1. **内维度**必须匹配：
  * `(3, 2) @ (3, 2)` 不可行
  * `(2, 3) @ (3, 2)` 可行
  * `(3, 2) @ (2, 3)` 可行
2. 结果矩阵的形状是**外维度**的形状：
 * `(2, 3) @ (3, 2)` -> `(2, 2)`
 * `(3, 2) @ (2, 3)` -> `(3, 3)`

> **注意：** 在Python中，"`@`"是矩阵乘法的符号。

> **资源：** 你可以在[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.matmul.html)中查看使用`torch.matmul()`进行矩阵乘法的所有规则。

让我们创建一个张量，并在其上执行逐元素乘法和矩阵乘法。

```Python
import torch
tensor = torch.tensor([1, 2, 3])
print(tensor.shape)
```
逐元素乘法和矩阵乘法之间的差异在于值的相加方式。

对于我们的 `tensor` 变量，其值为 `[1, 2, 3]`：

| 操作 | 计算 | 代码 |
| ----- | ----- | ----- |
| **逐元素乘法** | `[1*1, 2*2, 3*3]` = `[1, 4, 9]` | `tensor * tensor` |
| **矩阵乘法** | `[1*1 + 2*2 + 3*3]` = `[14]` | `tensor.matmul(tensor)` |

```Python
print(tensor * tensor)

print(torch.matmul(tensor, tensor))

print(tensor @ tensor)
```

你可以手工进行矩阵乘法，但不建议这样做。

内置的 `torch.matmul()` 方法更快。

```Python
%%time
# 手动计算 be like：
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
print(value)
```
```Python
%%time
# 用函数计算 be like：
torch.matmul(tensor, tensor)
```
### 3.9.3 深度学习中最常见错误之一（形状错误）

由于深度学习的大部分内容涉及矩阵的乘法和操作，而矩阵对于形状和大小的组合有严格的规则，因此你在深度学习中最常见的错误之一就是形状不匹配。
```Python
# 下面这个代码会报错
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # (this will error)
```
我们可以通过使 `tensor_A` 和 `tensor_B` 的内维度匹配来实现它们之间的矩阵乘法。

实现这一点的方法之一是使用**转置**（交换给定张量的维度）。

你可以使用以下任一方法在PyTorch中执行转置：
* `torch.transpose(input, dim0, dim1)` - 其中 `input` 是要转置的张量，`dim0` 和 `dim1` 是要交换的维度。
* `tensor.T` - 其中 `tensor` 是要转置的张量。

我们来尝试后者。
```Python
print(tensor_A)
print(tensor_B)
```
```Python
print(tensor_A)
print(tensor_B.T)
```
```Python
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")
```

您也可以使用 [`torch.mm()`](https://pytorch.org/docs/stable/generated/torch.mm.html)，它是 `torch.matmul()` 的简称。

`torch.mm()` 是一个用于矩阵乘法的便捷函数，它是 `torch.matmul()` 的别名，专门用于矩阵乘法操作。这个函数适用于两个二维张量的乘法，其中第一个张量的列数必须与第二个张量的行数相匹配。结果张量将具有与第一个张量相同的行数和与第二个张量相同的列数。
```Python
print(torch.mm(tensor_A, tensor_B.T))
```
### 3.9.4 神经网络中的矩阵运算
神经网络中充满了矩阵乘法和点积。

[`torch.nn.Linear()`](https://pytorch.org/docs/1.9.1/generated/torch.nn.Linear.html) 模块（我们稍后将实际看到它的应用），也称为前馈层或全连接层，实现了输入 `x` 和权重矩阵 `A` 之间的矩阵乘法。

$$
y = x \cdot A^T + b
$$

其中：
* `x` 是该层的输入（深度学习是像 `torch.nn.Linear()` 这样的层堆叠在一起）。
* `A` 是由该层创建的权重矩阵，这最初是随机数，随着神经网络学习以更好地表示数据中的模式而进行调整（注意 "`T`"，这是因为权重矩阵需要转置）。
  * **注意：** 你可能还会经常看到 `W` 或其他字母如 `X` 用来表示权重矩阵。
* `b` 是偏置项，用于轻微偏移权重和输入。
* `y` 是输出（对输入的操纵，希望发现其中的模式）。

这是一个线性函数（你可能在高中或其他场合见过类似 $y = mx + b$ 的东西），可以用来画一条直线！

让我们来玩转一下线性层。

尝试改变下面的 `in_features` 和 `out_features` 的值，看看会发生什么。

你有没有注意到与形状有关的任何事情？
```Python
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
print(torch.manual_seed(42))
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
print(x = tensor_A)
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```
> **问题：** 如果你将上面的 `in_features` 从 2 改为 3 会发生什么？会出现错误吗？你如何改变输入（`x`）的形状以适应这个错误？提示：我们之前对 `tensor_B` 做了什么？

### 3.9.5 寻找最小值、最大值、平均值、总和等（聚合）

现在我们已经看到了一些操作张量的方法，让我们来了解一些聚合它们的方法（从多个值变为较少的值）。

首先，我们将创建一个张量，然后找到它的最大值、最小值、平均值和总和。
```Python
x = torch.arange(0, 100, 10)
print(x)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
```

> **注意：** 你可能会发现一些方法，比如 `torch.mean()`，要求张量必须是 `torch.float32`（最常见的）或另一种特定的数据类型，否则操作将失败。

你也可以使用 `torch` 方法来执行上述相同的操作。
### 3.9.6 位置最小值/最大值

你还可以分别使用 [`torch.argmax()`](https://pytorch.org/docs/stable/generated/torch.argmax.html) 和 [`torch.argmin()`](https://pytorch.org/docs/stable/generated/torch.argmin.html) 来找到张量中最大值或最小值出现的位置索引。

这在你只想要最高（或最低）值的位置而不是实际值本身时非常有用（我们将在稍后使用[softmax激活函数](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)的章节中看到这一点）。

```Python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```

### 3.9.7 改变张量数据类型

正如前面提到的，深度学习操作中一个常见的问题是张量的数据类型不同。

如果一个张量是 `torch.float64` 而另一个是 `torch.float32`，你可能会碰到一些错误。

但是有一个解决办法。

你可以使用 [`torch.Tensor.type(dtype=None)`](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html) 来改变张量的数据类型，其中 `dtype` 参数是你想要使用的数据类型。

首先，我们将创建一个张量并检查其数据类型（默认是 `torch.float32`）。
```Python
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
```
现在我们将创建一个与之前相同的张量，但是将其数据类型更改为 `torch.float16`。
```Python
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)
```
我们也可以进行类似的操作来创建一个 `torch.int8` 类型的张量。
```Python
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)
```
> **注意：** 不同的数据类型一开始可能会让人感到困惑。但可以这样理解，数字越小（例如32、16、8），计算机存储该值的精度就越低。存储量越少，通常意味着更快的计算速度和更小的模型总体大小。基于移动设备的神经网络通常使用8位整数进行操作，它们比float32的对应物更小、运行更快，但准确度较低。想要了解更多，可以阅读关于[计算中的精度](https://en.wikipedia.org/wiki/Precision_(computer_science))的内容。

> **练习：** 到目前为止，我们已经介绍了一些张量方法，但在 [`torch.Tensor` 文档](https://pytorch.org/docs/stable/tensors.html) 中还有更多。我建议你花10分钟浏览一下，看看有没有引起你注意的方法。点击它们，然后在代码中自己尝试编写这些方法，看看会发生什么。

### 3.9.8 重塑、堆叠、压缩和取消压缩

很多时候，你可能想要重塑或改变你的张量的维度，而实际上并不改变它们内部的值。

为此，一些常用的方法如下：

| 方法 | 一句话描述 |
| ----- | ----- |
| [`torch.reshape(input, shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape) | 将 `input` 重塑为 `shape`（如果兼容），也可以使用 `torch.Tensor.reshape()`。 |
| [`Tensor.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | 返回原始张量的一个视图，其形状不同 `shape`，但与原始张量共享相同的数据。 |
| [`torch.stack(tensors, dim=0)`](https://pytorch.org/docs/1.9.1/generated/torch.stack.html) | 沿着一个新的维度（`dim`）连接一系列 `tensors`，所有的 `tensors` 必须大小相同。 |
| [`torch.squeeze(input)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | 压缩 `input` 以移除所有值为 `1` 的维度。 |
| [`torch.unsqueeze(input, dim)`](https://pytorch.org/docs/1.9.1/generated/torch.unsqueeze.html) | 返回 `input`，并在 `dim` 处添加一个值为 `1` 的维度。 |
| [`torch.permute(input, dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | 返回原始 `input` 的一个视图，其维度被置换（重新排列）为 `dims`。 |

为什么要使用这些方法？

因为深度学习模型（神经网络）都是以某种方式操作张量。由于矩阵乘法的规则，如果你有形状不匹配的问题，你会遇到错误。这些方法帮助你确保你的张量的正确的元素与其他张量的正确的元素混合。

让我们来尝试一下它们。

首先，我们将创建一个张量。

```Python
import torch
x = torch.arange(1., 8.)
print(x)
print(x.shape)
```

```Python
# Add an extra dimension
x_reshaped = x.reshape(1, 7)
print(x_reshaped)
print(x_reshaped.shape)
```

```Python
# Change view (keeps same data as original but changes view)
# See more: https://stackoverflow.com/a/54507446/7900723
z = x.view(1, 7)
print(z)
print(z.shape)
```

请记住，使用 `torch.view()` 改变张量的视图实际上只创建了同一个张量的*新视图*。

所以改变视图也会改变原始张量。

```Python
z[:, 0] = 5
print(z)
print(x)
```

```Python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)
```

那么，如何从张量中移除所有单一维度呢？

为此，你可以使用 `torch.squeeze()`（我记作*挤压*张量，使其只有大于1的维度）。

```Python
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
```

要执行 `torch.squeeze()` 的反向操作，你可以使用 `torch.unsqueeze()` 在特定索引处添加一个值为1的维度。

```Python
print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
```

你还可以使用 `torch.permute(input, dims)` 重新排列轴的顺序，其中 `input` 被转换成具有新 `dims` 的*视图*。

```Python
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
```

> **注意**：因为置换返回的是一个*视图*（与原始数据共享相同的数据），所以置换后的张量中的值将与原始张量的值相同，如果你在视图中改变值，原始张量的值也会改变。

## 3.10 索引（从张量中选择数据）

有时你想要从张量中选择特定的数据（例如，仅第一列或第二行）。

为此，你可以使用索引。

如果你曾经在Python列表或NumPy数组上进行过索引，那么在PyTorch中对张量进行索引是非常相似的。

```Python
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```
索引值从外维度到内维度（检查一下方括号）。
```Python
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")
```

你还可以使用 `:` 来指定“这个维度的所有值”，然后使用逗号（`,`）来添加另一个维度。

```Python
print(x[:, 0])
print(x[:, :, 1])
print(x[:, 1, 1])
print(x[0, 0, :])
```

索引一开始可能会相当令人困惑，特别是对于较大的张量（我仍然需要尝试多次索引才能正确）。但通过一些练习，并遵循数据探索者的座右铭（***可视化，可视化，再可视化***），你将开始掌握它。

