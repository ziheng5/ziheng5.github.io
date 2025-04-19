---
title: Airsim 笔记：Python API 总结
date: 2025-04-13 18:08:04
tags:
    - 强化学习
    - 仿真环境
categories:
    - 强化学习
description: |
    ✈️ 比较热门的无人机强化学习仿真平台之一
---

> 📚 小生的笔记参考西湖大学宁子安大佬的教程：https://www.zhihu.com/column/multiUAV
>
> 另：本文内容均在 Windows 操作系统下进行（Ubuntu 下的仿真环境折腾起来比较麻烦 💦）

## 0. AirSim APIs 简介
---
AirSim 封装了一些 API 接口，使用这些 API 接口，可以用程序跟仿真进行交互。例如可以使用 API 来获取图片、无人机状态、控制无人机/车辆的运动等。

AirSim 的 API 非常丰富，有很多可以调用的功能，可以将这些功能分成以下几类：

- **图像类 API**：获取各种类型的图像、控制云台等；
- **控制仿真运行**：可以控制仿真暂停或继续；
- **碰撞 API**：获取碰撞信息，包括碰撞次数、位置、表面信息、渗透深度等；
- **环境时间 API**：主要是控制太阳在天空中的位置；
- **环境天气 API**：控制环境中的天气：下雨、雪、灰尘、雾气、道路湿滑程度等；
- **环境风 API**：控制环境中的风速和风向等；
- **雷达 API**：添加和控制环境中的雷达传感器；
- **无人机或车辆的 API**：控制运动、获取状态等

AirSim 的 API 有 python 和 C++ 两种使用方式，可以根据自己的习惯任意选择，这里主要记录 python 的使用方式。

在虚拟环境中安装 `msgpack-rpc-python` 和 `airsim` 两个库即可使用

```bash
pip install msgpack-rpc-python
pip install airsim
```

## 1. 控制无人机的起飞和降落
---
先看下面这段代码：

```python
import airsim

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# get control
client.enableApiControl(True)

# unlock
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
client.landAsync().join()

# lock
client.armDisarm(False)

# release control
client.enableApiControl(False)
```

> 在运行上面这段代码之前，需要先将 AirSim 运行起来。
> 
> 先将 `C:\Users\[用户名]\Documents\AirSim` 路径下的 `settings.json` 文件中的仿真模式改为多旋翼，如下所示
>
> ```json
> {
>     "SettingsVersion": 1.2,
>     "SimMode": "Multirotor"
> }
> ```
> 
> 修改完成后，打开之前安装好的 `LandscapeMountains` 环境目录下的 `LandscapeMountains.sln` 文件；Visual Studio 中，选择编辑模式为 `Debug Game Editor` 和 `win64`，确保 `LandscapeMountains` 为启动项目。
>
> 然后点击 `本地Windows调试器`，这时就会打开 Unreal Editor；在 Unreal Editor 中点击 `播放(Play)` 按钮，仿真开始运行。
>
> 无人机初始位置设置的是在空中，所以刚一点击`播放`按钮时，无人机会向下落，最后触地，同时机翼也一直旋转。因为有两个物体的碰撞，所以画面中给出了一个碰撞警告。（初始位置是可以自己调的）
>
> 注意，在 `LandscapeMountain` 这个环境中，水面就是一个物体，它跟石头、树木等有相同的物理特性，所以无人机不会落到水中，而是在睡眠，就像在平地上着陆一样。所以在这个环境中的仿真，就可以把水面当做平地来看待。

代码详解：

```python
client = airsim.MultirotorClient()
```

与 AirSim 建立连接，并且返回句柄（client），后面的每次操作需要使用这个句柄。

如果是汽车仿真，代码是：`client = airsim.CarClient()`；

```python
client.confirmConnection()
```

检查通信是否建立成功，并且会在命令行中打印连接情况，这样就可以判断程序是否和 AirSim 连接正常，如果连接正常会在命令行中打印如下：

```Terminal
Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)
```

```python
client.enableApiControl(True)   # get control
client.enableApiControl(False)  # release control
```

因为安全问题， API 控制默认是不开启的，遥控器有全部的控制权限。所以必须要在程序中使用这个函数来获取控制权。遥控器的操作会抢夺 API 的控制权，同时让 API 获取的控制权失效。使用 `isApiControlEnabled` 可检查 API 是否具有控制权。

可能会有人问为什么最后结束的时候要释放控制权，反正都是仿真，结束仿真就好了。但是实际上 AirSim 的开发人员希望在仿真中的代码可以直接移到现实中使用，所以对于现实中的安全问题，还是开发了获取控制权和释放控制权、解锁和上锁等一系列安全操作。

```python
client.takeoffAsync().join()    # 起飞
client.landAsync().join()   # 降落
```

这两个函数可以让无人机起飞和降落。

很多无人机或者汽车控制的函数都有 `Async` 作为后缀，这些函数在执行的时候会立即返回，这样的话，虽然任务还没有执行完，但是程序可以继续执行下去，而不用等待这个函数的任务在仿真中有没有执行完。

如果你想让程序在这里等待任务执行完，则只需要在后面加上 `.join()`。本例子就是让程序在这里等待无人机起飞任务完成，然后再执行降落任务。

新的任务会打断上一个没有执行完的任务，所以如果 `takeoff` 函数没有加 `.join()`，则最后的表现是无人机还没有起飞就降落了，无人机是不会起飞的。

> **Python 与 AirSim 的通信机制**
>
> AirSim API 使用的是 `TCP/IP` 中的 `msgpack-rpc` 协议，这是一种网络协议，所以如果设置正确，其实可以用两台不同的电脑来做仿真，一台跑 AirSim 和 Unreal，另一台跑 python 程序。
>
>当 AirSim 开始仿真的时候，会打开 41451 端口，并监听这个端口的需求。python 程序使用 `msgpack serialization` 格式向这个端口发送 RPC 包，就可以与 AirSim 进行通信了。

## 2. 控制四旋翼的飞行（core API）
---
### 2.1 控制样例
示例代码：

```python
import airsim
import time

# connect to the Airsim simulator
client = airsim.MultirotorClient()

client.enableApiControl(True)   # get control
client.armDisarm(True)          # unlock
client.takeoffAsync().join()    # takeoff

# square flight
client.moveToZAsync(-3, 1).join()               # 上升到 3m 高度
client.moveToPositionAsync(5, 0, -3, 1).join()  # 飞到（5, 0）点坐标
client.moveToPositionAsync(5, 5, -3, 1).join()  # 飞到（5, 5）点坐标
client.moveToPositionAsync(0, 5, -3, 1).join()  # 飞到（0, 5）点坐标
client.moveToPositionAsync(0, 0, -3, 1).join()  # 飞到（0, 0）点坐标

client.landAsync().join()       # land
client.armDisarm(False)         # lock
client.enableApiControl(False)  # release control
```

上面这段代码实现的效果是：

- 第一阶段：起飞
- 第二阶段：上升到 3m 高度
- 第三阶段：飞正方形
  - 向前飞 5m（沿 x 轴正方向）
  - 向右飞 5m
  - 向后飞 5m
  - 向左飞 5m，回到起飞点
- 第四阶段：降落

> ⚠️ 注意，如果你的仿真在刚开始的时候，四旋翼并不是正前方，说明 player start 的位置是有角度的，将其改成 0 即可。

代码详解：

```python
client.moveToZAsync(-3, 1).join()   # 高度控制
```

`moveToAsync(z, velocity)` 是高度控制 API，第一个参数是**高度**，第二个参数是**速度**。实现的效果是无人机以 1m/s 的速度飞到 3m 高。`.join()` 后缀的意思是程序在这里等待直到任务完成，也就是四旋翼达到 3m 的高度。如果不加 `.join()` 后缀，则不用等待任务是否完成，函数直接返回，程序继续往下执行。

```python
client.moveToPositionAsync(5, 0, -3, 1).join()  # 飞到（5, 0）点坐标
client.moveToPositionAsync(5, 5, -3, 1).join()  # 飞到（5, 5）点坐标
client.moveToPositionAsync(0, 5, -3, 1).join()  # 飞到（0, 5）点坐标
client.moveToPositionAsync(0, 0, -3, 1).join()  # 飞到（0, 0）点坐标
```

`moveToPositionAsync(x, y, z, velocity)` 是水平位置控制 API，`x, y, z` 是全局坐标位置，`velocity` 是速度。实现的效果是以 1m/s 的速度飞到 (5, 0) 点，3m 高的位置。`.join()` 后缀的意思是程序在这里等待直到任务完成，也就是四旋翼到达目标位置点，同时到达设置的高度。如果不加 `.join()` 后缀，则不用等待任务是否完成，函数直接返回，程序继续往下执行。

### 2.2 四旋翼底层飞控通道简介
关于四旋翼的非线性建模和底层控制器设计，在后面详细讲解，目前先简单介绍一下四旋翼的底层飞控可以控制什么量。当四旋翼低速飞行时，其底层飞控可以解耦为 3 个通道：
- 水平通道
- 高度通道
- 偏航通道

这 3 个通道可以分别控制，可以理解为通道之间相互不会影响。水平通道可以控制四旋翼的水平位置、水平速度、水平姿态角；高度通道可以控制垂直高度、垂直速度、垂直加速度；偏航通道可以控制偏航角度、偏航角速度、偏航角加速度。

本文例子程序中 `x, y` 是水平通道的指令，控制四旋翼水平方向的飞行；`z` 是高度通道指令，控制四旋翼的高度。本文例子程序中没有偏航通道指令，默认的偏航是 0，也就是四旋翼自身不会水平旋转，其朝向始终朝向前方。

### 2.3 水平位置控制函数
函数定义

```python
def moveToPositionAsync(
    self,
    x, 
    y,
    z,
    velocity,
    timeout_sec=3e38,
    drivetrain=DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=YawMode(),
    lookahead=-1,
    adaptive_lookahead=1,
    vehicle_name="",
)
```

输入参数包括：

- x, y, z：位置坐标（全局坐标系-北东地）
- velocity：飞行速度（m/s）
- timeout_sec：如果没有相应，超时时间
- drivetrain, yaw_mode：设置飞行朝向模式和 yaw 角控制模式
- lookahead, adaptive_lookahead：设置路径飞行的时候的 yaw 角控制模式
- vehicle_name：控制的四旋翼的名字

`x, y, z, velocity` 这四个参数是必须要设置的量，指示四旋翼以多大的速度飞往哪个坐标点。后面的几个参数都有其默认值，不用设置也可以。

`lookahead` 和 `adaptive_lookahead` 这两个参数是设置当四旋翼飞轨迹的时候的朝向，目前还用不到。

`vehicle_name` 是指将指令发送给哪个四旋翼，当做多个四旋翼协同飞行控制的时候，这个参数就派上用场了，后面会有多机协同编队的内容。

`drivetrain` 和 `yaw_mode` 这两个参数的组合可以设置四旋翼的偏航角控制模式，下面详细介绍。

### 2.4 偏航角控制模式详解

drivetrain 参数可以设置为两个量：
- `airsim.DrivetrainType.ForwardOnly`：始终朝向速度方向
- `airsim.DrivetrainType.MaxDegreeOfFreedom`：手动设置 yaw 角度

`yaw_mode` 必须设置为 `YawMode()` 类型的变量，这个结构体类型包含两个属性：

- `YawMode().is_rate`：True - 设置角速度；False - 设置角度。
- `YawMode().yaw_or_rate`：可以是任意浮点数

下面分几种情况讨论：

#### 情况1（不允许）
```python
drivetrain = airsim.DrivetrainType.ForwardOnly
yaw_mode = airsim.YawMode(True, 0)
client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
```
当 `drivetrain = airsim.DrivetrainType.ForwardOnly` 时，四旋翼始终朝向其飞行的方向，这时 `yaw_mode` 不允许设置为 `YawMode().is_rate = True`。因为前面的参数要求四旋翼朝向运动方向，而 `yaw_mode` 要求四旋翼以一定的角速度旋转，这是矛盾的。

#### 情况2
```python
drivetrain = airsim.DrivetrainType.ForwardOnly
yaw_mode = airsim.YawMode(False, 90)
client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
```
这种情况下，四旋翼的朝向始终与前进方向相差 90 度，也就是四旋翼始终向左侧方向运动。例如：当四旋翼在绕着一个圆心转圈时，其朝向始终指向圆心（这种飞行状态的代码在后面给出）。这里的 90 度可以任意设置。

#### 情况3
```python
drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
yaw_mode = airsim.YawMode(False, 0)
client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
```
这种情况下，不管速度方向是什么，四旋翼的 yaw 角始终等于 0，也就是其朝向始终指向正北方向。如果是 90 度，则始终指向正东方向，而 -90 度，则始终指向正西方向。

#### 情况4
```python
drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
yaw_mode = airsim.YawMode(True, 10)
client.moveToPositionAsync(x, y, z, velocity, drivetrain=drivetrain, yaw_mode=yaw_mode).join()
```

这种情况下，四旋翼不管速度方向是什么，yaw 角以 10 度/秒的速度旋转。

下面总结一下这两个参数的设置对效果的影响：

||ForwardOnly|MaxDegreeOfFreedom|
|---|---|---|
|is_rate=True|不允许|yaw 角以 yaw_or_rate 度/秒旋转|
|is_rate=False|yaw 角相对于速度方向偏差 yaw_or_rate 度|yaw 角相对正北方向偏差 yaw_or_rate 度|

> **AirSim 坐标系定义**
>
> 为什么 0 度是正北方向呢，这就涉及到坐标系的定义了，下面先简单介绍一下坐标系的定义
>
> Unreal 引擎中的坐标系与 AirSim 定义的坐标系是不同的，甚至长度单位都不同。Unreal 的长度单位是厘米，而 AirSim 的长度单位是米。不过不用担心，AirSim 已经非常好地处理了这个问题，你不用在意 Unreal 的坐标系是什么，只需要按照 AirSim 的坐标系设置即可，AirSim 会帮你自动转换的。
>
> 本文先说明两个坐标系的定义：**全局坐标系、机体坐标系**。
>
> **全局坐标系**是固连到大地的，x, y, z 三个坐标轴的指向分别是北、东、地，也就是朝北是 x 轴的正方向，朝南是 x 轴的负方向。全局坐标系的原点位置是大地的某一点（可以在 settings 文件中设置）。
>
> **机体坐标系**是固连到四旋翼机身的，x, y, z 三个坐标轴的指向分别是前、右、下，也就是飞机的前方是 x 轴的正方向，飞机后方是 x 轴的负方向。机体坐标系的原点位置是机体的重心位置。