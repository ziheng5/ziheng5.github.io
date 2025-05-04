---
title: Airsim 笔记：Python API 总结
date: 2025-04-13 18:08:04
tags:
    - 仿真
categories:
    - 食用指南
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

## 3. 速度控制
---

### 3.1 控制样例
老样子，先上示例：

```python
import airsim
import time

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.enableApiControl(True)       # 获得控制权
client.armDisarm(True)              # 解锁
client.takeoffAsync().join()        # 第一阶段：起飞

client.moveToZAsync(-2, 1).join()   # 第二阶段：上升到 2m 高度

# 飞正方形
client.moveByVelocityZAsync(1, 0, -2, 8).join() # 第三阶段：以 1m/s 速度向前飞 8 秒钟
client.moveByVelocityZAsync(0, 1, -2, 8).join() # 第三阶段：以 1m/s 速度向右飞 8 秒钟
client.moveByVelocityZAsync(-1, 0, -2, 8).join() # 第三阶段：以 1m/s 速度向后飞 8 秒钟
client.moveByVelocityZAsync(0, -1, -2, 8).join() # 第三阶段：以 1m/s 速度向左飞 8 秒钟

# 悬停 2 秒钟
client.hoverAsync().join()      # 第四阶段：悬停 6 秒
time.sleep(6)

client.landAsync().join()       # 第五阶段：降落
client.armDisarm(False)         # 上锁
client.enableApiControl(False)  # 释放控制权
```

> 在视频演示的时候，可以设置固定视角来观察四旋翼运动的轨迹，方法如下。
>
> 点击 `运行/Play` 按钮后，中间的视角默认是跟随视角，视角的设置决定了摄像机如何跟随四旋翼，对于四旋翼来说，默认是 `跟随/Flywithme` 模式，对于汽车来说，默认是 `SpringArmChase` 模式。下面列出这些模式：
>
> - `跟随/FlyWithMe`：以 6 自由度跟随四旋翼
> - `FPV`：机载摄像头视角
> - `地面观察者/GroundObserver`：在地面上以 XY 平面自由度跟随四旋翼
> - `手动/Manual`：手动设置摄像机的位置
> - `弹性机臂跟随/SpringArmChase`：摄像机固定在一个隐形的与汽车连在一起的弹性机臂上，跟随汽车，所以会有一些时延。
> - `NoDisplay`：不显示画面，这可以提高渲染性能，而且不影响 APIs
>
> 把鼠标移动到中间的视野中随便一个位置，点一下鼠标左键，这时鼠标就消失了。这时可以设置视角模式：
> - `F` 按键：FPV
> - `B` 按键：跟随/FlyWithMe
> - `\` 按键：地面观察者/GroundObserver
> - `/` 按键：弹性机臂跟随/SpringArmChase
> - `M` 按键：手动/Manual
>
> 按 `M` 按键进入手动设置模式，可以设置摄像机的位置：
>
> - 方向键：前进、后退、向左、向右移动
> - page up/down：上下移动
> - W，S 按键：俯仰转动
> - A，D 按键：偏航转动

### 3.2 代码详解

```python
client.moveByVelocityZAsync(vx, vy, z, duration).join()
```
速度控制函数，让四旋翼在 `z` 的高度，以 `vx`，`vy` 的速度，飞行 `duration` 秒。

```python
client.hoverAsync().join()
```

这句指令的功能是让四旋翼在当前位置悬停。

### 3.3 速度控制方法 API
```python
def moveByVelocityZAsync(
    self,
    vx,
    vy,
    z,
    duration,
    drivetrain=DrivetrainType.MaxDegreeOfFreedom,
    yaw_mode=YawMode(),
    vehicle_name="",
)
```

### 3.4 速度控制误差
四旋翼是一个非线性系统，给一个速度指令，它是不可能瞬时到达的，而且这个速度指令与当前的速度之差越大，达到这个速度指令的调节时间就越长。所以在上面的程序中，最后的四旋翼并没有回到起点位置。

为了做对比，可以把速度增大一些，下面的代码是四旋翼以 8m/s 的速度飞行 2 秒钟，分别向前右后左四个方向各飞行一次，最后离起点位置偏差更大：

```python
import airsim
 import time
 ​
 client = airsim.MultirotorClient()  # connect to the AirSim simulator
 client.enableApiControl(True)       # 获取控制权
 client.armDisarm(True)              # 解锁
 client.takeoffAsync().join()        # 第一阶段：起飞
 ​
 client.moveToZAsync(-2, 1).join()   # 第二阶段：上升到2米高度
 ​
 # 飞正方形
 client.moveByVelocityZAsync(8, 0, -2, 2).join()     # 第三阶段：以8m/s速度向前飞2秒钟
 client.moveByVelocityZAsync(0, 8, -2, 2).join()     # 第三阶段：以8m/s速度向右飞2秒钟
 client.moveByVelocityZAsync(-8, 0, -2, 2).join()    # 第三阶段：以8m/s速度向后飞2秒钟
 client.moveByVelocityZAsync(0, -8, -2, 2).join()    # 第三阶段：以8m/s速度向左飞2秒钟
 ​
 # 悬停 2 秒钟
 client.hoverAsync().join()          # 第四阶段：悬停6秒钟
 time.sleep(6)
 ​
 client.landAsync().join()           # 第五阶段：降落
 client.armDisarm(False)             # 上锁
 client.enableApiControl(False)      # 释放控制权
```

## 4. 四旋翼飞圆形及画图 API
---
四旋翼飞圆形和之前将的飞正方形有本质的不同，为了能够飞一个完美的圆形，只靠飞点是不够的，需要获取无人机当前的位置状态来计算速度指令。接下来介绍以速度指令的方式，让四旋翼飞一个圆形。

完整代码如下：

```python
import airsim
import numpy as np
import math
import time

client = airsim.MultirotorClient()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()

center = np.array([[0], [5]])   # 圆心设置
speed = 2                       # 速度设置
radius = 5                      # 半径设置
clock_wise = True               # 顺时针或逆时针设置

pos_reserve = np.array([[0.], [0.], [-3.]])

# 速度控制
for i in range(2000):
    # 获取无人机当前的位置
    state = client.simGetGroundTruthKinematics()
    pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])

    # 计算径向速度的方向向量
    dp = pos[0:2] - center
    if np.linalg.norm(dp) - radius > 0.1:
        vel_dir_1 = -dp
    elif np.linalg.norm(dp) - radius < 0.1:
        vel_dir_1 = dp

    # 计算切向速度的方向向量
    theta = math.atan2(dp[1, 0], dp[0, 0])
    if clock_wise:
        # 如果是顺时针
        theta += math.pi/2
    else:
        # 如果是逆时针
        theta -= math.pi/2
    v_dir_2 = np.array([[math.cos(theta)], [math.sin(theta)]])

    # 计算最终速度的方向向量
    v_dir = 0.08 * vel_dir_1 + v_dir_2

    # 计算最终速度指令
    v_cmd = speed * v_dir/np.linalg.norm(v_dir)

    # 速度控制
    client.moveByVelocityZAsync(v_cmd[0, 0], v_cmd[1, 0], -3, 1)

    # 画图
    point_reserve = [airsim.Vector3r(pos_reserve[0, 0], pos_reserve[1, 0], pos_reserve[2, 0])]
    point = [airsim.Vector3r(pos[0, 0], pos[1, 0], pos[2, 0])]
    point_end = pos + np.vstack((v_cmd, np.array([[0]])))
    point_end = [airsim.Vector3r(point_end[0, 0], point_end[1, 0], point_end[2, 0])]
    client.simPlotArrows(point, point_end, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
    client.simPlotLineList(point_reserve+point, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)

    # 循环
    pos_reserve = pos
    time.sleep(0.02)
```

- **第一步**：计算径向速度向量方向
    计算向量的方向，也就是不用管大小是多少，只要向量方向正确即可。

    因为无人机很可能并没有在圆周上，所以要给它径向的指令，来限制它在圆周上运动，不能偏离太远。
- **第二步**：计算切向速度向量方向
    切向速度向量计算我是先求的当前方位角（相对于圆心的），然后加（或减）90 度，得到的向量。

- **第三步**：两个向量加权相加并单位化，得到最终单位速度方向向量
    两个向量加权平均，k 值是需要调节的，这样曲线才圆润。

- **第四步**：单位向量乘速度，就是最终速度指令


## 5. 多无人机协同集群编队（分布式控制算法）
---
### 5.1 多无人机编队控制理论
集群编队控制有集中式和分布式两种，集中式控制受限于中心计算资源的限制，无法做到大规模集群编队。而分布式控制，理论上可以做到无限规模的集群编队。这部分实现的集群控制算法就属于一种分布式的控制算法。

分布式集群控制最早是由 Reynolds 在 1987 年提出的分布式协同的三定律：避碰、速度一致、中心聚集。只要每个无人机都满足这三个条件就能形成集群飞行的效果。后续的集群研究都是在三定律的基础上进行的，或者说后续的集群算法都基本满足这三个定律，只是满足的形式各有区别。

这里参考论文中的集群算法，在 AirSim 中实现多无人机集群飞行的效果。论文中的集群控制算法是三个速度指令相加的：
- 避碰：$v_i^{sep}=-\frac{k^{sep}}{||N_i||} \Sigma \frac{r_{ij}}{||r_{ij}||^2}$
- 中心聚集：$v_i^{coh}=\frac{k^{coh}}{||N_i||} \Sigma r_{ij}$
- 整体迁移（速度一致）：$v_i^{mig}=k^{mig} \frac{r_i^{mig}}{||r^{mig}_i||}$

其中：

- $k$ 是系数
- $N_i = {agents \quad j:j \ne i \wedge ||r_{ij}|| < r^{max}}$
  - $N_i$ 表示的是每个无人机的邻居是如何选择的，也就是说对于每个无人机来说，其周围半径范围内的其他无人机都是自己的邻居。
  - 所以 $||N_i||$ 表示的是第 $i$ 个无人机周围的邻居个数。
  - $r_{ij}=p_j-p_i$ 表示的是两架无人机之间的距离
  - $r_i^{mig}=p^{mig}-p_i$ 表示的是第 $i$ 个无人机到目标点的距离。

这个集群算法比较简单，避碰和中心聚集这两项的公式很好理解。论文将速度一致项变形为了整体迁移，因为论文想要实现的目标是让无人机集群到达全局的一个固定位置。

添加速度限幅后，最终无人机的速度指令为：

$v_i=\frac{\bar{v_i}}{||\bar{v_i}||} min(||\bar{v_i}||, v^{max})$

其中：$\bar{v_i}=v_i^{sep} + v_i^{coh} + v_i^{mig}$

其中的参数设置如下：

- $r^{max}=20m$
- $v^{max}=2m/s$
- $k^{sep}=7$
- $k^{coh}=1$
- $k^{mig}=1$

最后形成的集群效果应该是每两个相邻的无人机都保持同样的距离编队。

### 5.2 AirSim 集群编队实现
#### 修改 settings.json 文件
首先需要修改 `settings.json` 文件，添加多个无人机。初始位置每个无人机的位置都是自己随便设置的。

```json
"Vehicles": {
    "UAV1": {
        "VehicleType": "SimpleFlight",
        "X":0, "Y":0, "Z":0,
        "Yaw": 0
    },
    "UAV2": {
      "VehicleType": "SimpleFlight",
      "X": 2, "Y": 0, "Z": 0,
      "Yaw": 0
    },
    "UAV3": {
      "VehicleType": "SimpleFlight",
      "X": 4, "Y": 0, "Z": 0,
      "Yaw": 0
    },
    "UAV4": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": -3, "Z": 0,
      "Yaw": 0
    },
    "UAV5": {
      "VehicleType": "SimpleFlight",
      "X": 2, "Y": -2, "Z": 0,
      "Yaw": 0
    },
    "UAV6": {
      "VehicleType": "SimpleFlight",
      "X": 4, "Y": -3, "Z": 0,
      "Yaw": 0
    },
    "UAV7": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 3, "Z": 0,
      "Yaw": 0
    },
    "UAV8": {
      "VehicleType": "SimpleFlight",
      "X": 2, "Y": 2, "Z": 0,
      "Yaw": 0
    },
    "UAV9": {
      "VehicleType": "SimpleFlight",
      "X": 4, "Y": 3, "Z": 0,
      "Yaw": 0
    }
}
```

#### 编队算法代码

```python
import airsim
import time
import numpy as np

origin_x = [0, 2, 4, 0, 2, 4, 0, 2, 4]       # 无人机初始位置
origin_y = [0, 0, 0, -3, -2, -3, 3, 2, 3]

def get_UAV_pos(client, vehicle_name="SimpleFlight"):
    global origin_x
    global origin_y
    state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    x = state.position.x_val
    y = state.position.y_val
    i = int(vehicle_name[3])
    x += origin_x[i - 1]
    y += origin_y[i - 1]
    pos = np.array([[x], [y]])
    return pos


client = airsim.MultirotorClient()  # connect to the AirSim simulator
for i in range(9):
    name = "UAV"+str(i+1)
    client.enableApiControl(True, name)     # 获取控制权
    client.armDisarm(True, name)            # 解锁（螺旋桨开始转动）
    if i != 8:                              # 起飞
        client.takeoffAsync(vehicle_name=name)
    else:
        client.takeoffAsync(vehicle_name=name).join()

for i in range(9):                          # 全部都飞到同一高度层
    name = "UAV" + str(i + 1)
    if i != 8:
        client.moveToZAsync(-3, 1, vehicle_name=name)
    else:
        client.moveToZAsync(-3, 1, vehicle_name=name).join()


# 参数设置
v_max = 2     # 无人机最大飞行速度
r_max = 20    # 邻居选择的半径
k_sep = 7     # 控制算法系数
k_coh = 1
k_mig = 1
pos_mig = np.array([[25], [0]])   # 目标位置
v_cmd = np.zeros([2, 9])

for t in range(500):
    for i in range(9):   # 计算每个无人机的速度指令
        name_i = "UAV"+str(i+1)
        pos_i = get_UAV_pos(client, vehicle_name=name_i)
        r_mig = pos_mig - pos_i
        v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
        v_sep = np.zeros([2, 1])
        v_coh = np.zeros([2, 1])
        N_i = 0
        for j in range(9):
            if j != i:
                name_j = "UAV"+str(j+1)
                pos_j = get_UAV_pos(client, vehicle_name=name_j)
                if np.linalg.norm(pos_j - pos_i) < r_max:
                    N_i += 1
                    r_ij = pos_j - pos_i
                    v_sep += -k_sep * r_ij / np.linalg.norm(r_ij)
                    v_coh += k_coh * r_ij
        v_sep = v_sep / N_i
        v_coh = v_coh / N_i
        v_cmd[:, i:i+1] = v_sep + v_coh + v_mig

    for i in range(9):   # 每个无人机的速度指令执行
        name_i = "UAV"+str(i+1)
        client.moveByVelocityZAsync(v_cmd[0, i], v_cmd[1, i], -3, 0.1, vehicle_name=name_i)

# 循环结束
client.simPause(False)
for i in range(9):
    name = "UAV"+str(i+1)
    if i != 8:                                              # 降落
        client.landAsync(vehicle_name=name)
    else:
        client.landAsync(vehicle_name=name).join()
for i in range(9):
    name = "UAV" + str(i + 1)
    client.armDisarm(False, vehicle_name=name)              # 上锁
    client.enableApiControl(False, vehicle_name=name)       # 释放控制权
```

## 6. 状态读取 API & 多无人机位置读取函数封装
---
### 6.1 无人机真值状态读取 API
无人机的真值状态读取接口可以得到无人机的真值状态（无误差），此接口同样适用于无人车，属于 vehicle 类的通用 API。接口的调用格式如下：

```python
kinematic_state_groundtruth = client.simGetGroundTruthKinematics(vehicle_name='')
```

其中返回值 `kinematic_state_groundtrutth` 包含 6 个属性：

```python
# state_groundtruth 的 6 个属性
kinematic_state_groundtruth.position            # 位置信息
kinematic_state_groundtruth.linear_velocity     # 速度信息
kinematic_state_groundtruth.linear_acceleration # 加速度信息
kinematic_state_groundtruth.orientation         # 姿态信息
kinematic_state_groundtruth.angular_velocity    # 姿态角速度信息
kinematic_state_groundtruth.angular_acceleration  # 姿态角加速度信息
```

以上 6 个属性中，除了 `orientation` 其他几个都包含了 `x_val`、`y_val` 和 `z_val` 三个属性，分别代表 x,y,z 3 个方向的值。例如位置真值的读取如下：

```python
# 无人机全局位置坐标真值
x = kinematic_state_groundtruth.position.x_val  # 全局坐标系下，x 轴方向的坐标
y = kinematic_state_groundtruth.position.y_val  # 全局坐标系下，y 轴方向的坐标
z = kinematic_state_groundtruth.position.z_val  # 全局坐标系下，z 轴方向的坐标
```

同理，速度、加速度、姿态角速度、姿态角加速度真值的读取如下：

```python
# 无人机全局速度真值
vx = kinematic_state_groundtruth.linear_velocity.x_val    # 无人机 x 轴方向（正北）
vy = kinematic_state_groundtruth.linear_velocity.y_val    # 无人机 y 轴方向（正东）
vz = kinematic_state_groundtruth.linear_velocity.z_val    # 无人机 z 轴方向（垂直）

# 无人机全局加速度真值
ax = kinematic_state_groundtruth.linear_acceleration.x_val
ay = kinematic_state_groundtruth.linear_acceleration.y_val
az = kinematic_state_groundtruth.linear_acceleration.z_val

# 机体角速度
kinematic_state_groundtruth.angular_velocity.x_val    # 机体俯仰角速率
kinematic_state_groundtruth.angular_velocity.y_val    # 机体翻滚角速率
kinematic_state_groundtruth.angular_velocity.z_val    # 机体偏航角速率

# 机体角加速度
kinematic_state_groundtruth.angular_acceleration.x_val
kinematic_state_groundtruth.angular_acceleration.y_val
kinematic_state_groundtruth.angular_acceleration.z_val
```

而对于姿态的读取，姿态信息是用四元数表示的，而 AirSim 同时也提供了四元数转换为欧拉角的接口：

```python
# 无人机姿态真值的四元数表示
kinematic_state_groundtruth.orientation.x_val
kinematic_state_groundtruth.orientation.y_val
kinematic_state_groundtruth.orientation.z_val
kinematic_state_groundtruth.orientation.w_val

# 四元数转换为欧拉角，单位 rad
(pitch, roll, yaw) = airsim.to_eularian_angles(kinematic_state_groundtruth.orientation)
```

### 6.2 无人机状态估计值读取 API
在实际的无人机中，无法获取无人机的状态真值，状态的读取是无人机上传感器融合得到的估计值。AirSim APIs 还提供了获取无人机状态（估计值）的功能接口，其调用格式如下：

```python
state_multirotor = client.getMultirotorState(vehicle_name='')
```

这里的无人机状态 `state_multirotor` 包含：时间戳、运动状态信息（估计值）、碰撞信息、GPS 经纬度信息、遥控器信息、降落信息等。读取方式如下：

```python
state_multirotor.collision            # 碰撞信息
state_multirotor.kinematics_estimated # 运动信息
state_multirotor.gps_location         # GPS 经纬度信息
state_multirotor.timestamp             # 时间戳
state_multirotor.landed_state         # 降落信息
state_multirotor.rc_data              # 遥控器信息
```

其中的降落信息 `state_multirotor.landed_state` 是整数类型，其值为 0 表示在地上，值为 1 表示在空中。其中的时间戳 `state_multirotor.timestamp` 是仿真开始后的时间，单位是**纳秒**。

无人机状态中的运动信息 `state_multirotor.kinematics_estimated` 是由多个传感器测量值融合得到的，包含 6 种运动信息，读取方式相同：

```python
# 无人机运动信息的估计值有 6 个属性
state_multirotor.kinematic_estimated.position             # 位置信息估计值
state_multirotor.kinematic_estimated.linear_velocity      # 速度信息估计值
state_multirotor.kinematic_estimated.linear_acceleration  # 加速度信息估计值
state_multirotor.kinematic_estimated.orientation          # 姿态信息估计值
state_multirotor.kinematic_estimated.angular_velocity     # 姿态角速率信息估计值
state_multirotor.kinematic_estimated.angular_acceleration # 姿态角加速度信息估计值
```

但是遗憾的是，`SimpleFlight` 不支持模拟传感器，所以如果使用的是 `SimpleFlight`，则使用 `getMultirotorState()` 得到的运动状态与使用 `simGetGroundTruthKinematics()` 得到的运动状态真值是一模一样的。如果使用硬件在环仿真（如 PX4 等），则可以使用 `getMultirotorState()` 获得无人机的估计状态信息。

这里我们暂时只介绍时间戳、降落信息和运动信息，对于碰撞信息和 GPS 经纬度信息在后面有具体的讲解。

AirSim APIs 还提供了获取电机状态的功能接口，调用格式如下：

```python
state_rotor = client.getRotorStates(vehicle_nane='')
```

函数的返回值 `state_rotor` 包含时间戳和每个螺旋桨的转速、拉力、力矩信息。使用 `print()` 将其打印出来（无人机悬停状态下），可以得到如下信息：

```python
<RotorStates> {   'rotors': [   {   'speed': 516.8734130859375,
                       'thrust': 2.4884119033813477,
                       'torque_scaler': -0.03308132290840149},
                   {   'speed': 516.8734130859375,
                       'thrust': 2.4884119033813477,
                       'torque_scaler': -0.03308132290840149},
                   {   'speed': 516.8734130859375,
                       'thrust': 2.4884119033813477,
                       'torque_scaler': 0.03308132290840149},
                   {   'speed': 516.8734130859375,
                       'thrust': 2.4884119033813477,
                       'torque_scaler': 0.03308132290840149}],
     'timestamp': 1632472243084013312}
```

如果想要单独得到第 0 个螺旋桨的转速信息，可以用如下代码：

```python
state_rotors = client.getRotorStates(vehicle_name="")
print(state_rotors.rotors[0]['speed'])
```

这样可以直接打印出电机的转速。使用电机转速控制可以实现对无人机运动控制精度较高的应用，如高精度轨迹跟踪。

### 6.3 位置读取函数封装（重点）
**无人机位置读取时使用的是全局坐标系，这里的全局坐标系定义是北东地，全局坐标系的原点是无人机的初始位置。**

也就是说当有多个无人机时，而且每个无人机的初始位置不同时，此接口读取到的不同的无人机的状态是以其读的无人机的初始位置为原点的，所以在读取的时候需要加上其初始位置的补偿。

基于此可以写出如下位置读取的函数封装：

```python
orig_pos = np.array([[0., 0., 0.,], 
                     [3., 0., 0.]])
def get_uav_pos(client, name):
    uav_state = client.getMultirotorState(vehicle_name=name).kinematics_estimated
    num = int(name[-1])
    pos = np.array([[uav_state.position.x_val + orig_pos[num, 0]], 
                    [uav_state.position.x_val + orig_pos[num, 1]], 
                    [uav_state.position.x_val + orig_pos[num, 2]]])
    return pos
```

其中 `orig_pos` 是每个无人机的初始位置，上面的代码中，有两个无人机，无人机 0 的初始位置是 [0, 0, 0]，无人机 1 的初始位置是 [3, 0, 0]。使用上面的方法添加补偿后，就可以将每个无人机的全局坐标系统一了。

## 7. 航路点飞行 API & 多无人机位置控制函数封装
---
AirSim 提供的轨迹跟踪 API 是基于位置控制的，所以严格意义上并不能算是轨迹跟踪，而应该称之为连续航路点飞行。无人机会依次飞过多个航路点，形成特定的飞行轨迹，其调用格式如下：

```python
# 航路点轨迹飞行控制
client.moveOnPathAsync(path, velocity, 
                    drivetrain = DrivetrainType.MaxDegreeOfFreedom, 
                    yaw_mode = YawMode(), 
                    lookahead = -1, adaptive_lookahead = 1, 
                    vehicle_name = '')
```

参数说明：

- path(list[Vector3r]):多个航路点的 3 维坐标
- velocity(float)：无人机的飞行速度大小；
- lookahead，adaptive_lookahead：设置路径跟踪算法的参数

参数 path 包含了多个航路点的全局位置坐标，也就是说无人机要飞过 path 中包含的每一个坐标点（也就是航路点）。将 path 中的航路点依次连接起来，就会组成一条路径，无人机依次飞过航路点，就像是沿着特定路径飞行一样，形成路径跟踪的效果。

这里需要注意的是全局坐标系的原点是无人机的初始位置，所以当有多架无人机同时仿真的时候，每架无人机的原点是不同的，所以当使用统一的全局坐标系的时候，需要对不同的无人机添加位置补偿。

`moveOnPathAsync` 中使用的算法是 **Carrot Chasing Algorithm**，这是一个非常经典的路径跟踪算法，其中需要设置 `lookahead` 参数，这个参数的意义是设置在路径上向前看的距离。`lookahead` 设置的越大，在路径拐弯的时候，无人机就越提前转弯；如果 `lookahead` 参数设置的过小，则可能出现超调，无人机会飞跃航路点，然后才拐弯。

任何时候，`lookahead` 值的设置要大于位置控制允许的误差，并且小于整个路径的总长度。如果 `lookahead` 值小于位置控制允许的误差，则无人机会认为已经到达了第一个目标点，最终效果是无人机悬停在原地。如果 `lookahead` 值大于轨迹的总长度，则无人机会认为已经达到了最后一个航路点，最终效果还是无人机悬停在原地。

当 `lookahead` 参数小于 0 时，`adaptive_lookahead` 设置为大于 0 的数值，这样可以让算法根据当前的无人机飞行速度自动设置 `lookahead` 的值，通常按照默认设置（`lookahead=1`，`adaptive_lookahead=-1`）就可以达到比较好的效果。

如果航路点组成的轨迹是一条直线，不存在转弯的情况，那么`lookahead` 值的大小对最终轨迹的影响将减小很多，此时 `lookahead` 的值只要小于轨迹的总长度即可。例如 `moveToPositionAsync()` 接口函数内容就是调用的 `moveOnPathAsync` 接口，也就是只有一个航路点的轨迹，此时轨迹就是一条直线，只要 `lookahead` 值小于当前到唯一航路点的距离，最终的飞行轨迹都是一样的。

## 8. 图像 APIs
---
### 8.1 拍摄图像前的基本知识

#### 图像位置

AirSim 平台中的无人机和无人车都默认放有 5 个相机，只是位置不同，这里主要介绍无人机上的相机。下表列出的是相机的位置和命名。旧版本的 AirSim 对相机的命名是使用的数字，目前 AirSim 兼容旧版本，所以使用数字也是可以的。

| 相机位置 | ID | 旧版本 ID |
|---|---|---|
| 前向中间 | `front_center` | `0` |
| 前向右边 | `front_right` | `1` |
| 前向左边 | `front_left` | `2` |
| 底部中间 | `bottom_center` | `3` |
| 后向中间 | `back_center` | `4` |

#### 图像种类
AirSim 中默认可以获得 8 种不同类型的图片，下表总结了 10 种类型图片的名称、解释和调用方式。

|图像类型|解释|调用方法|
|---|---|---|
|Scene|彩色图|airsim.ImageType.Scene|
|DepthPlanar|深度图，像素值代表到相平面的距离|airsim.ImageType.DepthPlanar|
|DepthPerspective|深度图，像素值代表透视投影下到相平面的距离|airsim.ImageType.DepthPerspective|
|DepthVis|深度图，为了更好的展示距离的远近，100 米对应白色，0 米对应黑色|airsim.ImageType.DepthVis|
|DisparityNormalized|深度视差图，每个像素值是浮点型|airsim.ImageType.DisparityNormalized|
|Segmentation|分割图，不同 ID 号的物体用不同的颜色表示|airsim.ImageType.Segmentation|
|SurfaceNormals|表面法线，包含了比较丰富的细节信息|airsim.ImageType.SurfaceNormals|
|Infrared|红外图，与分割图类似，知识颜色变为 ID 号|airsim.ImageType.Infrared|

#### 图像数据的格式
AirSim 中获得的图像有 3 种格式：PNG 格式、Array 格式、浮点型格式。通过在程序中设置不同的参数即可得到不同格式的图像。

PNG 格式的图片是一种无损压缩的图片格式，由于其具有连续读出和写入的特性，所以非常适合于网络传播。其文件结构中包括：文件署名域和数据块（关键数据块、辅助数据块）。文件署名域的 8 个固定字节用来识别该文件是不是 PNG 格式的文件。数据块中包含了图像的大小、色深、颜色类型、压缩方式等信息。PNG 格式的图像是用字节的形式保存的，可以用图片查看器等软件打开。

Array 格式是最原始的图像格式，在图像处理时都需要将图片转换成此格式。Array 是以向量的形式存储，彩色图像的每个像素点都有 RGB（红绿蓝）三个通道，所以向量的维度为：height \times width\times 3；而灰度图像每个像素点仅有一个通道，向量维度为：height\times width。每个值的取值范围是 0 ~ 255，其中0代表亮度最低，255代表亮度最高。例如第i行第j列的像素是红色的，则有 `img_rgb[i, j, :]=[255, 0, 0]`。另外需要注意的是在使用 OpenCV 库时，其默认的通道顺序为 BGR（蓝绿红），所以当图像的第i行第j列像素是红色时，则有 `img_rgb[i, j, :]=[0, 0, 255]`。

浮点型格式的图像同样以向量的形式保存，维度为：height\times width，每个像素点的值是浮点型，适合深度图的保存。例如 DepthPlanar 图像中每个像素点的值表示此像素点到相平面的距离。

将这3种图片保存格式的特点总结如下表所示。

|图像类型|存储形式|是否压缩|适合保存的图片类型|
|---|---|---|---|
|PNG 格式|bytes|无损压缩|彩色图、分割图、表面法线图、红外图|
|Array 格式|bytes|无|彩色图、分割图、表面法线图、红外图|
|浮点型格式|float|无|深度图|

通过一个简单的例子来进一步说明，下面的代码实现的功能是：使用同一个相机分别读取 3 种类型的图片： PNG 格式彩色图、Array 格式彩色图、浮点型深度图，最后打印出图片的一些信息。

```python
response = client.simGetImages([
    # png format
    airsim.ImageRequest(0, airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    airsim.ImageRequest(0, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
    airsim.ImageRequets(0, airsim.ImageType.DepthPlanar, pixels_as_float=True)
])

print("PNG 格式彩色图的前 8 个字节：", np.frombuffer(response[0].image_date_uint8[0:8], np.uint8))
print("PNG 格式彩色图的字节个数：", len(response[0].image_data_uint8))
print("Array 格式彩色图的字节个数：", len(response[1].image_data_uint8))
print("浮点型格式深度图任一像素的值：", response[2].image_data_float[random.randint(0, 10000)])
```

#### 图片文件的格式
首先要区分图像数据与图片文件的不同。

图像数据是保存在内存中的数据，大部分是在程序运行的时候通过 API 读取到的，可以从 AirSim APIs 获得，也可以直接从文件中获得，当程序运行结束，或者内存释放掉的时候，这些数据也就随之消失了。

图片文件是保存在硬盘中的文件，有不同的后缀名格式的文件，可以用图像查看器等软件打开。常用的图片文件格式有：

- `.jpg` 格式：不带透明通道的有损压缩格式，广泛应用于互联网和数码相机领域；
- `.png` 格式：便携式网络图形，无损压缩的位图，有较高的压缩比；
- `.tif` 格式：非失真的压缩格式，占用空间较大，通常用于书籍和海报等数专业的领域;
- `.bmp` 格式:是 Windows 操作系统中的标准图像文件格式,通常不压缩,文件所占空间较大.