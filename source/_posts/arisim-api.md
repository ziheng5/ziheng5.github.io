---
title: Airsim 笔记：Python API 总结
date: 2025-04-13 18:08:04
tags:
    - 强化学习
    - 仿真环境
categories:
    - 强化学习
description: |
    简单介绍深度学习，并简单讲解PyTorch的基本语法
---
## 0. AirSim APIs 简介
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

## 2. Python 与 AirSim 的通信机制
AirSim API 使用的是 `TCP/IP` 中的 `msgpack-rpc` 协议，这是一种网络协议，所以如果设置正确，其实可以用两台不同的电脑来做仿真，一台跑 AirSim 和 Unreal，另一台跑 python 程序。

当 AirSim 开始仿真的时候，会打开 41451 端口，并监听这个端口的需求。python 程序使用 `msgpack serialization` 格式向这个端口发送 RPC 包，就可以与 AirSim 进行通信了。