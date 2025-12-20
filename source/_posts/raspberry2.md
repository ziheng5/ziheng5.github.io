---
title: Raspberry Pi, start!
date: 2025-01-18 21:01:04
tags: 
    - 树莓派
    - 硬件
    - Ubuntu
categories: 
    - 树莓派
description: |
    树莓派折腾记其二 —— GPIO
---
继上一次小生在树莓派上安装完操作系统并进行了相关配置后，小生开始准备折腾树莓派上的 GPIO 了。

# 1. 📦 前置准备
## 1.1 WiringPi
### 方法一（B站教程，其实不推荐）
在树莓派上写 C 语言代码，最重要的就是安装 WiringPi 这个库

安装指令如下：

```bash
sudo apt install wiringpi
```

然而，对于高版本的树莓派（比如 4B），按照上面指令下载到的 wiringpi，运行 `gpio readall` 指令会报错，原因是作者已经很久没有更新这个库了

那么我们需要另寻他法。解决方案如下。

`cd` 想要下载安装包的位置，然后运行下面的命令：

```bash
wget https://gitee.com/LJYSCIENTIST/raspberry-pi-software-storage/raw/master/wiringpi-2.60-1_arm64.deb
```

接着安装上面下载到的安装包：

```bash
sudo apt-get install  ./wiringpi-2.60-1_arm64.deb
```

安装完成后，可以运行 `gpio -v` 和 `gpio readall` 来检查是否正确安装

> ⚠️ 注意：Ubuntu 24 版本过新，上述方法均无法正确安装，小生也没有找到合适的安装包。这里建议大家 Ubuntu 安装为旧版本，比如 22 系列的。

### 方法二（用 git，推荐）
> 如果没有事先安装好 git 的话，可以执行以下命令安装 git
>
> ```bash
> sudo apt install git
> ```
>
> 等待安装完成即可

首先克隆仓库：

```bash
git clone https://github.com/WiringPi/WiringPi
cd WiringPi
```

然后编译并安装：

```bash
./build
```

安装完成后，执行下面的命令验证安装：

```bash
gpio -v
gpio readall
```

若正常输出，则安装成功

## 1.2 python——RPi.GPIO
先给树莓派上的 python 配置硬件操作相关的包

> 这里不需要激活 conda 环境，直接使用系统内置的 python3 即可
>
> （你问为什么？其实小生用 conda 无论如何都没法把 `RPi.GPIO` 配置下来，各位巨佬有兴趣的话可以去折腾折腾 💦）

我们直接在终端运行下面的指令：

```bash
sudo apt-get -y install python3-rpi.gpio
```

等待下载完成即可在 python3 中使用 `RPi.GPIO`

# 2. 关于树莓派的 GPIO
树莓派的 GPIO 有三种编码方式，分别是板载编码、BCM 编码、Wiringpi 编码

这里直接附上一张图

![gpio_encode](/images/raspberrypi/pic3.png)

三种编码方式的不同主要体现在代码中。Python 采用的是 BCM 编码，而 C 采用的是 wiringpi 编码。今天小生主要讲讲如何用 Python 脚本控制 GPIO。

# 3. 树莓派 GPIO ——点灯
准备好面包板、杜邦线和发光二极管，完成接线后，小生便准备开始点灯了。

这里直接附上 Python 脚本代码：

```Python
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)
for i in range(10):
    GPIO.output(26, GPIO.HIGH)
    sleep(1)
    GPIO.output(26, GPIO.LOW)
    sleep(1)

GPIO.cleanup()
```

利用 vim 编辑器创建 led.py 文件后，将上述代码输入其中，保存退出。

运行 `sudo su` 进入管理员模式，然后直接运行 led.py：

```bash
python3 led.py
```

运行命令后，Led 灯开始闪烁，点灯成功 ✌