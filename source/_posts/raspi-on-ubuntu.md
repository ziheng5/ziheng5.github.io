---
title: 在树莓派的 Ubuntu 系统上安装 RaspberryPi OS 上硬件相关的配置
date: 2025-03-10 11:27:34
tags:
    - 树莓派
    - 硬件
    - Ubuntu
categories: 
    - 树莓派
description: |
    ❓ 如何在 Ubuntu 上安装 RaspberryPi OS 上硬件相关的配置
---

> 正常情况下，如果要用树莓派做开发的话，最好使用树莓派官方的 RaspberryPi OS 操作系统。
>
>但如果实在想使用其他的 Linux 发行版（ Ubuntu、Kali 等），需要在系统上自行安装相关配置。


## 1. raspi-config
利用 raspi-config 来进行硬件相关配置极为方便，奈何只有 RaspberryPi OS 上才会自带 raspi-config。

raspi-config 历史版本安装包地址：http://archive.raspberrypi.org/debian/pool/main/r/raspi-config

选择自己需要的版本后，在 Ubuntu 的终端中输入以下命令来获取安装包：

```Terminal
# 这里我选择的是 raspi-config_20200707_all 版本的
wget http://archive.raspberrypi.org/debian/pool/main/r/raspi-config/raspi-config_20200707_all.deb
```

下载完成后，执行下面的命令安装：

```Terminal
sudo dpkg -i raspi-config_20200707_all.deb

sudo apt install -f -y
```

检查安装是否成功：

```Terminal
sudo raspi-config
```

出现 raspi-config 配置界面后即为安装成功：

![raspi-config](/images/raspberrypi/raspi_config.png)

## 2. GPIO 权限配置
首先更新 apt：

```bash
sudo apt update && sudo apt upgrade -y
```

接着安装编译工具和依赖：

```bash
sudo apt install -y git python3-dev python3-pip
```

然后创建 `gpio` 用户组（如果没有创建的话）：

```bash
sudo groupadd gpio
sudo usermod -a -G gpio $USER # 将当前用户加入 gpio 组
``` 

创建 udev 规则文件：

```bash
sudo vim /etc/udev/rules.d/99-gpio.rules
```

在其中添加以下内容：

```plaintext
SUBSYSTEM=="gpio", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpiomem", GROUP="gpio", MODE="0660"
```

重新加载规则并重启服务：

```bash
sudo udevadm control --reload
sudo udevadm trigger
```

最后重新启动一下就 OK 了。

## 3. SPI（Serial Peripheral Interface）权限配置
在 Linux 系统中，SPI 设备的权限管理与 GPIO 类似，但默认情况下不存在名为 `spi` 的用户组。

首先，确认 SPI 设备是否已启用并出现在 `/dev` 目录中：

```bash
ls /dev/spidev*
```

> 如果这里没有输出，那么说明 SPI 内核模块未启用，此时需要手动启用 SPI：
>
> ```bash
> # 临时加载内核模块
> sudo modprobe spidev
> 
> # 永久启用 SPI
> echo "dtparam=spi=on" | sudo tee -a /boot/firmware/config.txt
> sudo reboot
> ```

接着，创建 `spi` 用户组：

```bash
sudo groupadd spi
```

然后配置 SPI 设备权限：

```bash
# 创建 udev 规则文件
sudo vim /etc/udev/rules.d/99-spi.rules
```

在其中添加以下内容：

```plaintext
SUBSYSTEM=="spidev", GROUP="spi", MODE="0660"
```

然后重新加载 udev 规则：

```bash
sudo udevadm control --reload
sudo udevadm trigger
```

将用户添加到 `spi` 组：

```bash
sudo usermod -a -G spi $USER
```

最后重启一下即可