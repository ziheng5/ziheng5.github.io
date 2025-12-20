---
title: 树莓派的初次尝试
date: 2025-01-17 18:38:51
tags: 
    - 树莓派
    - 硬件
    - Ubuntu
categories: 
    - 树莓派
description: |
    树莓派折腾记其一 —— 操作系统与相关配置
---

> 大二上半学期期末周的时候突然心血来潮，想要做一些硬件开发（人脸识别锁、指纹锁和无人机啥的）
>
> 由于之前除了 ESP8266 的板子以及 MicroPython 这个工具外，其他硬件相关的东西基本上没接触过。
>
> 于是小生买了一块似乎比较好上手的树莓派 4b，准备熬过期末周之后好好折腾一番。

# 1. 🔥 烧录系统
## 1.1 准备 SD 卡和读卡器
不同于普通计算机，树莓派采用 SD 卡来存储系统及其内部信息

所以，想要在树莓派上运行操作系统，首先需要在一张 SD 卡中烧入系统。

将 SD 卡插入读卡器，再将读卡器插入电脑的 USB 接口，便可以在电脑上对 SD 卡进行操作

（这里缺图）

> 这里小生建议多准备一张 SD 卡，以备不时之需

## 1.2 Raspberry Pi Imager
如果你此前有在电脑上装操作系统的经验，那么这个时候估计你第一个想到的方法就是从网上把某个操作系统的镜像下载下来之后再装到 SD 卡上。

然而，树莓派官方提供了更为便捷的途径：通过树莓派官方提供的 [Raspberry Pi Imager](https://www.raspberrypi.com/software/) 将适配的操作系统直接安装到 SD 卡上。Raspberry Pi Imager 提供了树莓派官方操作系统以及各种其他 Linux 发行版的镜像，安装也十分迅速便捷。

于是小生采用 Raspberry Pi Imager 在我的第一张 SD 卡上装一个 **Ubuntu Desktop 24.04.1** 系统。

> ⚠️：不建议装 Ubuntu 24 版本，后面做各种兼容太麻烦了，这里小生建议装老版本，比如 Ubuntu 22

将 Raspberry Pi Imager 下载下来之后，点击运行，出现以下界面：

![raspberrypiimager](/images/raspberrypi/pic1.png)

- 点击 "CHOOSE DEVICE" 后，选择 "Raspberry Pi 4"。
- 点击 "CHOOSE OS" 后，选择 "Other general-purpose OS"，接着选择 "Ubuntu"，再选择 "Ubuntu Desktop 24.04.1 LTS(64-bit)"
- 点击 "CHOOSE STORAGE"后，选择插到电脑上的那张 SD 卡。

> ⚠️ 注意：
> 
> 1. 也可以选择在 "CHOOSE OS" 界面选择其他操作系统。如果没有自己想要安装的操作系统，也可以自行安装其他系统镜像
>
> 2. 安装操作系统的 SD 卡尽量用空卡，因为烧录的时候会先把 SD 卡格式化，删除其中所有数据。如果使用之前用过的 SD 卡，则务必确保其中没有存储重要信息！

接下来点击 "NEXT"，等待系统安装完毕即可。

# 2. 💻 配置系统
## 2.1 接入外设
经过上面的操作，小生的 SD 卡中已经有了一个 Ubuntu Desktop 24.04.1 操作系统，接下来需要将 SD 卡插入到树莓派上的卡槽中，并在树莓派上配置操作系统。

配置操作系统需要给树莓派外接显示器、键盘和鼠标。

树莓派上连接显示器的接口采用 **Micro HDMI**，即小号的 HDMI，所以我们需要准备一根 **HDMI-Micro HDMI 转接线**。

键盘和鼠标直接通过树莓派上的 USB 接口连接即可。

> ⚠️ 树莓派上运行操作系统后，CPU 发热极其厉害，建议先在树莓派上贴好散热片，或者安装散热风扇。
## 2.2 开始配置
由于 Ubuntu 配置过程极其简单，这里就不再赘述，需要注意的一点是选择语言的时候尽量选择英文。

## 2.3 配置 apt 镜像
这里小生采用清华源，首先打开配置文件：

```Terminal
sudo nano /etc/apt/sources.list
```

在配置文件中粘贴如下代码：

```Terminal
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```

保存退出后，运行以下指令来更新软件列表到本地：

```Terminal
sudo apt-get update
```

```Terminal
sudo apt-get upgrade
```

等待更新完毕，镜像源就配置完毕了。

## 2.4 配置必要的功能
### 2.4.1 SSH 连接
如果需要使用 SSH 连接到树莓派，树莓派上的操作系统要安装一下 openssh-server：

```Terminal
sudo apt install openssh-server
```

而客户端（这里小生客户端使用的是 ArchLinux）需要安装 openssh-client：

```Terminal
# 这里小生使用的是 ArchLinux，如果是其他操作系统请使用对应的指令
paru -S openssh
```

安装完成后，开启手机热点或电脑热点（如果哟的话），在手机上查看连接设备以获取树莓派的 IP 地址（或者在树莓派的 Ubuntu 终端中输入 `ip addr` 以查看 IP 地址）。

确保树莓派和电脑在同一个网络下，先ping一下：

```Terminal
ping IP地址
```

如果ping得通，在电脑终端（这里小生用的是 ArchLinux）中输入以下指令：

```Terminal
ssh 用户名@IP地址
```

如果成功连接，则会要求你输入密码。如果长时间没反应或者直接报错，说明哪里没有配置好。

### 2.4.2 vim
Ubuntu 系统自带一个 vim,但是这个 vim 其实是 vim-common，不好用，我们将它卸载掉，然后重新安装一个 vim。

```Terminal
sudo apt-get remove vim-common
sudo apt-get install vim
```

## 2.5 配置 conda 环境
Ubuntu 24.04.1 系统自带 Python2 和 Python3，但是环境管理不太方便，所有小生决定安装一个 miniconda 

在安装 miniconda 前，**务必**查看 CPU 信息：

```Terminal
lscpu
```

在输出内容中找到树莓派 CPU 信息（这里小生的 CPU 是 **aarch64** 的）。

然后根据 CPU 型号，选择下载对应的 miniconda 版本：

```Terminal
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
```

下载完成后，目录下出现 Miniconda3-latest-Linux-aarch64.sh 文件。

接下来添加可执行权限：

```Terminal
sudo chmod +x Miniconda3-latest-Linux-aarch64.sh
```

然后运行安装脚本：

```Terminal
sudo ./Miniconda3-latest-Linux-aarch64.sh
```

接下来按照提示完成安装操作即可。

# 3. 🚢 文件传输
## 3.1 SCP 上传文件
SCP，即 Security Copy，通过 SCP 命令可以在本地主机和远程主机之间安全且便捷地复制文件和目录

具体实现方法采用下面的命令格式：

```Terminal
scp local_file_location remote_username@remote_ip:remot_file_location
```

举个例子：

比如说，现在我们本地计算机的用户名为 lengyu,我们在 Desktop 文件夹下建立一个 `local_file.txt` 文件。

远程服务器用户名为 coldrain,现在小生想要把 `local_file.txt` 从本地计算机传输到远程服务器上，那么这个时候小生需要在本地计算机上运行如下指令：

```Terminal
scp /home/lengyu/Desktop/local_file.txt coldrain@xxx.xxx.xxx.xxx:/home/coldrain/Desktop
```

运行结果如下：

```Terminal
scp /home/lengyu/Documents/local_file.txt coldrain@192.168.37.122:/home/coldrain/Desktop
coldrain@192.168.37.122's password:
local_file.txt                                             100%    0     0.0KB/s   00:00
```

然后我们就可以在远程服务器上的 `/home/coldrain/Desktop` 目录下发现 `local_file.txt` 文件了

如果需要上传文件夹，记得在 `scp` 后面加上 `-r`，即：

```Termminal
scp -r local_folder_location remote_username@remote_ip:remot_file_location
```

## 3.2 SCP 下载文件
网上查到的具体实现方法采用下面的命令格式：

```Terminal
scp remote_name@remote_ip:remote_file_location local_file_location
```

然而实际操作的时候，小生这里**报了一个错** 😨：

```Terminal
scp: open local "/usr/test.txt": Permission denied
```

查找了一会儿资料，发现是没有权限在本地计算机的文件夹上写入文件。于是小生更改了本地计算机目标文件夹的写入权限：

```Terminal
chmod 777 /home/local_name/Documents
```
> 注：`chmod 777` 命令可以更改文件夹权限为可写入

运行上述代码后，再次尝试下载文件，下载成功 ✌️

