---
title: JGAurora A5S 3D 打印机零基础食用指南
date: 2025-04-21 16:29:02
tags:
    - 硬件
    - 杂谈
categories:
    - 食用指南
description: |
    ❓ 如何使用 JGAurora A5S 3D 打印机呢
---
>  😢 学院社团活动室里的 3D 打印机尘封已久，查找使用教程比较麻烦，如今重见天光，遂作此篇，以便后人。
>
> 注：该 3D 打印机设备型号为 **JGAurora A5S**，其他使用这种打印机的朋友也可以参考本文。
>
> 本指南参考教程：
> - https://mp.weixin.qq.com/s/DJKIi99eGjai1Nu09lgsNQ

![printer](/images/3d_printer/printer.png)

## 1. 打印前的硬件调试
---
### 1.1 调平
点击 3D 打印机上的 `归零` 按钮，即可进行调平操作，将喷嘴调整到合适位置即可。

> 🔎 这台 3D 打印机在被尘封之前已经调试好了，目前暂时还没有遇到需要重新调试的情况

### 1.2 预热
点击 3D 打印机上的 `预热` 按钮，点击 `+` 按钮增大预期温度，然后点击 `预热` 按钮开始预热。

> 参考温度值：$PLA \approx 200 \degree C$，$ABS \approx 240 \degree C$

> ⚠️ 预热后**千万不要**用身体接触**喷嘴**！高温危险！⚠️

### 1.3 检查上料情况
**预热完成后**，需要检查耗材是否正确进料。

如果耗材并没有连接到**进料器**上，需要手动连接一下，如下图所示：


![install2](/images/3d_printer/install2.png)

![install](/images/3d_printer/install.png)

**用力**将耗材插入进料器，再退出 `预热` 界面，点击 `挤出` 按钮，使得进料器不停吸入耗材，待 3D 打印机可以稳定挤出后，即为进料成功。

> ⚠️ 如果喷嘴位置过低，记得调高一点，否则点击 `挤出` 的时候喷嘴可能会被堵住（

## 2. 由 .STL 模型文件生成 .gcode 切片文件
---
> `.STL` 文件为 **3D 模型文件**，既然都用到 3D 打印机了，应该不会不知道这个文件吧 💦

由 .STL 生成 .gcode 切片文件需要用到切片软件，这里使用最热门的一个软件：**Cura**

### 2.1 安装 Cura（切片软件）
> **Cura** 是一款由荷兰公司 Ultimaker 开发的 **开源 3D 打印切片软件**，广泛应用于 FDM（熔融沉积成型）3D 打印机用户中。它以其 **易用性、强大功能** 和 **高度可定制性** 而闻名，适用于初学者和高级用户。

#### Windows
Windows 系统下安装 Cura 可以直接从官网上获取安装包：https://ultimaker.com/software/ultimaker-cura/

![cura_website](/images/3d_printer/pic2.png)

#### Ubuntu
Ubuntu 系统下执行以下命令以安装：
```bash
sudo apt-get install cura
```

#### Archlinux
Archlinux 系统下执行以下命令以安装：

```bash
sudo pacman -S cura
```

或者使用 `aur`：

```bash
sudo paru -S cura
```

> ⚠️ 注：如果安装时要求选择安装 `cura` 或 `cura-bin`，理论上两者都可以正常使用。（但是小生看了一下，好像两种安装的**版本不太一样**）

### 2.2 注册 Cura 账号并登陆
使用邮箱注册即可（小生用的是 **Gmail**），然后填写必要信息即可。

注册完成后，可以先去激活**教育认证**（即 `Free EDU`），激活教育认证后即可**免费**使用 Cura

![free_edu](/images/3d_printer/pic3.png)

### 2.3 添加打印设备（Add printer）
- 选择 `Non UltiMaker printer`
- 选择 `Add a non-networked printer`
- 选择 `JGAurora` → `JGAurora A5 & A5S`
- 点击 `Add`

![add_printer](/images/3d_printer/pic4.png)

### 2.4 切片 .STL 文件
点击文件夹图标后，选择自己的 `.STL` 模型文件，导入后点击 `Slice` 按钮即可进行切片操作。

切片完成后可以选择 SD 卡导出。

> ⚠️ 注意：如果想将文件导出到 SD 卡上，需要先将 SD 卡插到读卡器上，然后再将读卡器插入电脑的 USB 接口，如下所示：
>
> ![sd_card](/images/3d_printer/sd_card.png)


## 3. 导入切片文件 & 开始打印
---
从上文中的读卡器中取出 SD 卡，将 SD 卡插入到**大号读卡器**（小生也不知道应该叫什么💦）上，如下所示：

![reader](/images/3d_printer/big_reader.png)

再将这个大号读卡器插到 3D 打印机上，点击 `打印` 按钮，正常情况下应该就可以看到 `.gcode` 文件了，选择对应文件即可开始打印～