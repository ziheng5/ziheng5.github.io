---
title: Linux 用户的 Windows 改造之旅
date: 2025-08-30 02:07:11
tags:
    - Windows
categories:
    - Windows
description: |
    🔍 如何让你的 Windows 变得像你的 Linux 一样便捷、高效且美观
---


> Coldrain 在暑假的尾声里突然有了使用 `Windows` 进行仿真的需求。但是当 Coldrain 在已经习惯 `Linux` 的情况下回到阔别已久的 `Windows` 后，顿时觉得 `Windows` 的图形化界面操作已经不再适合自己，但又不能不用 `Windows`，于是 Coldrain 开始改造起自己的 `Windows`...

注：作为 Linux 用户，小生在本文中的大部分操作、终端配置等都将在终端中进行。如果你没有接触过 Linux，或是对 Windows 终端指令知之甚少的话，建议先学习最基本的终端指令，譬如 `ls`、`cd <path>` 等基本指令

## 1. 包管理器及其配置
### 1.1 Windows 下常见的包管理器
在 `Linux` 下已经习惯了使用包管理器来快捷安装并管理应用程序、插件，那么改造 `Windows`，首先需要的就是一个便捷的软件包管理器。

`Windows` 下的软件包管理有很多，其中最常用的是 `winget`、`scoop` 与 `chocolatey`：

- `winget`：微软官方的包管理器，由微软官方维护，速度快，且和微软商店联动。
- `scoop`：第三方包管理器，生态灵活，默认安装在用户目录下，不需要管理员权限，似乎很受开发者欢迎。
- `chocolatey`：`Windows` 上最老牌的包管理器，更偏向系统管理员和 DevOps 场景。

在本文中，小生将采用 `scoop` 来安装各种应用程序与插件配置。

### 1.2 Scoop 的安装

> Scoop 官方仓库：https://github.com/ScoopInstaller/Scoop#readme

在 `Windows` 上安装 `scoop` 可以在 PowerShell 下执行如下命令：

```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

> `scoop` 的默认安装位置位于 `C:\Users\<USERNAME>\scoop`

安装完成后，即可使用 `Scoop` 在终端实现软件包的快速安装了。

`Scoop` 安装软件包的指令为：

```powershell
scoop install XXX
```

### 1.3 给 Scoop 配置 aria2 以加速下载
但是实际安装的时候，经常会遇到**下载速度较慢**的情况（比如即便已经打开了代理却依然和没开代理一样慢）

这个时候，我们可以给 `scoop` 配置 `aria2` 来实现**多线程下载**，进而加快下载速度。

```powershell
scoop install aria2
scoop config aria2-enabled true
```

配置完成后，再次安装软件的时候会自动启用多线程下载。

### 1.4 扩展软件清单
有的时候你想要的软件包并没有被官方仓库登记在册，这时候就需要配置一些扩展仓库了，比如 `extra` 等。

首先，查看 `scoop` 现有的仓库：

```powershell
scoop bucket list
```

这会显示你已经添加过的仓库。Scoop 官方和社区维护了很多仓库，可以按需添加常用的 bucket：

```powershell
# 官方的扩展仓库
scoop bucket add extras
scoop bucket add versions
scoop bucket add games
scoop bucket add java
scoop bucket add nerd-fonts

# 社区大仓库（收录量非常大，极力推荐）
scoop bucket add dorado https://github.com/chawyehsu/dorado
```

其中，`extra` 用于安装各种常见工具（比如 VSCode、7zip 等），`dorado` 是中文社区维护的大仓库，很多国人常用软件都在里面（比如微信、QQ 等）。

配置完扩展仓库后，你就可以通过包管理器下载到各种常用软件了，比如 QQ：

```PowerShell
scoop install qq
```

当然，如果你不确定扩展仓库后，你想要的软件到底存不存在（比如微信），你可以直接搜索：

```PowerShell
scoop search wechat
```


## 2. PowerToys 开发者工具箱
> PowerToys 官方文档：https://learn.microsoft.com/zh-cn/windows/powertoys/

PowerToys 是微软官方开发的一套 **Windows 实用工具集合**，可以提升生产力和个性化体验。它包含很多模块，例如窗口管理、快捷键映射、文件工具等。

这里在终端使用 `scoop` 直接安装：

```powershell
scoop install powertoys
```

安装完成后，你会看到开始菜单的“最近添加”部分出现了一个 **PowerToys 图标**，可以点开进入设置界面。

> 绝对路径：`C:\Users\<USERNAME>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Scoop Apps\PowerToys.lnk`

PowerToys 中包含了各种各样的实用功能：
- **高级粘贴**：可以将剪贴板内容转换为所需的任何格式。
- **Awake**：使电脑保持唤醒状态，而不会进入睡眠（很适合挂机跑代码）
- **命令面板**：快速启动器（和 Linux 很像，再也不需要手动点击了！）
- **键盘管理器**：自定义键盘映射，创建快捷键。
- **文本提取器**：使用 OCR 将文本从图片中提取出来。
- **...**

下面拿**键盘管理器**为例，看看如何用 PowerToys 创建终端启动快捷键：

- 打开 PowerToys，在**键盘管理器**一栏下打开设置界面，选择“启用键盘管理器”，接着打开“重新映射快捷键”
- 选择“快捷键”，同时键入`Win(left)`+`Enter` 两个按键，然后选择“允许组合键”和“精准匹配”。
- 接着在“操作”选项中选择“运行程序”
- 选择应用路径为 `C:\Users\<USERNAME>\AppData\Local\Microsoft\WindowsApps\wt.exe`
- 参数不设置也行
- 启动路径可以简单设置为 `C:\Users\<USERNAME>`
- 权限级别选择“正常”，如果正在运行选择“启动另一个”，可见性选择“正常”
- 点击确定后退出

接下来，按下 `Win`+`Enter` 键就可以快速唤出终端了~

## 3. 终端美化
刚才我们已经配置好了快捷键，现在我们按下 `Enter`+`Win` 来唤出一个终端界面，可以看到目前的终端界面只有黑白双色，十分单调。该怎么让它变得更好看呢？

### 3.1 字体、透明背景与高斯模糊
点击标签栏上的“**向下箭头**”，打开设置界面，选择 “Windows PowerShell”，接着在“其他设置”中打开外观选项。

在字体栏中输入 `JetBrainsMono Nerd Font` 来设置终端字体为 JetBrainsMono。

> 当然，这里你也可以设置其他你喜欢的字体，前提是你在系统中已经安装。
> 
> 什么？你不知道如何安装？自己到设置里面去搜“**字体**”（

接下来，在透明度栏下设置背景不透明度为 **80%**，并选择下面的“**启用亚克力材料**”来开启高斯模糊（个人觉得这样的配置很好看，当然具体如何设置完全看你的喜好）

### 3.2 oh-my-posh 终端主题的配置
> oh-my-posh 官方文档：https://ohmyposh.dev/docs/

首先，在终端界面使用 `scoop` 安装 Oh My Posh：

```powershell
scoop install oh-my-posh
```

安装完成后，前往[主题文档界面](https://ohmyposh.dev/docs/themes)选择你喜欢的主题（当然，如果你有实力的话也可以参照官方文档自己写一套主题配置，只不过比较麻烦）

比如这里小生选择了 [catppuccin_mocha](https://github.com/JanDeDobbeleer/oh-my-posh/blob/main/themes/catppuccin_mocha.omp.json) 主题，在官方仓库中下载主题文件 `catppuccin_mocha.omp.json` 至 `C:\Users\<USERNAME>\config_dir\` 路径下（当然，你也可以随便找一个路径放这个配置文件，只要你下次想修改的时候方便找到它就可以）

下载完成后，回到终端界面，查找你的 PowerShell 配置文件 `Microsoft.PowerShell_profile.ps1`：

```powershell
$PROFILE
```

如果文件不存在的话，使用下面这行命令创建配置文件：

```powershell
if (!(Test-Path -Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force }
```

接下来编辑配置文件：

```powershell
notepad $PROFILE
```

然后再文件末尾加入（或修改为）：

```powershell
oh-my-posh init pwsh --config "C:\Users\<USERNAME>\config_dir\catppuccin_mocha.omp.json" | Invoke-Expression
```

保存并重新打开终端，这时候就会自动加载 Oh My Posh 主题。


## 4. 其他终端工具
### 4.1 neovim

> 官方网址：https://neovim.io/

是的，你在 Windows 上也可以使用 neovim 来在终端中编辑文本，方便又快捷。

直接使用 `scoop` 安装即可：

```PowerShell
scoop install neovim
```

配置过程和在 Linux 上一致，详情请参考官方文档，这里不再赘述

### 4.2 fastfetch
快速显示设备信息的便捷工具，安装操作如下：

```PowerShell
scoop install fastfetch
```

安装完成后，就可以在终端界面使用 `fastfetch` 命令来快速显示设备信息啦：

```PowerShell
fastfetch
```

至于更高级的配置，请参考官方文档，这里不再赘述。