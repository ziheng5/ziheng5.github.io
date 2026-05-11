---
title: Hyprland 常见问题
date: 2025-10-18 12:31:06
tags:
    - Archlinux
categories:
    - Archlinux
description: |
    🔍 记录配置 Hyprland 时遇到的问题及其解决方案
---
## 1. 桌面环境问题
### 1.1 启动失败（回到登陆界面）
目前 Coldrain 已经遇到过一次 **从 SDDM 登陆界面启动 Hyprland 后又回到了 SDDM 登陆界面**的赛博鬼打墙（

Coldrain 切换至 tty3 中，试图查找 Hyprland 的运行日志来调查问题的根本原因，但是发现 Hyprland 根本没有生成运行日志，也就是说，Hyprland 还没来得及写日志，就已经崩溃了。

于是，Coldrain 尝试从 tty 中启动 Hyprland，看看是怎么崩溃的：

```bash
Hyprland
```

结果根据报错信息发现是 abseil 版本过新，导致 hyprland 无法找到原本配置文件的位置（人话就是：滚包把 hyprland 滚炸了💦）

Coldrain 接着前往 Hyprland 官方文档的 Q&A 中寻找解决方案，发现官方已经注意到这个问题。那么解决方案就是：把你系统中所有包含 `hypr*` 的包及依赖这些包的包全部卸载掉，然后重新用 `cmake` 编译 Hyprland，Coldrain 亲测有效（虽然有点麻烦就是了）。

> ⚠️ 记得备份好你现有的配置文件

### 1.2 启动后各种配件消失（但是应用程序和终端可以唤出）
遇到这种情况一般是在你滚包之后，各种配件更新后与 Hyprland 官方的配置不兼容导致

可以先切换到 tty 界面，将当前启动的 Hyprland 关闭：

```bash
killall Hyprland
```

接下来手动启动 Hyprland，并并将输出重定向到一个文件以便查看：

```bash
Hyprland > ~/hyprland.log 2>&1
```

启动后，在终端打开你的 `hyprland.log`，看看问题出在哪里。

当然，如果你实在找不到问题，或者实在懒得查问题的话，完全可以把 Hyprland 重新编译安装一遍（


## 2. 应用程序问题
### 2.1 linuxqq 中无法正常使用 fcitx5 输入法
Hyprland 不依赖传统的 X11 协议，而是使用 Wayland，这意味着应用程序要想在 Hyprland 下运行，必须通过 Wayland 原生支持或 XWayland 兼容层（用于运行老的 X11 应用）

`linuxqq` 是一个 Electron 应用，Electron 在 Linux 上同时支持两种后端：`X11` 和 `Wayland`。默认情况下，`linuxqq` 会根据当前环境变量自动选择运行模式，也就是说 `linuxqq` 在 `Hyprland` 上默认通过 `Wayland` 启动，然而 `linuxqq` 默认未启用 `--enable-wayland-ime`，导致 fcitx5 无法检测到输入窗口，最终表现为：在 linuxqq 中无法正常使用 fcitx5 输入法

解决方案如下：

- 在 `/usr/bin/linuxqq` 中加入如下内容（直接复制粘贴到最后一行即可）：

```bash
exec /opt/QQ/qq ${QQ_USER_FLAGS[@]} "$@" --ozone-platform-hint=x11 --enable-wayland-ime --wayland-text-input-version=3
```


### 2.2 Steam 启动后 GUI 界面模糊
> ArchWiki 有相关讨论与解决方案：[Blurry text and graphics with Xwayland and HiDPI](https://wiki.archlinux.org/title/Steam/Troubleshooting#Blurry_text_and_graphics_with_Xwayland_and_HiDPI)

当 Steam 作为 Xwayland 客户端在一个使用 HiDPI 缩放的合成器下运行时，你可能会发现 Steam 和游戏以一半的分辨率渲染，然后缩放以适应 HiDPI 屏幕。这会导致图形模糊，如下图所示：

![Steam Blur](/images/hypr_problems/steam_blur.png)

在 Hyprland 下如果遇到这个问题，可以在 `~/.config/hypr/hyprland.conf` 文件（或者你的其他 Hyprland 配置文件中）添加如下内容：

```json
xwayland {
    force_zero_scaling = true
}
```

> 但是这么修改之后，窗口又有点太小了？暂时没想到如何解决这个问题...