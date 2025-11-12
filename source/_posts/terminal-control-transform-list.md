---
title: 终端控制转义序列：可以用任意文件格式执行的终端操作 
date: 2025-11-12 00:12:40
tags:
    - Archlinux
categories:
    - Archlinux
description: |
    是的，这个玩意可以放在任何文件里，cat 一下就可以被执行（
---

## 0. 故事背景
Coldrain 的 Hyprland 桌面环境使用了 [illogical-impulse](https://ii.clsty.link/en/) 的主题，配置文件自然也是使用人家编辑好的方案。

有一天，Coldrain 在微调配置文件的时候，偶然在 fish 的配置文件 `~/.config/fish/config.fish` 中发现了这么一段内容：

```shell
if test -f ~/.local/state/quickshell/user/generated/terminal/sequences.txt
    cat ~/.local/state/quickshell/user/generated/terminal/sequences.txt
end
```

为什么配置文件里要 `cat` 一个 `.txt` 文件呢？Coldrain 于是前去查看该文件的内容，使用 `nvim ~/.local/state/quickshell/user/generated/terminal/sequences.txt` 后，发现文件内部是一堆乱七八糟的代码

Coldrain 看不懂，于是默默退出，然后重新打开一个新的 tty 后（配置不同），尝试执行 `cat ~/.local/state/quickshell/user/generated/terminal/sequences.txt` 这一操作，突然间，新的 tty 背景变色了，透明度与高斯模糊都被更改了，好神奇！


## 1. 这是什么？
Coldrain 查阅资料后发现，这种东西叫做**终端控制转义序列（ANSI/OSC/xterm 序列）**。当 Coldrain 在 `config.fish` 里 `cat` 这个文件时，这些控制序列会被终端解释，从而动态修改终端配色（包括前景/背景/调色板），所以背景颜色会变。


（编辑中...）