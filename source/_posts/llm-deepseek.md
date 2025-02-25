---
title: 在 Archlinux 上本地部署 Deepseek-R1
date: 2025-02-05 13:23:11
tags:
    - LLM
    - Archlinux
categories:
    - LLM
description: |
    🔍 记录自己第一次本地部署大模型的过程
---

# 1. 安装 Ollama
可以直接在官网上下载 LM Studio 的安装包：https://lmstudio.ai/

但是既然小生用的是 Archlinux，这里就直接用 aur 拉包即可：

```Terminal
paru -S ollama
```

接下来跟随指引安装。