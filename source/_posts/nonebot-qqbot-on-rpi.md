---
title: 从零开始在树莓派上部署 Nonebot2 QQ Bot
date: 2026-04-04 16:41:18
tags:
    - 树莓派
    - Bot
categories: 
    - 树莓派
description: |
    ✨ 真的“从零开始”，在树莓派上利用 NapCat + NoneBot2 部署 QQ Bot 服务喵
---

> **硬件准备**：
> - 树莓派 4B（`8GB` 内存）
> - 散热片 + 小风扇（考虑到 Bot 服务需要长时间运行）
> - SD 卡一张（存储最好大一点）
> - USB-SD 卡转接器（用于在 SD 卡上安装操作系统）

## 1. 方案总览

基于 NapCat + NoneBot 的 QQ 机器人通信链路如下图所示（AI 画图真好用啊💦）：

![communication](/images/raspberrypi/qqbot_communication.png)



## 2. 操作系统选择与安装
我们的需求目前只是将一个机器人服务跑在树莓派上，因此操作系统最好使用**无桌面**的版本以减轻树莓派负担。

由于[树莓派官方](https://www.raspberrypi.com/documentation/computers/os.html)明确写了，**Raspberry Pi OS 是官方支持、并且“推荐用于大多数树莓派场景”的系统**；它本身又是**基于 Debian、并针对树莓派硬件做了优化**。同时官方提供了 **64 位 Lite 版**，并明确说明 Lite 没有图形桌面，非常适合 **headless servers（无头服务器）**这种用途。

于是这里我们选择 **Raspberry Pi OS Lite (64-bit)** 作为我们使用的操作系统。

接下来，我们将采用树莓派官方的 **Raspberry Pi Imager** 软件包来安装操作系统。

对于 **Archlinux** 用户，包管理器仓库里的软件包如下所示：

![rpi-imager](/images/raspberrypi/rpi-imager.png)

执行下方指令安装 **rpi-imager**：

```bash
# 使用 pacman 安装
sudo pacman -S rpi-imager

# 或者 paru
paru -S rpi-imager
```

安装完成后，先将 SD 卡插入转接器，再将转接器插入电脑 USB 接口，打开 Raspberry Pi Imager，按照提示在 SD 卡中安装 Raspberry Pi OS Lite (64-bit)，如下所示：

![install-system](/images/raspberrypi/install-system.png)

等待安装完成后，将 SD 卡插入树莓派，再给树莓派接上外接显示器，接下来就可以开始操作系统的简单配置了。

## 3. 服务器的简单配置

由于通过 rpi-imager 安装的 Raspberry Pi OS Lite 系统本身就已经具有了一定的配置，所以我们需要配的并不多

### 3.1 更新系统

```bash
sudo apt update
sudo apt full-upgrade -y

# 更新完之后重新启动
reboot
```

### 3.2 安装常用工具
虽然这些不是树莓派官方强制要求，但是后面部署 NoneBot 的时候会用上：

```bash
sudo apt update
sudo apt install -y \
git curl wget vim nano tmux htop tree unzip zip ca-certificates \
python3-venv python3-pip pipx sqlite3 jq lsof rsync dnsutils net-tools \
build-essential ufw
pipx ensurepath
```


### 3.3 安装 zsh
相比于 bash，zsh 的优点如下：
- 命令补全体验更好
- 提示符更灵活
- 历史记录和自动建议更舒服
- 交互使用更顺手
- 脚本配置语言和 bash 几乎一致

话不多说，直接安装：
```bash
sudo apt update
sudo apt install -y zsh
chsh -s /usr/bin/zsh

# 退出重新登录
reboot
```

zsh 的配置建议写在 `~/.zshrc` 里

> 此外，也可以尝试一下 Coldrain Dotfiles 里的 `.zshrc` 配置喵～
> 地址：https://github.com/ziheng5/dotfiles/blob/master/dotfiles/zsh/.zshrc

### 3.4 终端美化（可选）
这里主要是使用 **oh-my-posh** 对终端 prompt 进行美化，具体操作参考 Coldrain 的[这篇文章](https://coldrain.top/2025/04/11/llm-deepseek/)

当然，oh-my-posh 的配置 Coldrain 也有一套（[在这里](https://github.com/ziheng5/dotfiles/blob/master/dotfiles/oh-my-posh/catppuccin_coldrain.omp.json)），配置文件为 `catppuccin_coldrain.omp.json`，将这个文件塞到你的 `~/.config/oh-my-posh/themes/` 路径下面

如果你使用的是 Coldrain 的 `~/.zshrc`，那么接下来就没有什么配置了，否则需要在你的 `~/.zshrc` 中添加如下配置：

```shell
eval "$(oh-my-posh init zsh --config ~/.config/oh-my-posh/themes/catppuccin_coldrain.omp.json)"
```


### 3.5 Neovim + LazyVim
我们后续需要通过 SSH 远程连接到树莓派上进行开发，那么使用终端集成开发环境会很方便，这里我们选择使用 **Neovim + LazyVim**。Neovim 是 vim 的升级版，极大扩展了 vim 的可配置性，而 LazyVim 是 Neovim 的一套第三方个性化配置方案，其配置程度差不多是将整个 VSCode 塞进终端里。

⚠️ 安装的时候要注意：Raspberry Pi OS Lite 上通过 `apt` 从仓库里安装到的 neovim 版本过低，似乎很久没有更新了，需要从 Neovim 官方的 Github 仓库下载最新版本：

```bash
# 注意，要下载 arm64 版的，因为树莓派是 arm64
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-arm64.appimage

# 赋予 appimage 可执行权限
chmod u+x nvim-linux-x86_64.appimage

# 运行安装
./nvim-linux-x86_64.appimage

# 全局配置
mkdir -p /opt/nvim
mv nvim-linux-x86_64.appimage /opt/nvim/nvim
```

然后接下来将下面这一行配置添加到 `~/.bashrc` 或 `~/.zshrc`里（看你使用哪一个 sh 了）：

```shell
export PATH="$PATH:/opt/nvim/"
```

Neovim 安装完成！接下来给 Neovim 安装 LazyVim 配置（也可以参考[官方文档](https://www.lazyvim.org/installation)）

首先对原有的 Neovim 配置进行备份（如果你有的话，否则没必要）：
```bash
# required
mv ~/.config/nvim{,.bak}

# optional but recommended
mv ~/.local/share/nvim{,.bak}
mv ~/.local/state/nvim{,.bak}
mv ~/.cache/nvim{,.bak}
```

接着从 Github 上 clone 配置文件：

```bash
git clone https://github.com/LazyVim/starter ~/.config/nvim
```

接着删除克隆文件夹下的 `.git` 文件：

```bash
rm -rf ~/.config/nvim/git
```

此时启动 Neovim，你会发现你的 Neovim 界面发生巨变：

```bash
nvim
```

试试看编辑当前文件夹（`nvim ./`），你会发现它长得和 VSCode 几乎一模一样：

![lazyvim](/images/raspberrypi/lazy.png)


### 3.6 SSH 配置与连接

> SSH 之前，建议在 [Tailscale](https://login.tailscale.com) 上为树莓派搞一个虚拟 IP，这样以后即使没有局域网，也可以走 Tailscale 远程 SSH 连接到你的树莓派。
> 具体操作请参考官方文档，这里不作展开

首先，先在自己的设备上生成密钥：

```bash
# 这里也可以换成你自己的名字
ssh-keygen -t ed25519 -C "coldrain-to-pi"
```

一路回车即可，接下来上传公钥：

```bash
# 这里要换成你树莓派上的用户名和你树莓派的局域网 IP 或 Tailscale 虚拟 IP
ssh-copy-id coldrain@192.169.101.240
```

接下来测试能否免密登录：
```bash
# 依旧替换成你自己的用户名和 IP
ssh coldrain@192.168.101.240
```

如果可以登录进去，则回到树莓派上备份 SSH 配置：

```bash
sudo cp /etc/ssh/sshd_config /etc/ssh/ssh_config.bak.$(date +%F-%H%M%S)
```

然后修改 SSH 配置：

```bash
sudo nvim /etc/ssh/sshd_config
```

确认这几项存在且为下面的值：

```shell
PubkeyAuthentication yes
PasswordAuthentication no
PermitRootLogin no
ChallengeResponseAuthentication no
UsePAM yes
```

接下来检查配置语法：
```bash
sudo sshd -t
```

如果没有任何输出，则可以说明配置语法没有问题，接下来我们重启 SSH 服务即可完成配置：

```bash
sudo systemctl restart ssh
```

> 此外，如果你在自己的设备上通过 **kitty** SSH 到树莓派，则可能会出现下述情况：
> “kitty 默认把 `TERM` 设置为 `xterm-kitty`，但你在终端里 SSH 到树莓派上时，树莓派一端没有正确识别 `xterm-kitty` 对应的 terminfo 能力描述，导致 zsh 的行编辑变成退格键、光标移动到行为解释出错，进而按下删除键可能反而打出空格”
> 有两种解决方案：
> - 1. 将树莓派端更改为 `xterm-256color`
> ```bash
> # 这个配置远端是一定能认识的通用终端类型
> export TERM=xterm-256color
> ```
> - 2. 在树莓派上补上 `kitty-terminfo`（推荐）
> ```bash
> sudo apt update
> sudo apt install -y kitty-terminfo
> ```

## 4. Python 工具链配置
在这个项目里，我们使用 `pipx` 命令行工具来配置 Python 环境，且每个项目单独配置一个 `venv`

首先，我们将项目的工作区建立好：

```bash
mkdir -p ~/apps ~/apps/qqbot ~/src ~/venvs
```

其中，
- `~/apps/qqbot` 用作 NoneBot 项目路径
- `~/src` 作为平时测试脚本、拉代码的路径
- `~/venvs` 放置独立虚拟环境


接下来，我们先创建一个虚拟环境并激活它：

```bash
cd ~/apps/qqbot
python3 -m venv .venv
source ~/apps/qqbot/.venv/bin/activate
```

激活后，升级虚拟环境里的打包工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

接下来安装我们的项目运行依赖，以及全局 CLI 工具：

```bash
pip install nonebot2 nonebot-adapter-onebot

pipx install nb-cli
```

> 💡 上述工作完成后，可以顺手改一下 pip 配置：
> 先创建配置目录：
> ```bash
> mkdir -p ~/.config/pip
> ```
> 接着写配置文件
> ```bash
> cat > ~/.config/pip/pip.conf <<'EOF'
> [global]
> disable-pip-version-check = true
> timeout = 120
> EOF
> ```
> 该配置可以让 pip 不总是提示版本检查，且网络慢一点的时候更“宽容”

## 5. 创建最小 NoneBot 项目

先创建 `.env`：

```bash
cat > ~/apps/qqbot/.env <<'EOF'
DRIVER=~fastapi+~websockets
HOST=127.0.0.1
PORT=8080
LOG_LEVEL=INFO
COMMAND_START=["/"]
COMMAND_SEP=["."]
ONEBOT_ACCESS_TOKEN=改成你自己的长随机字符串
EOF
```

其中，
- `HOST` 默认是 `127.0.0.1`
- `PORT` 默认是 `8080`
- `COMMAND_START` 默认是 `["/"]`
- `COMMAND_SEP` 默认是 `["."]`
- `DRIVER` 可以通过 `.env` 来指定
- `ONEBOT_ACCESS_TOKEN` 改成一串自己的随机字符串，比如 `ONEBOT_ACCESS_TOKEN=9sYw4mR2Pq7XcK8nF5uL1aZ6tH3vB0`

接下来，创建入口文件 `~/apps/qqbot/bot.py`：

```python
import nonebot
from nonebot.adapters.onebot.v11 import Adapter

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(Adapter)

nonebot.load_plugins("plugins")

if __name__ == "__main__":
    nonebot.run()
```

接下来我们创建插件目录：

```bash
mkdir -p ~/apps/qqbot/plugins
touch ~/apps/qqbot/plugins/__init__.py
```

然后创建一个最小测试插件 `~/apps/qqbot/plugins/ping.py`：

```python
from nonebot import on_command
from nonebot.adapters.onebot.v11 import MessageEvent

ping = on_command("ping", priority=5, block=True)

@ping.handle()
async def _(event: MessageEvent):
    await ping.finish("pong")

```

## 6. 接入 NapCat
[NapCat 官方文档](https://napneko.github.io/guide/napcat)当前给出的 WebUI 基础操作是：
- 启动 NapCat
- 访问 `http://ip:port/webui/`
- 进入 QQ 登陆
- 用 QRCode 扫码
- 登陆后进入**网络配置**
- 点击新建
- 建立对应的服务器或客户端

> 此外，文档还特别提醒：如果是公网部署，务必启用 Token

话不多说，这里我们再开启一个终端进程，先安装一下 NapCat

NapNeko 的 `napcat-linux-installer` [仓库](https://github.com/NapNeko/napcat-linux-installer)当前给出了 Linux 的一键安装命令；Napcat 另一个安装器仓库也写明 Linux 支持 Debian/Ubuntu 这类发行版，并提供 `ws / reverse_ws / reverse_http` 这些模式

接下来我们在终端里执行安装：

```bash
cd ~
curl -o napcat.sh https://raw.githubusercontent.com/NapNeko/napcat-linux-installer/refs/heads/main/install.sh && sudo bash napcat.sh
```

安装完成后，使用 `bash` 运行安装目录下的 `launcher.sh`：

```bash
# 具体路径看你安装在哪
bash ~/launcher.sh
```

运行成功之后，按照终端给出的提示，我们需要登录 WebUI，在浏览器中访问 `http://你的服务器 IP:6099/webui/`，然后进入 QQ 登录

> 这里注意 ⚠️：请使用你想要用来作为机器人的 QQ 号登录 WebUI！

登录完成后，按照页面提示更新 WebUI Token / 修改密码，然后接下来我们在 NapCat WebUI 里配置 **OneBot 反向 WebSocket**：
- 进入 “网络配置”
- 点击 “新建”
- 选择 WebSocket 客户端
- URL 填：`ws://127.0.0.1:8080/onebot/v11/ws`
- Token 里填写你在 `.env` 里随便创建的 `ONEBOT_ACCESS_TOKEN`
- 勾选 “保存时启用”


当 NapCat 成功连接上后，`python bot.py` 那个终端应该会开始出现连接日志，此时你去 QQ 里向你的机器人账号发送 `/ping`，如果返回 `/pong`，那么恭喜！🎉 你的整条机器人服务链路已经全通了！


## 7. 优化服务链路
### 7.1 将 NoneBot 做成 systemd 服务
如果你确认 `/ping` 已经成功，那么就接着往下面看吧～

首先创建服务文件：

```bash
sudo nvim /etc/systemd/system/nonebot-qqbot.service
```

写入下面的内容：

```INI
[Unit]
Description=NoneBot QQ Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=你的用户名
WorkingDirectory=/home/你的用户名/apps/qqbot
Environment=PATH=/home/你的用户名/apps/qqbot/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/home/你的用户名/apps/qqbot/.venv/bin/python /home/coldrain/apps/qqbot/bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

然后启用：

```bash
sudo systemctl daemon-reload
sudo systemctl enable nonebot-qqbot
sudo systemctl start nonebot-qqbot
sudo systemctl status nonebot-qqbot --no-pager
```

### 7.2 最小 NoneBot 项目改进

由于：
- NoneBot 支持用 `SUPERUSERS` 配置超级用户；权限控制里也直接提供了 `SUPERUSER` 权限。
- NoneBot 官方建议，本地文件存储统一用 `nonebot-plugin-localstore` 管理。
- NapCat 当前也明确建议把 `message_id / user_id / group_id` 这类 ID 按字符串处理；而 NoneBot 配置里的 `SUPERUSERS` 也是 `set[str]`。

所以接下来我们可以改进一下我们之前创建的最小 NoneBot 项目

首先补充一下 `~/apps/qqbot/.env` 配置为下面的样子：

```env
DRIVER=~fastapi+~websockets
HOST=127.0.0.1
PORT=8080
LOG_LEVEL=INFO
COMMAND_START=["/"]
COMMAND_SEP=["."]
ONEBOT_ACCESS_TOKEN=abc123456
SUPERUSERS=["你的 QQ 号"]
NICKNAME=["小雨","机器人"]
```

接下来我们安装两个依赖包：

```bash
cd ~/apps/qqbot
source ./.venv/bin/activate
pip install psutil nonebot-plugin-localstore
```

其中，`psutil` 用来读取 CPU / 内存 / 进程信息，`nonebot-plugin-localstore` 是 NoneBot 官方推荐的本地数据目录管理方案

接下来我们将入口文件 `~/apps/qqbot/bot.py` 修改如下：

```python
import nonebot
from nonebot.adapters.onebot.v11 import Adapter
from nonebot import require

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(Adapter)

# require 和 load_plugins 都是 NoneBot 官方提供的插件加载方式
require("nonebot_plugin_localstore")
nonebot.load_plugins("plugins")

if __name__ == "__main__":
    nonebot.run()
```

修改完成之后，我们可以在此基础上，试试为我们的机器人添加一个**管理员插件**

首先我们新建文件：

```bash
nvim ~/apps/qqbot/plugins/admin.py
```

在其中写入如下内容：

```python
import platform
import time
from pathlib import Path
import subprocess
import psutil
from nonebot import on_command, require
from nonebot.permission import SUPERUSER
from nonebot.plugin import PluginMetadata
from nonebot.adapters.onebot.v11 import MessageEvent, GroupMessageEvent
from pathlib import Path
require("nonebot_plugin_localstore")
import nonebot_plugin_localstore as store

__plugin_meta__ = PluginMetadata(
    name="基础管理",
    description="提供 whoami / status / paths 三个基础命令",
    usage=(
        "/whoami 查看自己的 user_id 和当前群号\n"
        "/status 查看机器人运行状态（超级用户）\n"
        "/paths 查看本地存储目录（超级用户）"
    ),
    type="application",
    supported_adapters={"~onebot.v11"},
)

START_TIME = time.time()

whoami = on_command("whoami", priority=5, block=True)
status = on_command("status", permission=SUPERUSER, priority=5, block=True)
paths = on_command("paths", permission=SUPERUSER, priority=5, block=True)


@whoami.handle()
async def handle_whoami(event: MessageEvent):
    user_id = event.get_user_id()
    if isinstance(event, GroupMessageEvent):
        msg = f"user_id={user_id}\ngroup_id={event.group_id}"
    else:
        msg = f"user_id={user_id}\nprivate_chat=true"
    await whoami.finish(msg)


@status.handle()
async def handle_status():
    proc = psutil.Process()
    uptime = int(time.time() - START_TIME)
    mem_mb = proc.memory_info().rss / 1024 / 1024
    cpu_percent = psutil.cpu_percent(interval=0.2)
    load_avg = " / ".join(f"{x:.2f}" for x in psutil.getloadavg())

    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    temp_c = int(temp_path.read_text().strip()) / 1000.0

    msg = (
        "机器人状态：\n"
        f"Python: {platform.python_version()}\n"
        f"平台: {platform.platform()}\n"
        f"运行时长: {uptime}s\n"
        f"CPU 温度: {temp_c:.2f} °C\n"
        f"进程内存: {mem_mb:.1f} MB\n"
        f"系统 CPU: {cpu_percent:.1f}%\n"
        f"LoadAvg: {load_avg}"
    )
    await status.finish(msg)


@paths.handle()
async def handle_paths():
    data_dir: Path = store.get_plugin_data_dir()
    cache_dir: Path = store.get_plugin_cache_dir()
    config_dir: Path = store.get_plugin_config_dir()

    msg = (
        "localstore 路径：\n"
        f"data: {data_dir}\n"
        f"cache: {cache_dir}\n"
        f"config: {config_dir}"
    )
    await paths.finish(msg)
```

接下来重启服务，即可让我们刚才做的更改生效：

```bash
sudo systemctl restart nonebot-qqbot
sudo systemctl status nonebot-qqbot --no-pager
```

接下来在 QQ 里尝试向机器人发送 `/whoami`、`/status`、`/paths` 指令试试看吧！😀

### 7.3 将 NapCat 做成 systemd 服务

首先我们新建服务文件：

```bash
sudo nvim /etc/systemd/system/napcat.service
```

写入服务内容：

```INI
[Unit]
Description=NapCat QQ Server
Documentation=https://napneko.github.io/guide
After=network.target

[Service]
Type=simple
User=coldrain
WorkingDirectory=/home/coldrain
Restart=on-failure
ExecStart=/bin/bash /home/你的用户名/你的 NapCat 安装路径/launcher.sh

[Install]
WantedBy=multi-user.target
```

启用服务并启动：
```bash
sudo systemctl daemon-reload
sudo systemctl enable napcat.service --now
sudo systemctl status napcat.service --no-pager
```

做成服务之后，每次开启服务器就不需要手动启动 NapCat 了，而只需要在 WebUI 中登录你的机器人 QQ 号即可！

---

截至目前，我们已经部署好了一个基础的 NoneBot + NapCat 的 QQ 机器人项目！ 🎉