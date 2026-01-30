---
title: Arch Linux 运行 AUR 版腾讯会议（wemeet）踩坑复盘：ZLIB_1.2.9 / Qt_5 / segfault 一条龙
date: 2026-01-30 19:02:09
tags:
    - Archlinux
categories:
    - Archlinux
description: |
    ✍ 一篇关于 Arch Linux 下工具链错误调用的记录

---

> 故事背景：
> 
> Coldrain 在 Arch Linux 上装了 AUR 版腾讯会议 wemeet，本来想开个会，结果被一条“工具链抢库”给按在地上摩擦。最后 Coldrain 用一种**不污染系统**、但又非常 “Linux 现实主义” 的办法把它救活了。

---

## 0. 开局暴击：会议软件为什么会去读 RISC-V 工具链的库？

Coldrain 在终端里敲下 `wemeet`，它回敬 Coldrain 一条极其具有 “Linux 真实感” 的报错：

```
/opt/wemeet/bin/wemeetapp: /opt/wch/Toolchain/RISC-V Embedded GCC/bin/libz.so.1:
version `ZLIB_1.2.9' not found (required by /usr/lib/wemeet/libQt5Gui.so.5)
```

这句话信息量很大：

* wemeet 需要 `ZLIB_1.2.9`
* 需求来自它的 Qt 库：`/usr/lib/wemeet/libQt5Gui.so.5`
* 但它实际加载到的 `libz.so.1` 居然来自：
  **`/opt/wch/Toolchain/RISC-V Embedded GCC/bin/libz.so.1`**

这就离谱了：一个桌面会议软件，为什么要去吃Coldrain  **RISC-V 嵌入式工具链**里的 zlib？

> 事后 Coldrain 才反应过来 RISC-V 这玩意好像是上个月自己玩沁恒的板子时装的 💦

**第一反应：库路径被污染了。**

---

## 1. 第一轮推理：是不是 LD_LIBRARY_PATH / LD_PRELOAD 的问题

动态链接器（ld.so）找库时，最常见的“邪路”就是：

* `LD_LIBRARY_PATH`
* `LD_PRELOAD`

所以 Coldrain 第一步尝试最简单的 “清空环境变量启动”：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD wemeet
```

结果——**完全没用**，还是同样的 `ZLIB_1.2.9 not found`，还在命中 `/opt/wch/.../libz.so.1`。

这就说明：
问题不只是 “当前 shell 环境变量污染”。它可能来自更深的层（比如启动脚本、系统 ld 缓存、RPATH/RUNPATH、甚至某些全局 profile）。

但Coldrain 当时没急着去大拆系统；Coldrain 先想的是：**先让它跑起来。**

---

## 2. 第二轮踩坑：强行 preload 系统 zlib，Qt 直接炸给Coldrain 看

Coldrain 的想法很直接：

> 你老是加载错 zlib？那 Coldrain 用 `LD_PRELOAD` 把正确的 zlib 先塞进去，抢占符号解析优先级。

于是Coldrain 尝试：

```bash
env LD_LIBRARY_PATH=/usr/lib:/lib \
    LD_PRELOAD=/usr/lib/libz.so.1 \
    /opt/wemeet/bin/wemeetapp
```

结果新错误来了：

```
symbol lookup error: /usr/lib/wemeet/libwemeet_framework.so:
undefined symbol: _ZN7QWidget11eventFilterEP7QObjectP6QEvent, version Qt_5
```

这行属于 Qt 用户的噩梦：`version Qt_5` + `undefined symbol`。

**翻译成人话：**
Coldrain 为了救 zlib，把 `LD_LIBRARY_PATH` 指向系统库路径，结果导致 wemeet 的 Qt（它自带一套 Qt）和系统 Qt 混用了。
Qt 这种体系一旦混用（QtCore/QtGui/QtWidgets 或 plugins 版本不一致），要么直接 `abort`，要么 segfault，基本没救。

这时候 Coldrain 意识到一个非常重要的原则：

> **桌面闭源软件的自带 Qt：要么全用它那套，要么全不用。最怕“掺着来”。**

---

## 3. 第三轮定位：它不是随机崩，是 Qt 自己 `qFatal()` 然后 `abort()`

Coldrain 后来还遇到了 segfault，于是用 gdb 抓 backtrace。关键栈长这样：

```
QMessageLogger::fatal(...)
QGuiApplicationPrivate::createPlatformIntegration()
...
abort()
```

这不是“内存乱飞”那种随机崩，这是 Qt **明确判定环境不对，直接 fatal 退出**。

`createPlatformIntegration()` 这一步典型对应：

* 平台插件加载失败（`xcb` / `wayland`）
* 插件依赖缺失
* 插件来自系统 Qt，但 Qt 库来自 wemeet 自带（或反过来）

到这 Coldrain 基本确认：
**Coldrain 不能用粗暴的 LD_LIBRARY_PATH=/usr/lib 这种方式。否则 Qt 混用必炸。**

---

## 4. 最终解法：不去和系统抢，直接在 wemeet 的私有目录里“放一个它要的 zlib”

核心矛盾一直没变：

> wemeet 总能“莫名其妙”加载到 `/opt/wch/.../libz.so.1`（老的），然后找不到 `ZLIB_1.2.9`。

所以 Coldrain 反向操作：
既然它喜欢从自己的库目录（`/usr/lib/wemeet`）加载东西，那Coldrain 就把“正确版本的 zlib”放进 `/usr/lib/wemeet`，让它**优先命中正确库**，从而绕开 `/opt/wch` 的抢库。

这招的好处是：

* 不动系统 `/usr/lib/libz.so.1`（不污染系统）
* 不改工具链目录（不破坏开发环境）
* 只影响 wemeet 自己

### 4.1 Coldrain 踩的坑：复制后 chmod 失败，提示 dangling symlink

Coldrain 一开始把 `/usr/lib/libz.so.1` “复制”过去，然后：

```bash
sudo chmod 755 /usr/lib/wemeet/libz.so.1
```

结果报：

```
chmod: cannot operate on dangling symlink '/usr/lib/wemeet/libz.so.1'
```

这个坑非常典型，也很“Linux”：

* 在 Arch 上，`/usr/lib/libz.so.1` 往往是个 **符号链接**
  指向真实文件，比如 `libz.so.1.3.1`（版本可能不同）
* Coldrain 如果用 `cp -a` 把它复制过去，复制的是 **symlink 本身**
* 但它指向的真实文件并没被复制到 `/usr/lib/wemeet`
* 所以 `/usr/lib/wemeet/libz.so.1` 变成了**断链**（dangling symlink）
* 断链当然 chmod 不动

### 4.2 正确姿势：复制“真实文件”到 wemeet 目录，再重建 symlink

最终可复现、不会断链的做法如下：

```bash
# 删除断链（如果存在）
sudo rm -f /usr/lib/wemeet/libz.so.1

# 找到系统 zlib 的真实文件路径（不是 symlink）
real=$(readlink -f /usr/lib/libz.so.1)
echo "real zlib = $real"

# 复制真实文件进 wemeet 私有目录
sudo cp -v "$real" /usr/lib/wemeet/

# 给真实文件正确权限
sudo chmod 755 "/usr/lib/wemeet/$(basename "$real")"

# 在 wemeet 目录里重建 libz.so.1 软链接
sudo ln -s "$(basename "$real")" /usr/lib/wemeet/libz.so.1

# 验证
ls -l /usr/lib/wemeet/libz.so.1
```

然后—— Coldrain 直接运行 `wemeet`：

**成功启动。**

那一刻非常舒爽，属于“你可以和闭源软件讲道理，但最好还是用 ld.so 能听懂的语言”。

---

## 5. 验证：Coldrain 不是碰巧跑起来，而是真的把抢库路线改掉了

为了确保不是侥幸（比如某次恰好没撞上工具链库），Coldrain 用 `LD_DEBUG=libs` 看动态链接器到底加载了谁：

```bash
LD_DEBUG=libs wemeet 2>&1 | grep -E "libz\.so\.1|wch|/usr/lib/wemeet" | head -n 60
```

理想结果是能看到类似：

* `libz.so.1 => /usr/lib/wemeet/libz.so.1`
* 不再出现 `/opt/wch/Toolchain/.../libz.so.1`

只要这两点成立，就说明问题从“路径抢库”层面被稳稳解决。

---

## 6. 硬核解释：这类问题本质是什么？

一句话：

> **动态链接器按规则找库，但你机器上“库太多、路径太脏”。**

你的工具链（WCH RISC-V）里带了 `libz.so.1`，而且它在某种路径优先级上压过了系统 zlib。

当软件需要 `ZLIB_1.2.9` 的符号版本时：

* 系统 zlib：有
* 工具链 zlib：没有
* 于是报错

而 Qt 那个阶段的崩溃，属于次生灾害：
你为了救 zlib 改了 `LD_LIBRARY_PATH`，结果 Qt 自带库和系统 Qt 插件/依赖混在一起，直接 fatal/abort。

---

## 7. 后记：最建议做的“根治”检查（可选）

Coldrain 这次用“往 `/usr/lib/wemeet/` 塞依赖”救活了 wemeet，但从系统健康角度，最好还是查一下 **为什么 `/opt/wch` 会被拿来当动态库搜索路径**。

建议跑：

```bash
sudo grep -R "/opt/wch" /etc/ld.so.conf /etc/ld.so.conf.d 2>/dev/null
ldconfig -p | grep -E "opt/wch|libz\.so\.1"
```

如果你真看到 `/opt/wch/...` 被写进了 `ld.so.conf.d`，那就意味着未来别的软件也可能被它抢库。
这时候把它从系统 ld 配置里移走、再 `sudo ldconfig`，才是更“长期”的清爽方案。

---

## 8. 结语：Linux 的浪漫就是——你总能找到一个让它听话的方式

这次的最终解法并不花哨，就是一句话：

> **不要替换系统库；把正确库放进应用自己的库目录，让它优先命中。**

它不优雅，但很有效；
它不抽象，但很现实；
它不需要争论“哪个包该负责”，只需要让 ld.so 在运行时做出正确选择。

## 9. 参考文献

[1] https://stackoverflow.com/questions/48306849/lib-x86-64-linux-gnu-libz-so-1-version-zlib-1-2-9-not-found