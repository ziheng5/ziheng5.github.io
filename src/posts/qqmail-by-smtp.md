---
title: 通过 SMTP 协议发送一封 QQ 邮件吧
date: 2026-10-12 10:51:45
tags:
    - 网络
categories:
    - 网络
description: |
    跨到计算机之后，动手学习计算机网络～
---

> 💻 实验环境
> - 操作系统：`Arch Linux x86_64`
> - 终端：`zsh`
> - 终端工具：`Telnet`、`OpenSSL`、`dig`
> - 其他：`QQ 邮箱 * 2`（这里使用了大号和小号喵）


> ❓ 二刷计网的时候突发奇想，想试试看能不能用最原始的 `SMTP` 协议来发送 QQ 邮件...


## 1. 关于 SMTP（RFC 5321）
### 1.1 什么是 SMTP

**SMTP**（Simple Mail Transfer Protocol）是因特网电子邮件中主要的**应用层协议**，它使用 `TCP` 可靠数据传输服务，从发送方的邮件服务器向接收方的邮件服务器发送邮件。

像大多数应用层协议一样， `SMTP` 也有两个部分，运行在发送方邮件服务器的客户端和运行在接收方邮件服务器的服务器端，每台邮件服务器上既运行 `SMTP` 的客户端也运行 `SMTP` 的服务器端。

- 当一个邮件服务器向其他邮件服务器发送邮件时，它就表现为 `SMTP` 的客户；
- 当邮件服务器从其他邮件服务器上接收邮件时，它就表现为一个 `SMTP` 的服务器。

### 1.2 SMTP 的工作流程
`SMTP` 是如何将一个报文从发送邮件服务器传送到接收邮件服务器的呢？

- 1. 首先，客户 `SMTP`（运行在发送邮件服务器主机上）在 `25` 号端口建立一个到服务器 `SMTP`（运行在接收邮件服务器主机上）的 `TCP` 连接
- 2. 如果服务器没有开机，客户会在稍后继续尝试连接；一旦连接建立，服务器和客户执行某些应用层的握手，就像人们在互相交流前先进行自我介绍一样，`SMTP` 的客户和服务器在传输信息前先相互介绍。在 `SMTP` 握手的阶段， `SMTP` 客户指示发送方的邮件地址（产生报文的那个人）和接收方的邮件地址
- 3. 一旦该 `SMTP` 客户和服务器彼此介绍之后，客户发送该报文
- 4. `SMTP` 能依赖 `TCP` 提供的可靠数据传输无差错地将邮件投递到接收服务器
- 5. 该客户如果有另外的报文要发送到该服务器，就在该相同的 `TCP` 连接上重复这种处理；否则，它指示 `TCP` 关闭连接



## 2. SMTP 协议发送 QQ 邮件

在看书（即《计算机网络：自顶向下方法（原书第七版）》）的过程中，作者时常提及 `Telnet`，所以一开始我能够想到的途径似乎只有通过 `Telnet` 连接邮件服务器 `25` 号端口，并手动输入 `SMTP` 指令进而实现发送 QQ 邮件

> ❓ **Telnet**（Telecommunication Network，远程登录协议） 是一种基于 TCP/IP 的网络应用层协议，主要用于通过网络远程连接另一台计算机或网络设备，并使用命令行与其交互，默认使用 TCP 23 端口。因为没有安全相关的功能，所以大概相当于 SSH 青春版（雾

那么查阅资料后发现，发送 QQ 邮件具体可分成两种路径：
- 1. 采用 SMTP 原始工作流，查询 qq.com 的 MX（Mail Exchanger，邮件交换器）记录，连接其 `25` 号端口，然后直接向 QQ 的收信服务器发送邮件。但由于现代电子邮件做了很多信息安全相关的工作，所以直接采用这种方法很可能被视为伪造等风险行为，并被拒绝发送
- 2. 登录 QQ 官方的 smtp.qq.com，通过身份认证后，使用自己的邮箱发送 QQ 邮件


对于上述两种路径，其区别在于是否有安全认证。对于前者，我们当然可以直接采用 `Telnet` 连接；而对于后者，由于中间要经历 TLS 加密认证，所以不能简单采用 `Telnet` 来实现，这里我们可以尝试使用 `OpenSSL`

> ❓ **OpenSSL** 是一个开源的密码学与安全通信工具包，主要用于实现 TLS/SSL 安全通信以及各种密码学操作。其主要包含以下三个部分：
> - libcrypto：提供加密、哈希、数字签名等密码学功能
> - libssl：提供 TLS 等安全通信协议的实现
> - openssl 命令行工具：供用户在终端中进行证书管理、加密操作、TLS 连接测试


### 2.1 Telnet 尝试

首先，我们先使用 `dig` 查询 QQ 邮箱的 MX 记录：

```bash
dig MX qq.com
```

> ❓ `dig` 是 Domain Information Groper 的缩写，是一个用于查询 DNS 信息的命令行工具，它可以查询 DNS 中不同类型的资源记录。在 Arch Linux 中，`dig` 由 `bind` 软件包提供

得到的结果如下所示（这里将大量无关重复信息用 "......" 替代）：

```plaintext
; <<>> DiG 9.20.29 <<>> MX qq.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 53028
;; flags: qr rd ra; QUERY: 1, ANSWER: 3, AUTHORITY: 13, ADDITIONAL: 27

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 65494
;; QUESTION SECTION:
;qq.com.				IN	MX

;; ANSWER SECTION:
qq.com.			1262	IN	MX	10 mx3.qq.com.
qq.com.			1262	IN	MX	20 mx2.qq.com.
qq.com.			1262	IN	MX	30 mx1.qq.com.

;; AUTHORITY SECTION:
com.			86188	IN	NS	k.gtld-servers.net.
......（省略部分 DNS 记录）
com.			86188	IN	NS	l.gtld-servers.net.

;; ADDITIONAL SECTION:
l.gtld-servers.net.	45907	IN	A	192.41.162.30
......（省略部分 DNS 记录）
e.gtld-servers.net.	45907	IN	AAAA	2001:502:1ca1::30

;; Query time: 9 msec
;; SERVER: 127.0.0.53#53(127.0.0.53) (UDP)
;; WHEN: Mon Oct 12 17:41:58 CST 2026
;; MSG SIZE  rcvd: 891

```

注意到 "ANSWER SECTION" 一栏中输出了查询到的邮件服务器，我们将其整理到下表中

| 邮件服务器 | 优先级 | TTL（秒）|
| :---: | :---: | :---: |
| `mx3.qq.com` | 10 | 1262 |
| `mx2.qq.com` | 20 | 1262 |
| `mx1.qq.com` | 30 | 1262 |

其中：
- MX 优先级的数字越小，优先级越高
- TTL（Time To Live）表示 DNS 解析结果可以被缓存的剩余时间


接下来，我们按照 MX 优先级，选择优先级最高的 `mx3.qq.com` 进行连接，这里可以继续使用 `dig` 获取服务器的 IP 地址：

```bash
dig A mx3.qq.com +short 
# +short 是指精简输出
```

输出如下 IP 地址：

```plaintext
120.233.20.111
120.241.149.185
112.60.15.191
```

这里查询出三个 IP 地址，恰好涉及到 DNS 中一个非常重要的知识点：**一个域名可以对应多个 IP 地址，这通常与负载均衡、服务器冗余的高可用性有关**

接着我们直接开始使用 `Telnet` 尝试连接服务器 `25` 号端口并发送邮件：

```bash
telnet mx3.qq.com 25
```

接下来，Coldrain 的命令行交互如下所示：

```plaintext
Trying 112.60.15.191...
Connected to mx3.qq.com.
Escape character is '^]'.
220 newxmmxszgpub6-1.qq.com MX QQ Mail Server.
EHLO localhost
250-newxmmxszgpub6-1.qq.com
250-STARTTLS
250-8BITMIME
250-SMTPUTF8
250-SIZE 66060288
250 OK
MAIL FROM:<2199325776@qq.com>
550 SPF check failed. http://service.mail.qq.com/detail/122/72 [diagnostic token redacted; IP: 203.0.113.10]
Connection closed by foreign host.
```

> ⚠️ 为保护隐私，以上 SMTP 日志中的客户端公网 IP 已替换为 RFC 5737 定义的文档示例地址 `203.0.113.10`，并省略了服务器返回的诊断标识符，其余内容保持原样。下文中会对同样的情况重复此操作

交互情况大概可以描述如下：
- DNS 解析：成功 ✅
- TCP 25 端口连接：成功 ✅
- SMTP 欢迎消息 `220`：成功 ✅
- `EHLO` 协商：成功 ✅
- `MAIL FROM`：SPF 验证失败 ❌
- 邮件正文传输：尚未进行

交互结果中，`550 SPF check failed.` 意味着 QQ 的邮件接收服务器不认可我当前使用的公网 IP 作为 qq.com 域名的授权发信源

那么 SPF 为什么会拒绝我的邮件呢？在 SMTP 协议中，声明自己是某个邮箱的发件人，不等于证明自己有权使用该邮箱发件，否则随便找一个陌生人来，都可以用你的邮箱随意发件，或者伪装成其他个人、机构发送信件，造成严重的安全事故。

因此，现代邮件服务器需要防止邮件地址伪造。而 SPF 的核心思想就是 “域名所有者通过 DNS 发布规则，声明哪些 IP 地址或服务器被授权代表该域名发送邮件”。

至此，我们的 `Telnet` 尝试失败了，因为无法通过 SPF 验证。接下来我们将尝试使用 OpenSSL 来实现 QQ 邮件的发送

> ❓ 那么课本上为什么没有提及这一点呢？小生认为《计算机网络：自顶向下方法》这本书需要讲解的内容过于丰富，对于具体的例子如果都过多展开，那么内容过多将不再适合作为教材来带领学生快速入门计算机网络这一领域，所以只是引导学生尝试使用 `Telnet` 实现，那么在实现的过程中自然会发现各种各样的实际问题！

### 2.2 OpenSSL 尝试

在使用 OpenSSL 之前，我们需要知道，QQ 有提供专门的邮件提交服务器 `smtp.qq.com`，通过它可以完成加密连接和身份认证，并继续手动执行 `SMTP` 命令

首先，我们需要在 QQ 邮箱的 Web 客户端上为自己的邮箱账号开启 `SMTP` 服务。打开 QQ 邮箱并登陆，点击 `设置` -> `账号与安全` -> `安全设置` -> `POP3/IMAP/SMTP/Exchange/CardDAV 服务`，如下图所示

![pic1](/images/smtp/qq-smtp-identify.png)

按照提示完成授权并开启服务后，将会得到一份**授权码**（用于身份认证，相当重要，请妥善保管！）

得到授权码后，我们就可以尝试连接 `smtp.qq.com` 并完成身份认证了，使用 `OpenSSL` 进行如下操作：

```bash
openssl s_client -connect smtp.qq.com:465 -servername smtp.qq.com -crlf -quiet
```

其中：
- `s_client` 意味着启动 `OpenSSL` 的 TLS 客户端
- `-connect smtp.qq.com:465` 意味着连接 QQ SMTP 服务器的 TCP 465 端口
- `-servername smtp.qq.com` 意味着发送 TLS SNI 服务器名称
- `-crlf` 意味着将终端输入的换行转换为 `SMTP` 使用的 CRLF
- `-quiet` 意味着减少调试输出，方便手动输入 `SMTP` 命令

连接成功后，命令行出现以下信息，表示此时服务器已经准备好进行 `SMTP` 通信了：

```plaintext
Connecting to 120.233.18.201
depth=2 C=US, O=DigiCert Inc, OU=www.digicert.com, CN=DigiCert Global Root G2
verify return:1
depth=1 C=US, O=DigiCert, Inc., CN=DigiCert Secure Site OV G2 TLS CN RSA4096 SHA256 2022 CA1
verify return:1
depth=0 C=CN, ST=Guangdong Province, L=Shenzhen, O=Shenzhen Tencent Computer Systems Company Limited, CN=*.mail.qq.com
verify return:1
220 newxmesmtplogicsvrszb51-1.qq.com XMail Esmtp QQ Mail Server.
```

接下来尝试输入 `EHLO localhost` 可以查看服务器支持哪些身份认证机制：

```plaintext
EHLO localhost
250-newxmesmtplogicsvrszb51-1.qq.com
250-PIPELINING
250-SIZE 73400320
250-AUTH LOGIN PLAIN XOAUTH XOAUTH2
250-AUTH=LOGIN
250-MAILCOMPRESS
250-SMTPUTF8
250 8BITMIME
```

根据返回内容，服务器声明支持多种认证机制。其中我们可以选择 `AUTH LOGIN`，其工作流程如下所示：

```plaintext
客户端                         QQ SMTP 服务器
   |                                |
   | -------- AUTH LOGIN ---------->|
   |<------ 334 用户名提示 -----------|
   |                                |
   |------ Base64 编码的邮箱 -------->|
   |<------ 334 密码提示 ------------|
   |                                |
   |------ Base64 编码的授权码 ------>|
   |<------ 235 认证成功 ------------|
```

注意，`AUTH LOGIN` 流程中输入的内容需要进行 Base64 编码，而不能直接将文本输入进去。下面先对 Coldrain 的 QQ 邮箱进行 Base64 编码格式转换：

```bash
printf '%s' '2199325776@qq.com' | base64 -w 0
```

输出结果为 `MjE5OTMyNTc3NkBxcS5jb20=`。接下来将之前得到的授权码拿出来，进行同样的操作后转换为 Base64 编码格式，假设得到的结果为 `xxxxxxxxxx`

接下来我们就可以开始执行 `AUTH LOGIN` 认证了：

```plaintext
AUTH LOGIN
334 VXNlcm5hbWU6
MjE5OTMyNTc3NkBxcS5jb20=
334 UGFzc3dvcmQ6
xxxxxxxxxx
```

> 💡 在上面这一段内容中，`VXNlcm5hbWU6` 和 `UGFzc3dvcmQ6` 都是 Base64 格式，解码后的结果分别为 `Username` 和 `Password`

认证成功后，将会输出如下提示：

```plaintext
235 Authentication successful
```

之后我们就可以按照 `SMTP` 协议开始发送邮件了！如下所示（小生的小号就不展示了嘿嘿）：

```plaintext
MAIL FROM:<2199325776@qq.com>
250 OK
RCPT TO:<**********@qq.com>
250 OK
DATA
354 End data with <CR><LF>.<CR><LF>.
From: 2199325776@qq.com
To: **********@qq.com
Subject: SMTP Manual Test

Hello!
This is my first complete SMTP email.
Sent from Arch Linux using OpenSSL.
.
250 OK: queued as.
QUIT
221 Bye.

```

在上述内容中，`MAIL FROM`、`RCPT TO`、`DATA`、`QUIT` 以及正文格式均为 `SMTP` 协议的内容，可以参考教材或 [RFC 5321: Simple Mail Transfer Protocol](https://www.rfc-editor.org/info/rfc5321/)

然后你将在你的小号上收到你大号发送的邮件！（如下图所示）

![pic2](/images/smtp/mail-rcpt.png)


### 2.3 自动化邮件发送脚本（Python 实现）

在清楚具体的流程之后，大概就可以尝试编程实现自动化发送邮件了。这里我们采用 Python 实现：

```python
import smtplib
import ssl
import getpass
from email.message import EmailMessage

SENDER = "2199325776@qq.com"
RECEIVER = "**********@xx.com"  # 这里输入自己的小号，或者其他人的邮箱地址

# 安全输入 SMTP 授权码，不回显
auth_code = getpass.getpass("请输入 QQ 邮箱 SMTP 授权码：")

# 构造邮件
msg = EmailMessage()
msg["From"] = SENDER
msg["To"] = RECEIVER
msg["Subject"] = "SMTP 网络协议实验"
msg.set_content(
    "你好！\n\n"
    "这封邮件通过 Arch Linux 发送。\n"
    "使用 SMTP 协议及 TLS 加密连接。\n\n"
    "网络实验成功！"
)

context = ssl.create_default_context()

try:
    with smtplib.SMTP_SSL(
        "smtp.qq.com",
        465,
        context=context,
        timeout=30
    ) as smtp:
        smtp.login(SENDER, auth_code)
        smtp.send_message(msg)

    print("邮件已被 SMTP 服务器接受！")

except smtplib.SMTPAuthenticationError:
    print("认证失败：请检查 SMTP 是否开启、授权码是否正确。")
except Exception as e:
    print("发送失败：", e)
```

接下来再画一个 GUI 界面，就可以制作出私人的邮件 Client 了（


## 参考文献
[1] 《计算机网络：自顶向下方法（原书第七版）》（美）詹姆斯. F. 库罗斯（James F. Kurose），（美）基思. w. 罗斯（Keith W. Ross）著；陈鸣译 ．一北京：机械工业出版社，2018.5（计算机科学丛书）（P76～P83）

[2] [RFC 5321: Simple Mail Transfer Protocol](https://www.rfc-editor.org/info/rfc5321/)，J. Klensin

[3] [RFC 5737: IPv4 Address Blocks Reserved for Documentation](https://www.rfc-editor.org/info/rfc5737/), J. Arkko, M. Cotton, L. Vegoda