<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [介绍](#%E4%BB%8B%E7%BB%8D)
- [登入](#%E7%99%BB%E5%85%A5)
- [免密登入](#%E5%85%8D%E5%AF%86%E7%99%BB%E5%85%A5)
- [ssh 本地端口反向代理到公网服务器](#ssh-%E6%9C%AC%E5%9C%B0%E7%AB%AF%E5%8F%A3%E5%8F%8D%E5%90%91%E4%BB%A3%E7%90%86%E5%88%B0%E5%85%AC%E7%BD%91%E6%9C%8D%E5%8A%A1%E5%99%A8)
  - [场景&需求](#%E5%9C%BA%E6%99%AF%E9%9C%80%E6%B1%82)
  - [命令以及参数](#%E5%91%BD%E4%BB%A4%E4%BB%A5%E5%8F%8A%E5%8F%82%E6%95%B0)
  - [case：本地的flask服务在服务器端访问](#case%E6%9C%AC%E5%9C%B0%E7%9A%84flask%E6%9C%8D%E5%8A%A1%E5%9C%A8%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%AB%AF%E8%AE%BF%E9%97%AE)
  - [case 反向代理本地的ssh端口到服务器](#case-%E5%8F%8D%E5%90%91%E4%BB%A3%E7%90%86%E6%9C%AC%E5%9C%B0%E7%9A%84ssh%E7%AB%AF%E5%8F%A3%E5%88%B0%E6%9C%8D%E5%8A%A1%E5%99%A8)
  - [反向代理失败的场景](#%E5%8F%8D%E5%90%91%E4%BB%A3%E7%90%86%E5%A4%B1%E8%B4%A5%E7%9A%84%E5%9C%BA%E6%99%AF)
  - [SSH反向代理掉线解决方案](#ssh%E5%8F%8D%E5%90%91%E4%BB%A3%E7%90%86%E6%8E%89%E7%BA%BF%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88)
- [scp](#scp)
  - [从本地复制文件到远程主机](#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%A4%8D%E5%88%B6%E6%96%87%E4%BB%B6%E5%88%B0%E8%BF%9C%E7%A8%8B%E4%B8%BB%E6%9C%BA)
  - [从本地复制文件夹到远程主机](#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%A4%8D%E5%88%B6%E6%96%87%E4%BB%B6%E5%A4%B9%E5%88%B0%E8%BF%9C%E7%A8%8B%E4%B8%BB%E6%9C%BA)
  - [从远程主机复制文件到本地](#%E4%BB%8E%E8%BF%9C%E7%A8%8B%E4%B8%BB%E6%9C%BA%E5%A4%8D%E5%88%B6%E6%96%87%E4%BB%B6%E5%88%B0%E6%9C%AC%E5%9C%B0)
  - [从远程主机复制文件夹到本地](#%E4%BB%8E%E8%BF%9C%E7%A8%8B%E4%B8%BB%E6%9C%BA%E5%A4%8D%E5%88%B6%E6%96%87%E4%BB%B6%E5%A4%B9%E5%88%B0%E6%9C%AC%E5%9C%B0)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 介绍
SSH是一种网络协议，用于计算机之间的加密登录。
1. SSH 是一种加密的网络传输协议。
2. SSH 最常见的用途是远程登录系统。
OpenSSH 工具分为两种，一种是服务器端，另一种则是客户端。

## 登入
```bash
ssh  -p ${Port} ${User}@${服务器的IP}
```

## 免密登入
所谓"公钥登录"（免密登入），原理很简单，就是用户将自己的公钥储存在远程主机上。登录的时候，远程主机会向用户发送一段随机字符串，用户用自己的私钥加密后，再发回来。远程主机用事先储存的公钥进行解密

1. ssh-keygen，本地生成ssh的key
```bash
ssh-keygen
```
> 一直回车即可
> 
在~/.ssh会生成`id_rsa` （私钥）和`id_rsa.pub` (公钥)`文件
2. 上传公钥文件到服务器(~/.ssh/authorized_keys 文件中)
```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub ${User}@${服务器的IP}
```
3. 免密登入测试
```bash
ssh -p ${Port} ${User}@${服务器的IP}
```

如果无法进行免密登入，需要在服务器端检查ssh配置；检查`/etc/ssh/sshd_config`文件并确保下列配置
```file
RSAAuthentication yes # 允许使用 RSA 密钥进行身份验证。这是 SSH 支持的一种公钥身份验证方法。
PubkeyAuthentication yes # 启用公钥身份验证。这意味着用户可以使用公钥进行身份验证，而不是仅依赖密码。
AuthorizedKeysFile .ssh/authorized_keys #指定存放授权公钥的文件路径。只有在此文件中列出的公钥才能用于身份验证。
```
然后重启ssh服务
```bash
systemctl restart ssh
```

## ssh 本地端口反向代理到公网服务器
### 场景&需求
* 有一台`公网服务器B`，能够对外开启服务并让其他人访问
* 有一台`本地电脑A`，无公网IP，但是部署了多个服务

需求：现在需要通过`公网服务器B`作为**跳板**（中介）访问`本地电脑A`部署的服务，这就是`反向代理`

### 命令以及参数
```bash
ssh -NfR ${公网服务器B的端口，不得与现有的端口想冲突}:${本地电脑A的IP地址}:${本地电脑A的端口} ${公网服务器B的登入用户名}@${公网服务器B的地址}
```
* -N: 不执行远程命令，仅用于端口转发。这通常用于只进行转发而不需要交互式登录。
* -f: 在后台运行 SSH，而不是在前台。这使得 SSH 会在建立连接后立即返回控制权给用户。
* -R `${公网服务器B的端口，不得与现有的登入端口想冲突}:${本地电脑A的IP地址}:${本地电脑A的端口}`: 设置远程端口转发

example
```
ssh -NfR 5007:localhost:3003 user@119.1.2.3
```
> localhost:自动使用本地的IP地址

### case：本地的flask服务在服务器端访问
1. 在本地启动 Flask 应用
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask on port 3003!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3003)
```
将上述代码保存为 app.py，然后在终端中运行：
```
python app.py
```

`curl`命令访问 
```bash
curl http://localhost:3003
```
返回结果
```markdown
Hello, Flask on port 3003!
```

2. 在本地运行 SSH 命令
在另一个终端窗口中，运行以下 SSH 命令来设置远程端口转发：

```bash
ssh -NfR 5007:localhost:3003 user@119.1.2.3
```
3. 在服务器端访问转发的端口
`curl`命令访问 
```bash
curl http://localhost:5007
```
返回结果
```markdown
Hello, Flask on port 3003!
```

> ref:https://blog.csdn.net/qq_36154886/article/details/131970485


### case 反向代理本地的ssh端口到服务器
除了能够反向代理http的`服务`，还能将本地的ssh端口到服务器上

1. 验证本地的user是否能够登入
```bash
ssh ${USER}@localhost
```
> 与前面的http的服务反向代理相比，需要验证下用户是否能登入

如果不能登入，检查`/etc/ssh/sshd_config`文件并确保下列配置
```file
PermitRootLogin yes  # 允许 root 用户通过 SSH 登录，通常建议在必要时启用此选项以增加安全性(如果不是root登入可忽略)
PasswordAuthentication yes  # 允许使用密码进行身份验证，建议在不使用公钥认证时谨慎启用
```
> `PermitRootLogin yes` 是SSH服务器配置中的一个选项，用于控制是否允许root用户通过SSH登录系统。可选值及含义
* `yes`：允许root用户通过SSH登录系统，可以使用密码或公钥身份验证。
* `no`：禁止root用户通过SSH登录系统。这是一种安全措施，可以防止未经授权的用户直接使用root权限登录到服务器。
* `without-password`：只允许root用户使用公钥身份验证登录，禁止使用密码登录。
* `prohibit-password`：只允许root用户使用公钥身份验证登录，禁止使用密码登录。与without-password相同，但更推荐使用这个选项，因为它提供了更明确的语义。
> 安全建议:在实际应用中，为了增加系统的安全性，通常推荐将`PermitRootLogin`的值设置为`prohibit-password`或`without-password`，并使用公钥身份验证登录root用户。这样可以避免使用密码登录root用户所带来的安全风险。如果确实需要允许root用户使用密码登录系统，可以将该值设置为yes，但要格外注意密码的安全性。

> ref:https://blog.csdn.net/xie__jin__cheng/article/details/142655337


> root 修改密码 
> 执行如下操作修改root用户密码。
```bash
passwd root
```



随后重启ssh服务
```bash
systemctl restart ssh
```
或者
```bash
sudo /etc/init.d/ssh restart
```

2. 反向代理建立
```bash
ssh -N -f -R 2230:localhost:22 ${服务器B的USER}@${服务器B的IP地址}
```
> 2230可以换成服务器B上任何不冲突的端口

3. （另一台电脑）访问localhist
```bash
ssh ${服务器B的USER}@${服务器B的IP地址}
```
随后，在服务B上通过以下命令访问localhost

```bash
ssh -p 2230 ${USER}@localhost
```

这样就可以直接先连接服务器B，然后在连接localhost了，但是这样要访问两次，有没有什么命令一次就能搞定

配置服务器端口转发到本地
```bash
ssh -NfL 2230:localhost:2230 ${服务器B的USER}@${服务器B的IP地址}
```
* -N: 不执行远程命令，仅用于端口转发。此选项用于确保 SSH 会话不会执行任何命令，只是保持连接。
* -f: 在后台运行 SSH。这意味着 SSH 在建立连接后会立即返回控制权给用户，而不会在前台保持连接。
* -L 2230:localhost:2230: 设置本地端口转发：2230 是本地机器上的端口。
localhost:2230 表示将本地的 2230 端口转发到远程服务器（服务器 B）上的 localhost 的 2230 端口。
${服务器B的USER}@${服务器B的IP地址}: 指定要连接的远程主机，${服务器B的USER} 是用户名，${服务器B的IP地址} 是远程主机的 IP 地址。
总结
这个命令的作用是将本地机器的 2230 端口转发到远程服务器 B 的 2230 端口，并在后台运行该 SSH 会话。这样，您可以**通过访问本地的 2230 端口，来间接访问远程服务器上的同一端口**。

随后就可直接访问了
```
ssh -p 2230 ${USER}@localhost
```

4. vscode配置`ssh/config`文件
```
Host jump-server
    HostName 服务器B的IP地址
    User 服务器B的USER

# 本地机器配置，使用跳板机
Host local-machine
    HostName localhost
    Port 2230
    User 本地用户
    ProxyJump jump-server
```

如果上述配置无法访问需要添加`ProxyCommand`进行访问
```
# 跳板机配置
Host jump-server
    HostName 服务器B的IP地址
    User 服务器B的USER

# 本地机器配置，使用跳板机
Host local-machine
    HostName localhost
    Port 2230
    User 本地用户
    ProxyCommand "C:\\Windows\\System32\\OpenSSH\\ssh.exe" -W %h:%p jump-server
```
> "C:\\Windows\\System32\\OpenSSH\\ssh.exe"需要换成能够运行ssh的文件或者程序
> ref:https://blog.csdn.net/weixin_42096901/article/details/105193366

如果服务器的公网没有开放2230端口进行访问，或者不想暴露2230端口，则使用一下命令
```bash
ssh -J ${jump-server-user}@${jump-server} -p ${jump-server-port} ${USER}@localhost -p ${2230或者其他端口}
```
vscode config文件
```file
Host local_4090(叫啥都可以)
  HostName localhost
  Port ${2230或者其他端口}
  User lingjiebao
  ProxyCommand "C:\\Windows\\System32\\OpenSSH\\ssh.exe" -W %h:%p ${jump-server-user}@${jump-server} -p ${jump-server-port}
```
> "C:\\Windows\\System32\\OpenSSH\\ssh.exe"需要换成能够运行ssh的文件或者程序

### 反向代理失败的场景
检查跳板机的 SSH 配置文件`/etc/ssh/sshd_config`：
```bash
cat /etc/ssh/sshd_config | grep -E "GatewayPorts|AllowTcpForwarding"
```
检查`AllowTcpForwarding`
>AllowTcpForwarding 是 OpenSSH 服务器（sshd）配置文件（/etc/ssh/sshd_config）中的一个重要参数，用于控制是否允许通过 SSH 建立 ​​TCP 端口转发​​（包括本地转发、远程转发和动态转发）

设置
```bash
AllowTcpForwarding yes
```
| 值       | 含义                                                                 |
|----------|----------------------------------------------------------------------|
| `yes`    | **允许所有 TCP 转发**（默认值）。客户端可以建立本地、远程或动态转发隧道。 |
| `no`     | **禁止所有 TCP 转发**。即使客户端尝试 `-L`、`-R` 或 `-D` 参数，SSH 服务器也会拒绝。 |
| `local`  | **仅允许本地转发（`-L`）**，禁止远程转发（`-R`）和动态转发（`-D`）。 |
| `remote` | **仅允许远程转发（`-R`）**，禁止本地转发（`-L`）和动态转发（`-D`）。 |


检查`GatewayPorts`
> GatewayPorts 是 OpenSSH 服务器（sshd）的一个配置选项，用于控制 ​​反向隧道（Remote Forwarding）​​ 的端口是否可以被外部网络访问。它决定了哪些主机可以连接到通过 -R 参数转发的端口。

```bash
GatewayPorts yes
```

| 值 | 含义 |
|----|------|
| `no`（默认） | **仅允许本地访问**（`127.0.0.1`），外部网络无法连接。 |
| `yes` | **允许所有主机访问**（`0.0.0.0`），外部网络可以直接连接。 |
| `clientspecified` | **由客户端指定**（`-R` 参数可以指定 `bind_address`，如 `0.0.0.0` 或 `127.0.0.1`）。 |


### SSH反向代理掉线解决方案
```bash
#!/bin/bash
# while true;do
#     RET=`ps ax | grep "ssh -NfR 5007:localhost:3003" | grep -v "grep"`
#     if [ "$RET" = "" ]; then
#         echo "restart ssh server"
#         ssh -NfR 5007:localhost:3003 user@119.1.2.3
#     fi
#     sleep 10
# done
```
> 有问题

autossh
```
autossh -M 7281 -NfR 5007:localhost:3003
```


## scp
在 Linux 系统中，scp 命令是一个非常实用的工具 scp 是 "secure copy" 的缩写，它基于 SSH（Secure Shell）协议，确保数据传输的安全性。

### 从本地复制文件到远程主机
假设你有一个文件 example.txt，你想将它复制到远程主机的 /home/user/ 目录下，可以使用以下命令：
```bash
scp -P ${PORT} example.txt user@remote_host:/home/user/
```
### 从本地复制文件夹到远程主机
假设你有一个文件夹 example，你想将它复制到远程主机的 /home/user/ 目录下，可以使用以下命令：
```bash
scp -P ${PORT} -r example user@remote_host:/home/user/
```

### 从远程主机复制文件到本地
如果你想将远程主机上的文件 example.txt 复制到本地的当前目录，可以使用以下命令：
```bash
scp user@remote_host:/home/user/example.txt .
```

### 从远程主机复制文件夹到本地
如果你想将远程主机上的文件夹 example 复制到本地的当前目录，可以使用以下命令：
```bash
scp -r user@remote_host:/home/user/example .
```
