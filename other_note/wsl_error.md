## wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。
### 解决方案
在Windows中的`C:\Users\<your_username>`目录下创建一个.wslconfig文件（如果已经存在则进行修改），然后在文件中写入如下内容
```markdown
[experimental]
autoMemoryReclaim=gradual  
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```
然后用wsl --shutdown关闭WSL，之后再重启，提示就消失了。

> ref:https://www.cnblogs.com/hg479/p/17869109.html
