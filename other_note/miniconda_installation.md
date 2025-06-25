## installation of miniconda 
### installlation
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
### add to envirnment
* (1) 临时生效（仅当前终端会话）​​
```bash
export PATH=~/miniconda3/bin:$PATH
```
这样当前终端就可以使用 conda 命令，但关闭终端后失效。
​
* (2) 永久生效（推荐）​​
# 将 export 命令添加到 ~/.bashrc（Bash Shell）或 ~/.zshrc（Zsh Shell）文件中：
```bash
echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
# 然后​​重新加载配置​​：
source ~/.bashrc
```