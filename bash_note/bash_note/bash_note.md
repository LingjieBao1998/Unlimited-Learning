## error
在Linux系统中，可以查看 /var/log/syslog 或 /var/log/messages，获取更多错误信息。

## 清除cache
```bash
sudo sync && sudo sysctl -w vm.drop_caches=1
```

