## 利用pytorch的钩子（`hook`）机制来获取某一层的输出
以yolo11x模型为例
```python
## 钩子函数
def make_hook_fn(model):
    outputs = []
    def hook_fn(module, input, output):
        outputs.append(output)

    hooks = []  # 用于存储注册的钩子

    if len(hooks)==0:
        c3k2_modules = [module for module in model.model.modules() if module.__class__.__name__ == 'C3k2']

        hooks = []  # 用于存储注册的钩子
        for c3k2 in c3k2_modules[-3:]:
            hook = c3k2.register_forward_hook(hook_fn)
            hooks.append(hook)

    return outputs, hooks

## 注册钩子
yolo_model = ......
outputs, hooks = make_hook_fn(yolo_model)
print("outputs",outputs)
## 推理或者运行model.forward
with torch.no_grad():
    yolo_model(images)

# 移除注册的钩子,不然会出现额外的结果存储在outputs中的
for hook in hooks:
    hook.remove()  
## 自动将结果存储在outputs里面了
print("outputs",outputs)
```
> * 注意，对于一些共享layer的模型，比如`BART`的encoder和decoder的embedding层，所以如果用钩子函数获取embedding层的输出结果，会有多个输出，此时需要加以甄别。
> * outputs中的结果是按照model前向推理的顺序进行记录的