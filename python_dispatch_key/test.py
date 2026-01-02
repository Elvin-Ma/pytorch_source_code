import torch

# 定义一个实现 __torch_dispatch__ 的 Tensor 子类
class MyTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(f"触发 __torch_dispatch__, 处理算子: {func.__name__}")
        return func(*args, **kwargs)

# 创建实例时，PyTorch 自动为其添加 DispatchKey::Python
a = MyTensor(torch.tensor([1,2,3]))
print(a.dispatch_keys())  # 输出中会包含 DispatchKey.Python