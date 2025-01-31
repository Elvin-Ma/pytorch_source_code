# 1 dispatcher
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 PyTorch 中，__torch_dispatch__ 是一个非常强大的钩子函数，它允许用户自定义张量操作的行为。__torch_dispatch__ 主要有两种常见的用法：算子重载和自动混合精度（AMP）模拟等自定义计算流程，下面分别进行详细介绍。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;分发器的一个配置是所有这些特性和后端键的调用顺序。最新的列表及其顺序可以在DispatchKey.h文件中的DispatchKey枚举里找到。为了扩展PyTorch，本讨论中重要的顺序子集是：<br>

**vmap -> Autocast -> Autograd -> ZeroTensor -> Neg/Conj -> Functionalize -> Python -> Backends**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于本次讨论而言，最重要的key是Python，因为**每个定义了__torch_dispatch__方法的Tensor子类都会调用这个特性。** 正是在这里调用了用户定义的方法，并且可以在这里任意重写行为。从这里开始，再次调用提供的函数func将执行一次"redispatch"。<br>

# 2 算子重载
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;算子重载是指通过 __torch_dispatch__ 改变 PyTorch 中现有算子的默认行为。例如，你可以在执行加法操作时添加额外的逻辑。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这种实现的一些重要含义是：<br>

1. 这段代码运行在“所有特性之下”。因此，它仅负责（像一个常规后端一样）生成每个Tensor的输出值（并且可以也应该忽略所有高级特性，如自动微分、自动类型转换等）。<br>

2. 如果任何高级特性实现了某个函数而没有进行重新分发，那么它将永远不会到达Python键，因此__torch_dispatch__回调也永远不会被触发。这种情况特别发生在CompositeImplicitAutograd函数中，这些函数在自动微分级别进行评估，而无需重新分发。这是因为CompositeImplicitAutograd函数通过隐式调用其他原生操作来指定其自动微分公式，所以在自动微分级别，该函数被分解为其原生操作，并且这些操作被评估执行。<br>

3. 当回调到Python并对结果进行包装时，使用的转换与常规的PyTorch Python/C++绑定相同。特别地，有些对象在Python中无法表示，需要特殊处理（例如，未定义的Tensor会变成None）。<br>

4. 我们的原生函数被惰性地填充为torch.ops.{namespace}.{func_name}.{overload_name}，作为可调用的Python对象，以便能够从Python轻松地与它们进行交互。传递给__torch_dispatch__的func对象总是来自这个命名空间的一个条目。这个命名空间可以用来**直接调用原生操作**，并绕过常规的Python API和绑定代码。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;类似地，正如__torch_function__能够插入到PyTorch的所有Python API和Tensor方法之间一样，__torch_dispatch__能够拦截对aten原生API的所有调用。需要注意的是，在进入分发器之前，Tensor上的所有方法都会被转换成函数调用，因此在这里它们会作为函数调用出现：torch.add(a, 2)和a + 2会导致完全相同的aten调用。这些函数中的大多数在native_functions.yaml中定义，该文件指定了这些函数的属性以及它们的后端实现。然后，这些函数的实现以及指定的特性会通过代码生成自动注册。一些更特殊的函数或特性也会在C++代码库的其他地方或用户定义的C++扩展中注册。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;也可以使用torch.library来添加新的原生函数。这个Python功能允许为原生函数定义和/或添加新的实现。这可以用于添加缺失的内核、替换现有的内核或定义全新的原生函数。<br>


# 3 Extending torch native API

```python
import torch

class MyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        # 创建一个新的张量实例
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # 处理加法操作
        if func.__name__ == 'add':
            print("Custom add operation detected!")
            # 调用原始的加法操作
            result = func(*args, **kwargs)
            # 在结果上添加额外的逻辑，例如乘以 2
            result = result * 2
            return result
        # 对于其他操作，直接调用原始函数
        return func(*args, **kwargs)

# 创建自定义张量
x = MyTensor([1, 2, 3])
y = MyTensor([4, 5, 6])

# 执行加法操作
z = x + y
print(z)
```

# 4 Extending all torch API with Modes
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;不幸的是，有些函数并不接受Tensor作为输入。这意味着上述的子类化方法无法用于覆盖PyTorch中所有函数的行为。此外，如果用例要求拦截每一个函数调用，那么将每个Tensor都更改为子类可能会过于侵入性。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了解决这个用例，我们引入了“模式”的概念。这些模式用于__torch_function__和__torch_dispatch__的重写，分别通过子类化torch.overrides.TorchFunctionMode和torch.utils._python_dispatch.TorchDispatchMode来创建，并作为上下文管理器使用。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了简化描述这种模式如何与子类和其他模式交互，每当进入模式的上下文管理器时，每个函数的行为就好像在参数列表的开头多了一个Tensor参数，且该参数的模式作为子类存在。这特别意味着，所有模式处理程序都会在任何子类处理程序之前被调用，并且与内部上下文管理器对应的模式将始终首先运行。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同样重要的是要注意，在给定的模式处理程序中，这个特定的模式是禁用的，并且可以通过使用with self:手动重新启用。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是一个示例，展示了每种类型的日志记录模式：<br>

```python
import torch
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

class FunctionLog(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        print(f"Function Log: {resolve_name(func)}(*{args}, **{kwargs})")
        return func(*args, **(kwargs or {}))

class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
        return func(*args, **(kwargs or {}))

def f():
    a = torch.rand(10, requires_grad=True)
    b = a * 2
    b.sum().backward()

# print("TorchFunctionMode logging:")
# with FunctionLog():
#     f()

print("TorchDispatchMode logging:")
with DispatchLog():
    f()
```