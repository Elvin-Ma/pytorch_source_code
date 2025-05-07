# 1 计算原理
当然可以！以下是整理后的 Markdown 格式内容：

---

## 1.1  PyTorch 中 ReLU 的 Kernel 调用 `abs` 的原因

1. **数学表达式优化**
ReLU 的数学定义为：
$$
\text{ReLU}(x) = \max(x, 0)
$$
该公式可以等价表示为：
$$
\text{ReLU}(x) = \frac{x + |x|}{2}
$$
通过引入绝对值运算 `abs`，避免了显式的条件分支（如 `if-else`），从而更适合并行计算。

---

2. **避免条件分支**
GPU 等并行设备对条件分支敏感：
- 分支预测失败会导致线程束（warp）内的线程分化，降低吞吐量。
- 使用 `abs` 和算术运算替代 `if (x > 0)`，保证所有线程执行相同的指令流，提升性能。

---

3. **代码复用（与 LeakyReLU 统一实现）**
PyTorch 可能复用 LeakyReLU 的 Kernel 逻辑。LeakyReLU 的通用形式为：
$$
\text{LeakyReLU}(x) = \begin{cases}
x & x \ge 0 \\
\alpha x & x < 0
\end{cases}
$$
当 $\alpha = 0$ 时，LeakyReLU 退化为标准 ReLU。其统一实现可写为：
$$
\text{LeakyReLU}(x) = \frac{x + |x|}{2} + \alpha \cdot \frac{x - |x|}{2}
$$
此时仅需计算前一项（$\alpha=0$），直接复用 `abs` 实现。

---

4. **硬件友好性**
- **绝对值指令优化**：硬件（如 CUDA）对 `abs` 操作有专用指令（如 `__fabsf`），计算效率极高。
- **内存连续性**：无分支的向量化操作更适配 GPU 的 SIMD 架构，减少线程等待。

---

## 1.2 总结
| 优化目标               | 实现手段                          |
|------------------------|-----------------------------------|
| 消除分支               | 用 `abs` 替代 `if-else`           |
| 统一代码逻辑           | 复用 LeakyReLU 的 Kernel          |
| 最大化硬件并行效率     | 依赖无分支算术运算 + 专用指令      |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过上述优化，PyTorch 的 ReLU 实现在保持数学等价性的同时，显著提升了 GPU 上的计算性能。

# 2 relu 函数调用栈

## 2.1 python 栈

- gdb 中查看

```shell
# apt-get install python3-dbg
source /usr/share/gdb/auto-load/usr/bin/python3.10-gdb.py
```

## 2.2 调度到at::native::relu
```c++
#0  at::native::relu (self=...) at /root/projects/pytorch/aten/src/ATen/native/Activation.cpp:512
#1  0x00007fffce7ac838 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu (self=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:13464
#2  0x00007fffce9f5620 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >::operator() (args#0=..., this=0x4110840)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#3  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x4110840, args#0=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#4  0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffce9f55b1 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>, functor=0x4110840, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#5  0x00007fffde32df35 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1139558)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#6  c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, c10::DispatchKeySet, at::Tensor con
--Type <RET> for more, q to quit, c to continue without paging--
st&) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#7  0x00007fffdf00fb62 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&) const (args#0=...,
    currentDispatchKeySet=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#8  at::_ops::relu::redispatch (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:3763
#9  0x00007fffe27ed082 in at::redispatch::relu (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:6622
#10 0x00007fffe27191ae in <lambda()>::operator()(void) const (__closure=0x7fffffffd4f0)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_4.cpp:15278
#11 0x00007fffe271945c in torch::autograd::VariableType::(anonymous namespace)::relu (ks=..., self=...)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_4.cpp:15279
#12 0x00007fffe27b9077 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >::operator() (args#1=...,
    args#0=..., this=0x3285cd0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#13 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x3285cd0, dispatchKeySet=...,
    args#0=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#14 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffe27b8fec <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>,
    functor=0x3285cd0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#15 0x00007fffdf00f922 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x113a0d8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#16 c10::Dispatcher::call<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#17 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::call(at::Tensor const&) const (args#0=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#18 at::_ops::relu::call (self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:3756
#19 0x00007ffff4eeec5a in at::Tensor::relu (this=0x7fffffffd828) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:3294
#20 0x00007ffff50a3be2 in <lambda(const at::Tensor&)>::operator()(const at::Tensor &) const (__closure=0x7fffffffd827, self=...)
--Type <RET> for more, q to quit, c to continue without paging--
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:6551
#21 0x00007ffff50a3e6b in torch::autograd::THPVariable_relu (self_=0x0, Python Exception <class 'AssertionError'>:
args=, kwargs=0x0)
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:6553
#22 0x000000000054c584 in cfunction_call (func=<built-in method relu of type object at remote 0x7ffff6f84540>, Python Exception <class 'AssertionError'>:
args=, kwargs=0x0)
    at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
```

## 2.3 一直到kernel执行
```c++
#0  at::native::(anonymous namespace)::clamp_min_scalar_kernel_impl (iter=..., min=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:86
#1  0x00007fffddc76ede in at::native::DispatchStub<void (*)(at::TensorIteratorBase&, c10::Scalar), at::native::clamp_min_scalar_stub_DECLARE_DISPATCH_type>::operator()<at::native::structured_clamp_min_out&, c10::Scalar const&> (this=0x7ffff39d2280 <at::native::clamp_min_scalar_stub>,
    device_type=c10::DeviceType::CUDA) at /root/projects/pytorch/aten/src/ATen/native/DispatchStub.h:250
#2  0x00007fffddc72f9a in at::native::structured_clamp_min_out::impl (this=0x7fffffffcaf0, self=...,
    min=..., result=...) at /root/projects/pytorch/aten/src/ATen/native/TensorCompare.cpp:746
#3  0x00007fffce7748c4 in at::(anonymous namespace)::wrapper_CUDA_clamp_min (self=..., min=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:5591
#4  0x00007fffce9d78f0 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_clamp_min>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const c10::Scalar&> >::operator() (args#1=...,
    args#0=..., this=0x40c6640)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#5  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFu
--Type <RET> for more, q to quit, c to continue without paging--
nctionPointer<at::Tensor(const at::Tensor&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_clamp_min>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const c10::Scalar&> >, at::Tensor(const at::Tensor&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const c10::Scalar &) (functor=0x40c6640, args#0=..., args#1=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#6  0x00007fffde4adcd4 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::Scalar const&> (
    unboxed_kernel_func=0x7fffce9d7856 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_clamp_min>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const c10::Scalar&> >, at::Tensor(const at::Tensor&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const c10::Scalar &)>,
    functor=0x40c6640, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#7  0x00007fffded16d14 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::Scalar const&> (dispatchKeySet=..., opHandle=..., this=0x110eea8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#8  c10::Dispatcher::call<at::Tensor, at::Tensor const&, c10::Scalar const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::Scalar const&)> const&, at::Tensor const&, c10::Scalar const&) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#9  c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::Scalar const&)>::call(at::Tensor const&, c10::Scalar const&) const (args#1=..., args#0=...,
    this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#10 at::_ops::clamp_min::call (self=..., min=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:1677
#11 0x00007fffdd21ad7d in at::clamp_min (self=..., min=...) at /root/projects/pytorch/build/aten/src/ATen/ops/clamp_min.h:27
#12 0x00007fffdd72202a in at::native::relu (self=...) at /root/projects/pytorch/aten/src/ATen/native/Activation.cpp:513
#13 0x00007fffce7ac838 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu (self=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:13464
#14 0x00007fffce9f5620 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >::operator() (args#0=..., this=0x4110840)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#15 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x4110840, args#0=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#16 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
--Type <RET> for more, q to quit, c to continue without paging--
    unboxed_kernel_func=0x7fffce9f55b1 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__relu>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>, functor=0x4110840, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#17 0x00007fffde32df35 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1139558)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#18 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, c10::DispatchKeySet, at::Tensor const&) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#19 0x00007fffdf00fb62 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&) const (args#0=...,
    currentDispatchKeySet=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#20 at::_ops::relu::redispatch (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:3763
#21 0x00007fffe27ed082 in at::redispatch::relu (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:6622
#22 0x00007fffe27191ae in <lambda()>::operator()(void) const (__closure=0x7fffffffd4f0)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_4.cpp:15278
#23 0x00007fffe271945c in torch::autograd::VariableType::(anonymous namespace)::relu (ks=..., self=...)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_4.cpp:15279
#24 0x00007fffe27b9077 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >::operator() (args#1=...,
    args#0=..., this=0x3285cd0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#25 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x3285cd0, dispatchKeySet=...,
    args#0=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#26 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffe27b8fec <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::relu>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>,
--Type <RET> for more, q to quit, c to continue without paging--
    functor=0x3285cd0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#27 0x00007fffdf00f922 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x113a0d8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#28 c10::Dispatcher::call<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#29 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::call(at::Tensor const&) const (args#0=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#30 at::_ops::relu::call (self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:3756
#31 0x00007ffff4eeec5a in at::Tensor::relu (this=0x7fffffffd828) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:3294
#32 0x00007ffff50a3be2 in <lambda(const at::Tensor&)>::operator()(const at::Tensor &) const (__closure=0x7fffffffd827, self=...)
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:6551
#33 0x00007ffff50a3e6b in torch::autograd::THPVariable_relu (self_=0x0, Python Exception <class 'AssertionError'>:
args=, kwargs=0x0)
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:6553
#34 0x000000000054c584 in cfunction_call (func=<built-in method relu of type object at remote 0x7ffff6f84540>, Python Exception <class 'AssertionError'>:
args=, kwargs=0x0)
```

## 2.4 kernel 执行也要经过多个调用栈

```c++
#0  at::native::launch_vectorized_kernel<at::native::AbsFunctor<float>, std::array<char*, 2> > (N=120, f=..., data=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh:152
#1  0x00007fffcb39c115 in at::native::gpu_kernel_impl_nocast<at::native::AbsFunctor<float> > (iter=..., f=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh:309
#2  0x00007fffcb39bb2c in at::native::gpu_kernel_impl<at::native::AbsFunctor<float> > (iter=..., f=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh:323
#3  0x00007fffcb397e78 in at::native::gpu_kernel<at::native::AbsFunctor<float> > (iter=..., f=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/Loops.cuh:120
--Type <RET> for more, q to quit, c to continue without paging--
#4  0x00007fffcb388631 in <lambda()>::operator()(void) const (__closure=0x7fffffffbb90) at /root/projects/pytorch/aten/src/ATen/native/cuda/AbsKernel.cu:39
#5  0x00007fffcb388865 in <lambda()>::operator()(void) const (__closure=0x7fffffffbc10) at /root/projects/pytorch/aten/src/ATen/native/cuda/AbsKernel.cu:39
#6  0x00007fffcb388a68 in at::native::abs_kernel_cuda (iter=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/AbsKernel.cu:39
#7  0x00007fffddd430fd in at::native::DispatchStub<void (*)(at::TensorIteratorBase&), at::native::abs_stub_DECLARE_DISPATCH_type>::operator()<at::TensorIterator&>
    (this=0x7ffff39d2560 <at::native::abs_stub>, device_type=c10::DeviceType::CUDA) at /root/projects/pytorch/aten/src/ATen/native/DispatchStub.h:250
#8  0x00007fffddd3e908 in at::native::unary_op_impl_out<at::native::abs_stub_DECLARE_DISPATCH_type> (result=..., self=..., stub=...)
    at /root/projects/pytorch/aten/src/ATen/native/UnaryOps.cpp:407
#9  0x00007fffddd3d556 in at::native::unary_op_impl_with_complex_to_float_out<at::native::abs_stub_DECLARE_DISPATCH_type> (result=..., self=..., stub=...,
--Type <RET> for more, q to quit, c to continue without paging--
    _integer_to_float=false) at /root/projects/pytorch/aten/src/ATen/native/UnaryOps.cpp:455
#10 0x00007fffddd398aa in at::native::abs_out (self=..., result=...) at /root/projects/pytorch/aten/src/ATen/native/UnaryOps.cpp:544
#11 0x00007fffce757ba8 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_out_abs_out (self=..., out=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:1454
#12 0x00007fffce9cead8 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor&(const at::Tensor&, at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_out_abs_out>, at::Tensor&, c10::guts::typelist::typelist<const at::Tensor&, at::Tensor&> >::operator() (
    args#1=..., args#0=..., this=0x40d1840) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#13 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor&(const at::Tensor&, at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_out_abs_out>, at::Tensor&, c10::guts::typelist::typelist<const at::Tensor&, at::Tensor&> >, at::Tensor&(const at::Tensor&, at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, at::Tensor &) (functor=0x40d1840, args#0=...,
    args#1=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#14 0x00007fffde49e5bb in c10::callUnboxedKernelFunction<at::Tensor&, at::Tensor const&, at::Tensor&> (
    unboxed_kernel_func=0x7fffce9cea5b <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor&(const at::Tensor&, at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA_out_abs_out>, at::Tensor&, c10::guts::typelist::typelist<const at::Tensor&, at::Tensor&> >, at::Tensor&(const at::Tensor&, at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, at::Tensor &)>,
    functor=0x40d1840, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#15 0x00007fffde5ab2b2 in c10::KernelFunction::call<at::Tensor&, at::Tensor const&, at::Tensor&> (dispatchKeySet=..., opHandle=..., this=0x180c568)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:145
#16 c10::Dispatcher::call<at::Tensor&, at::Tensor const&, at::Tensor&>(c10::TypedOperatorHandle<at::Tensor& (at::Tensor const&, at::Tensor&)> const&, at::Tensor const&, at::Tensor&) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#17 c10::TypedOperatorHandle<at::Tensor& (at::Tensor const&, at::Tensor&)>::call(at::Tensor const&, at::Tensor&) const (args#1=..., args#0=...,
    this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#18 at::_ops::abs_out::call (self=..., out=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_1.cpp:1009
#19 0x00007fffddd3fac4 in at::abs_out (out=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/ops/abs.h:37
#20 0x00007fffddd3d75a in at::native::unary_op_impl_with_complex_to_float<at::Tensor&(at::Tensor&, const at::Tensor&)>(const at::Tensor &, at::Tensor &(&)(at::Tensor &, const at::Tensor &)) (self=...,
    out_impl=@0x7fffddd3fa9d: {at::Tensor &(at::Tensor &, const at::Tensor &)} 0x7fffddd3fa9d <at::abs_out(at::Tensor&, at::Tensor const&)>)
    at /root/projects/pytorch/aten/src/ATen/native/UnaryOps.cpp:479
--Type <RET> for more, q to quit, c to continue without paging--
#21 0x00007fffddd398e9 in at::native::abs (self=...) at /root/projects/pytorch/aten/src/ATen/native/UnaryOps.cpp:547
#22 0x00007fffdf5711c6 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__abs (self=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp:1337
#23 0x00007fffdf6b51b8 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__abs>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >::operator() (args#0=...,
    this=0x1f45560) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#24 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__abs>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x1f45560, args#0=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#25 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffdf6b5149 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__abs>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>, functor=0x1f45560, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#26 0x00007fffde32df35 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1134698)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#27 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, c10::DispatchKeySet, at::Tensor const&) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#28 0x00007fffde5aa5c0 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&) const (args#0=...,
    currentDispatchKeySet=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#29 at::_ops::abs::redispatch (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_1.cpp:974
#30 0x00007fffe2107c5e in at::redispatch::abs (dispatchKeySet=..., self=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:407
#31 0x00007fffe1fcadb8 in <lambda()>::operator()(void) const (__closure=0x7fffffffcaa0)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_1.cpp:3964
#32 0x00007fffe1fcb0b4 in torch::autograd::VariableType::(anonymous namespace)::abs (ks=..., self=...)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_1.cpp:3965
#33 0x00007fffe20c8c06 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::au--Type <RET> for more, q to quit, c to continue without paging--
tograd::VariableType::(anonymous namespace)::abs>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >::operator() (args#1=...,
    args#0=..., this=0x2cb5be0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#34 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::abs>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x2cb5be0, dispatchKeySet=...,
    args#0=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#35 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffe20c8b7b <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::abs>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>,
    functor=0x2cb5be0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#36 0x00007fffde5aa380 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1135218)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#37 c10::Dispatcher::call<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#38 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::call(at::Tensor const&) const (args#0=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#39 at::_ops::abs::call (self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_1.cpp:967
#40 0x00007ffff4ee8ce8 in at::Tensor::abs (this=0x7fffffffd178) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:1564
#41 0x00007fffddc6ed51 in <lambda()>::operator()(void) const (__closure=0x7fffffffcdf0) at /root/projects/pytorch/aten/src/ATen/native/TensorCompare.cpp:415
#42 0x00007fffddc6f2e8 in <lambda()>::operator()(void) const (__closure=0x7fffffffce80) at /root/projects/pytorch/aten/src/ATen/native/TensorCompare.cpp:415
#43 0x00007fffddc6f55b in at::native::isfinite (self=...) at /root/projects/pytorch/aten/src/ATen/native/TensorCompare.cpp:415
#44 0x00007fffdf968117 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__isfinite (self=...)
    at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp:5685
#45 0x00007fffdfa9341c in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__isfinite>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >::operator() (args#0=...,
    this=0x2713e30) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#46 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__isfinite>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(cons--Type <RET> for more, q to quit, c to continue without paging--
t at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &) (functor=0x2713e30, args#0=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#47 0x00007fffde49d941 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&> (
    unboxed_kernel_func=0x7fffdfa933ad <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__isfinite>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&> >, at::Tensor(const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &)>, functor=0x2713e30, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#48 0x00007fffdee04172 in c10::KernelFunction::call<at::Tensor, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x14d3ff8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#49 c10::Dispatcher::call<at::Tensor, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)> const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#50 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&)>::call(at::Tensor const&) const (args#0=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#51 at::_ops::isfinite::call (self=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:9405
#52 0x00007ffff4ef7b36 in at::Tensor::isfinite (this=0x7fffffffd178) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:5564
#53 0x00007ffff50e01c8 in <lambda(const at::Tensor&)>::operator()(const at::Tensor &) const (__closure=0x7fffffffd177, self=...)
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:11013
#54 0x00007ffff50e0451 in torch::autograd::THPVariable_isfinite (self_=0x0, args=0x7ffff7951390, kwargs=0x0)
```

# 3 AbsKernel.cu
- **真正执行的kernel 是AbsFunctor**；
- __forceinline__ 提示编译器尽量将函数内联，减少调用开销.

> 注意：在 CUDA 编程中，std::abs 也可以在设备代码中使用，前提是编译器支持（如 NVIDIA 的 NVCC 支持标准库函数std在设备代码中的调用）

```c++
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {

template<typename scalar_t>
struct AbsFunctor {
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    return std::abs(a);
  }
};

constexpr char abs_name[] = "abs_kernel";
void abs_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto abs_string = jiterator_stringify(
        template <typename T> T abs_kernel(T x) { return std::abs(x); });
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/abs_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, abs_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, AbsFunctor<opmath_t>());
    });
#endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_cuda",
        [&]() { gpu_kernel(iter, AbsFunctor<scalar_t>()); });
  }
}

  REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda)

} // namespace at::native
```

# 3 launch kernel 的函数
- **/root/projects/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh**
- <<<gridSize, blockSize, sharedMemBytes, stream>>>。
- num_threads() = 4 * warp_size(32) = 128


```c++
// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int64_t grid = (N + io_block_work_size<io_size>() - 1) / io_block_work_size<io_size>();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);

  switch (vec_size) {
    case 4:
      vectorized_elementwise_kernel<4, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 2:
      vectorized_elementwise_kernel<2, func_t, array_t>
          <<<grid, num_threads(), 0, stream>>>(N, f, data);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();
      unrolled_elementwise_kernel<func_t, array_t, elems_per_thread<io_size>()>
          <<<grid, num_threads(), 0, stream>>>(
              N, f, data, input_calc, output_calc, loader, storer);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
```


# 4 最终kernel : vectorized_elementwise_kernel
- func_t : type = at::native::AbsFunctor<float>
- f : $15 = (const at::native::AbsFunctor<float> &) @0x7fffffffbb57: {<No data fields>}
- N : 120

```c++
// /root/projects/pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh
template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  int remaining = N - io_block_work_size<io_size>() * blockIdx.x;

  if (remaining < io_block_work_size<io_size>()) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast,
        elems_per_thread<io_size>()>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t, elems_per_thread<io_size>()>(data));
  }
}
```
**elementwise_kernel_helper**

```c++
// /root/projects/pytorch/aten/src/ATen/native/cuda/Loops.cuh
template<typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;
  constexpr int elems_per_thread = policy_t::tws;

  int idx = blockIdx.x;

  return_t results[elems_per_thread];
  args_t args[elems_per_thread];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = c10::guts::apply(f, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}
```

# 5 数据加载有统一的加载函数

```c++
namespace policies {

template<typename data_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t, int elems_per_thread, int num_outputs=1>
struct unroll {

  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  static constexpr int tws = elems_per_thread;

  __device__ unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc, loader_t l, storer_t s):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc), loader(l), storer(s) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((int)(threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size_v<args_t>;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + elems_per_thread * num_threads() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads();
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + elems_per_thread * num_threads() * idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      thread_idx += num_threads();
    }
  }
};
```
