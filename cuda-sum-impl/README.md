# 1 Reduce 的完整调用栈
python 侧启动函数，用的是pytorch 2.6.0 版本， cuda: 12.4

```python
import torch

data = torch.randn(2,3,4,5).cuda()

data2 = torch.sum(data, 1, keepdim=True)
```

- sum 算子注册处
```c++
// /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu
REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda)
```

- **对应的c++调用栈：**

```c++
#0  at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float, float, float>::*)(at::TensorIterator&), &at::native::sum_functor<float, float, float>::operator(), 1u>, float (float, float)> >, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float, float, float>::*)(at::TensorIterator&), &at::native::sum_functor<float, float, float>::operator(), 1u>, float (float, float)> >, unsigned int, float, 4>) (reduction=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Reduce.cuh:222
#1  0x00007fffcccd125b in at::native::launch_reduce_kernel<512, at::native::ReduceOp<float, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float>::*)(at::TensorIterator&), &at::native::sum_functor<float>::operator(), 1>, float(float, float)> >, unsigned int, float, 4> >(const at::native::ReduceConfig &, const at::native::ReduceOp<float, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float, float, float>::*)(at::TensorIterator&), &at::native::sum_functor<float, float, float>::operator(), 1>, float(float, float)> >, unsigned int, float, 4> &) (config=..., reduction=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Reduce.cuh:890
#2  0x00007fffcccdbd3f in at::native::gpu_reduce_kernel<float, float, 4, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float, float, float>::*)(at::TensorIterator&), &at::native::sum_functor<float, float, float>::operator(), 1u>, float (float, float)> >, double>(at::TensorIterator&, at::native::func_wrapper_t<float, __nv_hdl_wrapper_t<false, true, false, __nv_dl_tag<void (at::native::sum_functor<float, float, float>::*)(at::TensorIterator&), &at::native::sum_functor<float, float, float>::operator(), 1u>, float (float, float)> > const&, double, at::native::AccumulationBuffer*, long) (iter=..., ops=..., ident=0, acc_buf_ptr=0x676e2e0, base_idx=0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Reduce.cuh:1273
#3  0x00007fffcccd6023 in at::native::sum_functor<float, float, float>::operator() (this=0x7fffffffcbb7, iter=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:16
#4  0x00007fffcccc83c7 in <lambda()>::operator()(void) const (__closure=0x7fffffffcbf0) at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:175
#5  0x00007fffcccc8643 in <lambda()>::operator()(void) const (__closure=0x7fffffffcc60) at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:175
#6  0x00007fffcccc8795 in <lambda(at::TensorIterator&)>::operator()(at::TensorIterator &) const (__closure=0x7fffffffcc87, iter=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:175
#7  0x00007fffccccfa67 in at::native::reduce_dispatch<at::native::sum_functor, at::native::sum_kernel_cuda(at::TensorIterator&)::<lambda(at::TensorIterator&)> >(at::TensorIterator &, <lambda(at::TensorIterator&)>) (iter=..., op=...)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:170
#8  0x00007fffcccc87d7 in at::native::sum_kernel_cuda (iter=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/ReduceSumProdKernel.cu:181
#9  0x00007fffddb6ef93 in at::native::DispatchStub<void (*)(at::TensorIterator&), at::native::sum_stub_DECLARE_DISPATCH_type>::operator()<at::TensorIterator&> (this=0x7ffff39d1000 <at::native::sum_stub>, device_type=c10::DeviceType::CUDA)
    at /root/projects/pytorch/aten/src/ATen/native/DispatchStub.h:250
#10 0x00007fffddb4fbd6 in at::native::structured_sum_out::impl (this=0x7fffffffd110, self=..., opt_dim=..., keepdim=true, opt_dtype=std::optional<c10::ScalarType> [no contained value], result=...) at /root/projects/pytorch/aten/src/ATen/native/ReduceOps.cpp:1218
#11 0x00007fffce7bc1b3 in at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList (self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:15727
#12 0x00007fffce9f9d2d in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >::operator() (args#3=std::optional<c10::ScalarType> [no contained value], args#2=true, args#1=..., args#0=..., this=0x411b740)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#13 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) (functor=0x411b740, args#0=..., args#1=..., args#2=true, args#3=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#14 0x00007fffdec5a80c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    unboxed_kernel_func=0x7fffce9f9c08 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>, functor=0x411b740, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#15 0x00007fffdeb22e86 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (dispatchKeySet=..., opHandle=..., this=0x10eea38)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#16 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> >(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)> const&, c10::DispatchKeySet, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#17 0x00007fffde9c4e6c in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (args#3=std::optional<c10::ScalarType> [no contained value], args#2=true, args#1=..., args#0=..., currentDispatchKeySet=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#18 at::_ops::sum_dim_IntList::redispatch (dispatchKeySet=..., self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/Operators_2.cpp:5526
#19 0x00007fffe235ef92 in at::redispatch::sum (dispatchKeySet=..., self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:7377
#20 0x00007fffe22a65b1 in <lambda()>::operator()(void) const (__closure=0x7fffffffd610) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_2.cpp:18956
#21 0x00007fffe22a6917 in torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList (ks=..., self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_2.cpp:18957
#22 0x00007fffe232f67b in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >::operator() (args#4=std::optional<c10::ScalarType> [no contained value], args#3=true,
    args#2=..., args#1=..., args#0=..., this=0x31b25d0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#23 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) (functor=0x31b25d0, dispatchKeySet=..., args#0=..., args#1=..., args#2=true,
    args#3=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#24 0x00007fffdec5a80c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    unboxed_kernel_func=0x7fffe232f51c <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>, functor=0x31b25d0, dispatchKeySet=...)
--Type <RET> for more, q to quit, c to continue without paging--
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#25 0x00007fffde9c4b53 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (dispatchKeySet=..., opHandle=..., this=0x10ef5b8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#26 c10::Dispatcher::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> >(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)> const&, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#27 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>::call(at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (
    args#3=std::optional<c10::ScalarType> [no contained value], args#2=true, args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#28 at::_ops::sum_dim_IntList::call (self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/Operators_2.cpp:5519
#29 0x00007ffff4ef0133 in at::Tensor::sum (this=0x7fffffffdad0, dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:3629
#30 0x00007ffff50ab35a in <lambda(const at::Tensor&, at::OptionalIntArrayRef, bool, std::optional<c10::ScalarType>)>::operator()(const at::Tensor &, at::OptionalIntArrayRef, bool, std::optional<c10::ScalarType>) const (__closure=0x7fffffffdac8, self=..., dim=...,
    keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:7137
#31 0x00007ffff50aba3c in torch::autograd::THPVariable_sum (self_=0x0, args=0x7ffff79fdcc0, kwargs=0x7ffece889c00) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:7139
#32 0x000000000054c584 in cfunction_call (func=0x7ffff7486390, args=0x7ffff79fdcc0, kwargs=0x7ffece889c00) at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
```

# 2 blockDim 和 GridDim 设置

## 2.1 形状计算
- **计算形状**：是为了内核实现的便利性和性能优化设计的。
- **维度重排（Permutation）**：通过调用 permute_dimensions 方法，将维度重新排列为计算效率最高的顺序（而非用户指定的逻辑顺序）；
- **维度合并（Coalescing）**：通过调用 coalesce_dimensions 方法，将相邻的小维度合并为一个大维度，以减少显式迭代的维度数量。例如，对于连续存储的张量，点对点操作可以简化为仅在一个维度上进行;
- 用户最终看到的输出形状可能是未经重排或合并的原始形状。

---

最终output shape_ 的变化过程如下：<br>

1. 用户显式声明形状：如果用户通过 declare_static_shape() 提供了初始形状，则`直接使用该形状`;
2. 计算真实形状：如果没有显式声明形状，则通过 `compute_shape() 计算用户指定的真实形状`;
3. 重排维度：调用 **reorder_dimensions()** 方法，根据计算效率重新排列维度顺序**;
4. 合并维度：调用 **coalesce_dimensions()** 方法，将相邻的小维度合并为一个大维度.

> shape_ 是 TensorIterator 内部用于优化计算的核心变量。
> 它的值从用户指定的形状出发，经过一系列变换（如重排和合并），最终成为适合内核实现的计算形状。
> 这种设计允许 PyTorch 在保持用户接口简单的同时，最大化底层计算的性能。
> 通过这种方式，TensorIterator 能够高效地处理广播、类型转换和多维张量操作，同时隐藏了复杂的实现细节。

## 2.2 setReduceConfig的定义和设置

## 2.2.1 struct ReduceConfig
```c++
struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int input_vec_size = 4;

  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  template <int output_vec_size> // BlockIdx : {32, 4, 1}
  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x; // 一个warp ???
    int warp = threadIdx.y; // 几个warp ???
    int cta1 = blockIdx.x;  // GridDim : {128, 1, 1}
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size; // output_mult : {1, 32}
  }

};
```

### 2.2.2 setReduceConfig
- block width 对应 input tensor 变化最快的维度, 可以起到对HBM 的合并访问;
- dim0 和 dim1 这两个参数定义了线程块维度（blockDim.x 和 blockDim.y）的最大值，但不直接决定实际启动配置；
- 实际block size可能小于或等于这些上限值，具体由运行时参数 config.input_mult 和 config.output_mult 动态控制;
- config.input_mult：控制输入数据处理的并行度（如每个线程处理多少输入元素）。
- config.output_mult：控制输出数据的合并策略（如多个线程协作写入单个结果）。
- **为什么优先最大化dim1（y维度）而非dim0（x维度）？**
> 硬件调度机制：
> GPU的线程调度以线程束（Warp，通常32线程）为单位，优先在dim0（x）上连续分配线程。若dim0已较大（如256），继续增加可能导致资源争用。
>
> 内存访问模式：
> dim0通常与变化最快的维度对齐（如矩阵的列方向），已用于优化内存合并。此时扩展dim1（y方向）可增加线程数，而不影响内存访问连续性。
>
> 避免资源浪费：
> GPU对线程块维度有上限（如每块最多1024线程）。若dim0=256，则dim1=4（总线程数256×4=1024）可占满上限；而单纯增大dim0可能超过硬件限制。

**CTA（Cooperative Thread Array） 是 线程块（Thread Block）的同义词.**

---

```c++
template<typename arg_t, typename scalar_t, int vt0>
ReduceConfig setReduceConfig(const TensorIterator& iter){
  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements(); // mtn: 40
  int64_t inputs_per_output = iter.numel() / num_outputs; // mtn: 3 --> 3 个input element 对应一个output
  int input_index = iter.ntensors() - 1; // 1

  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output); // mtn: arg_t : float

  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    // Adjust block size to map block width to fastest changing dimension of input
    // tensor. This grants the best possible memory accessing pattern, given that
    // for non-contiguous tensor with space in between, we cannot have perfect
    // memory coalescing.
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(/*arg=*/input_index)[0] <
        iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);
    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
      // Map block.x to the fastest reducing dimension. It implies:
      //   1. block_x_reduce is required.
      //   2. block.y now max out to num_outputs.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
    } else {
      // Map block.x to the fastest non reducing dimension. It implies:
      //   1. block_x_reduce is turned off.
      //   2. block.y now max out to inputs_per_output.
      dim0 = num_outputs;       // mtn : 40
      dim1 = inputs_per_output; // mtn : 3 --> iter.strides(0) : （80， 4， 240）--> （20， 1， 60）
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]; // mtn : 4, output compute shape: (3, 20, 2);
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = sizeof(scalar_t);
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access, there are two cases which we call
  // "vectorize along input" and "vectorize along output". Note that the "input/output"
  // here does not mean we are vectorizing load/store instructions. We always only vectorize
  // load instructions.
  //
  // Case 1: "vectorize along input"
  // This case happens when we are reducing along fastest moving dimesion. In such case, threads
  // with the same threadIdx.y works on the same reduction cooperatively and will produce results
  // for the same output. In such case, values in each loaded vector always correspond to the same output.
  //
  // Case 2: "vectorize along output"
  // This case happens when the fastest moving dimesion is not the dimension of reduction. In such case,
  // threads with different threadIdx.x are independent and will produce results for different outputs.
  // In such case, values in each loaded vector always correspond to different outputs.
  if (fastest_moving_stride == sizeof(scalar_t)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 && iter.num_reduce_dims() == 1 && vt0 >= ReduceConfig::input_vec_size) {
      // Case 1: "vectorize along input"
      // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
      // we should avoid vectorization.
      config.vectorize_input = true;
      dim0 /= config.input_vec_size;
    } else if (!reduction_on_fastest_striding_dimension) {
      // Case 2: "vectorize along output"
      config.output_vec_size = get_output_vec_size<scalar_t>(iter); // mtn: 4
      dim0 /= config.output_vec_size; // mtn: 40/4 = 10
    }
  }

  // Adjust block_width and block_height
  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width; // mtn: 10
  int block_height = config.block_height; // mtn: 2

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > warpSize).
    config.input_mult[0] = config.split_input(block_width);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width); // mtn: {1, 0}
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  const int warp_split_threshold =
      std::min<int>(block_height * 16, max_values_per_thread);
  const int num_mp =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  bool force_splitting_output = false;
#ifdef USE_ROCM
  force_splitting_output = iter.ndim() == 2 &&
      reduction_on_fastest_striding_dimension &&
      config.values_per_thread() < 1024 && num_mp < 100;
#endif

  if (!force_splitting_output &&
      config.values_per_thread() >= warp_split_threshold) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height); // mtn : {1, 8}
  }

  int max_threads_per_mp =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor; // mtn : 2048
#ifdef USE_ROCM
  // Control the number of threadblocks by adjusting the maximum number of
  // threads per multi-processor. These numbers better reflect the maximum
  // theoretical achievable threads per MP for the reduction operation.
  if (iter.ndim() == 1)
    max_threads_per_mp = 512;
  if (iter.ndim() == 2)
    max_threads_per_mp = 256;
#endif
  const int blocks_per_sm = max_threads_per_mp / config.num_threads; // mtn: 2048 / 16 = 128
  const int target_grid_size = num_mp * blocks_per_sm; // mtn : 13824
  int grid = config.grid().x; // mtn : 1
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    // If we decide to split input across blocks, as long as we can get enough
    // number of blocks (`target_grid_size`) to balance SM, we should still
    // make the number of values per thread large for best performance.
    int ctas_per_output1 = div_up(target_grid_size, grid);
    int ctas_per_output2 = div_up(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 = div_up(config.values_per_thread(), max_values_per_thread);
    // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
    // a large number of values to deal with. But we don't want values_per_thread to be larger than
    // max_values_per_thread
    config.ctas_per_output = std::max(std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
#ifdef USE_ROCM
    // In cases where a number of threadblocks along the y direction of the grid
    // is needed then make sure they are reduced to the number of MPs. For
    // smaller sizes, use half the number of MPs. For smaller sizes than half
    // the number of MPs use the original value unless the value is less than 16
    // blocks in which case it is more profitable to use just 1 block.
    if (config.ctas_per_output > num_mp)
      if (num_mp < 128)
        config.ctas_per_output =
            num_mp * (config.ctas_per_output > 512 ? 4 : 2);
      else
        config.ctas_per_output = num_mp;
    else if (config.ctas_per_output > div_up(num_mp, 2))
      config.ctas_per_output = div_up(num_mp, 2);
    else if (config.ctas_per_output < 16)
      config.ctas_per_output = 1;
#endif
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
};
```

### 2.2.3 最终的config 参数

```c++
$218 = {static BLOCK_X = 0, static BLOCK_Y = 1, static CTA = 2, static input_vec_size = 4, element_size_bytes = 4, num_inputs = 32, num_outputs = 65536,
  step_input = 1, step_output = 128, ctas_per_output = 1, input_mult = {0, 0, 0}, output_mult = {1, 32}, block_width = 32, block_height = 4, num_threads = 128,
  vectorize_input = false, output_vec_size = 4}
(gdb) p config.grid()
$219 = {x = 128, y = 1, z = 1}
```

- output shape : {8, 1, 64, 128} --> {8, 1, 64 * 128} --> {64 * 128, 1, 8} --> 提取sum dim --> {1, 64 * 128, 8};
- output stride : {8, 1, 8192} --> {32768, 0, 4} --> {4, 0, 32768} --> 提取sum dim --> {0, 4, 32768}
- iter shape : {32, 8192, 8} # dim 逆排列的
- input stride : {32768, 4, 1048576}

**block and grid value** <br>
```c++
(gdb) p block
$229 = {x = 32, y = 4, z = 1}
(gdb) p grid
$230 = {x = 128, y = 1, z = 1}
```

# 3 最终实现的函数

## 3.1 global function
```c++
template<int nt, int output_vec_size, typename R>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void reduce_kernel(R reduction) {
  reduction.template run<output_vec_size>();
}
```

## 3.2 reduce device function

```c++
template <int output_vec_size>
C10_DEVICE void run() const { // #define C10_DEVICE __device__
  extern __shared__ char shared_memory[];
  index_t output_idx = config.output_idx<output_vec_size>(); //
  index_t input_idx = config.input_idx();
  auto base_offsets1 = output_calc.get(output_idx)[1];

  using arg_vec_t = std::array<arg_t, output_vec_size>;
  arg_vec_t value;

  if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
    const scalar_t* input_slice = (const scalar_t*)((const char*)src + base_offsets1);
    value = thread_reduce<output_vec_size>(input_slice);
  }

  if (config.should_block_y_reduce()) {
    value = block_y_reduce<output_vec_size>(value, shared_memory);
  }
  if (config.should_block_x_reduce()) {
    value = block_x_reduce<output_vec_size>(value, shared_memory);
  }

  using out_ptr_vec_t = std::array<out_scalar_t*, output_vec_size>;
  using offset_vec_t = std::array<index_t, output_vec_size>;
  offset_vec_t base_offsets;
  out_ptr_vec_t out;

  #pragma unroll
  for (int i = 0; i < output_vec_size; i++) {
    base_offsets[i] = output_calc.get(output_idx + i)[0];
    out[i] = (out_scalar_t*)((char*)dst[0] + base_offsets[i]);
  }

  arg_vec_t* acc = nullptr;
  if (acc_buf != nullptr) {
    size_t numerator = sizeof(arg_t);
    size_t denominator = sizeof(out_scalar_t);
    reduce_fraction(numerator, denominator);
    acc = (arg_vec_t*)((char*)acc_buf + (base_offsets[0] * numerator / denominator));
  }

  if (config.should_global_reduce()) {
    value = global_reduce<output_vec_size>(value, acc, shared_memory);
  } else if (config.should_store(output_idx)) {
    if (accumulate) {
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.translate_idx(value[i], base_idx);
      }
    }

    if (acc == nullptr) {
      if (accumulate) {
        value = accumulate_in_output<output_vec_size, can_accumulate_in_output>(out, value);
      }
      if (final_output) {
        set_results_to_output<output_vec_size>(value, base_offsets);
      } else {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          *(out[i]) = get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
        }
      }
    } else {
      if (accumulate) {
        #pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.combine((*acc)[i], value[i]);
        }
      }
      if (final_output) {
        set_results_to_output<output_vec_size>(value, base_offsets);
      } else {
        *acc = value;
      }
    }
  }
}
```

# 4 example
**input and sum**
- input shape : {16, 128, 64, 128}
- sum dim : 1

**TensorIterator**
- iter input shape : {16, 128, 64, 128}
- iter input stride : {1048576, 8192, 128, 1}
- iter output shape :  {16, 1, 64, 128}
- iter output stride : {8192, 8192, 128, 1}
- iter.shape : {128, 8192, 16}
- iter.perm_ : {1, 3, 2, 0}

**operantInfo_**
- iter output operant_ stride-bytes : {0, 4, 32768}       # stride 从内向外递增, 单位： bytes
- iter input operant_ stride-bytes : {32768, 4, 4194304}  # {8192， 128， 16} --> {4, 32768, 4194304}

**RduceConfig**
- BLOCK_X = 0    # block x 对应 dim 0
- BLOCK_Y = 1
- CTA = 2        # block 数量 ？是每个SM里最大线程块的数量吗？
- input_vec_size = 4
- output_vec_size = 4
- element_size_bytes = 4
- num_inputs = 128 # 进行reduce 的 input 的元素个数
- num_outputs = 131072 # 16 * 1 * 64 * 128
- step_input = 4   # 128 / 32 ? 需进行4次step ...
- step_output = 32 # parallelism : 32
- block_width = 32 # {1, 8192, 16}
- block_height = 4
- num_threads = 128
- vectorize_input = false
- ctags_per_output = 1 # 每个输出元素（或输出张量）分配的CUDA线程块数量
- input_mult = {0, 1, 0}
- output_mult = {1, 0}
- block dim3 = {x=32, y=4, z=1}
- grid dim3 = {x=1024, y=1, z=1}
-

**grid and block**
```c++
  dim3 block() const {
    return dim3(block_width, block_height); // 32, 4
  }

  dim3 grid() const {
    // {16 * 1 * 64 * 128 / 4 / 32 = 1024, 1}
    return dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }
```

**should_block_x_reduce and should_block_y_reduce**
```c++
C10_HOST_DEVICE bool should_block_x_reduce() const {
  return input_mult[BLOCK_X] != 0; // false
}

C10_HOST_DEVICE bool should_block_y_reduce() const {
  return input_mult[BLOCK_Y] != 0; // true
}
```

**input 和 output 索引的计算**
- block : {x = 32, y = 4, z = 1}
- grid :  {x = 1024, y = 1, z = 1}

```c++
C10_HOST_DEVICE int input_idx() const {
  int lane = threadIdx.x;
  int warp = threadIdx.y;
  int cta2 = blockIdx.y;
  // blockIdx.y + threadIdx.y
  return (lane * input_mult[BLOCK_X] +
          warp * input_mult[BLOCK_Y] +
          cta2 * input_mult[CTA]);
}

template <int output_vec_size>
C10_HOST_DEVICE int output_idx() const {
  int lane = threadIdx.x; // warp : {}
  int warp = threadIdx.y;
  int cta1 = blockIdx.x; // 0 ~ 1023

  // 131072 = 16 * 64 * 128 --> (32 * blockIdx.size + threadIdx.x ) * 4 --> 每个线程处理4个elem
  return (lane * output_mult[BLOCK_X] +
          warp * output_mult[BLOCK_Y] +
          cta1 * step_output) * output_vec_size;
}
```

**set input_mult and output_mult**
```c++
int split_input(int parallelism) {
  int step = step_input;     // 是否进行multistep
  step_input *= parallelism; // 步长设定
  return step;
}

int split_output(int parallelism) {
  int step = step_output;
  step_output *= parallelism;
  return step;
}
```

**offsetCalculator**
-  int dims; // mtn : how many dims to reduce : 1
-  at::cuda::detail::IntDivider<index_t> sizes_[MAX_DIMS];
-  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];

> 注意：保存的stride_ 和 strides 做了 转置：strides_[i][arg] = strides[arg][i] / element_size;

```c++
template <typename index_t> // uint32_t
static OffsetCalculator<2, index_t> make_output_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();        // mtn : 1 有几个reduce dims
  int num_output_dims = iter.ndim() - num_reduce_dims; // mtn : 2 输出是几个dims
  int input_index = iter.ntensors() - 1;               // mtn : 1 输入的dim
  int output_index = 0;
  std::array<const int64_t*, 2> strides = {
    iter.strides(output_index).data() + num_reduce_dims, // {0, 4, 32768} --> {4, 32768}
    iter.strides(input_index).data() + num_reduce_dims,  // {32768, 4, 4194304} --> {4, 4194304}
  };
  auto shape = iter.shape().data() + num_reduce_dims;  // mtn : {128, 8192, 16} --> shape: {8192, 16}
  return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data()); // NARGS  = 2
}

template <typename index_t>
static OffsetCalculator<1, index_t> make_input_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims(); // mtn : 1
  int input_index = iter.ntensors() - 1;        // mtn : 1
  std::array<const int64_t*, 1> strides = {     // mtn : {32768, 4, 4194304}
    iter.strides(input_index).data(),
  };
  return OffsetCalculator<1, index_t>(num_reduce_dims, iter.shape().data(), strides.data()); // NARGS  = 1
}
```

**通过offsetcalculator获取offset**
```c++
C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
  offset_type offsets;
  #pragma unroll
  for (int arg = 0; arg < NARGS; arg++) {
    offsets[arg] = 0; // 先填充0
  }

  #pragma unroll
  for (int dim = 0; dim < MAX_DIMS; ++dim) { // dims = 2
    if (dim == dims) { // dims = 1
      break;
    }
    // output : sizes_ = {8192, 16}
    auto divmod = sizes_[dim].divmod(linear_idx); // dim0: 100 / 8192 = 0, 100 % 8192 = 100
    linear_idx = divmod.div; // 0 ~ 8192 : linar_idx = 0, 8192 ~ 131072 : linar_idx = 1 ~ 15

    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] += divmod.mod * strides_[dim][arg]; // 8192 --> {32768, 4194304}
    }

  }
  return offsets;
}
```

**thread_reduce_impl**
```c++
  template <int output_vec_size, typename offset_calc_t>
  C10_DEVICE std::array<arg_t, output_vec_size> thread_reduce_impl(const scalar_t* data_, offset_calc_t calc) const {
    // data : base offset : {8192, 16} 中每个元素开始的偏移
    index_t idx = config.input_idx();
    const index_t end = config.num_inputs;    // 128
    const index_t stride = config.step_input; // 4

    using arg_vec_t = std::array<arg_t, output_vec_size>; // 长度为4的数组
    using load_t = at::native::memory::aligned_vector<scalar_t, output_vec_size>;

    // Multiple accumulators to remove dependency between unrolled loops.
    // vt0 : Vectorization Thread Unroll Factor
    arg_vec_t value_list[vt0]; // vt0 = 4

    #pragma unroll
    for (int i = 0; i < vt0; i++) {
      #pragma unroll
      for (int j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ident; // idtent = 0, 填充默认值 0 ???
      }
    }

    load_t values[vt0]; // load_t 一次load的是一个vector : size : 4

    while (idx + (vt0 - 1) * stride < end) { // threadIdx.y + 3 * 4 < 128
      #pragma unroll
      for (index_t i = 0; i < vt0; i++) {
        // calc = [&](index_t idx) { return idx * 32768; })
        const auto offset = calc(idx + i * stride) / output_vec_size;
        values[i] = memory::load_vector<output_vec_size>(data_, offset);
      }
      #pragma unroll
      for (index_t i = 0; i < vt0; i++) {
        #pragma unroll
        for (index_t j = 0; j < output_vec_size; j++) {
          value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
        }
      }
      idx += stride * vt0;
    }

    // tail
    int idx_ = idx;
    #pragma unroll
    for (index_t i = 0; i < vt0; i++) {
      if (idx >= end) {
        break;
      }
      const auto offset = calc(idx) / output_vec_size;
      values[i] = memory::load_vector<output_vec_size>(data_, offset);
      idx += stride;
    }
    idx = idx_;
    #pragma unroll
    for (index_t i = 0; i < vt0; i++) {
      if (idx >= end) {
        break;
      }
      #pragma unroll
      for (index_t j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
      }
      idx += stride;
    }

    // combine accumulators
    #pragma unroll
    for (int i = 1; i < vt0; i++) {
      #pragma unroll
      for (index_t j = 0; j < output_vec_size; j++) {
        value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
      }
    }
    return value_list[0];
  }

```

**block_y_reduce** <br>
```c++
template <int output_vec_size>
C10_DEVICE std::array<arg_t, output_vec_size> block_y_reduce(std::array<arg_t, output_vec_size> value, char* shared_memory) const {
  using args_vec_t = std::array<arg_t, output_vec_size>;
  args_vec_t* shared = (args_vec_t*)shared_memory;
  shared[config.shared_memory_offset(0)] = value;
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      args_vec_t other = shared[config.shared_memory_offset(offset)];
      #pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        value[i] = ops.combine(value[i], other[i]);
      }
      shared[config.shared_memory_offset(0)] = value;
    }
  }
  return value;
}
```
