# 如何调度到TensorIterator

- 以sum 为例, 会调度到TensorIterator 的 reduce_op,调用栈如下:

```c++
#0  at::TensorIterator::reduce_op (out=..., a=...) at /root/projects/pytorch/aten/src/ATen/TensorIterator.cpp:1125
#1  0x00007fffddb66926 in at::meta::make_reduction (self=..., result=..., opt_dims=..., keepdim=true, in_dtype=c10::ScalarType::Float)
    at /root/projects/pytorch/aten/src/ATen/native/ReduceOpsUtils.h:423
#2  0x00007fffddb66a9d in at::meta::make_reduction_from_out_ty (self=..., result=..., opt_dims=..., keepdim=true, out_dtype=c10::ScalarType::Float)
    at /root/projects/pytorch/aten/src/ATen/native/ReduceOpsUtils.h:465
#3  0x00007fffddb4fa0a in at::native::structured_sum_out::impl (this=0x7fffffffd110, self=..., opt_dim=..., keepdim=true,
    opt_dtype=std::optional<c10::ScalarType> [no contained value], result=...) at /root/projects/pytorch/aten/src/ATen/native/ReduceOps.cpp:1203
#4  0x00007fffce7bc1b3 in at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList (self=..., dim=..., keepdim=true,
    dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:15727
#5  0x00007fffce9f9d2d in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >::operator() (args#3=std::optional<c10::ScalarType> [no contained value], args#2=true,
    args#1=..., args#0=..., this=0x411b740) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#6  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) (
    functor=0x411b740, args#0=..., args#1=..., args#2=true, args#3=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#7  0x00007fffdec5a80c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    unboxed_kernel_func=0x7fffce9f9c08 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), at::(anonymous namespace)::wrapper_CUDA_sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>, functor=0x411b740, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#8  0x00007fffdeb22e86 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    dispatchKeySet=..., opHandle=..., this=0x10eea38) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#9  c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> >(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)> const&, c10::DispatchKeySet, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#10 0x00007fffde9c4e6c in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (
    args#3=std::optional<c10::ScalarType> [no contained value], args#2=true, args#1=..., args#0=..., currentDispatchKeySet=..., this=<optimized out>)
--Type <RET> for more, q to quit, c to continue without paging--
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#11 at::_ops::sum_dim_IntList::redispatch (dispatchKeySet=..., self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/build/aten/src/ATen/Operators_2.cpp:5526
#12 0x00007fffe235ef92 in at::redispatch::sum (dispatchKeySet=..., self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:7377
#13 0x00007fffe22a65b1 in <lambda()>::operator()(void) const (__closure=0x7fffffffd610)
    at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_2.cpp:18956
#14 0x00007fffe22a6917 in torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList (ks=..., self=..., dim=..., keepdim=true,
    dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_2.cpp:18957
#15 0x00007fffe232f67b in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >::operator() (
    args#4=std::optional<c10::ScalarType> [no contained value], args#3=true, args#2=..., args#1=..., args#0=..., this=0x31b25d0)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#16 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) (functor=0x31b25d0, dispatchKeySet=..., args#0=..., args#1=..., args#2=true,
    args#3=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#17 0x00007fffdec5a80c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    unboxed_kernel_func=0x7fffe232f51c <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>, functor=0x31b25d0, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#18 0x00007fffde9c4b53 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    dispatchKeySet=..., opHandle=..., this=0x10ef5b8) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#19 c10::Dispatcher::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> >(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)> const&, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#20 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>::call(at::Tensor const&, c10::Optiona--Type <RET> for more, q to quit, c to continue without paging--
, torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::[11/9572]
e> > >::operator() (
    args#4=std::optional<c10::ScalarType> [no contained value], args#3=true, args#2=..., args#1=..., args#0=..., this=0x31b25d0)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#16 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional
<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optio
nal<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, c10:
:OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) (functor=0x31b25d0, dispatchKeySet=..., args#0=..., args#1=..., args#2=true,
    args#3=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#17 0x00007fffdec5a80c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    unboxed_kernel_func=0x7fffe232f51c <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArr
ayRef<long int>, bool, std::optional<c10::ScalarType>), torch::autograd::VariableType::(anonymous namespace)::sum_dim_IntList>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, c10::Optional
ArrayRef<long int>, bool, std::optional<c10::ScalarType> > >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, c10::OptionalArrayRef<long int>, bool, std::optional<c10::ScalarType>)>::call(c10::OperatorKernel *, c10::Disp
atchKeySet, const at::Tensor &, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>, functor=0x31b25d0, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#18 0x00007fffde9c4b53 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> > (
    dispatchKeySet=..., opHandle=..., this=0x10ef5b8) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#19 c10::Dispatcher::call<at::Tensor, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType> >(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::op
tional<c10::ScalarType>)> const&, at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#20 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>)>::call(at::Tensor const&, c10::Optiona--Type <RET> for more, q to quit, c to continue without pa
ging--
lArrayRef<long>, bool, std::optional<c10::ScalarType>) const (args#3=std::optional<c10::ScalarType> [no contained value], args#2=true, args#1=..., args#0=...,
    this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#21 at::_ops::sum_dim_IntList::call (self=..., dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/build/aten/src/ATen/Operators_2.cpp:5519
#22 0x00007ffff4ef0133 in at::Tensor::sum (this=0x7fffffffdad0, dim=..., keepdim=true, dtype=std::optional<c10::ScalarType> [no contained value])
    at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:3629
#23 0x00007ffff50ab35a in <lambda(const at::Tensor&, at::OptionalIntArrayRef, bool, std::optional<c10::ScalarType>)>::operator()(const at::Tensor &, at::OptionalIntArrayRef, bool, std::optional<c10::ScalarType>) const (__cl
osure=0x7fffffffdac8, self=..., dim=..., keepdim=true,
    dtype=std::optional<c10::ScalarType> [no contained value]) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:7137
#24 0x00007ffff50aba3c in torch::autograd::THPVariable_sum (self_=0x0, args=0x7ffff79fdc40, kwargs=0x7ffecea7bf40)
    at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_2.cpp:7139
#25 0x000000000054c584 in cfunction_call (func=0x7ffff7482390, args=0x7ffff79fdc40, kwargs=0x7ffecea7bf40)
    at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
```

## 1.1 cuda op 在此注册
```cpp
// /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp

at::Tensor wrapper_CUDA_sum_dim_IntList(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  // No device check
structured_sum_out_functional op;   // 创建一个结构体对象，并调用其meta方法 和 impl方法
op.meta(self, dim, keepdim, dtype);
op.impl(self, dim, keepdim, dtype, op.outputs_[0]);
return std::move(op.outputs_[0]);
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
m.impl("sum.dim_IntList", TORCH_FN(wrapper_CUDA_sum_dim_IntList));
}
```

## 1.2 structured_sum_out_functional 类

- ouptut 在此保存;
- set_output_strided 和 set_output_raw_strided override

```c++
struct structured_sum_out_functional final : public at::native::structured_sum_out {
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        auto current_device = guard_.current_device();
        if (C10_UNLIKELY(current_device.has_value())) {
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),
            "structured kernels don't support multi-device outputs");
        } else {
          guard_.reset_device(options.device());
        }
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        auto current_device = guard_.current_device();
        if (C10_UNLIKELY(current_device.has_value())) {
          TORCH_INTERNAL_ASSERT(*current_device == options.device(),
            "structured kernels don't support multi-device outputs");
        } else {
          guard_.reset_device(options.device());
        }
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<Tensor, 1> outputs_;
    c10::cuda::OptionalCUDAGuard guard_;
};
```

## 1.3 structured_sum_out_functional 相关类

### 1.3.1 MetaBase

```c++
struct TORCH_API MetaBase {
  MetaBase() = default;
  MetaBase(const MetaBase&) = default;
  MetaBase& operator=(const MetaBase&) = default;
  MetaBase(MetaBase&&) noexcept = default;
  MetaBase& operator=(MetaBase&&) noexcept = default;
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;

  // Use this function whenever the kernel requires specific strides for the
  // output. If `strides` does not match the given output strides, proxy outputs
  // will be created and passed to the IMPL function.
  virtual void set_output_strided(
      int64_t output_idx [[maybe_unused]],
      IntArrayRef sizes [[maybe_unused]],
      IntArrayRef strides [[maybe_unused]],
      TensorOptions options [[maybe_unused]],
      DimnameList names [[maybe_unused]] = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function whenever the kernel requires specific strides for the
  // output. If `strides` does not match the given output strides, proxy outputs
  // will be created and passed to the IMPL function.
  virtual void set_output_strided(
      int64_t output_idx [[maybe_unused]],
      IntArrayRef sizes [[maybe_unused]],
      IntArrayRef strides [[maybe_unused]],
      TensorOptions options [[maybe_unused]],
      DimnameList names [[maybe_unused]] = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function whenever the kernel knows how to handle arbitrary strided
  // outputs. This function has the same behavior as the old `set_output`: it
  // will only re-stride if the given output was resized.
  virtual void set_output_raw_strided(
      int64_t output_idx [[maybe_unused]],
      IntArrayRef sizes [[maybe_unused]],
      IntArrayRef strides_hint [[maybe_unused]],
      TensorOptions options [[maybe_unused]],
      DimnameList names [[maybe_unused]] = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }
};
```

```markdown
| 方法                  | 性能  | 灵活性 | 核函数要求               | 典型场景                     |
|-----------------------|-------|--------|--------------------------|------------------------------|
| `set_output_raw_strided` | 最高  | 最高   | 需处理任意步长           | 逐元素操作（如加法、乘法）   |
| `set_output_contiguous` | 较低  | 最低   | 仅支持连续内存           | 依赖内存连续性的优化核函数   |
| `set_output_strided`    | 中等  | 中等   | 需处理广播后的规则步长   | 广播操作（如矩阵乘、张量扩展） |
```

### 1.3.2 meta 函数的实现：父类-->父类 里的函数
```cpp
// meta 声明处: /root/projects/pytorch/build/aten/src/ATen/ops/sum_meta.h
struct TORCH_API structured_sum_dim_IntList : public at::impl::MetaBase {
    void meta(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype);
};
```

```c++
// meta 实现处: /root/projects/pytorch/aten/src/ATen/native/ReduceOps.cpp

#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta

TORCH_META_FUNC2(sum, dim_IntList)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}
```

**resize_reduction** <br>

> 处理的是在哪个维度上进行reduce, 那个维度由 opt_dims来决定，

> **resize_reduction 里meta.set_output_raw_strided 里才真正创建了输出的tensor**

```c++
// /root/projects/pytorch/aten/src/ATen/native/ReduceOpsUtils.h
inline void resize_reduction(
    impl::MetaBase& meta,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype,
    bool allow_empty_dims=false) {
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  maybe_wrap_dims(dims_, self.dim()); // 处理reduce 哪个dims里的负数index
  // (8, 1, 64, 128)
  auto shape = get_reduction_shape(self, dims_, keepdim, allow_empty_dims);
  if (self.layout() == kStrided) {
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  } else if (shape.empty()) {
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype).layout(kStrided));
  } else {
    TORCH_CHECK(false, "resize_reduction: support for output with ", self.layout(), " layout is not implemented yet");
  }
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(), self, dims_, keepdim);
}
```

### 1.3.3 impl 函数的实现

- 声明处
```cpp
// /root/projects/pytorch/build/aten/src/ATen/ops/sum_native.h
struct TORCH_API structured_sum_out : public at::meta::structured_sum_dim_IntList {
void impl(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, const at::Tensor & out);
};
```

- 定义处
```cpp
// /root/projects/pytorch/aten/src/ATen/native/ReduceOps.cpp

// 宏定义不在上文件
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

TORCH_IMPL_FUNC(sum_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  // 在此创建 TensorIterator
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    // Here is a limitation of TensorIterator reductions for permuted input with lower precision on CPU.
    // Consider the case: TensorIterator coalesces such input and output to >= 2 dims tensors,
    // and the output stride is [0, 0, x, x, ...] with x >= 0 (two reduced dimensions and non-reduced dims).
    // Since the reduction loop only operates on two dimensions at a time,
    // the intermediate sums is forced to do accumulation in the second reduced dim with lower precision.
    // See https://github.com/pytorch/pytorch/issues/83149
    if (should_use_acc_buffer(iter)) {
      auto tmp_output = at::empty(result.sizes(), result.options().dtype(kFloat));
      at::sum_outf(self.to(ScalarType::Float), opt_dim, keepdim, /*dtype=*/std::nullopt, tmp_output);
      result.copy_(tmp_output);
    } else{
      sum_stub(iter.device_type(), iter);
    }
  }
}
```

- 数据类型统一
```c++
// [[maybe_unused]] 避免了编译器对该函数未被使用时的警告
[[maybe_unused]] inline TensorIterator make_reduction_from_out_ty(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype) {
  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  const bool gpu_lowp_to_f32 =
      (self.is_cuda() &&
       (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) &&
       out_dtype == kFloat);
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(self, result, opt_dims, keepdim, in_dtype);
}
```

- TensorIterator reduce_op 的调用处
```c++
inline TensorIterator make_reduction(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef opt_dims, // {1}
    bool keepdim,
    ScalarType in_dtype) {
  int64_t ndim = self.dim(); // 4
  // std::bitset 简化了固定大小位序列的操作，提供类型安全和丰富的接口，
  // 适用于需高效处理位标志、掩码或二进制数据的场景。
  // 其编译时确定大小的特性确保了性能，但在需要动态调整时需选择替代方案。
  auto mask = at::native::make_dim_mask(opt_dims, ndim); // std::bitset
  auto viewed_result =
      at::native::review_reduce_result(result, ndim, mask, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}
```

**review_reduce_result** <br>
```c++
inline Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (const auto dim : c10::irange(ndim)) {
    if (mask[dim]) {
      // 恢复被移除的维度（大小为 1）。
      shape.insert(shape.begin() + dim, 1);
      // 标记该维度为广播维度（所有元素共享内存中的同一值）
      stride.insert(stride.begin() + dim, 0); // 0代表: broadcast
    }
  }
  return result.as_strided(shape, stride); //
}
```

**重点** <br>
> 在 PyTorch 中，Tensor 的 stride 表示沿某个维度移动时需要在内存中跳过的元素数量。如果某个维度的 stride 为 0，意味着该维度上的所有元素在内存中共享同一个数据值。这是 PyTorch 实现广播（broadcasting）或扩展（expand）时的优化手段，无需真正复制数据即可“虚拟扩展”维度。

```python
import torch
data0 = torch.randn(3, 1) # shape : (3, 1), stride : (1, 1)
data1 = data0.expand(3, 4) # shape : (3, 4), stride : (1, 0)
```

### 1.3.4 TensorIterator::reduce_op 的调用
```c++
// /root/projects/pytorch/aten/src/ATen/TensorIterator.cpp:1125
TensorIterator TensorIterator::reduce_op(TensorBase& out, const TensorBase& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out)
    .add_owned_const_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
    .promote_inputs_to_common_dtype(true)
    .build();
}
```

### 1.3.5 TensorIterator::build
```c++
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // Check that input dimensions are aligned correctly & compute outnames.
  compute_names(config);
  // compute the broadcasted shape
  compute_shape(config);
  // mark outputs for resizing if necessary
  mark_resize_outputs(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config); // output:  (64*128*4, 0, 512*4, 4*4)
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    if (!is_meta_) coalesce_dimensions();
  }

  if (is_meta_) return;

  auto has_storage = true;
  for (auto& op : operands_) {
    has_storage &= op.tensor_base().has_storage();
  }
  auto privateuse1_without_storage =
     common_device_.type() == DeviceType::PrivateUse1 &&
     !has_storage;

  // XLA and lazy tensors don't have storage, so they don't have an underlying data pointer.
  // Nothing beyond this point is important for meta functions, so it's fine to exit early here.
  // Extend the condition to MAIA tesnors as MAIA tensors also don't have storage.
  if (privateuse1_without_storage  ||
      common_device_.type() == DeviceType::MTIA ||
      common_device_.type() == DeviceType::XLA  ||
      common_device_.type() == DeviceType::IPU  ||
      common_device_.type() == DeviceType::Lazy ||
      common_device_.type() == DeviceType::MAIA  ||
      common_device_.type() == DeviceType::HPU) return;

  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    if (op.is_const) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      op.data = const_cast<void*>(op.tensor_base().const_data_ptr());
    } else {
      op.data = op.tensor_base().mutable_data_ptr();
    }
  }

  // zero out offsets
  // If the tensor is a scalar, we leave room for it
  // So index translations in reduction can access
  // a valid value for the offset
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_ = DimVector(ndim_offsets, 0);
}
```

> permuste dims
> reorder_dimensions
> allocate_or_resize_outputs()
> coalesce_dimensions
**翻转维度和stride, 之后将reduce维度提前**



