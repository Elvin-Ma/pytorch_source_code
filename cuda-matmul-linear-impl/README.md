# 1 cuda matmul

- CUBLASS : CUDA Basic Linear Algebra Subprograms Lite (CUDA 基础线性代数子程序-轻量版)
- CUTLASS : CUDA Template Library Accelerated Software Stack

![alt text](image.png)

# 2 整个调用栈

```c++
#0  at::cuda::blas::bgemm_internal_cublas<float> (transa=110 'n', transb=110 'n', m=128, n=128, k=32, alpha=1, a=0x7ffea32c8000, lda=128, stridea=4096, b=0x7ffea3200000, ldb=32, strideb=4096, beta=0, c=0x7ffe8e000000, ldc=128, stridec=16384, num_batches=40)
    at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:489
#1  0x00007fffcecff2e2 in at::cuda::blas::bgemm_internal<float> (transa=110 'n', transb=110 'n', m=128, n=128, k=32, alpha=1, a=0x7ffea32c8000, lda=128, stridea=4096, b=0x7ffea3200000, ldb=32, strideb=4096, beta=0, c=0x7ffe8e000000, ldc=128, stridec=16384,
    num_batches=40) at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:620
#2  0x00007fffcecff8de in at::cuda::blas::bgemm<float> (transa=110 'n', transb=110 'n', m=128, n=128, k=32, alpha=1, a=0x7ffea32c8000, lda=128, stridea=4096, b=0x7ffea3200000, ldb=32, strideb=4096, beta=0, c=0x7ffe8e000000, ldc=128, stridec=16384, num_batches=40)
    at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:746
#3  0x00007fffcee199fc in <lambda()>::operator()(void) const (__closure=0x7fffffffcae0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:549
#4  0x00007fffcee1ab44 in <lambda()>::operator()(void) const (__closure=0x7fffffffccb0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:549
#5  0x00007fffcee1b73e in at::native::(anonymous namespace)::baddbmm_out_cuda_impl (result=..., self=..., batch1=..., batch2=..., beta=..., alpha=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:549
#6  0x00007fffcee1bb68 in at::native::structured_bmm_out_cuda::impl (this=0x7fffffffce20, batch1=..., batch2=..., result=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:616
#7  0x00007fffce76c35a in at::(anonymous namespace)::wrapper_CUDA_bmm (self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:4476
#8  0x00007fffce9d5cd4 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_bmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >::operator() (args#1=..., args#0=..., this=0x40ccca0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#9  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_bmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x40ccca0, args#0=..., args#1=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#10 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffce9d5c3a <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_bmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>, functor=0x40ccca0, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#11 0x00007fffde3321aa in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1084ec8) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#12 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&) const (
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#13 0x00007fffded121a2 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., currentDispatchKeySet=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#14 at::_ops::bmm::redispatch (dispatchKeySet=..., self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:1495
#15 0x00007fffe259fe5b in at::redispatch::bmm (dispatchKeySet=..., self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:1482
#16 0x00007fffe2464b92 in <lambda()>::operator()(void) const (__closure=0x7fffffffd280) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:8012
#17 0x00007fffe24650b2 in torch::autograd::VariableType::(anonymous namespace)::bmm (ks=..., self=..., mat2=...) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:8013
#18 0x00007fffe255db13 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::bmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >::operator() (args#2=..., args#1=..., args#0=..., this=0x21c34f0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#19 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::bmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x21c34f0,
    dispatchKeySet=..., args#0=..., args#1=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#20 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffe255da5d <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::bmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>, functor=0x21c34f0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#21 0x00007fffded11f32 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1085a48) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#22 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#23 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::call(at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#24 at::_ops::bmm::call (self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:1488
#25 0x00007ffff4eea58a in at::Tensor::bmm (this=0x7fffffffd620, mat2=...) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:1989
#26 0x00007fffdd9965ac in at::native::_matmul_impl (out=..., tensor1=..., tensor2=...) at /root/projects/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:2163
#27 0x00007fffdd996bd7 in at::native::matmul (tensor1=..., tensor2=...) at /root/projects/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:2183
#28 0x00007fffdf95d8da in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul (self=..., other=...) at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp:2766
#29 0x00007fffdfa612b4 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >::operator() (args#1=..., args#0=..., this=0x25accf0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#30 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x25accf0, args#0=..., args#1=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#31 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffdfa6121a <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_Composit--Type <RET> for more, q to quit, c to continue without paging--
eImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>,
    functor=0x25accf0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#32 0x00007fffdeff069e in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1138008) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#33 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#34 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::call(at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#35 at::_ops::matmul::call (self=..., other=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:2769
#36 0x00007ffff4eed57a in at::Tensor::matmul (this=0x7fffffffdb80, other=...) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:2899
#37 0x00007ffff4f43f4a in <lambda(const at::Tensor&, const at::Tensor&)>::operator()(const at::Tensor &, const at::Tensor &) const (__closure=0x7fffffffdb78, self=..., other=...) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_0.cpp:4921
#38 0x00007ffff4f442c2 in torch::autograd::THPVariable_matmul (self_=0x0, args=0x7ffff79ffa40, kwargs=0x0) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_0.cpp:4923
```

# 3 重点函数

## 3.1 CUDA bmm 算子注册

```c++
// */pytorch/build/aten/src/ATen/RegisterCUDA.cpp: 4469
at::Tensor wrapper_CUDA_bmm(const at::Tensor & self, const at::Tensor & mat2) {
std::optional<Device> common_device = std::nullopt;
(void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_CUDA_bmm", "self");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper_CUDA_bmm", "mat2");
structured_bmm_out_cuda_functional op;
op.meta(self, mat2);
op.impl(self, mat2, op.outputs_[0]); // 这里是真正执行的kernel, 具体实现见下节
return std::move(op.outputs_[0]);
}

...

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
m.impl("bmm", TORCH_FN(wrapper_CUDA_bmm));
}
```

## 3.2 cuda bmm 算子impl处

```c++
// */pytorch/aten/src/ATen/native/cuda/Blas.cpp
TORCH_IMPL_FUNC(bmm_out_cuda)(const Tensor& batch1, const Tensor& batch2, const Tensor &result) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
    NoNamesGuard guard;
    baddbmm_out_cuda_impl(result, result, batch1, batch2, beta, alpha);
  }
}
```

## 3.3 baddbmm_out_cuda_impl 实现

根据batch调用 CUDABlas.h 中 at::cuda::blas::gemm 或 at::cuda::blas::bgemm.

```c++
// */pytorch/aten/src/ATen/native/cuda/Blas.cpp
const Tensor& baddbmm_out_cuda_impl(const Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) || (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = resolve_conj_if_indicated(result, true);
  } else if ((result_strides[2] == 1) &&
    (result_sizes[1] == 1 || (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ = c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2).clone(at::MemoryFormat::Contiguous).transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda = 0, ldb = 0, ldc = 0;
  bool transpose_batch1 = false, transpose_batch2 = false;
  auto batch1_ = prepare_batch_matrix_for_cublas(transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
  auto batch2_ = prepare_batch_matrix_for_cublas(transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "baddbmm_cuda", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t alpha_val = alpha.to<opmath_t>();
    opmath_t beta_val = beta.to<opmath_t>();
    const scalar_t* batch1_ptr = batch1_->const_data_ptr<scalar_t>();
    const scalar_t* batch2_ptr = batch2_->const_data_ptr<scalar_t>();
    scalar_t* result_ptr = result_->mutable_data_ptr<scalar_t>();
    const auto transa = transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n';
    const auto transb = transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n';
    // If batch is 1 call gemm rather than bgemm
    if (num_batches == 1) {
      at::cuda::blas::gemm<scalar_t>(
          transa, transb,
          m, n, k,
          alpha_val,
          batch1_ptr, lda,
          batch2_ptr, ldb,
          beta_val,
          result_ptr, ldc);
    } else {
      at::cuda::blas::bgemm<scalar_t>(
        transa, transb,
        m, n, k,
        alpha_val,
        batch1_ptr, lda, batch1_->strides()[0],
        batch2_ptr, ldb, batch2_->strides()[0],
        beta_val,
        result_ptr, ldc, result_->strides()[0],
        num_batches
      );
   }
  });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}
```

## 3.4 at::cuda::blas::bgemm

```c++
template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    bgemm_internal<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}


template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
  else {
    bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}
```

## 3.5 调度到bgemm_internal 里

```c++
template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    bgemm_internal_cublaslt<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}
```

## 3.6 cublass api 的调用

**根据数据类型来判断是用TensorCore 还是 SIMT.**

```c++
template <>
void bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}
```

- BFloat16 会调度到TensorCore : CUBLAS_GEMM_DEFAULT_TENSOR_OP 指定用TensorCore.

```c++
template <>
void bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                  opa, opb, (int)m, (int)n, (int)k,
                                  (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                                  b, CUDA_R_16BF, (int)ldb, strideb,
                                  (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
                                  (int)num_batches,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
```

# 4 torch.matmul 4维 x 2维

```c++
#0  at::cuda::blas::gemm_internal_cublas<c10::BFloat16> (transa=110 'n', transb=110 'n', m=128, n=5120, k=32, alpha=1, a=0x7ffea3204000, lda=128, b=0x7ffea32a0000, ldb=32, beta=0, c=0x7ffe8e000000, ldc=128) at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:959
#1  0x00007fffced02ab5 in at::cuda::blas::gemm_internal<c10::BFloat16> (transa=110 'n', transb=110 'n', m=128, n=5120, k=32, alpha=1, a=0x7ffea3204000, lda=128, b=0x7ffea32a0000, ldb=32, beta=0, c=0x7ffe8e000000, ldc=128)
    at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:1100
#2  0x00007fffced0302d in at::cuda::blas::gemm<c10::BFloat16> (transa=110 'n', transb=110 'n', m=128, n=5120, k=32, alpha=1, a=0x7ffea3204000, lda=128, b=0x7ffea32a0000, ldb=32, beta=0, c=0x7ffe8e000000, ldc=128)
    at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:1207
#3  0x00007fffcee182ab in <lambda()>::operator()(void) const (__closure=0x7fffffffcaa0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:449
#4  0x00007fffcee184a4 in <lambda()>::operator()(void) const (__closure=0x7fffffffcbb0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:449
#5  0x00007fffcee19132 in at::native::(anonymous namespace)::addmm_out_cuda_impl (result=..., self=..., mat1=..., mat2=..., beta=..., alpha=..., activation=at::native::(anonymous namespace)::Activation::None)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:449
#6  0x00007fffcee1b9ae in at::native::structured_mm_out_cuda::impl (this=0x7fffffffcdc0, self=..., mat2=..., result=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:601
#7  0x00007fffce7a3356 in at::(anonymous namespace)::wrapper_CUDA_mm (self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:12178
#8  0x00007fffce9ef639 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_mm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >::operator() (args#1=..., args#0=..., this=0x204a8f0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#9  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_mm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x204a8f0, args#0=..., args#1=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#10 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffce9ef59f <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::wrapper_CUDA_mm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>, functor=0x204a8f0, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#11 0x00007fffde3321aa in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x107b638) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#12 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&) const (
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#13 0x00007fffded4fc66 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., currentDispatchKeySet=..., this=<optimized out>)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#14 at::_ops::mm::redispatch (dispatchKeySet=..., self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:3469
#15 0x00007fffe25a0df1 in at::redispatch::mm (dispatchKeySet=..., self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:5222
#16 0x00007fffe24afdbe in <lambda()>::operator()(void) const (__closure=0x7fffffffd280) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:13514
#17 0x00007fffe24b051e in torch::autograd::VariableType::(anonymous namespace)::mm (ks=..., self=..., mat2=...) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_3.cpp:13515
#18 0x00007fffe25693f1 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::mm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >::operator() (args#2=..., args#1=..., args#0=..., this=0x3213320) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#19 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::mm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x3213320,
    dispatchKeySet=..., args#0=..., args#1=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613
#20 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffe256933b <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&), torch::autograd::VariableType::(anonymous namespace)::mm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>, functor=0x3213320, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#21 0x00007fffded4f9f6 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x107c1b8) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#22 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#23 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::call(at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#24 at::_ops::mm::call (self=..., mat2=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_3.cpp:3462
#25 0x00007ffff4eedd0e in at::Tensor::mm (this=0x7fffffffd5e0, mat2=...) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:2999
#26 0x00007fffdd99590a in at::native::_matmul_impl (out=..., tensor1=..., tensor2=...) at /root/projects/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:2072
#27 0x00007fffdd996bd7 in at::native::matmul (tensor1=..., tensor2=...) at /root/projects/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:2183
#28 0x00007fffdf95d8da in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul (self=..., other=...) at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp:2766
#29 0x00007fffdfa612b4 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >::operator() (args#1=..., args#0=..., this=0x25ac9c0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#30 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &) (functor=0x25ac9c0, args#0=..., args#1=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#31 0x00007fffde4a291c in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&> (
    unboxed_kernel_func=0x7fffdfa6121a <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&), at::(anonymous namespace)::(anonymous namespace)::wrapper_Composit--Type <RET> for more, q to quit, c to continue without paging--
eImplicitAutograd__matmul>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&> >, at::Tensor(const at::Tensor&, const at::Tensor&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &)>,
    functor=0x25ac9c0, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#32 0x00007fffdeff069e in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&> (dispatchKeySet=..., opHandle=..., this=0x1137a38) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#33 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)> const&, at::Tensor const&, at::Tensor const&) const (op=...,
    this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#34 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&)>::call(at::Tensor const&, at::Tensor const&) const (args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#35 at::_ops::matmul::call (self=..., other=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_4.cpp:2769
#36 0x00007ffff4eed57a in at::Tensor::matmul (this=0x7fffffffdb80, other=...) at /root/projects/pytorch/build/aten/src/ATen/core/TensorBody.h:2899
#37 0x00007ffff4f43f4a in <lambda(const at::Tensor&, const at::Tensor&)>::operator()(const at::Tensor &, const at::Tensor &) const (__closure=0x7fffffffdb78, self=..., other=...) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_0.cpp:4921
#38 0x00007ffff4f442c2 in torch::autograd::THPVariable_matmul (self_=0x0, args=0x7ffff79ffa40, kwargs=0x0) at /root/projects/pytorch/torch/csrc/autograd/generated/python_torch_functions_0.cpp:4923
#39 0x000000000054c584 in cfunction_call (func=0x7ffff7467d30, args=0x7ffff79ffa40, kwargs=0x0) at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
```

# 5 torch.nn.Linear 调用栈会不同吗？

## 5.1 torch.nn.Linear 调用栈 和 torch.matmul 区别

- torch.nn.Linear 完整函数调用栈 和 torch.matmul 二维情况下调用栈基本相同，都会调度到Blas.cpp::addmm_out_cuda_impl;

**在addmm_out_cuda_impl 中分布进行调度** <br>

```c++
Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, targs);

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  IntArrayRef self__sizes;
  bool useLtInterface = false;
#if defined(USE_ROCM)
  // When hipBLASLt is not supported on the architecture,
  // disable_addmm_cuda_lt will always be to set to true
  static bool disable_addmm_cuda_lt =
    !isSupportedHipLtROCmArch(self.device().index()) || getDisableAddmmCudaLt();
#else
  static bool disable_addmm_cuda_lt = getDisableAddmmCudaLt();
#endif
  at::ScalarType scalar_type = self.scalar_type();
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self) {
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11040)) || defined(USE_ROCM)
    // Strangely, if mat2 has only 1 row or column, we get
    // CUBLAS_STATUS_INVALID_VALUE error from cublasLtMatmulAlgoGetHeuristic.
    // self.dim() == 1 && result.dim() == 2 && self.sizes()[0] == mat2_sizes[1]
    // is to use lt interface only when self is bias.
    // for cuda 11.4, cublasLtMatmul is activated
    // the last two conditions is to skip 16b transA and non-trans-B having
    // leading dim >> rows when they are sliced from a large tensor
    // see fbcode/caffe2/test/test_linalg.py:test_corner_cases_of_cublasltmatmul
    if (!disable_addmm_cuda_lt) {
      useLtInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
          result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
          self.is_contiguous() && result.is_contiguous() &&
#ifdef USE_ROCM
          (scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#else
          (scalar_type == at::ScalarType::Double ||
           scalar_type == at::ScalarType::Float ||
           scalar_type == at::ScalarType::Half ||
           scalar_type == at::ScalarType::BFloat16) &&
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12010 || defined(USE_ROCM))
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1;
#else
          mat2_sizes[0] > 1 && mat2_sizes[1] > 1 &&
          mat2_sizes[0] < 65535 * 32 && mat2_sizes[1] < 65535 * 32 &&
          mat1_sizes[0] < 65535 * 32 && mat1_sizes[1] < 65535 * 32 &&
          // avoid leading dim >> rows bugs
          ((mat1.strides()[0] == 1 && mat1.strides()[1] == mat1_sizes[0]) ||
           (mat1.strides()[1] == 1 && mat1.strides()[0] == mat1_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16)) &&
          ((mat2.strides()[0] == 1 && mat2.strides()[1] == mat2_sizes[0]) ||
           (mat2.strides()[1] == 1 && mat2.strides()[0] == mat2_sizes[1]) ||
           (scalar_type != at::ScalarType::Half &&
            scalar_type != at::ScalarType::BFloat16));
#endif
    }
#endif
    if (!useLtInterface) {
      self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    }
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&result != &self) {
    at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
    if (beta.toComplexDouble() != 0.0 && !useLtInterface) {
      at::native::copy_(result, *self_);
    }
  }


  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  cublasCommonArgs args(mat1, mat2, result);

  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and infs
    // should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    // TODO: We could squeeze some perf by calling at::cuda::mul_out here instead, to bypass the dispatcher.
    // That requires some fixing some internal build dependencies though.
    return at::mul_out(
        result,
        self.expand(result.sizes()),
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            std::nullopt /* layout */,
            at::kCPU,
            std::nullopt /* pin_memory */));
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!args.result->is_conj());

  if (useLtInterface) {
#if defined(USE_ROCM)
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          launchTunableGemmAndBias<scalar_t>(
              args,
              alpha,
              (&result != &self) ? self.const_data_ptr<scalar_t>() : nullptr,
              activation_to_gemm_and_blas_arg(activation));
        }
        else {
          at::cuda::blas::gemm_and_bias<scalar_t>(
              args.transa == 't',
              args.transb == 't',
              args.m,
              args.n,
              args.k,
              alpha.to<at::opmath_type<scalar_t>>(),
              args.mata->const_data_ptr<scalar_t>(),
              args.lda,
              args.matb->const_data_ptr<scalar_t>(),
              args.ldb,
              // This condition is needed for mm case on ROCm for hipblasLt path.
              // Passing the bias ptr as null to avoid accuracy issues for mm case.
              (&result != &self) ? self.const_data_ptr<scalar_t>() : nullptr,
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_to_gemm_and_blas_arg(activation)
          );
        }});
#else
    auto activation_epilogue = activation_to_gemm_and_blas_arg(activation);
#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11080))
    // GELU is not supported (and does not compile!) prior
    // to CUDA 11.4. Have observed accuracy issues with
    // GELU epilogue in 11.4; disabling the GELU epilogue
    // path for CUDA version < 11.8.
    if (activation == Activation::GELU)
      activation_epilogue = cuda::blas::GEMMAndBiasActivationEpilogue::None;
#endif

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda_lt",
        [&] {
        auto tuning_ctx = at::cuda::tunable::getTuningContext();
        if (tuning_ctx->IsTunableOpEnabled()) {
          launchTunableGemmAndBias<scalar_t>(
              args,
              alpha,
              self.const_data_ptr<scalar_t>(),
              activation_epilogue);
        }
        else {
          at::cuda::blas::gemm_and_bias<scalar_t>(
              args.transa == 't',
              args.transb == 't',
              args.m,
              args.n,
              args.k,
              alpha.to<at::opmath_type<scalar_t>>(),
              args.mata->const_data_ptr<scalar_t>(),
              args.lda,
              args.matb->const_data_ptr<scalar_t>(),
              args.ldb,
              self.const_data_ptr<scalar_t>(),
              args.result->data_ptr<scalar_t>(),
              args.result_ld,
              activation_epilogue
          );
        }});
#endif
  } else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        scalar_type,
        "addmm_cuda",
        [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t alpha_val = alpha.to<opmath_t>();
          opmath_t beta_val = beta.to<opmath_t>();
          const scalar_t* mat1_ptr = args.mata->const_data_ptr<scalar_t>();
          const scalar_t* mat2_ptr = args.matb->const_data_ptr<scalar_t>();
          scalar_t* result_ptr = args.result->mutable_data_ptr<scalar_t>();
          at::cuda::blas::gemm<scalar_t>(
              args.transa,
              args.transb,
              args.m,
              args.n,
              args.k,
              alpha_val,
              mat1_ptr,
              args.lda,
              mat2_ptr,
              args.ldb,
              beta_val,
              result_ptr,
              args.result_ld);
        });
    switch (activation) {
      case Activation::RELU:
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        at::relu_(const_cast<Tensor&>(*args.result));
        break;
      case Activation::GELU:
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        at::gelu_(const_cast<Tensor&>(*args.result), "tanh");
        break;
      default: break;
    }
  }

// Preprocessor gate here needs to match the inverse of the check
// gating activation_to_gemm_and_blas_arg above; here we are manually
// performing a post-GELU because we weren't able to use the GELU
// epilogue above.
#if !(defined(CUDA_VERSION) && CUDA_VERSION >= 11080) && !defined(USE_ROCM)
  if (useLtInterface && activation == Activation::GELU) {
    at::gelu_(const_cast<Tensor&>(*args.result), "tanh");
  }
#endif

  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
  }
  return result;
}
```

## 5.2 torch.nn.Linear 完整函数调用

```c++
#0  at::cuda::blas::gemm_and_bias<c10::BFloat16> (transpose_mat1=true, transpose_mat2=false, m=128, n=5120, k=32, alpha_val=1, mat1_ptr=0x7ffea3204200, mat1_ld=32, mat2_ptr=0x7ffea32a0000, mat2_ld=32, bias=0x7ffea3200000, result_ptr=0x7ffe8e000000, result_ld=128,
    activation=at::cuda::blas::None) at /root/projects/pytorch/aten/src/ATen/cuda/CUDABlas.cpp:1311
#1  0x00007fffcee17867 in <lambda()>::operator()(void) const (__closure=0x7fffffffbee0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:414
#2  0x00007fffcee17a30 in <lambda()>::operator()(void) const (__closure=0x7fffffffbff0) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:414
#3  0x00007fffcee190e6 in at::native::(anonymous namespace)::addmm_out_cuda_impl (result=..., self=..., mat1=..., mat2=..., beta=..., alpha=..., activation=at::native::(anonymous namespace)::Activation::None)
    at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:414
#4  0x00007fffcee1b8ba in at::native::structured_addmm_out_cuda::impl (this=0x7fffffffc1d0, self=..., mat1=..., mat2=..., beta=..., alpha=..., result=...) at /root/projects/pytorch/aten/src/ATen/native/cuda/Blas.cpp:591
#5  0x00007fffce7cb284 in at::(anonymous namespace)::wrapper_CUDA_addmm (self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/RegisterCUDA.cpp:17856
#6  0x00007fffcea01d48 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_addmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >::operator() (args#4=..., args#3=..., args#2=..., args#1=..., args#0=..., this=0x41386f0)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#7  c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_addmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >, at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::Scalar &, const c10::Scalar &) (functor=0x41386f0, args#0=..., args#1=..., args#2=..., args#3=..., args#4=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#8  0x00007fffde4cb040 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&> (
    unboxed_kernel_func=0x7fffcea01c04 <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), at::(anonymous namespace)::wrapper_CUDA_addmm>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >, at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::Scalar &, const c10::Scalar &)>, functor=0x41386f0, dispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#9  0x00007fffde352d39 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&> (dispatchKeySet=..., opHandle=..., this=0x1080b28)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#10 c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)> const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>, op=..., currentDispatchKeySet=...)
    at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:714
#11 0x00007fffde215ab0 in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)>::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (args#4=..., args#3=..., args#2=..., args#1=..., args#0=..., currentDispatchKeySet=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:536
#12 at::_ops::addmm::redispatch (dispatchKeySet=..., self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_0.cpp:6219
#13 0x00007fffe1ed8400 in at::redispatch::addmm (dispatchKeySet=..., self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/RedispatchFunctions.h:8672
#14 0x00007fffe1daad5a in <lambda()>::operator()(void) const (__closure=0x7fffffffc930) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_0.cpp:7222
#15 0x00007fffe1dab89e in torch::autograd::VariableType::(anonymous namespace)::addmm (ks=..., self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/torch/csrc/autograd/generated/VariableType_0.cpp:7223
#16 0x00007fffe1e956d8 in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), torch::autograd::VariableType::(anonymous namespace)::addmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >::operator() (args#5=..., args#4=..., args#3=..., args#2=..., args#1=..., args#0=...,
    this=0x193b9b0) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#17 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), torch::autograd::VariableType::(anonymous namespace)::addmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::Scalar &, const c10::Scalar &) (functor=0x193b9b0, dispatchKeySet=..., args#0=...,
    args#1=..., args#2=..., args#3=..., args#4=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613

#18 0x00007fffde4cb040 in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&> (
    unboxed_kernel_func=0x7fffe1e9555d <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&), torch::autograd::VariableType::(anonymous namespace)::addmm>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&> >, at::Tensor(c10::DispatchKeySet, const at::Tensor&, const at::Tensor&, const at::Tensor&, const c10::Scalar&, const c10::Scalar&)>::call(c10::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::Scalar &, const c10::Scalar &)>, functor=0x193b9b0,
    dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#19 0x00007fffde215761 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&> (dispatchKeySet=..., opHandle=..., this=0x10816a8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#20 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)> const&, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#21 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)>::call(at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (args#4=...,
    args#3=..., args#2=..., args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#22 at::_ops::addmm::call (self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_0.cpp:6212
#23 0x00007fffdd5613e4 in at::addmm (self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/ops/addmm.h:36
#24 0x00007fffdd96e9d5 in at::native::_flatten_nd_linear (input=..., weight=..., bias=...) at /root/projects/pytorch/aten/src/ATen/native/Linear.cpp:65
#25 0x00007fffdd96f010 in at::native::linear (input=..., weight=..., bias_opt=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/aten/src/ATen/native/Linear.cpp:104
#26 0x00007fffdf95d154 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__linear (input=..., weight=..., bias=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp:2619
--Type <RET> for more, q to quit, c to continue without paging--
    dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76                                                                                                                                                                   [13/6623]
#19 0x00007fffde215761 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&> (dispatchKeySet=..., opHandle=..., this=0x10816a8)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#20 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)> co
nst&, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#21 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&)>::call(at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&) const (args#4=...,
    args#3=..., args#2=..., args#1=..., args#0=..., this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#22 at::_ops::addmm::call (self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/Operators_0.cpp:6212
#23 0x00007fffdd5613e4 in at::addmm (self=..., mat1=..., mat2=..., beta=..., alpha=...) at /root/projects/pytorch/build/aten/src/ATen/ops/addmm.h:36
#24 0x00007fffdd96e9d5 in at::native::_flatten_nd_linear (input=..., weight=..., bias=...) at /root/projects/pytorch/aten/src/ATen/native/Linear.cpp:65
#25 0x00007fffdd96f010 in at::native::linear (input=..., weight=..., bias_opt=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/aten/src/ATen/native/Linear.cpp:104
#26 0x00007fffdf95d154 in at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__linear (input=..., weight=..., bias=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp:2619
--Type <RET> for more, q to quit, c to continue without paging--
#27 0x00007fffdfa5ecce in c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__l
inear>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&> >::operator() (args#2=std::optional<at::Tensor> = {...}, args#1=..., args#0=..., this=0x2595c60)
    at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17
#28 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&), at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeI
mplicitAutograd__linear>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&> >, at::Tensor(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&)>::call(c10::OperatorKernel *, c10::DispatchKey
Set, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &) (functor=0x2595c60, args#0=..., args#1=..., args#2=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579
#29 0x00007fffde4b4e5e in c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&> (
    unboxed_kernel_func=0x7fffdfa5ec0c <c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&), at::(anonymous namespace)::(anon
ymous namespace)::wrapper_CompositeImplicitAutograd__linear>, at::Tensor, c10::guts::typelist::typelist<const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&> >, at::Tensor(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&)>::call(c10
::OperatorKernel *, c10::DispatchKeySet, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &)>, functor=0x2595c60, dispatchKeySet=...) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76
#30 0x00007fffde1bca63 in c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&> (dispatchKeySet=..., opHandle=..., this=0x122e988) at /root/projects/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:149
#31 c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&)> const&, at::Tensor const&, at::Tensor const&, std::o
ptional<at::Tensor> const&) const (op=..., this=0x7ffff39c9960 <c10::Dispatcher::realSingleton()::_singleton>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:698
#32 c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&)>::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&) const (args#2=std::optional<at::Tensor> = {...}, args#1=..., args#0=...,
    this=<optimized out>) at /root/projects/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:531
#33 at::_ops::linear::call (input=..., weight=..., bias=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/build/aten/src/ATen/Operators_0.cpp:3146
#34 0x00007ffff512c3fe in at::linear (input=..., weight=..., bias=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/build/aten/src/ATen/ops/linear.h:27
#35 0x00007ffff510e86f in <lambda(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&)>::operator()(const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &) const (__closure=0x7fffffffd328, input=..., weight=...,
    bias=std::optional<at::Tensor> = {...}) at /root/projects/pytorch/torch/csrc/autograd/generated/python_nn_functions.cpp:1780
#36 0x00007ffff510ec0d in torch::autograd::THPVariable_linear (self_=0x7fff9619c900, args=0x7ffff79dfe00, kwargs=0x0) at /root/projects/pytorch/torch/csrc/autograd/generated/python_nn_functions.cpp:1782
#37 0x000000000054c584 in cfunction_call (func=0x7fff9619d8a0, args=0x7ffff79dfe00, kwargs=0x0) at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
```

## 5.3 主要调度点
## 5.3.1 CUDA kernel 调用处

```c++
at::Tensor wrapper_CUDA_addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
std::optional<Device> common_device = std::nullopt;
(void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_CUDA_addmm", "self");
  c10::impl::check_and_update_common_device(common_device, mat1, "wrapper_CUDA_addmm", "mat1");
  c10::impl::check_and_update_common_device(common_device, mat2, "wrapper_CUDA_addmm", "mat2");
structured_addmm_out_cuda_functional op;
op.meta(self, mat1, mat2, beta, alpha);
op.impl(self, mat1, mat2, beta, alpha, op.outputs_[0]);
return std::move(op.outputs_[0]);
}
```

## 5.3.2 Blas.cpp

- Basic Linear Algebra Subprograms ： 基本线性代数子程序;

```c++
Tensor& addmm_out_cuda_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Activation activation=Activation::None) {
  // Make sure to keep addmm_cuda below in sync with this code; it
  // preflights a check to try to avoid actually needing to call
  // expand().
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
  TORCH_CHECK(
    mat1.dtype() == mat2.dtype(),
    "expected mat1 and mat2 to have the same dtype, but got: ", mat1.dtype(), " != ", mat2.dtype()
  )

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{{result, "out", 0}, {self, "self", 1}, {mat1, "mat1", 2}, {mat2, "mat2", 3}};
  checkAllSameGPU(__func__, targs);

  ...

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "addmm_cuda_lt",
      [&] {
      auto tuning_ctx = at::cuda::tunable::getTuningContext();
      if (tuning_ctx->IsTunableOpEnabled()) {
        launchTunableGemmAndBias<scalar_t>(
            args,
            alpha,
            self.const_data_ptr<scalar_t>(),
            activation_epilogue);
      }
      else {
        at::cuda::blas::gemm_and_bias<scalar_t>(
            args.transa == 't',
            args.transb == 't',
            args.m,
            args.n,
            args.k,
            alpha.to<at::opmath_type<scalar_t>>(),
            args.mata->const_data_ptr<scalar_t>(),
            args.lda,
            args.matb->const_data_ptr<scalar_t>(),
            args.ldb,
            self.const_data_ptr<scalar_t>(),
            args.result->data_ptr<scalar_t>(),
            args.result_ld,
            activation_epilogue
        );
      }});

  ...
```

## 5.3.3 gemm_and_bias

```c++
template <typename Dtype>
void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<Dtype> alpha_val,
    const Dtype* mat1_ptr,
    int64_t mat1_ld,
    const Dtype* mat2_ptr,
    int64_t mat2_ld,
    const Dtype* bias,
    Dtype* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation) {
  using opmath_t = at::opmath_type<Dtype>;
  opmath_t beta_val = 0; // bias is added in epilogue

  ...

  cublasStatus_t cublasStatus = cublasLtMatmul(
    ltHandle,
    computeDesc.descriptor(),
    &alpha_val,
    mat1_ptr,
    Adesc.descriptor(),
    mat2_ptr,
    Bdesc.descriptor(),
    &beta_val,
    result_ptr,
    Cdesc.descriptor(),
    result_ptr,
    Cdesc.descriptor(),
    &heuristicResult.algo,
    workspace.mutable_data_ptr(),
    workspaceSize,
    at::cuda::getCurrentCUDAStream());
  }
```

# 6 cublas 在那里进行转置的呢？

$$C_{mn} = A_{mk} * B_{nk} 等价于 C_{nm}^T = B_{nk}^T * A_{km}^T$$

将AB矩阵位置互换, 按照互换后的dim来指定m,k,n即可:

```c++
#include <cublas_v2.h>
 #include <cuda_runtime.h>
 #include <iostream>

 int main() {
     cublasHandle_t handle;
     cublasCreate(&handle);

     int m = 16, n = 32, k = 64;
     float alpha = 1.0f;
     float beta = 0.0f;
     std::vector<float> A(m*k, 1);
     std::vector<float> B(k*n, 2);
     std::vecotr<float> C(m*n, 0);

     float *d_A, *d_B, *d_C;
     cudaMalloc((void**)&d_A, m*k*sizeof(float));
     cudaMalloc((void**)&d_B, k*n*sizeof(float));
     cudaMalloc((void**)&d_C, m*n*sizeof(float));

     cudaMemcpy(d_A, A, m*k*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, B, k*n*sizeof(float), cudaMemcpyHostToDevice);

     cublasSgemm(handle,
                 CUBLAS_OP_N, // 不进行转置
                 CUBLAS_OP_N, // 不进行转置
                 n, m, k,     // 注意这里是 n, m, k 而不是 m, n, k
                 &alpha,
                 d_B,         // 先传
                 n,
                 d_A,
                 k,
                 &beta,
                 d_C,
                 n);

     cudaMemcpy(C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);

     std::cout << "结果矩阵 C:" << std::endl;
     for (int i = 0; i < m * n; i++) {
         std::cout << C[i] << " ";
         if ((i + 1) % n == 0) std::cout << std::endl;
     }

     cublasDestroy(handle);
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);

     return 0;
 }
```

- 源码接口

```c++
// /usr/local/cuda/targets/x86_64-linux/include/cublas_api.h
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const float* alpha,
                                                     const float* A,
                                                     int lda,
                                                     const float* B,
                                                     int ldb,
                                                     const float* beta,
                                                     float* C,
                                                     int ldc);
```


# 7 Q & A
- 1. 4维 和 2维的矩阵相乘，kernel 如何处理呢？ <br>

> [4, 10, 128, 32] x [32, 128] ==> [5120, 32] x [32, 128] = [5120, 128]
> 走的是addmm_out_cuda_impl

- 2. 4 维和 3 维的矩阵相乘， kernel 如何处理？

> [4, 10, 128, 32] x [1, 32, 128] ==> [40, 128, 32] x [40, 128, 32] = [40, 128, 128]
> 可见右矩阵进行了broadcast

- 3. 进行broadcast 后，数据真实扩充 copy 了吗？

> 没有，只是shape 和 stride 变了而已，stride : {0, 128, 1}, batch 维度stride 为 0， 表示步长为0;

- 4. 调用cublas算子 什么时候走TensorCore 什么时候走 SIMT 呢？

> 根据Tensor的数据类型，最后会选择到不同的模板函数中，Bfloat16/Half 等会走TensorCore, float/double 会走SIMT.

- 5. Linear 算子和matmul 走的后端一致吗？

> Linear 最终CUDABlass.cpp 调度到 addmm_out_cuda_impl，matmul 会调度到: mm_out_cuda_impl

- 6. culbass 会调度到 cutlass 吗？

> torch.nn.Linear 会调度到cutlass 的 **cublasLtMatmul**，cuBLASLt 运行时（可能通过 cublasLtMatmulAlgoGetHeuristic 或类似机制）会根据你的矩阵尺寸、数据类型、布局、指针对齐、Compute Capability 等信息，从它内部注册的内核集合中选择一个最优的算法（cublasLtMatmulAlgo_t）来执行。这个被选中的内核极有**可能就是基于 CUTLASS 模板实例化、编译并高度优化过的**。

- 7. 其它cuBlass 接口有可能调度到cutlass 吗？

> cublasGemmEx 在Bfloat16 下调度到了：

```c++
Slice void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align8::Params) at 81.328 ms
```

> torch.nn.Linear 调度 cublasLtMatmul

```c++
Slice void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8>(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align8::Params) at 87.139 ms
```

- 8. cublass 和 cublassLT 有何区别呢？使用时如何选择呢？

**核心区别：定位、范围与设计哲学**

1.  **cuBLAS (传统 cuBLAS 库 - `libcublas.so`):**
    *   **定位：** **完整的 BLAS 标准实现。** 它的目标是提供符合标准的、稳定的、涵盖所有 BLAS 层级（Level 1: 向量操作, Level 2: 矩阵-向量操作, Level 3: 矩阵-矩阵操作）的功能。
    *   **范围：** 非常广泛。包含：
        *   Level 1: `cublas(I/D/S/C/Z)axpy`, `cublas(I/D/S/C/Z)dot`, `cublas(I/D/S/C/Z)scal`, `cublas(I/D/S/C/Z)nrm2` 等。
        *   Level 2: `cublas(I/D/S/C/Z)gemv`, `cublas(I/D/S/C/Z)symv`, `cublas(I/D/S/C/Z)trmv` 等。
        *   Level 3: `cublas(I/D/S/C/Z)gemm`, `cublas(I/D/S/C/Z)trsm`, `cublas(I/D/S/C/Z)symm`, `cublas(I/D/S/C/Z)trmm` 等。
        *   **扩展功能：** 批量版本 (如 `cublasGemmStridedBatchedEx`)，混合精度支持。
    *   **API 风格：** **传统的、基于函数参数的 API。** 大部分操作通过一个函数调用完成，所有参数（矩阵指针、维度、lda/ldb/ldc、alpha/beta 标量、操作类型等）都作为参数传递给该函数。相对直接，但配置选项有限。
    *   **灵活性：** **较低。** 主要提供标准 BLAS 操作，对核心 GEMM 算法的选择和精细控制能力较弱。虽然有一些选项（如 `CUBLAS_GEMM_DEFAULT`, `CUBLAS_GEMM_ALGO0` 到 `23` 等），但远不如 cuBLASLt 灵活。
    *   **性能：** 对于标准操作，性能通常很好，因为它内部也使用了高度优化的内核（包括基于 CUTLASS 的）。但对于需要特定调优或融合操作的非标准场景，潜力不如 cuBLASLt。
    *   **易用性：** **相对简单直接。** 对于实现标准 BLAS 操作非常方便。

2.  **cuBLASLt (轻量级 cuBLAS - `libcublasLt.so`):**
    *   **定位：** **专注于 GEMM（广义矩阵乘法）及其相关融合变体的高性能、高灵活性库。** 它不是完整的 BLAS 实现，而是专门为需要极致 GEMM 性能和定制化的场景（尤其是深度学习）设计的。
    *   **范围：** **狭窄但深入。** 几乎只围绕 `cublasLtMatmul` 这个核心函数展开。它通过提供强大的配置机制来扩展 GEMM 的功能。
    *   **API 风格：** **现代的、基于描述符的 API。** 核心思想是创建和配置多个**描述符对象**来描述：
        *   `cublasLtMatmulDesc_t`: 描述矩阵乘法操作本身（opA, opB, 计算类型，是否有 epilogue 操作等）。
        *   `cublasLtMatrixLayout_t`: 描述输入/输出矩阵的布局（数据类型、维度、ld、批处理 stride 或指针数组、向量化宽度等）。
        *   `cublasLtMatmulPreference_t`: 描述算法选择的偏好（如 tile 大小、拆分策略、数值特性等）。
        *   主函数 `cublasLtMatmul` 接收这些描述符作为参数。
    *   **灵活性：** **极高。** 这是 cuBLASLt 的核心价值：
        *   **算法选择：** 可以枚举 (`cublasLtMatmulAlgoGetHeuristic`, `cublasLtMatmulAlgoGetIds`)、初始化 (`cublasLtMatmulAlgoInit`)、配置 (`cublasLtMatmulAlgoConfigSetAttribute`) 并指定 (`cublasLtMatmulAlgo_t`) 具体的 GEMM 内核算法。这是进行深度性能调优的关键。
        *   **Epilogue 融合：** 在 GEMM 计算完成后，**直接在核函数内部**执行额外的逐点操作，避免了额外的核函数启动和内存读写开销。支持的 epilogue 操作包括：
            *   乘以标量 (alpha/beta 的扩展)
            *   加 Bias 向量 (常用于 DNN)
            *   应用激活函数 (ReLU, GELU, Sigmoid, 等等)
            *   进行量化操作
        *   **高级批处理：** 支持标准的 Strided Batch（连续存储）和 **Array of Pointers Batch**（非连续存储）。
        *   **精细控制：** 控制累加顺序、是否允许降低精度计算等数值特性。
        *   **混合精度支持：** 灵活配置输入、输出、计算（累加）数据类型。
    *   **性能：** 在 GEMM 操作上，尤其是需要特定调优或融合 epilogue 的场景下，**通常能达到比传统 cuBLAS GEMM 函数更高的性能**。它集成了针对最新硬件架构优化的算法（包括大量基于 CUTLASS 构建的内核）。
    *   **易用性：** **相对复杂。** 需要创建和管理多个描述符对象，学习曲线较陡峭。

**总结对比表：**

| 特性                | cuBLAS (传统)                                    | cuBLASLt (轻量级)                                  |
| :------------------ | :----------------------------------------------- | :------------------------------------------------ |
| **定位**            | 完整的 BLAS 标准实现 (Level 1, 2, 3)             | 专注于高性能、高灵活性的 GEMM 及其融合变体         |
| **范围**            | 广泛：向量、矩阵-向量、矩阵-矩阵操作             | 狭窄：几乎只有 GEMM (`cublasLtMatmul`) 及其配置    |
| **核心价值**        | 标准兼容性、稳定性、易用性                       | **极致 GEMM 性能、算法选择灵活性、Epilogue 融合** |
| **API 风格**        | 传统函数参数式                                   | 现代描述符对象式                                   |
| **灵活性**          | 较低 (标准 BLAS 参数)                            | **极高** (算法选择、Epilogue 融合、批处理模式、精细控制) |
| **GEMM 性能潜力**   | 良好 (标准操作)                                  | **通常更高** (尤其调优后或融合场景)                |
| **Epilogue 融合**   | 不支持 (需额外核函数)                            | **原生支持** (Bias, Activation, Scaling 等)        |
| **批处理模式**      | Strided Batch (连续)                             | **Strided Batch + Array of Pointers (非连续)**     |
| **算法选择/调优**   | 非常有限 (少数预定义 `CUBLAS_GEMM_ALGO*`)        | **强大** (枚举、启发式推荐、手动配置算法属性)      |
| **易用性**          | **较高** (函数调用直接)                          | 较低 (需管理描述符)                               |
| **典型应用场景**    | 通用科学计算，需要完整 BLAS 功能                 | **深度学习核心算子**，需要最优 GEMM 性能或融合操作 |

**如何选择？**

选择 `cuBLAS` 还是 `cuBLASLt` 主要取决于你的具体需求：

1.  **选择 `cuBLAS` 当：**
    *   你需要实现 **完整的 BLAS 功能**，包括 **Level 1 (向量)** 或 **Level 2 (矩阵-向量)** 操作。cuBLASLt **不提供**这些。
    *   你只需要进行 **标准的、不需要特殊调优的 Level 3 (矩阵-矩阵) 操作**，特别是 `gemm`, `trsm`, `symm` 等。
    *   你的应用对 **API 的简洁性和易用性** 要求很高，不想处理复杂的描述符设置。
    *   你的 GEMM 操作 **非常标准**，且性能已经满足要求，不需要追求极限优化。
    *   你不需要在 GEMM 后 **融合执行 Bias 加法或 Activation 函数** 等操作。

2.  **选择 `cuBLASLt` 当：**
    *   你的核心计算瓶颈是 **GEMM (矩阵乘法)**，并且你追求 **极致的性能**。
    *   你需要 **在 GEMM 之后立即融合执行其他操作**，特别是 **添加 Bias 向量** 或 **应用 Activation 函数 (ReLU, GELU 等)**。这是 cuBLASLt 的 **杀手锏**，能显著减少显存带宽和核函数启动开销。
    *   你需要 **对 GEMM 内核算法进行精细的选择和调优** 以适应特定的问题规模、数据类型或硬件。
    *   你使用了 **非标准的批处理模式**（如 Array of Pointers）。
    *   你主要工作在 **深度学习领域**，构建高性能的算子库或框架核心。cuBLASLt 是该领域的 **首选**。
    *   你愿意投入时间学习 **更复杂的基于描述符的 API** 以获得性能和灵活性优势。

**简单决策树：**

1.  **你需要做向量操作 (Level 1) 或 矩阵-向量操作 (Level 2) 吗？**
    *   **是** -> 必须用 **cuBLAS** (cuBLASLt 不支持)。
    *   **否** -> 进入下一步。
2.  **你的主要操作是 GEMM (`gemm`) 吗？并且你需要以下任一功能吗？**
    *   在 GEMM 后**融合 Bias 或 Activation**？
    *   对 GEMM 进行**深度算法选择和性能调优**？
    *   使用**非连续的批处理 (Array of Pointers)**？
    *   追求**该 GEMM 操作的绝对最高性能**？
    *   **是** (任一) -> 强烈推荐使用 **cuBLASLt**。
    *   **否** -> **cuBLAS** 的 `cublasGemmEx` / `cublasGemmStridedBatchedEx` 可能更简单够用。
3.  **你需要 `trsm`, `symm`, `trmm` 等非-GEMM 的 Level 3 操作吗？**
    *   **是** -> 主要用 **cuBLAS** (cuBLASLt 不直接提供这些)。
    *   **否** -> 参考第 2 步。

**总结：**

*   **cuBLAS** 是你的 **通用 BLAS 工具箱**。当你需要标准的向量、矩阵-向量或矩阵-矩阵操作时，它是可靠且相对易用的选择。
*   **cuBLASLt** 是你的 **GEMM 手术刀**。当你需要榨干硬件性能、实现复杂的融合算子（GEMM+Bias+Activation）、或对 GEMM 内核进行精细控制时，它是更强大（但也更复杂）的工具，尤其是在深度学习等高性能计算领域。对于纯 GEMM 密集型任务，cuBLASLt 通常是性能最优的选择。