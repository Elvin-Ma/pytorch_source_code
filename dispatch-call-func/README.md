# dispatcher

```c++
// dispatcher.h
// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
    detail::_print_dispatch_trace("[call]", toString(op.operator_name()), dispatchKeySet);
  }
#endif
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {
    return callWithDispatchKeySlowPath<Return, Args...>(op, *step_callbacks, dispatchKeySet, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING

#ifdef FBCODE_CAFFE2
  if(profilingOperatorEvents()) {
    struct FireOpRAII {
       FireOpRAII(at::RecordFunction::schema_ref_t schema_ref) : schema_ref_(schema_ref) {
           fireOpStartUSDT(schema_ref);
        }
       ~FireOpRAII() { fireOpEndUSDT(schema_ref_); }
       at::RecordFunction::schema_ref_t schema_ref_;
    } event(op.schema());
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  } else {
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  }
#else
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
#endif // FBCODE_CAFFE2
}

// See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  // do not use RecordFunction on redispatch
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
    detail::_print_dispatch_trace("[redispatch]", toString(op.operator_name()), currentDispatchKeySet);
  }
#endif
  const KernelFunction& kernel = op.operatorDef_->op.lookup(currentDispatchKeySet);
  return kernel.template call<Return, Args...>(op, currentDispatchKeySet, std::forward<Args>(args)...);
}
```

# 2 launch func kernel

**真正的算子调用处** <br>

```c++
// pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:143
template <class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Args... args) const {
  // note: Args above is intentionally not Args&&. We don't want perfect
  // forwarding, which would require Args to be deduced, but instead we
  // want callers to explicitly specify the Args.

  if constexpr (std::disjunction_v<has_symint<Args>...>) {
    if (sym_unboxed_kernel_func_ != nullptr) {
      auto* functor = boxed_kernel_func_.getFunctor();
      return callUnboxedKernelFunction<Return, Args...>(
          sym_unboxed_kernel_func_,
          functor,
          dispatchKeySet,
          std::forward<Args>(args)...);
    }

    if (unboxed_kernel_func_ != nullptr) {
      auto* functor = boxed_kernel_func_.getFunctor();
      return callUnboxedKernelFunction<
          Return,
          typename remove_symint<Args>::type...>(
          unboxed_kernel_func_,
          functor,
          dispatchKeySet,
          unpackSymInt<Args>(args)...);
    }
  } else {
    if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
      auto* functor = boxed_kernel_func_.getFunctor();
      return callUnboxedKernelFunction<Return, Args...>(
          unboxed_kernel_func_,
          functor,
          dispatchKeySet,
          std::forward<Args>(args)...);
    }
  }

  return impl::BoxedKernelWrapper<Return(Args...)>::call(
      boxed_kernel_func_,
      opHandle,
      dispatchKeySet,
      std::forward<Args>(args)...);
}
```
