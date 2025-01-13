# 0 register operaters
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pytorch 中算子注册适合算子分发（dispatcher）密切相关的，算子注册其实是注册到pytorch的dispatcher 类中的某个数据结构。pytorch中算子注册有三种交互方式：<br>
- m.def 定义一个关于算子的 schema（后面会介绍）;
- m.impl 将带有 dispatch key 信息的算子实现注册到 dispatch table 中;
- m.fallback 为所有的算子都注册上同一个 dispatch key, 方便对算子进行统一操作.

![registry](./images/figure0.png)

# 1 m.def 定义算子 schema
m.def 的实现都是通过TORCH_LIBRARY 这个宏来实现的。

## 1.1 宏定义
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;该方式通过TORCH_LIBRARY 宏来实现, 具体宏的定义如下：
```c++
// \torch\library.h
#include <torch/library.h>
#define TORCH_LIBRARY(ns, m)                                                   \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);                        \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF,                                                     \
      &TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      c10::nullopt,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)
```
- 原理为：先声明TORCH_LIBRARY_init_##ns函数，然后初始化TorchLibraryInit的静态变量，该静态遍历在编译期间就要完成初始化，在初始化该静态遍历的时候需要调用TORCH_LIBRARY_init_##ns函数，该函数的定义发生在上述宏的最后一行。<br>

- TorchLibraryInit 类定义如下：
```c++
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      c10::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};
```

## 1.2 调度到library中
- 在 class library 的 _def 函数里完成向dispatcher单例模式中中注册算子的schema。
```c++
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema),
          debugString(file_, line_),
          tags
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForDef(schema);
      break;
  }
```

- 在dispatcher的registerDef中完成算子的注册，其实就是注册OperatorDef_, operatorDef 里有OperatorEntry, OperatorEntry 里才是kernel真正保存之处, 而OperatorHandle 只是OperaterDef的handle，于kerenl不直接相关。<br>
```c++
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(guard_->mutex);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    // we need a lock to avoid concurrent writes
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterDef_(op, op_name);
  });
}
```
**RegistrationHandleRAII 对象，用于在函数作用域结束时自动调用 deregisterDef_ 函数，解除操作符的定义注册** <br>

# 2 m.impl 注册算子实现
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;真正向distpather的operatorDef中注册算子的实现，也就是kernel的实现。

## 2.1 注册宏及方法
```c++
/// Macro for defining a function that will be run at static
/// initialization time to define operator overrides for dispatch key
/// `k` (must be an unqualified enum member of c10::DispatchKey) in
/// namespace `ns` (must be a valid C++ identifer, no quotes).  Use this
/// macro when you want to implement a preexisting set of custom
/// operators on a new dispatch key (e.g., you want to provide CUDA
/// implementations of already existing operators).  One common usage
/// pattern is to use TORCH_LIBRARY() to define schema for all new
/// operators you want to define, and then use several
/// TORCH_LIBRARY_IMPL() blocks to provide implementations of the
/// operator for CPU, CUDA and Autograd.
///
/// In some cases, you need to define something that applies to all namespaces,
/// not just one namespace (usually a fallback).  In that case, use the reserved
/// namespace _, e.g.,
///
/// ```
/// TORCH_LIBRARY_IMPL(_, XLA, m) {
///    m.fallback(xla_fallback);
/// }
/// ```
///
/// Example usage:
///
/// ```
/// TORCH_LIBRARY_IMPL(myops, CPU, m) {
///   // m is a torch::Library; methods on it will define
///   // CPU implementations of operators in the myops namespace.
///   // It is NOT valid to call torch::Library::def()
///   // in this context.
///   m.impl("add", add_cpu_impl);
/// }
/// ```
///
/// If ``add_cpu_impl`` is an overloaded function, use a
/// ``static_cast`` to specify which overload you want
/// (by providing the full type).
///
// NB: if the dispatch key is not whitelisted, we simply omit the Library
// call entirely
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

/// \private
///
/// The above macro requires an extra unique identifier (uid) to prevent
/// variable name collisions. This can happen if TORCH_LIBRARY_IMPL is called
/// multiple times with the same namespace and dispatch key in the same
/// translation unit.
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                                \
  static void C10_CONCATENATE(                                            \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);       \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(           \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(                 \
      torch::Library::IMPL,                                               \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)       \
           ? &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid) \
           : [](torch::Library&) -> void {}),                             \
      #ns,                                                                \
      c10::make_optional(c10::DispatchKey::k),                            \
      __FILE__,                                                           \
      __LINE__);                                                          \
  void C10_CONCATENATE(                                                   \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)
```

## 2.2 注册过程
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同样是通过TorchLibraryInit访问class Library 并利用Library 类的_impl 函数完成注册。<br>

```c++
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  // See Note [Redundancy in registration code is OK]
  TORCH_CHECK(!(f.dispatch_key_.has_value() &&
                dispatch_key_.has_value() &&
                *f.dispatch_key_ != *dispatch_key_),
    IMPL_PRELUDE,
    "Explicitly provided dispatch key (", *f.dispatch_key_, ") is inconsistent "
    "with the dispatch key of the enclosing ", toString(kind_), " block (", *dispatch_key_, ").  "
    "Please declare a separate ", toString(kind_), " block for this dispatch key and "
    "move your impl() there.  "
    ERROR_CONTEXT
  );
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          std::move(f.cpp_signature_),
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  return *this;
}
```

- class Dispatcher 中调用registerImpl完成kernel的注册, 最终是用OperatorEntry来完成注册的.
```c++
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}
```

- OperatorEntry 中的具体注册过程如下：
```c++
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // NB: cpp_signature doesn't get cleared even after the kernel that populated
  // it is deleted.  This means you could poison the value of cpp_signature_
  // with a bad signature value, and then it would permanently stay there until
  // you deregister the schema.  This can't really be fixed, because we
  // only do a typed() test once in the lifetime of a TypedOperatorHandle,
  // which means if you could validly change the type of a cpp_signature, then
  // that would also invalidate the old TypedOperatorHandles.
  if (cpp_signature.has_value()) {
    auto& local_cpp_signature = kernel.isValidSymUnboxed() ? sym_cpp_signature_ : cpp_signature_;
    if (local_cpp_signature.has_value()) {
      TORCH_CHECK(*cpp_signature == local_cpp_signature->signature,
        "\nMismatch in kernel C++ signatures\n",
        "  operator: ", (this->schema_.has_value() ? toString(this->schema_->schema) : toString(name_)), "\n",
        "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
        "  kernel 1: ", local_cpp_signature->signature.name(), "\n",
        "    dispatch key: ", toString(local_cpp_signature->dispatch_key), "\n",
        "    ", local_cpp_signature->debug, "\n",
        "  kernel 2: ", cpp_signature->name(), "\n",
        "    dispatch key: ", toString(dispatch_key), "\n",
        "    ", debug, "\n"
      );
    } else {
      local_cpp_signature = CppSignatureWithDebug { *cpp_signature, debug, dispatch_key };
    }
  }

  if (schema_ && inferred_function_schema) {
    checkSchema(name_, schema_->schema, schema_->debug, kernel, *inferred_function_schema, debug);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  // Redirect catchAll registrations to CompositeImplicitAutograd.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  if (k[0].kernel.isValid()) {
#else
  if (!k.empty()) {
#endif
    // Suppress the warning for Meta key as we are overriding C++ meta functions with python meta functions
    // for some ops
    if (dispatch_key != DispatchKey::Meta) {
      TORCH_WARN_ONCE("Warning only once for all operators,  other operators may also be overrided.\n",
            "  Overriding a previously registered kernel for the same operator and the same dispatch key\n",
            "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
            "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
            "  dispatch key: ", toString(dispatch_key), "\n",
            "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : (sym_cpp_signature_.has_value() ? sym_cpp_signature_->debug : "no debug info")), "\n",
            "       new kernel: ", debug
      );
    }
  }

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  k[0].kernel = std::move(kernel);
  k[0].inferred_function_schema = std::move(inferred_function_schema);
  k[0].debug = std::move(debug);
#else
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
#endif
  AnnotatedKernelContainerIterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}

// k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug)); 向operatorEntry 的kernels_ 中添加kernel；
// updateDispatchTable_(dispatcher, *dispatch_key); // 更新operatorEntry 的dispatch table；
```

- 重点在于updateDispatchTable_(dispatcher, *dispatch_key)，在这里将多个第一优先级的kernel（指的是多次注册相同key，优先用最后一个）按照dispatch_key来保存到std::array中。
```c++
// synchronizes the dispatch table entries for a given dispatch key *and its
// associated keys* with the current state of kernel registrations in the
// dispatcher.
// After a kernel has been registered to a dispatch key, a call to this
// function will synchronize the dispatcher state. See e.g. registerKernel()
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
  // See Note [Undefined in dispatchTable_]
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  // Registration to CompositeExplicitAutogradNonFunctional, CompositeExplicitAutograd and CompositeImplicitAutograd should be populated to Undefined.
  // We cannot do this above since Undefined cannot be represented in DispatchKeySet.
  if (dispatch_key == DispatchKey::CompositeImplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutograd
   || dispatch_key == DispatchKey::CompositeExplicitAutogradNonFunctional) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  // Note [Refresh Runtime Autograd entries in dispatchTable_]
  // Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
  // In theory, we should only have to check if the given runtime key has "dense" functionality,
  // e.g. DispatchKey::CPU (which is composed of DispatchKey::Dense and BackendComponent::CPUBit).
  // However, there are some backends that should be included in this set that don't have the dense key set.
  // E.g. DispatchKey::Meta, DispatchKey::ORT.
  if (c10::isBackendDispatchKey(dispatch_key)) {
    DispatchKey autograd_key = getAutogradKeyFromBackend(toBackendComponent(dispatch_key));
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}
```

- updateDispatchTableEntry_ 函数将kernel注册到operatorEntry的dispatchTable_中。
```c++
// synchronizes the dispatch table entry for a given dispatch key
// with the current state of kernel registrations in the dispatcher.
// note that this is not a complete update, due to relationships between
// dispatch keys (e.g. runtime keys and their associated autograd keys,
// or alias keys and their associated keysets).
// This function should be considered a private helper for updateDispatchTable_()
void OperatorEntry::updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  const auto dispatch_ix = getDispatchTableIndexForDispatchKey(dispatch_key);
  if (C10_UNLIKELY(dispatch_ix == -1)) {
    return;
  }
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
  dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
}
```

- 关键之处在于从dispatchKey 到 index 该如何获取？ 答：根据functionality_idx 和 backend_idx 以及 offset 和 mask 综合计算得到。要掌握这些，需要深入研究DispatchKey 和 DispatchKeySet。
```c++
// returns the index in the operator table of highest priority key in the the
// keyset Note that we could in theory implement this using
// highestPriorityTypeId(), but this code is very hotpath and we can do it
// faster without it.
int getDispatchTableIndexForDispatchKeySet() const {
auto functionality_idx =
    DispatchKeySet(repr_ >> num_backends).indexOfHighestBit();
auto offset_and_mask = offsetsAndMasks()[functionality_idx];
// Mask the functionality bits out first, then right-shift by 1.
// right-shifting by 1 because everything is zero-indexed.
// E.g. 000001 (CPU) should give us an offset of 0, 000010 (CUDA) should
// give us an offset of 1, etc.
auto backend_idx =
    DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
return offset_and_mask.offset + backend_idx;
}
```

**至此我们可以较为清晰的看到注册流程，深入的分析请看2-ops_dispatcher** <br>

# 3 各函数的注册处

## 3.1 first important registration
- [macro definition](torch/include/torch/library.h)
- [CPU op registration](build/aten/src/ATen/RegisterCPU.cpp)
- [Meta op registration](build/aten/src/ATen/RegisterMeta.cpp)
- [Autograd op registration](torch/csrc/autograd/generated/VariableType_4.cpp)
- [AutogradCUDA op registration](torch/csrc/autograd/generated/VariableType_*.cpp)
- [CUDA op registration](build/aten/src/ATen/RegisterCUDA.cpp)


## 3.2 autocast amp 相关注册
**通过设置dispatchkey 的exclude 来dispatch到amp算子上**<br>
- [autocast op registration](aten/src/ATen/autocast_mode.cpp)
- [AutocastMPS op registration](aten/src/ATen/autocast_mode.cpp)
- [AutocastCPU op registration](aten/src/ATen/autocast_mode.cpp)
- [AutocastXPU op registration](aten/src/ATen/autocast_mode.cpp)
- [autocast op registration](aten/src/ATen/autocast_mode.cpp)
- [autocast op registration](aten/src/ATen/autocast_mode.cpp)
- [autocast op registration](aten/src/ATen/autocast_mode.cpp)
- [autocast op registration](aten/src/ATen/autocast_mode.cpp)

## 3.3 合成算子注册 Composite op <br>
```python
# CompositeImplicitAutograd 是 PyTorch 自动微分系统中的一种机制，用于处理自定义算子的反向传播逻辑。
# 与 CompositeExplicitAutogradNonFunctional 不同，
# CompositeImplicitAutograd 表示一个算子在所有情况下都天然支持反向处理，并且其反向传播逻辑可以通过 PyTorch 的标准自动微分系统自动生成。

# CompositeExplicitAutograd
# 用途：
# CompositeExplicitAutograd 用于声明一个算子，该算子需要显式地为其定义反向传播逻辑。
# 这通常用于复杂的算子，这些算子不能通过简单的元素级操作或现有的自动微分规则来自动推导其梯度。
# 反向传播逻辑：
# 当使用 CompositeExplicitAutograd 时，开发者需要在 derivatives.yaml 文件中为相应的算子显式地指定反向传播函数。
# 这允许开发者完全控制反向传播过程中梯度的计算方式。
# 应用场景：
# 这种机制通常用于实现那些内部包含多个子步骤或复杂数学运算的算子，这些子步骤或运算需要特定的梯度处理逻辑。

#CompositeExplicitAutogradNonFunctional 是一个与 PyTorch 内部实现相关的概念，
# 它可能用于指示某些自定义算子不应该通过标准的自动微分路径（即自动生成 C++ 反向传播代码）来处理。
# 这个标记或机制的具体行为和可用性可能依赖于 PyTorch 的版本和实现细节，因此在不同的 PyTorch 版本或分支中可能会有所不同。
# torch.autograd.Function 是 PyTorch 提供的一个高级接口，用于定义自定义的自动梯度计算。
# 通过继承 torch.autograd.Function 类并重写其 forward 和 backward 方法，开发者可以实现自定义算子的正向和反向传播逻辑。
```
- [CompositeImplicitAutograd op registration](build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp)
- [CompositeExplicitAutograd op registration](build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp)
- [CompositeExplicitAutogradNonFunctional op registration](build/aten/src/ATen/RegisterCompositeExplicitAutogradNonFunctional.cpp)
- [RegisterCompositeImplicitAutogradNestedTensor op registration](build/aten/src/ATen/RegisterCompositeImplicitAutogradNestedTensor.cpp)

## 3.4 量化相关注册QuantizedMeta op
- [QuantizedCPU op registration](build/aten/src/ATen/RegisterQuantizedCPU.cpp)
- [QuantizedCUDA op registration](build/aten/src/ATen/RegisterQuantizedCUDA.cpp)
- [QuantizedMeta op registration](build/aten/src/ATen/RegisterQuantizedMeta.cpp)
- [cuda quantization op registration](torch/csrc/distributed/c10d/quantization/quantization_gpu.cu)
- [cpu quantization op registration](torch/csrc/distributed/c10d/quantization/quantization.cpp)
- [Quantized CPU / QuantizedCPU registration](aten/src/ATen/native/quantized/cpu/*.cpp) 
- [Quantized CUDA registration](aten/src/ATen/native/quantized/cuda/*.cu)
- [Quantized CUDA registration](aten/src/ATen/native/quantized/cudnn/*.cpp)
- [Quantized Meta registration](aten/src/ATen/native/quantized/cpu/*.cpp)


## 3.5 FullbackKernel
- [Functionalize FallbackKernel registration](aten/src/ATen/FunctionalizeFallbackKernel.cpp)
- [ZeroTensorFallbackKernel registration](aten/src/ATen/ZeroTensorFallback.cpp)
- [Conjugate op Fallback](aten/src/ATen/ConjugateFallback.cpp)
- [BackendSelect Fallthough](aten/src/ATen/core/BackendSelectFallbackKernel.cpp)
- [Python Fallback](aten/src/ATen/core/PythonFallbackKernel.cpp)
- [Python PythonDispatcher Fallback](aten/src/ATen/core/PythonFallbackKernel.cpp)
- [Python PythonTLSSnapshot Fallback](aten/src/ATen/core/PythonFallbackKernel.cpp)
- [Python PreDispatch Fallback](aten/src/ATen/core/PythonFallbackKernel.cpp)
- [Meta Fallback](aten/src/ATen/core/MetaFallbackKernel.cpp)
- [MPS Fallback](aten/src/ATen/mps/MPSFallback.mm) 
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**下列都fallback到Autograd上** <br>

- [AutogradOther Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradCPU Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradXPU Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradCUDA Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradXLA Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradLazy Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradMPS Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [ADInplaceOrView Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 
- [AutogradHPU Fallback](aten/src/ATen/core/VariableFallbackKernel.cpp) 

## 3.6 CatchAll
- [sparse CatchAll](aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_unpack.cpp)
- [quantized CatchAll](aten/src/ATen/native/RNN.cpp)
- [quantized CatchAll](aten/src/ATen/native/quantized/qlinear_unpack.cpp)

## 3.7 ADInplaceOrView op
- [ADInplaceOrView Op registration](torch/csrc/autograd/generated/ADInplaceOrViewType_1.cpp)
```python
#这段注释详细解释了ADInplaceOrView这个dispatch key在PyTorch中的用途和背后的设计理念。下面是对这段注释的解读：
#
# ADInplaceOrView key的作用
# ADInplaceOrView key主要用于原地（inplace）操作或view操作（不改变数据但改变tensor形状的操作）来注册一个特殊的kernel。
# 这个kernel负责为未来的自动微分（autograd）计算做一些额外的设置工作。
#
#对于原地操作：
# 这个kernel会执行版本更新（version bump）。在PyTorch中，原地操作会修改tensor的内容，这可能导致自动微分系统在计算梯度时遇到一些挑战，
# 因为梯度需要正确地回溯到修改前的状态。版本更新机制就是用来解决这个问题的。

# 对于view操作：
# 这个kernel会设置DifferentiableViewMeta，这是为了确保view tensor能够正确地参与自动微分计算。
# 在PyTorch中，view操作不会复制数据，而是创建一个新的tensor视图，这个视图与原tensor共享数据。
# 但是，在自动微分时，我们需要知道哪些tensor是通过view操作创建的，以便正确地处理它们。

# 对于其他操作
# 对于不是原地操作也不是view操作的其他操作，这个kernel是一个直通（fallthrough）kernel，即它不做任何额外的工作。
# 这是因为这些操作不需要为自动微分计算做额外的设置。
#
# 理想世界中的设计（Dream部分）
#注释中还提到了一个理想的设计方案，即在一个理想的世界中，我们可以为requires_grad=false的输入跳过VariableType kernel（这是PyTorch中处理tensor的一个核心部分，负责很多tensor的操作和自动微分的支持）。
# 但是，由于这会给所有操作增加一个额外的dispatch（分发）开销，并且在模型级别上会带来非微不足道的性能损失（几个百分点），因此这个方案目前被阻塞了。
#
# 当前的设计方案
# 当前的设计方案利用了这样一个事实：每个kernel都会首先通过VariableType kernel。
# 因此，他们将at::AutoDispatchBelowADInplaceOrView guard（一个用于控制dispatch行为的机制）上移到了VariableType kernel中。
# 这样，他们只对view/inplace操作添加了额外的dispatch，以最小化对实际模型性能的影响。
#
#总结
# 这段注释不仅解释了ADInplaceOrView key的用途，还揭示了PyTorch在设计自动微分系统时面临的一些挑战和权衡。
# 通过理解这些设计决策，我们可以更好地理解和使用PyTorch的自动微分功能。
```

## 3.8 通信算子的注册
- [communicate op registration](torch/csrc/distributed/c10d/Ops.cpp)

**step1 : 函数通过宏来实现** <br>
```c++
// Return input tensors as output tensors to make inplace allreduce look like
// a functional API, so that make_fx can correctly build the dependencies in
// the graph later.
#define IMPL_ALLREDUCE(DEV)                                                   \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>               \
      allreduce_##DEV(                                                        \
          at::TensorList tensors,                                             \
          const c10::intrusive_ptr<ProcessGroup>& process_group,              \
          const c10::intrusive_ptr<ReduceOp>& reduce_op,                      \
          const std::optional<at::Tensor>& sparse_indices,                    \
          int64_t timeout) {                                                  \
    auto tensor_vec = tensors.vec();                                          \
    auto work = process_group->getBackend(c10::DeviceType::DEV) -> allreduce( \
        tensor_vec,                                                           \
        AllreduceOptions{                                                     \
            *reduce_op.get(), std::chrono::milliseconds(timeout)});           \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(     \
        std::move(tensor_vec), work);                                         \
  }

IMPL_ALLREDUCE(CPU)
IMPL_ALLREDUCE(CUDA)
IMPL_ALLREDUCE(PrivateUse1)
```

**step2 : 注册** <br>
```c++
#define REGISTER_C10D_OP1(FUNC, DEV) \
  TORCH_LIBRARY_IMPL(c10d, DEV, m) { \
    m.impl(#FUNC, FUNC##DEV);        \
  }

// 1st level expansion
#define REGISTER_C10D_OP(FUNC)  \
  REGISTER_C10D_OP1(FUNC, CPU)  \
  REGISTER_C10D_OP1(FUNC, CUDA) \
  REGISTER_C10D_OP1(FUNC, PrivateUse1)

REGISTER_C10D_OP(allreduce_)
```

## 3.9 Other important registrations
- [Named dispatch registration](aten/src/ATen/core/NamedRegistrations.cpp) // 调度到下一个可用的dispatch
- [AutogradNestedTensor op registration](torch/csrc/autograd/generated/VariableType_*.cpp)
- [VariableTypeManual autograd registration](torch/csrc/autograd/VariableTypeManual.cpp)
- [VariableTypeManual ADInplaceOrView registration](torch/csrc/autograd/VariableTypeManual.cpp)
- [CPUCustomType op registration](build/out/RegisterCPUCustomOps.cpp)
- [Negative op registration](aten/src/ATen/native/NegateFallback.cpp)
- [Cathch all ops](torch/csrc/jit/runtime/register_distributed_ops.cpp)
- [inductor functionalize](torch/csrc/inductor/resize_storage_bytes.cpp)
- [aten trace](torch/csrc/autograd/generated/TraceType_1.cpp)
- [aten trace](torch/csrc/autograd/TraceTypeManual.cpp)
- [BackendSelect](build/aten/src/ATen/RegisterBackendSelect.cpp)
- [ZeroTensor op registration](build/aten/src/ATen/RegisterZeroTensor.cpp)
- [prepacked cpu](aten/src/ATen/native/xnnpack/RegisterOpContextClass.cpp)
- [BatchedTensor Implement](aten/src/ATen/LegacyBatchingRegistrations.cpp)

## 3.10 NestedTensor : 还在原型阶段<br>
```python
a, b = torch.arange(3), torch.arange(5) + 3
nt = torch.nested.nested_tensor([a, b])
```
- [NestedTensorCUDA op registration](build/aten/src/ATen/RegisterNestedTensorCUDA.cpp)
- [NestedTensorMeta op registration](build/aten/src/ATen/RegisterNestedTensorMeta.cpp)
- [NestedTensorCPU op registration](build/aten/src/ATen/RegisterNestedTensorCPU.cpp)

## 3.11 Functionalize
```python
在 PyTorch 中，DispatchKey 是用于定义操作在不同后端（如 CPU、CUDA、XLA 等）上如何执行的一种机制。functionalize 是与 PyTorch 的 Autograd 系统和图形转换相关的一个概念，特别是在将计算图从急切执行模式（eager execution mode）转换为图执行模式（graph execution mode）时起作用。

具体来说，functionalize 是指在将 PyTorch 的计算图转换为更容易优化和执行的形式时，将原始代码中的就地操作（in-place operations）和带有副作用的操作替换为纯函数式操作的过程。这种转换有助于确保计算图中的所有操作都是无状态的，并且可以安全地进行各种优化，如自动微分、梯度累积、图优化等。

在 PyTorch 的内部实现中，特别是在使用 torch.autograd.graph.Functionalize 类时，functionalize 的作用包括但不限于：

消除就地操作：将如 x += y 的就地操作替换为 x = x + y 的形式，以确保每次操作都返回一个新的张量，而不是修改现有的张量。
处理控制流：将 Python 的控制流结构（如 if 语句和 for 循环）转换为可以在图执行模式下更高效执行的等效形式。
优化梯度计算：通过消除计算图中的冗余操作和重新排列操作顺序，来优化反向传播过程中梯度的计算。
支持动态图到静态图的转换：在将动态计算图（eager graph）转换为静态计算图（static graph）时，functionalize 是关键步骤之一，使得转换后的图可以在不同的后端上高效执行。
总之，functionalize 在 PyTorch 中主要用于确保计算图的可优化性和后端执行的高效性，通过将就地操作和带有副作用的操作转换为纯函数式操作，为进一步的图优化和执行提供了基础。这在 PyTorch 的高性能计算和自动微分系统中扮演着重要角色。
```
- [Functionalize op registration](build/aten/src/ATen/RegisterFunctionalization_*.cpp)
- [Functionalize op registration](build/aten/src/ATen/RegisterFunctionalizationEverything.cpp)


## 3.12 SparseTensor
```python
SparseCsr key 对应于使用压缩稀疏行（Compressed Sparse Row，简称 CSR）格式存储的稀疏张量（Sparse Tensor）的情况。
CSR 是一种高效的稀疏矩阵存储格式，特别适用于那些非零元素相对较少且分布不规则的矩阵。
```
- [SparseCPU op registration](build/aten/src/ATen/RegisterSparseCPU.cpp)
- [SparseCUDA op registration](build/aten/src/ATen/RegisterSparseCUDA.cpp)
- [SparseCsrCUDA op registration](build/aten/src/ATen/RegisterSparseCsrCUDA.cpp)
- [SparseCsrMeta op registration](build/aten/src/ATen/RegisterSparseCsrMeta.cpp)
- [sparse QuantizedCPU](aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear.cpp)
- [sparse QuantizedCPU](aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_prepack.cpp)
- [sparse CPU](aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_dynamic.cpp)

## 3.13 functorch
```python
# functorch 是一个 PyTorch 的扩展库，它提供了一个可微分的张量库，并利用 PyTorch 的自动微分系统来实现自动化的函数变换。Functorch 是一个强大的工具，对于需要进行复杂微分计算和函数变换的深度学习研究者和开发者非常有用。
# 与 Google JAX 类似，functorch 是 PyTorch 中的一个库，提供可组合的 vmap（矢量化）和 autodiff 转换。
# 它支持高级的 autodiff 用例（在 PyTorch 中难以表达），包括：
# 
# 1. 模型集成 model ensembling
# 
# 2. 高效计算 Jacobian 和 Hessians
# 
# 3. 计算 per-sample-gradients 或其他 per-sample quantities
```
- [FuncTorchBatched registration](aten/src/ATen/functorch/Batch*.cpp)
- [FuncTorchGradWrapper registration](aten/src/ATen/functorch/*.cpp)
- [FuncTorchVmapMode registration](aten/src/ATen/functorch/*.cpp)
- [FuncTorchDynamicLayerFrontMode registration](aten/src/ATen/functorch/*.cpp)
- [FuncTorchDynamicLayerBackMode registration](aten/src/ATen/functorch/*.cpp)

## 3.14 Metal 相关算子
```python
# Metal 主要由 Apple 开发，并在其 macOS、iOS、tvOS 和 watchOS 等操作系统上得到支持。
# 当开发者需要在这些平台上运行 PyTorch 模型时，Metal dispatch key 就变得尤为重要。
# 它允许 PyTorch 模型利用 Metal 的高性能计算能力，实现跨平台的无缝计算。

# 另外：
# MPS（Metal Performance Shaders）对应于苹果公司的Metal图形API的后端，特别是在PyTorch等深度学习框架的上下文中。
# MPS是苹果公司为其硬件（特别是GPU）提供的一套高性能着色器和计算库，旨在加速图形渲染和通用计算任务。
```
- [Metal op registration](aten/src/ATen/native/metal/ops/Metal*.mm)
- [MPS Fallback](aten/src/ATen/mps/MPSFallback.mm)

## 3.15 mkldnn 相关算子
```python
# 一、MKLDNN简介
# MKLDNN是一个深度学习底层库，主要针对英特尔处理器、英特尔图形处理器以及Xe图形处理器，对深度神经网络进行op级以及指令集级的优化。
# 它能够成倍地提升神经网络在Intel CPU以及GPU下的推理速度，并大大简化了复杂神经网络的设计与实现过程。

# 二、MKLDNN的核心特性
# 高度优化：MKLDNN通过高度优化的算法，确保了计算资源的有效利用。它针对英特尔硬件平台进行了深度优化，能够充分发挥Intel CPU和GPU的性能。
# 开放性和兼容性：MKLDNN能够很好地与其他流行深度学习平台如TensorFlow、PyTorch或Caffe等协同工作。
# 这意味着开发者可以根据自身偏好自由选择最适合的开发环境，同时享受MKLDNN带来的性能增益。
# 多线程与多核处理：MKLDNN支持多线程与多核处理，这对于处理大规模数据集尤为重要。
# 通过并行计算，MKLDNN能够充分利用现代计算机中的多核处理器，进一步提升计算效率。
# 层融合技术：MKLDNN提供了层融合技术，以加速推理时间。通过将这些带宽限制型op与计算密集型op或者其它op进行融合，可以减小计算量并提高推理速度。
```
- [mkldnn op registration](aten/src/ATen/native/mkldnn/Linear.cpp)

## 3.16 Vulkan 相关算子
```python
# 在 PyTorch 中，dispatch Vulkan 主要指的是利用 Vulkan 图形和计算 API 来加速深度学习模型的推理和训练过程。
# Vulkan 是一种跨平台的图形和计算 API，它提供了对现代 GPU 的低级访问，并允许开发者更直接地控制硬件资源，从而实现高性能计算。
# 
# Vulkan Dispatch 在 PyTorch 中的应用
# 高性能计算：
# Vulkan 提供了高效的并行计算能力，这使得它成为加速深度学习模型推理和训练的理想选择。
# 通过利用 Vulkan 的低级访问和硬件加速功能，PyTorch 可以实现更快速的算子执行和更高的吞吐量。
# 跨平台支持：
# Vulkan 是一种跨平台的 API，这意味着它可以在多种操作系统和设备上运行。
# 这为 PyTorch 提供了更广泛的兼容性，使得开发者可以在不同的硬件平台上利用 Vulkan 的性能优势。
# 自定义算子支持：
# 对于需要在 Vulkan 上执行自定义算子的开发者来说，PyTorch 提供了必要的支持。
# 开发者可以通过 PyTorch 的自定义算子接口实现并注册支持 Vulkan 的算子，从而扩展 PyTorch 的功能并优化性能。
# 与硬件的紧密集成：
# Vulkan 与现代 GPU 硬件紧密集成，能够充分利用设备的硬件加速功能。
# 通过 Vulkan dispatch，PyTorch 可以更高效地利用这些硬件加速功能，提高算子执行的效率和速度。
```
- [vulkan op registration](aten/src/ATen/native/vulkan/ops/*.cpp)


# 4 参考文档
- [pytorch Registering a Dispatched Operator in C++](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page)
