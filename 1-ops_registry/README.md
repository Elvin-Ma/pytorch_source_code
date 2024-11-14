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

# 3 参考文档
- [pytorch Registering a Dispatched Operator in C++](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page)
