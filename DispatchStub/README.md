# 1 调度存根 DispatchStub.h

- /root/tiening.ma/torch_musa/torch_musa/share/generated_cuda_compatible/include/ATen/native/DispatchStub.h

## 1.1 声明
```c++
#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    name##_DECLARE_DISPATCH_type() = default;                                              \
    name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;            \
    name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
  };                                                                                       \
  extern TORCH_API struct name##_DECLARE_DISPATCH_type name;
```

## 1.2 定义
```c++
#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name
```

## 1.3 注册
```c++
#define REGISTER_MUSA_DISPATCH(name, fn) \
  static RegisterMUSADispatch<struct name##_DECLARE_DISPATCH_type> name ## __register(name, fn);
```

## 1.4 注册到哪里呢？？？

**通过DispatchStub 注册到 DispatchStubImpl**

```c++
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(const c10::DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , reinterpret_cast<void*>(ZVECTOR)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  #if defined(USE_XPU)
  void set_xpu_dispatch_ptr(FnPtr fn_ptr){
    impl.xpu_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }
  #endif

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mps_dispatch_ptr(FnPtr fn_ptr) {
    impl.mps_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_mtia_dispatch_ptr(FnPtr fn_ptr) {
    impl.mtia_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  // 用这个来注册
  void set_privateuse1_dispatch_ptr(FnPtr fn_ptr) {
    impl.privateuse1_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

private:
  DispatchStubImpl impl; // 注册到DispatchStubImpl
};
```

## 1.5 DispatchStubImpl

privateuse1_dispatch_ptr 指针存放处

```c++
struct TORCH_API DispatchStubImpl {

  // The DispatchStubImpl::try_get_call_ptr() method is used to get the call
  // pointer for a given device type. If the call pointer is not found,
  // DispatchStubImpl::try_get_call_ptr() returns an ErrorType.
  // The main difference between try_get_call_ptr() and get_call_ptr() is that
  // try_get_call_ptr() will return the ErrorType and not raise an exception.
  DispatchResult try_get_call_ptr(
    c10::DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
  );

  // Analogous to try_get_call_ptr(), but it will return the ErrorType and not
  // raise an exception.
  DispatchResult try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
  );


  void* get_call_ptr(
    c10::DeviceType device_type
    , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      , void *ZVECTOR
#endif
  );

  /**
   * The CPU Dispatch actual method is chosen in decreasing order of preference by
   * DispatchStubImpl::choose_cpu_impl() in case none is found by
   * DispatchStubImpl::get_call_ptr() in cpu_dispatch_ptr.
   */
  void* choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
  );

  // Fixing dispatch error in Windows debug builds.
  // See https://github.com/pytorch/pytorch/issues/22681 for more details.
  #if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
    void* cuda_dispatch_ptr;
    void* hip_dispatch_ptr;
    void* mps_dispatch_ptr;
    void* mtia_dispatch_ptr;
  #if defined(USE_XPU)
    void* xpu_dispatch_ptr;
  #endif
    void* privateuse1_dispatch_ptr;
  #else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
    void* hip_dispatch_ptr = nullptr;
    void* mps_dispatch_ptr = nullptr;
    void* mtia_dispatch_ptr = nullptr;
  #if defined(USE_XPU)
    void* xpu_dispatch_ptr = nullptr;
  #endif
    void* privateuse1_dispatch_ptr = nullptr;
  #endif
};
```

## 1.7 如何获取ptr 呢 ???

**return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;**

```c++
DispatchResult DispatchStubImpl::try_get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
) {
  constexpr auto supported_devices = c10::array_of<c10::DeviceType>(
        c10::DeviceType::CPU,
        c10::DeviceType::CUDA,
        c10::DeviceType::HIP,
        c10::DeviceType::MPS,
        c10::DeviceType::MTIA,
        c10::DeviceType::XPU,
        c10::DeviceType::PrivateUse1
    );
    // Check if the device type is supported.
    if (std::find(supported_devices.begin(), supported_devices.end(), device_type) == supported_devices.end()) {
        return ErrorType::DeviceNotSupported;
    }
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        auto result = try_choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
          , ZVECTOR
#endif
        );
        if (!std::holds_alternative<ErrorType>(result)) {
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
      return result;
      }
      return DispatchResult(fptr);
    }

    case DeviceType::CUDA:
      return cuda_dispatch_ptr != nullptr ? DispatchResult(cuda_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HIP:
      return hip_dispatch_ptr != nullptr ? DispatchResult(hip_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_MPS)
    case DeviceType::MPS:
      return mps_dispatch_ptr != nullptr ? DispatchResult(mps_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif
    case DeviceType::MTIA:
      return mtia_dispatch_ptr != nullptr ? DispatchResult(mtia_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_XPU)
    case DeviceType::XPU:
      return xpu_dispatch_ptr != nullptr ? DispatchResult(xpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif

    case DeviceType::PrivateUse1:
      return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    default:
      TORCH_INTERNAL_ASSERT(false, "An unexpected device type was provided ", device_type);
      return ErrorType::DeviceNotSupported;
  }
}
```

## 1.8 在哪里调用这个函数呢？？？

**其实还是在DispatchStub类里，在operator()函数里调用的**

```c++
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }
```

# 2 EXAMPLE

- 声明
```C++
DECLARE_DISPATCH(fused_sdp_choice_fn, _fused_sdp_choice_stub); // FN NAME

struct _fused_sdp_choice_stub_DECLARE_DISPATCH_type : DispatchStub<fused_sdp_choice_fn, _fused_sdp_choice_stub_DECLARE_DISPATCH_type> {
    _fused_sdp_choice_stub_DECLARE_DISPATCH_type() = default;
    _fused_sdp_choice_stub_DECLARE_DISPATCH_type(const _fused_sdp_choice_stub_DECLARE_DISPATCH_type&) = delete;
    _fused_sdp_choice_stub_DECLARE_DISPATCH_type& operator=(const _fused_sdp_choice_stub_DECLARE_DISPATCH_type&) = delete;
};

extern TORCH_API struct _fused_sdp_choice_stub_DECLARE_DISPATCH_type _fused_sdp_choice_stub;
```

- 定义
```C++
struct _fused_sdp_choice_stub_DECLARE_DISPATCH_type _fused_sdp_choice_stub
```

- 注册
```C++
template <typename DispatchStub>
struct RegisterMUSADispatch {
  RegisterMUSADispatch(DispatchStub &stub, typename DispatchStub::FnPtr value) {
    stub.set_privateuse1_dispatch_ptr(value); // 调用的是DistatchStub 里的对应函数
  }
};

static RegisterMUSADispatch<struct _fused_sdp_choice_stub_DECLARE_DISPATCH_type> _fused_sdp_choice_stub__register(_fused_sdp_choice_stub, fn);
```

- 调用
** 通过存根实例 + 对应device 可以直接调度到函数** <br>

```c++
fused_sgd_stub(
    kCPU,
    params[i],
    grads[i],
    no_momentum_buffer ? Tensor() : momentum_buffer_list[i],
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
    is_first_step,
    grad_scale_ptr);
```