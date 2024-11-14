# 1 DispatchKey 的基础数据结构及概念介绍

## 1.1 相关定义
- [Note: Per-Backend Functionality Dispatch Keys]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;检查一个dispatch key是否为按后端划分的functionality键,任何可以按后端进行自定义的functionality都应添加至此。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这些key对应于可以按后端单独自定义的功能。虽然它们在DispatchKeySet bitset中仅占用一个bit，但在operator table中，它们会映射到（#后端数量）个槽位(slots)。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于每个后端，这些key在dispatch enum中还有一组单独的“runtime keys”，这些runtime keys确实会映射到operator table中的单独槽位(slots)。<br>

**example:** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;“Sparse”键在DispatchKeySet中映射到一个单独的位，而SparseCPU、SparseCUDA等则分别映射到运行时运算符表中的单独槽位(slots)。<br>

- BackendComponent
enum class BackendComponent : uint8_t {
    InvalidBit = 0,    
    CPUBit, 
    CUDABit, 
    HIPBit, 
    XLABit, 
    MPSBit, 
    IPUBit, 
    XPUBit, 
    HPUBit, 
    VEBit, 
    LazyBit, 
    MTIABit, 
    PrivateUse1Bit, 
    PrivateUse2Bit, 
    PrivateUse3Bit, 
    MetaBit,
    EndOfBackendKeys = MetaBit,
}

- Out-of-tree vmap+grad prototype keys （非主仓需要用到的key）
以下键用于实现位于https://github.com/zou3519/functorch的out-of-tree可组合函数变换（vmap+grad）原型。我们计划最终将该原型整合进核心库，届时它将采用一个不同的设计，该设计应使用更少的键。<br>
```c++
case DispatchKey::FuncTorchDynamicLayerFrontMode:
    return "FuncTorchDynamicLayerFrontMode";
case DispatchKey::TESTING_ONLY_GenericWrapper:
    return "TESTING_ONLY_GenericWrapper";
case DispatchKey::TESTING_ONLY_GenericMode:
    return "TESTING_ONLY_GenericMode";
case DispatchKey::PreDispatch:
    return "PreDispatch";
case DispatchKey::PythonDispatcher:
    return "PythonDispatcher";
```

- [Functionalization Pass In Core]
```c++
// */pytorch/aten/src/ATen/FunctionalTensorWrapper.h
// The Functionalization pass is used to remove aliasing from a pytorch program.
//
// This is useful for backends that don't support aliasing, like XLA and Vulkan.
// It's also necessary in order to remove mutation from a program, which is
// needed in Functorch.
//
// Consider this program:
// a = torch.ones(...)
// b = a.view(...)
// b.add_(1)
//
// In this program, b is meant to alias with a due to the use of view(). At the
// end of the program, both a and b are full of 2's. However, backends that
// don't support aliasing aren't able to correctly implement the view()
// operator. Instead, they can opt into the Functionalization pass, which will
// sit between the user and the backend, and provide the necessary aliasing
// logic.
//
// The functionalization pass will turn the above program into a slightly
// different program that has the same semantics, transparently to the user,
// that backends like XLA/Vulkan are able to implement a = torch.ones(...) b =
// a.view_copy(...)  # view() replaced with view_copy(). Backends like
// XLA/Vulkan can implement this! b.add_(1) a.add_(1)  # Our functionalization
// pass machinery knows that a and b are aliased - it applies b's mutation to a
// too.
//
// So, how does the functionalization pass keep track of which tensors are
// aliased? The pass works by wrapping EVERY tensor in the program inside of a
// FunctionalTensorWrapper, which knows about its alias'd tensors.
//
// See Note [Functionalization: Alias Removal] for details on the aliasing
// machinery. See Note [Functionalization: Mutation Removal] for details on
// mutation removal.
```
- Fallthrough 机制
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fallthrough机制是指当某个特定分发键没有注册的内核实现时，操作会“落空”到一个默认行为。这是 PyTorch 分发机制的一部分，用于处理未实现特定后端的情况，确保操作在这种情况下不会崩溃或抛出错误。 PyTorch 的操作分发系统非常复杂，涉及多个后端（如 CPU、CUDA、XLA 等）和功能键（如 Autograd、Sparse 等）。每个操作（如矩阵乘法、卷积等）都可以有针对不同后端和功能键的具体实现。然而，并不是每个操作在每个后端或功能键下都有具体实现。这时，Fallthrough机制就起作用了。 比如我们在实现一个新的后端的时候，我们可能并不能一下子把所有的算子都按照新的后端在优化实现，这个时候就可以借助Fallthrough机制，让暂时没有实现的算子回落到其它后端比如CPU上执行。 比如:<br>
```c++
namespace at {
TORCH_LIBRARY(my_ops, m) {
  m.def("my_ext_op(Tensor input) -> Tensor");
}

// 注册 CPU 实现
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("my_ext_op", [](const at::Tensor& input) {
    return input * 2;
  });
}

// 为扩展的后端注册 Fallthrough
TORCH_LIBRARY_IMPL(my_ops, BackendComponent, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
} // namespace at
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在这个例子中，my_ext_op在 CPU 上有实现，但在某个扩展的后端（如 BackendComponent）上没有特定的实现。通过注册 Fallthrough，我们确保该操作在扩展后端上不会失败。<br>

- Alias Key
别名键是合成的dispacherkey，别名键不会在运行时被直接调用。 你可以将内核注册到别名键，在分发表计算期间，内核可能会被填充到映射的运行时键中。下面的代码就是一个合成的过程：<br>
```c++
DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      // See Note [autograd_dispatch_keyset Does Not Include Backend Bits]
      // 这就是为什么我们要在这里将它与一个后端位掩码进行或（OR）运算。
      // getRuntimeDispatchKeySet() 函数期望返回一个运行时分发键的键集，比如 AutogradCPU，但这需要backend bits。
      return autograd_dispatch_keyset | DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
    case DispatchKey::CompositeImplicitAutograd:
      return math_dispatch_keyset;
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      return nested_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutograd:
      return backend_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      return non_functional_backend_dispatch_keyset;
    default:
      return DispatchKeySet(t);
  }
}
```

# 2 DispatchKey 详解
```c++
// 从语义上讲，一个dispatch key标识了函数dispatch中的一个**可能的**级别，在这个级别上可以注册一个函数句柄。每个函数句柄对应functionality的一种类型。<br>
// 从实现的角度来看，dispatch key 标识了 DispatchKeySet中一个特定的bit。更高的位(bit)索引会优先被分发处理(因为我们在提取最高优先级的dispatch key时，会计算前导零(count leading zeros)的数量。）<br>
//
// 注释: [DispatchKey 的分类]
// 这个枚举实际上包含了多个类型的键，这些键的详细说明在后面会有更详细的解释：
// (1) 非可定制化的后端(例如 FPGA)
// (2) 非可定制化的functionalities(例如 Functionalize)
// (3) 可为每个后端定制功能化functionalized(例如 Dense, Quantized, Sparse, AutogradFunctionality, NestedTensor)
// (4) 上述key针对per-backend定制化实例化后的functionalities(CPU, QuantizedCPU, SparseCPU, AutogradCPU, NestedTensorCPU)
// (5) alias keys(例如 CompositeImplicitAutograd)
// 以上分类中，需要注意的是：
// (a) 哪些key被分配到 DispatchKeySet 中的单独的位
// (b) 哪些键被分配到运行时 operator 表中的单独的槽位("Runtime keys") --> 即那些key被分配到OperatorEntry中的operators_ array中。
// 答：
// (1), (2) 和 (3) 都有自己的bit位分配给它们。
// (1), (2) 和 (4) 都有自己的槽位(slots)分配给它们（OperatorEntry中的operators_中有位置）。
//
// 可查看 [DispatchKeySet 内部实现] 了解更多细节。
//
// 注释: 与 */pytorch/torchgen/model.py 中的 DispatchKey list 保持一致。
//
enum class DispatchKey : uint16_t {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 这不是一个“真实”的functionality，但它存在是为了让我们在DispatchKeySet不包含任何元素时，可以返回一个“nullopt”元素。你可以认为DispatchKey的一个更语义准确的定义是：
  //      using DispatchKey = optional<RealDispatchKey> and Undefined == nullopt.
  // 我们没有真正地表示它，因为optional<RealDispatchKey>将需要64bits，而DispatchKey只需要8bits。
  Undefined = 0,

  // 为 Undefined 定义一个别名以表示 CatchAll（长期来看这将被淘汰，但目前这样很方便）。
  CatchAll = Undefined,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ Functionality Keys ~~~~~~~~~~~~~~~~~~~~~~ //
  // 枚举中从此开始的每个值（直到EndOfFunctionalityKeys）都对应一个可以dispatch的单独“functionality”。
  // 在DispatchKeySet中，这是通过将这些枚举值中的每一个分配给剩余的（64 - len(BackendComponent)）bit来表示的。<br>
  // 这些functionality中的大多数都有一个单独的handler与之对应，使它们成为“runtime key”。这些runtime-key映射到运行时运算符表(runtime operator teble)中的一个单独槽位(slots)。
  // 少数几个functionalites 允许为每个后端定制化，例如Dense, Quantized, Sparse, AutogradFunctionality, NestedTensor。<br>
  // See [Note: Per-Backend Functionality Dispatch Keys] for details.
  Dense = 1, // Per-Backend Functionality

  // 以下是不可扩展的后端。这些后端目前没有针对Autograd/Sparse/Quantized内核的自定义实现，因此我们不会在运行时运算符表中为它们分配空间以避免浪费。
  // 如果将来这些后端中的任何一个需要自定义，例如Autograd，那么我们将需要为它们添加一个DispatchKey::*Bit。
  // TODO: put this in BackendComponents --> 将来可能会将其放到BackendComponents中
  FPGA = 2, // Xilinx support lives out of tree at https://gitlab.com/pytorch-complex/vitis_kernels  
  // TODO: put this in BackendComponents
  // ONNX Runtime, lives out of tree at https://github.com/pytorch/ort and
  // https://github.com/microsoft/onnxruntime, and is also used to test general
  // backend/extension machinery in the core. cf:
  // - test/cpp_extensions/ort_extension.cpp
  // - test/test_torch.py
  // - aten/src/ATen/test/extension_backend_test.cpp
  ORT = 3,       // onnx runtime backend
  Vulkan = 4,    // TODO: put this in BackendComponents
  Metal = 5,     // TODO: put this in BackendComponents
  
  Quantized = 6, // See [Note: Per-Backend Functionality Dispatch Keys]

  // 这个后端用于支持自定义的随机数生成器（RNGs）；如果你传入了一个不是传统CPUGeneratorImpl/CUDAGeneratorImpl的生成器，它允许你转向使用不同的内核。要使用这个键：
  // 1)在用户定义的伪随机数生成器（PRNG）类的at::Generator构造函数调用中，将其作为第二个参数设置。
  // 2)在注册自定义kernel（专为用户定义的PRNG类定制的模板化内核）时，将其用作dispatch key，这些kernel旨在用于树外（out of tree）使用；通过aten/src/ATen/test/rng_test.cpp进行测试。
  CustomRNGKeyId = 7,

  // TODO：将Mkldnn设为一个功能键，以便我们可以为其提供Meta支持
  // 以下是基于tensor layout指定更专业运算符(more specialized operators)的后端。
  // 请注意，稀疏后端是顺序很重要的一个案例：稀疏多分发与相应的稠密张量一起进行，并且必须在它们之前处理。
  // NB: 不要与MKLDNN混淆, which is Caffe2 only
  MkldnnCPU = 8, // registered at build/aten/src/ATen/RegisterMkldnnCPU.cpp  
  Sparse = 9,    // See [Note: Per-Backend Functionality Dispatch Keys]  
  SparseCsrCPU = 10,  // TODO: Make SparseCsr a functionality key
  SparseCsrCUDA = 11,
  NestedTensor = 12, // See [Note: Per-Backend Functionality Dispatch Keys]

  // 在某些情况下，并不总是能立即明确某个函数应该使用哪个后端，因为该函数可能没有任何“tensor”参数。
  // 在这种情况下，可以注册一个BackendSelect函数来实现对正确后端的自定义确定。
  BackendSelect = 13,

  Python = 14,

  // Out-of-core key for Fake Tensor in torchdistx.
  // See https://pytorch.org/torchdistx/latest/fake_tensor.html
  // TODO: delete this in favor of Python-implemented fake tensor
  Fake = 15, // 与meta tensor类似，但无数据有device

  // See Note [Out-of-tree vmap+grad prototype]. 
  // 这个键的目的是在“auto grad subsystem”运行之后插入代码，因此这个键应该紧跟在ADInplaceOrView和所有自动微分键之后。
  FuncTorchDynamicLayerBackMode = 16, 

  // 别名和mutation移除。如果某些后端仅希望选择别名移除或仅选择mutation移除，我们可以考虑添加专门用于这些单独遍历的key。
  // See Note [Functionalization Pass In Core] for details.
  Functionalize = 17,

  // 对于任何具有named dimensions的张量，都会设置named dispatch key。
  // 尽管我们为named tensor提供了一个分发键，但由于历史原因，这个分发键并不执行命名张量的任何实质性功能（尽管理论上它可以！）。
  // 目前，它仅负责在操作不支持命名张量时提供清晰的错误信息。
  // 注意：如果你考虑将named tensor的功能迁移到这个dispatch key中，请注意可能需要添加另一个在复合(composite)操作符之前触发的分发键。
  // 这是因为composite operator可能具有与其组成部分不匹配的命名维度传播方式。
  // TODO: 一旦torchdim在functorch 仓中ready，就可以删除这个key
  Named = 18,

  // 共轭分发键 : 为任何需要执行共轭操作的张量而设置。这是在任何后端运行之前，在分发层级上实现的。
  Conjugate = 19,  // implemented at a dispatch level right before any backends run
  Negative = 20,   // 为每个张量设置负数分发键 : implemented at a dispatch level right before any backends run
  ZeroTensor = 21, // registered at build/aten/src/ATen/RegisterZeroTensor.cpp

  // Note [ADInplaceOrView key] --> AD : AutoGrad
  // ADInplaceOrView key被用于在inplace/view ops中注册一个kernel，该kernel用于为未来的自动微分计算(autograd computation)做额外的设置。
  // 1. 对于 inplace ops，这个kernel执行版本升级;
  // 2. 对于 view ops，这个kernel执行 `as_view` 设置，其中我们正确设置了view张量的 DifferentiableViewMeta信息。
  // 对于其他操作，由于没有额外的工作要做，所以它们会使用默认的内核（fallthrough kernel）。
  //
  // Note [Dream: skip VariableType kernel when requires_grad=false]
  // 理想情况下对于inputs with requires_grad=false，我们可以跳过VariableType kernel，而不是访问一个fallthrough kernel.
  // 我们可以像如下这样为所有functional ops注册一个kernel：
  // torch::Tensor my_functional_op(...) {
  //   {
  //     // 对于VariableType中的每一个操作，你都需要恰好通过一次AutoDispatchBelowADInplaceOrView保护机制来将键添加到TLS排除集中。
  //     // 如果你完全没有通过这个保护机制，那么在你的后端kernel中通过at::调用的（inplace/view）ops将会dispatch到ADInplaceOrView内核，并执行大量的额外工作。
  //     at::AutoDispatchBelowADInplaceOrView guard;
  //     at::redispatch::my_functional_op(...);
  //   }
  // }
  // 但是，这项工作目前受到了阻碍，因为它会为所有操作增加一个额外的分发步骤，这在模型级别上会带来不小的开销（几个百分点）。
  // 因此，我们当前的方法利用了每个kernel首先都会经过VariableType kernel这一事实，并将函数操作的at::AutoDispatchBelowADInplaceOrView保护机制上移到VariableType内核中。
  // 这样，我们只给view/Inplace ops增加了额外的分发步骤，以最大限度地减少它对实际模型性能的影响。
  ADInplaceOrView = 22,

  // Note [Alias Dispatch Key : Autograd]
  // 所有后端都未感知到autograd, autograd 被处理为在所有backends的顶层。它检查所有输入的autograd metadata，
  // 确定output应该构造的autograd元数据，并将控制权转移到后端以实际执行数值计算。autograd包含了大部分此逻辑。
  // 
  // Autograd 现在是一个dispatch key 的别名，它默认映射到所有后端特定的自动微分（autograd）key。
  // 后端特定的key允许后端根据需要覆盖默认注册到 Autograd key的kernel。
  // 例如，XLA（一种加速线性代数运算的后端）希望直接为 einsum（爱因斯坦求和约定）op定义自动微分。
  // 但是，在 XLA 键下注册自定义的自动微分实现是行不通的，因为我们在处理 Autograd 之前就已经处理了 XLA。
  // Autograd 键具有更高的优先级，并且会首先被处理。因此，在处理完自动微分之后，通常不应该再进行重新调度.
  // 因为这会导致执行自动微分操作符，而这正是你想要避免的。
  // 在 AutogradXLA 的实现中，你需要自己负责处理自动微分，或者将其委托给其他支持自动微分的操作符。
  //  
  // 当前我们只为CPU/CUDA/XLA后端定义了特定的自动微分键，并为用户保留了user-defined backends的空间。
  // 其他in-tree kernel内核共享AutogradOther键。如果有需要，可以为其他后端添加特定的自动微分键。
  AutogradOther = 23,

  // See [Note: Per-Backend Functionality Dispatch Keys]
  AutogradFunctionality = 24,

  // NestedTensor 是一个并非“真实后端”的例子（因为它主要由redispatch kernel组成），但它希望在 C++ 中重写自动微分（autograd）功能。
  // 我们可以通过添加一个专门用于处理 NestedTensor 的自动微分功能键(functionality key)来处理此类情况。
  // 该项目在 PyTorch 主代码库之外(out-of-tree)维护，地址是：https://github.com/pytorch/nestedtensor。
  AutogradNestedTensor = 25,

  Tracer = 26, // 用于跟踪计算图的键，在计算图中，Tracer 节点表示计算图的输入，而非实际的计算。

  // TODO: make Autocast a functionality key
  // “自动类型转换优先于VariableTypeId，以确保类型转换能够暴露给自动微分（autograd）系统，
  // 并且转换后的类型中的输入被保存下来以供反向传播使用。”
  AutocastCPU = 27,
  AutocastXPU = 28,
  AutocastIPU = 29,
  AutocastHPU = 30,
  AutocastXLA = 31,
  AutocastCUDA = 32, // AutocastXLA 目前仅应用于TPU，XLA GPU仍然使用AutocastCUDA
  AutocastPrivateUse1 = 33,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ 装饰 ： WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // 在自动微分（autograd）之前，可能需要处理一些替代模式(alternative modes)，
  // 例如错误检查(error checking)、跟踪(tracing)、分析(profiling)或向量映射(vmap)。它们放在这里。

  FuncTorchBatched = 34,     // See Note [Out-of-tree vmap+grad prototype]
  FuncTorchVmapMode = 35,    // See Note [Out-of-tree vmap+grad prototype]
  Batched = 36,              // BatchedTensorImpl 对应的 dispatch key, 在vmap实现中被用于实现batching rules
  VmapMode = 37,             // 在vmap模式下，所有张量都被调度到这个key上。See Note: [DispatchKey::VmapMode usage] for more details.
  FuncTorchGradWrapper = 38, // See Note [Out-of-tree vmap+grad prototype]

  DeferredInit = 39,  // 延迟初始化，在torchdistx中使用，非核心key。See https://pytorch.org/torchdistx/latest/deferred_init.html

  // 由Python key逻辑使用，以了解进入dispatcher时的TLS(线程局部存储 thread local storage)集合。
  // 此kernel假定它是与functorch无关的最顶层的DispatchKey。
  // 如果您在其上方添加了一个键，请确保更新此key的fullback实现。
  PythonTLSSnapshot = 40,

  // This key should be at the very top of the dispatcher
  FuncTorchDynamicLayerFrontMode = 41, // See Note [Out-of-tree vmap+grad prototype]

  // 测试：这是一个旨在用于通用测试的张量类型标识符。不要将其用于任何实际用途；
  // 它仅在单个进程测试中使用是可接受的。使用方法是通过此DispatchKey创建一个TensorImpl，然后注册操作符以对此类型标识符进行操作。
  // 有关使用示例，请参阅aten/src/ATen/core/dispatch/backend_fallback_test.cpp。
  TESTING_ONLY_GenericWrapper = 42,

  // 测试：这是一个旨在用于通用测试的张量类型标识符。不要将其用于任何实际用途；
  // 它仅在单个进程测试中使用是可接受的。使用方法是通过TESTING_ONLY_tls_generic_mode_set_enabled函数来开启和关闭该模式，然后注册操作符以对此类型标识符进行操作。
  // 有关使用示例，请参阅aten/src/ATen/core/dispatch/backend_fallback_test.cpp。
  TESTING_ONLY_GenericMode = 43,

  // 这个key用于在make_fx中进行预分发追踪(pre-dispatch tracing)。它的优先级低于PythonDispatcher键，
  // 因为我们使用PythonDispatcher来从Python中拦截该键，从而避免在C++中实现它。
  PreDispatch = 44,

  // 这是一个旁路(bypass)，允许您完全跳过运行C++分发器。
  PythonDispatcher = 45,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  EndOfFunctionalityKeys = 46, // End of functionality keys.

// ~~~~~~~~~~~~~~ "Dense" Per-Backend Dispatch keys ~~~~~~~~~~~~~~~~~~~~ //
// 以下是您通常认为的后端，它们传统上指定了如何在某些设备上实现操作。

#define DEFINE_PER_BACKEND_KEYS_FOR_BACKEND(n, prefix) prefix##n,

#define DEFINE_PER_BACKEND_KEYS(fullname, prefix)      \
  StartOf##fullname##Backends,                         \
      C10_FORALL_BACKEND_COMPONENTS(                   \
          DEFINE_PER_BACKEND_KEYS_FOR_BACKEND, prefix) \
          EndOf##fullname##Backends = prefix##Meta,

  C10_FORALL_FUNCTIONALITY_KEYS(DEFINE_PER_BACKEND_KEYS)

#undef DEFINE_PER_BACKEND_KEYS
#undef DEFINE_PER_BACKEND_KEYS_FOR_BACKEND

      EndOfRuntimeBackendKeys = EndOfAutogradFunctionalityBackends, // 126 

  // ~~~~~~~~~~~~~~~~~~~~~~ Alias Dispatch Keys ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // Note [Alias Dispatch Keys]
  // Alias Dispatch key是合成的(synthetic)分发键，它们可以映射到多个运行时分发键。
  // 别名键具有优先级(precedence)，但它们的优先级总是低于运行时键。
  // 您可以将一个kernel注册到一个alias key上，在dispatch table计算期间，该kernel可能会被填充到映射的运行时键中。
  //
  // 如果一个运行时dispatch key从 alias key中获得了多个kernel，
  // 那么哪个kernel会被选中是基于alias key的优先级来决定的（但运行时键总是优先于别名键）。别名键在运行时不会被直接调用。
  // See Note [Alias Dispatch Key : Autograd]
  Autograd = 127,
  CompositeImplicitAutograd = 128,              // registered at */build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp

  // FuncTorchBatchedDecomposition的alias keyset与其他所有alias keyset别名键集都不相交，因此优先级顺序无关紧要。
  FuncTorchBatchedDecomposition = 129,          // registered at build/aten/src/ATen/RegisterFuncTorchBatchedDecomposition.cpp

  // CompositeImplicitAutogradNestedTensor的 alias keyset来自所有其他 alias keyset。
  CompositeImplicitAutogradNestedTensor =130,   // registered at */build/aten/src/ATen/RegisterCompositeImplicitAutogradNestedTensor.cpp
  CompositeExplicitAutograd = 131,              // registered at */build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
  // See Note [CompositeExplicitAutogradNonFunctional Key]
  CompositeExplicitAutogradNonFunctional =  132, // registered at */build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

  // Define an alias key to represent end of alias dispatch keys.
  // If you add new alias keys after Autograd, please also update it here.
  StartOfAliasKeys = Autograd,
  EndOfAliasKeys = CompositeExplicitAutogradNonFunctional,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't be used
  CPUTensorId = CPU,
  CUDATensorId = CUDA,
  DefaultBackend = CompositeExplicitAutograd,
  PrivateUse1_PreAutograd = AutogradPrivateUse1,
  PrivateUse2_PreAutograd = AutogradPrivateUse2,
  PrivateUse3_PreAutograd = AutogradPrivateUse3,
  Autocast = AutocastCUDA,
};
```

- per-backend 实例化后的结果
```cpp
StartOfDenseBackends,CPU,CUDA, HIP, XLA, MPS, IPU, XPU, HPU, VE, Lazy, MTIA, PrivateUse1, PrivateUse2, PrivateUse3, Meta, EndOfDenseBackends = Meta,
StartOfQuantizedBackends, QuantizedCPU, QuantizedCUDA, QuantizedHIP, QuantizedXLA, QuantizedMPS, QuantizedIPU, QuantizedXPU, QuantizedHPU, QuantizedVE, QuantizedLazy, QuantizedMTIA, QuantizedPrivateUse1, QuantizedPrivateUse2, QuantizedPrivateUse3, QuantizedMeta, EndOfQuantizedBackends = QuantizedMeta,
StartOfSparseBackends, SparseCPU, SparseCUDA, SparseHIP, SparseXLA, SparseMPS, SparseIPU, SparseXPU, SparseHPU, SparseVE, SparseLazy, SparseMTIA, SparsePrivateUse1, SparsePrivateUse2, SparsePrivateUse3, SparseMeta, EndOfSparseBackends = SparseMeta,
StartOfNestedTensorBackends, NestedTensorCPU, NestedTensorCUDA, NestedTensorHIP, NestedTensorXLA, NestedTensorMPS, NestedTensorIPU, NestedTensorXPU, NestedTensorHPU, NestedTensorVE, NestedTensorLazy, NestedTensorMTIA, NestedTensorPrivateUse1, NestedTensorPrivateUse2, NestedTensorPrivateUse3, NestedTensorMeta, EndOfNestedTensorBackends = NestedTensorMeta,
StartOfAutogradFunctionalityBackends, AutogradCPU, AutogradCUDA, AutogradHIP, AutogradXLA, AutogradMPS, AutogradIPU, AutogradXPU, AutogradHPU, AutogradVE, AutogradLazy, AutogradMTIA, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, AutogradMeta, EndOfAutogradFunctionalityBackends = AutogradMeta,
```

# 3 DispatchKeySet
```c++
// An undefined tensor is one with an empty tensor type set.
class DispatchKeySet final {
 public:
  enum Full { FULL };
  enum FullAfter { FULL_AFTER };
  enum Raw { RAW };
 private:
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;

 public:
    DispatchKey highestFunctionalityKey() const;
    BackendComponent highestBackendKey() const;
    DispatchKey highestPriorityTypeId() const;
    uint8_t indexOfHighestBit() const;  

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

  uint64_t getBackendIndex() const {
    return DispatchKeySet((repr_ & full_backend_mask) >> 1).indexOfHighestBit();
  }
};

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在runtime阶段会有一个根据需要将最高优先级的functionality index 和 backend index 映射成 DispatchTableIndex的过程.
DispatchKeySet::getDispatchTableIndexForDispatchKeySet中实际实现上述过程。在此过程中会有一个offset和mask的计算过程，这个过程是如何实现的呢？ <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如下述代码所示，每个functionality都会初始化一个offset和mask。<br>
```c++
// num_functionality_keys = 46
std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
  std::array<FunctionalityOffsetAndMask, num_functionality_keys>
      offsets_and_masks;
  // manually set the first entry, which corresponds to Undefined.
  offsets_and_masks[0] = FunctionalityOffsetAndMask(0, 0);                       // step1: 初始化第一个entry offset=0, mask=0
  // loop through every functionality key (aside from Undefined).
  for (const auto functionality_idx : c10::irange(1, num_functionality_keys)) {
    // functionality_idx should be Dense -> 1, ...
    auto prev_offset_and_mask = offsets_and_masks[functionality_idx - 1];        // step2: 获取前一个functionality的offset和mask
    auto k = static_cast<DispatchKey>(functionality_idx);                        // step3: index 转换成 DispatchKey

    // If the previous functionality was not per-backend, then we can just
    // increment the previous offset. Otherwise, the next offset =
    // previous_offset + num_backends.
    auto next_offset = prev_offset_and_mask.offset +
        (prev_offset_and_mask.mask == 0 ? 1 : num_backends);                    // step4: next_offset = 前entry的offset + 1 或者前entry的offset + num_backends
    // the mask is used in the runtime index calculation to find the offset of
    // the backend. For non-per-backend functionalities, this offset should
    // always be 0. Otherwise, we need to get the index of the backend (which we
    // can do using a backend mask).
    auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;    // step5: next_mask = is_per_backend_functionality ? full_backend_mask : 0
    offsets_and_masks[functionality_idx] =
        FunctionalityOffsetAndMask(next_offset, next_mask);
  }
  // Sanity check that the computed offset index of the last functionality key
  // is correct. This assumes that the highest priority functionality key is not
  // per backend.
  TORCH_INTERNAL_ASSERT(
      offsets_and_masks[num_functionality_keys - 1].offset ==
          (num_runtime_entries - 1),
      "num_runtime_entries: ",
      num_runtime_entries,
      "last_offset: ",
      offsets_and_masks[num_functionality_keys - 1].offset);
  return offsets_and_masks;
}
```


# 4 参考文档
- [lets-talk-about-the-pytorch-dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)