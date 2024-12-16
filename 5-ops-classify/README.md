# pytorch 算子分类汇总
- [参考自native_functions.yaml](https://github.com/pytorch/pytorch/tree/v2.5.0/aten/src/ATen/native/README.md)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ATen "native" 函数是向 ATen 添加operators和functions的现代机制。Native 函数在 native_functions.yaml 中**声明**，并在此目录中的一个 cpp 文件中定义其实现。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与所有ATen方法/函数一样，native functions(原生函数)在ATen的C++和Python API中均可用。在C++中，它们可以作为Tensor上的方法(如t.mymeth())或ATen命名空间中的函数（如at::myfunc()）来使用。在PyTorch中，它们可以作为Variable上的方法或**torch._C._FunctionBase**上的函数来使用。（用户有责任将这些函数重新导出到更面向用户的模块中。）<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文档的其余部分描述了如何实现一个ATen函数。<br>

# 1 在native_functions.yaml 中注册一个函数
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每个原生函数都必须在native_functions.yaml中有一个条目。其格式可以概括为：<br>
```yaml
- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是每个部分的更详细描述：<br>

## 1.1 func
```yaml
- func: func_name[.overload_name](ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**func** entry(条目)是一个字符串，描述了函数的名称及其类型签名。Argument types(参数类型)。如下这些类型是作为ArgType允许的:<br>

- **Tensor.** 一个Tensor参数在C++中转化为类型为**const Tensor&的参数(默认)**（除非该参数是 **“inplace”的，在这种情况下它是Tensor&** ）。跟在Tensor后面的?，如**Tensor?，表示该张量参数是可选的**，可以通过传递**std::nullopt**来省略。当一个函数接受多个Tensor参数时，这些张量被假定为同一类型（例如，如果一个参数是FloatTensor，那么所有其他参数**都会被检查是否为FloatTensors**）。有时需要对Tensor或Tensor?进行注释，以指示别名和可变性。一般来说，注释可以通过以下情况来定义: <br>
1. Tensor(a) - a是一组可能alias到相同数据的张量。这个集合的大小可能为一。<br>
2. Tensor(a!) - a中的成员可以被写入，从而修改底层数据。<br>
3. Tensor(a! -> a|b) - 张量在集合a中，被写入后，在写入之后同时属于集合a和b。关于何时以及为什么需要这样做，请参见关于注释的详细部分。<br>
- Tensor[]. 一个Tensor[]参数在C++中转化为类型为ArrayRef<Tensor>（也称为TensorList）的参数。
- int[]. int[]接受一个可选的长度说明符，例如int[2]，这在C++中没有影响，但扩展了我们的Python绑定，使其能够接受一个裸露的数字，该数字将通过重复来扩展为适当大小的列表。
- int. 可以将其视为Python中的int类型。这在C++中被转化为类型为int64_t的参数。
- float.可以将其视为Python中的float类型。它在C++中被转化为类型为double的参数。
- bool.
- str. 它被转化为C++中非拥有类型的参数c10::string_view.
- Scalar. Scalar支持从Python绑定到任何数值类型，包括整数类型、浮点类型以及零维张量。int和float绑定到相应的Python数值类型。然而，你可能并不需要使用Scalar；对于大多数算法来说，float和int参数类型应该就足够了(只有当操作符确实可以接受任意类型时，你才应该使用Scalar).
- Generator?，随机数生成器的状态.
- bool[N]（其中N为1-4）。这表示一个包含1到4个布尔值的数组。在C++中，这通常会被转化为一个具有固定大小的std::array<bool, N>或者类似的类型，具体取决于实现.
- * 是一个特殊的标记参数，它并不转化为实际的参数，而是指示在Python绑定中，任何后续参数都必须以关键字参数的形式指定（而不能以位置参数的形式提供）.
- ? 是一个尾随的问号，用于标注一个参数为**可选类型**。你可以通过搜索“optional”来找到一些使用示例。一般来说，大多数函数不需要使用这个标注，但在某些情况下，我们想要为不同类型的参数使用可选性：<br>
1. 你想要从Python向ATen函数/方法传递一个None，并在C++端处理None类型。例如，clamp(Tensor self, Scalar? min=None, Scalar? max=None)可以接受其min和max参数为None，但如果其中一个参数为None，它不会分发到不同的后端。可选类型可以从Python接受一个None类型（在C++中为nullopt），并**使用C++的Optional类来与参数进行交互**. <br>
2. 你想要一个默认值，这在Python中是可以的，但在C++中会引起歧义。例如，norm(Tensor self, Scalar p=2, int dim, bool keepdim=False)在C++中会引起歧义，因为c++默认参数必须相邻（当dim没有默认值时，p不能有默认值）。因此，我们需要将p设为可选的Scalar，并在未传入p时（即nullopt）将其设为2. <br>
3. 你想要一个参数的默认值与另一个参数相同(这在C++的默认参数中无法表达). <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没有张量输入的函数被称为**工厂函数**，并且通过代码生成来特殊处理。如果你的函数与另一个示例的行为不同，请首先检查它们之中是否一个是工厂函数而另一个不是。在一些罕见的情况下，工厂函数可能会有一个张量参数(**疑问：native_functions.yaml里为啥有多个tensor的情况？？？**)。在这种情况下，请明确地使用category_override: factory来标记它。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**参数名称(argument names)** 是有意义的；下游绑定代码可能会使用你提供的特定参数名称，并且参数名称的重命名被视为一个破坏向后兼容性（BC-breaking）的更改（例如，你可能需要至少更新tools/autograd/derivatives.yaml，并且它可能会影响Python关键字参数）。有关更多详细信息，请参阅关于variants的部分。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;按照惯例，我们使用“out”来表示输出参数。这与Python绑定是一致的。即使某个函数可能不会在Python绑定中使用，我们仍然建议遵循这一惯例。在更改参数名称时，请检查生成的代码，以确保你没有在重命名现有函数的参数名称时破坏API。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**默认值(defaults)**。参数的任何后缀都可以定义一个默认值；当未指定这些位置参数时，这些默认值将转换为C++/Python中的默认值并被应用。<br>
- 数字（例如，对于int、float以及具有明确长度的int[]（如int[2]），可以使用0或5.0等数字——在int[]的情况下，一个数字会被复制以填充整个长度(例如，int[2] x=2等价于int[2] x=[2,2]))<br>
- 数字列表（例如，[0, 0]）用于IntList. <br>
- 布尔值（例如，True）用于bool. <br>
- 空初始化列表(例如，[])用于Tensor（这会隐式地将一个Tensor参数更改为接受未定义的张量）. <br>
- 对于指针类型(例如，Generator?)，使用None.<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**返回值(return)**。以下是在返回（Return）中允许的内容：<br>
- 非元组返回：
```yaml
ReturnType [retarg0]
```
- 返回元组
```yaml
(ReturnType [retarg0], ReturnType [retarg1], ...)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在ReturnType中允许以下内容：<br>
- Tensor和Tensor[]，它们分别转换为C++类型中的Tensor和std::vector<Tensor>（除非操作是就地进行的，在这种情况下返回类型是Tensor&，即张量的引用）. <br>
- 任意数量的Tensor组成的元组，例如(Tensor, Tensor)，它转换为C++中的std::tuple<Tensor, Tensor>。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果你需要一种未在此列表中列出的类型，可能可以扩展ATen的代码生成来支持它。ATen支持类型的理念是，它只支持简单、通用的类型，以及少数基本的Tensor结构（例如，Tensor和Generator?），因为这些类型可以轻松地移植到任何与ATen绑定的语言（在实践中是C++和Python）中. <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;返回（Return）也支持指定（可选的）返回参数名称。这些名称有两个作用：<br>
- 它们允许你在tools/autograd/derivatives.yaml中轻松地根据返回参数编写导数。
- 它们对应于Python中可以引用的输出命名字段。(这意味着更改返回参数名称会破坏向后兼容性（BC-breaking），请小心！)

*(请注意，目前返回（Return）上不支持默认值（defaults）和可选（optional）等参数类型修饰符。)*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**重载(Overloads)**。如果你为它们指定了唯一的重载名称，那么你可以注册多个具有相同名称但函数签名不同的函数。**重载名称在函数名称之后指定，两者之间用句点分隔(eg. linspace.Tensor_Scalar)**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;重载名称不必在全局范围内是唯一的，但**在同一函数的所有重载集合中必须是唯一的**。出于向后兼容性的考虑，重载名称不能更改。请尽量使重载名称在语义上有意义。仅仅枚举所有参数类型的重载名称并没有帮助。在许多情况下，**根据重载所做的不同之处，可以清楚地得出一个语义名称。作为备选方案，你可以使用第一个不同参数的名称或类型作为重载名称**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果你向现有函数添加一个新的重载，请保持现有重载名称不变（以确保向后兼容性），但为新的重载指定一个新的、唯一的名称。虽然重载名称不直接被Python或C++ API使用，但它们是外部后端（注册到特定重载名称）和已部署的移动模型（将重载名称作为序列化格式的一部分使用）的公共API接口。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;不指定重载名称相当于指定一个空的重载名称。如果你添加了一个具有多个重载的新函数，请为它们指定唯一的重载名称，其中**最多只允许一个重载具有空的重载名称**。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;函数声明还支持以下属性。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**命名空间(namespace)**。用户可以通过在函数名称前添加自定义命名空间，来将运算符注册到不同于aten的命名空间中。当前，函数名称不支持嵌套命名空间。如果未指定命名空间，则所有函数都将注册在aten命名空间中。例如，假设我们要将my_op注册到自定义命名空间中，我们可以这样做：<br>

```yaml
- func: custom::my_op(Tensor(a) self, ...) -> Tensor(a)
  variants: function, method
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;请注意，我们有一次性的**TORCH_LIBRARY** API来实现将operator注册到自定义命名空间的相同目标。与该API相比，在native_functions.yaml中拥有自定义命名空间在以下情况下很有用：当函数实际上并不属于ATen但也被广泛使用，并且希望有一个共享的地方来注册它时。<br>

## 1.2 variants 变体
```yaml
variants: function, method
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此声明决定了是生成Tensor方法（如t.foo()）还是命名空间函数（如at::foo()）。如果声明的是一个方法，那么必须在方法的某个位置有一个**Tensor self参数**；在方法变体中，这个参数将从参数列表中省略。例如，给定声明where(BoolTensor cond, Tensor self, Tensor other)，这将生成函数at::where(cond, self, other)和方法self.where(cond, other)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认情况下，ATen只为原生函数生成函数变体。那么，何时还应该生成方法变体呢？将张量操作作为方法是适用于“核心”张量操作（例如，add、sub等）的，但并不适用于更复杂的神经网络层（例如，conv2d）以及专门为绑定而设计的内部函数（例如，cudnn_convolution）。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随着我们在函数模式与JIT签名模式的模式统一方面取得进展，我们必须引入一些功能来提高合规性。这些功能之一就是张量注解。到目前为止，我们使用命名约定(naming conventions)来指示函数的某个参数是否将被修改并返回。<br>

## 1.3 annotations 注解

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Python前端中，我们有两种典型的情况会修改参数的内存：<br>
- a) 对于inplace 操作(operations)，如self.abs_();
- b) 对于具有输出关键字参数的函数，如torch.abs(input, out=None)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了为这些Python函数提供实现，传统模式需要为三种情况提供C++实现：abs(Tensor self) -> Tensor、 abs_(Tensor self) -> Tensor 和 abs_out(Tensor out, Tensor self) -> Tensor。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;现在，随着我们向模式统一迈进，我们开始使用不同的语法来表示这一点，即使用**注解(annotation)**。最终，我们仍然会为下游消费者（如C++代码生成）翻译成传统模式(legacy schema)，但这种情况很快就会改变。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果两个张量携带相同的annotation，那么它们可能表示**相同的内存**。由感叹号表示的写注解表明它们也都可以被写入。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;让我们再次回顾之前的原生函数声明，并了解添加注解的惯例。<br>
- abs(Tensor self) -> Tensor 保持不变，因为它总是会分配新的内存. <br>
- abs_(Tensor(a!) self) -> Tensor(a!) 表示 self 可能会被写入并返回。此外，注解表明返回值可能与输入别名相同。这表示一个就地函数，并且按照约定，其名称以单个下划线 _ 结尾. <br>
- abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!) 在Python前端中，out 可以作为关键字参数传递，并且可能会被写入。在这种情况下，它表示一个必须接受 out 参数的函数的模式，因为这里没有提供默认参数。将其表示为可选参数的想法是为了记录预期的使用方式。这映射到传统的 abs_out(Tensor out, Tensor self) -> Tensor。与传统的 _out 函数一样，在多个参数的上下文中，你必须将参数命名为 Tensor out 或 Tensor out0、Tensor out1. <br>

*(注意：上述中 a 就是annotation)*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;还有一种情况我们会使用注解，那就是视图（views）。<br>
- transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)：这个函数可能会返回一个别名，该别名指向由 self 表示的内存，但是这块内存并不会被修改。<br>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当一个张量列表中包含张量视图时，我们需要表示**输出列表包含与输入别名的张量**。
- 函数：chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]。我们假设列表包含与堆别名的内存，因此为了正确设置输出和输入之间的别名关系，我们**注解annotation输入张量进入通配符集(wildcard set)(a -> *)**。有关更多详细信息，请参阅[JIT README](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#aliasing-and-mutation-annotations-in-functionschema)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们有一些断言来检查开发者是否正确使用了这些注解，并且如果他们没有正确使用，就会抛出断言错误。例如，**任何带有out参数的函数都必须使用上面描述的(a!)注解**。如果这造成了很大的困惑，请在你的PR（Pull Request，拉取请求）中添加@cpuhrsch。

## 1.4 dispatch 分发
```yaml
dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这指定了您**想要分发到的函数的实际名称**，因此您可以根据传递的张量属于哪个后端来分发到不同的函数。请注意，这些名称**支持自定义命名空间**，当列出的原生函数位于除默认命名空间at::native之外的其他命名空间中时，这非常有用。目前我们支持最多两级嵌套命名空间。例如：<br>

```yaml
dispatch:
    CPU: custom::ns::func_cpu
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面的例子暗示了原生函数可以在custom::ns::native命名空间下找到(末尾的 **::native是自动添加的**)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果dispatch table 被省略了，我们则假定使用默认的分发表。<br>

```yaml
# overload is ignored
func: func.overload(...) -> ...
dispatch:
    CompositeImplicitAutograd: func

# overload is ignored, but out functions get suffixed with _out in their name
# (NB: no out functions in PyTorch today actually support autograd, but if they
# did, you could call them here and autograd would be inferred)
func: func.out_overload(...) -> ...
dispatch:
    CompositeImplicitAutograd: func_out
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果两个后端有相同的调度函数，你可以分别为CPU和CUDA编写：func，以便在这两种情况下重用相同的函数名。通过搜索[codegen](https://github.com/pytorch/pytorch/blob/master/torchgen/gen.py) 中的dispatch_keys可以找到可用的后端选项。此外，还有三个特殊的"generic通用"后端：<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**CompositeExplicitAutograd（之前称为DefaultBackend）**：这是适用于所有后端的kernel实现，但**需要在derivatives.yaml中明确定义反向函数以支持自动微分（autograd）**。此键的**最典型用途是用于委托(delegating)函数**; 即那些只执行少量工作，然后**将实际的大量计算任务委托给另一个操作符的函数**。在底层实现中，将内核注册到CompositeExplicitAutograd相当于将该内核注册到每个后端（例如，CPU、CUDA）。注意：**调用DispatchStub的内核不应注册为CompositeExplicitAutograd，因为DispatchStub仅适用于CPU和CUDA。** <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**CompositeExplicitAutogradNonFunctional:** 与CompositeExplicitAutograd类似，但此键应在以下情况下使用：（1）您的kernel是为非别名操作符(non-aliasing operator)编写的。（2）并且它内部调用了一个别名操作符(aliasing operator)。这方面的一个例子是select_backward，它是非别名的，但会分解为select。我们希望区分“普通”的CompositeExplicitAutograd kernel和这些内核，因为一些后端不希望将一个非别名操作分解为别名操作。LazyTensor和XLA是当前的两个示例——由于它们操作的是函数式中间表示（IR），因此它们更倾向于直接使用自己的内核来实现非别名操作符，而不是使用会导致更多别名操作符的分解(decomposition)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**CompositeImplicitAutograd(之前称为Math)**：这是适用于所有后端的kernel implementations，并且能够**隐式支持自动微分（autograd）**，因为**它调用的所有操作都支持自动微分**。直接使用这个键的情况应该很少：如果您没有提供dispatch table，我们会默认将您的内核注册为CompositeImplicitAutograd。如果您有专门的CPU和CUDA实现，但希望为可能没有专门实现的外部后端提供一个(fallback lowering)选项，那么在现有的分发表中显式添加这个键可能是有用的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注册到复合后端(Composite backends)的函数应该能够在任何后端上工作，只要它们调用的嵌套函数(nested functions)也能在那些后端上工作。例如，假设my_op可以以以下方式实现：<br>

```yaml
at::Tensor my_op(const Tensor& self, const Tensor& other) {
  return self + 2 * other;
}
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果我们系统中已经知道了加法（+）和乘法（*）算子的inference kernel和 导数公式(derivative)，您**只需要**将my_op注册到CompositeImplicitAutograd中，推理和自动微分（autograd）就会正常工作。虽然这里看起来我们只写下了推理公式，但PyTorch的自动微分系统会利用链式法则以及加法和乘法算子的导数，正确地为my_op设置反向传播。换句话说, $\frac{d_out}{d_self} = 1$ 和 $\frac{d_out}{d_other} = 2$ 可以从my_op的inference kernel中自动推导出来。当然，如果我们没有为加法或乘法算子定义导数公式，那么my_op的反向传播就无法自动推导了。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;决定您的内核是使用隐式还是显式自动微分（autograd），可以通过以下步骤来确定: <br>
1. 如果可能的话，首先尝试使用由现有算子组合而成的CompositeImplicitAutograd内核。<br>
2. 如果您不希望使用CompositeImplicitAutograd内核推导出的**梯度公式来进行自动微分(仅仅对梯度而言)**，以获取更好的性能或数值稳定性，那么您应该使用CompositeExplicitAutograd来注册内核，以便它(组合的kernel)仅用于前向推理(inference)。之后，对于自动微分，根据您的自动微分kernel是否适用于所有后端，您可以将它们(自己的kernel)放在别名Autograd或特定键（如AutogradCPU）下(即反向kernel需要自己实现).<br>
3. 如果您更倾向于编写特定于后端的kernel(前反向都包含)，请使用为您的后端保留的分发键，例如CPU/AutogradCPU。<br>

**重要提示:** 因为没有 dispatch : section 的ops 会隐式注册CompositeImplicitAutograd kernel。因此，当您为这些ops之一添加一个特定于后端的内核(以及相应的dispatch:sect)时，您还必须添加一个CompositeImplicitAutograd:entry，该条目应命名旧的kernel实现（通常根据op命名，如果适用，会添加下划线_），以便**其他后端**仍然可以使用它。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果您在C++中实现了一个原生函数，并且想要了解在native_functions.yaml中应该使用哪个分发关键字（dispatch keyword），请遵循分发关键字的相关步骤(下文Choosing the right dispatch keyword)。<br>

## 1.5 Composite Compliance 复合合规性
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;定义：“复合函数(composite function)”是指注册为CompositeImplicitAutograd的算子，或者是由PyTorch operations组成的（Python或C++）函数。后者的示例包括反向公式和前向模式自动微分公式。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在PyTorch库中定义的复合函数必须适用于大多数（如果不是全部）**后端/子类**。这意味着我们施加了一系列约束，使得在PyTorch库代码中编写复合函数比用户编写PyTorch代码更加困难。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果您希望执行被禁止的操作（可能是出于性能原因），请**为您的函数编写一个反向公式**，以便它不再将函数的部分内容隐藏在不是CompositeImplicitAutograd的新aten算子中。<br>

复合函数不得：<br>
1. 调用resize_或其等效操作。这些操作对于许多后端（如vmap和meta）来说很难处理.
2. 调用带有out=参数的操作。vmap无法处理这类操作，而且可能导致派发到Python的对象失去其子类化特性.
3. 在不执行分发的情况下更改张量的元数据。这类操作的例子包括直接访问TensorImpl API来修改张量的大小/步长/元数据。
4. 与最后一点相似，不允许访问data_ptr或进行元素访问。这些操作不会通过分发器。
5. copy_是一个边缘情况。如果您能够重写您的操作而不使用copy_，那么您绝对应该这样做；如果您不是将内容复制到一个视图（view）中，这应该是很容易做到的。否则，保持代码原样也是可以接受的。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们在test/test_ops.py中有CompositeImplicitAutograd合规性测试。这些测试并不完美（因为检查上述所有情况相当困难），所以如果发现有任何问题，请大声指出来。<br>


## 1.6 device_guard
```yaml
device_guard: False
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认情况下，ATen代码生成会生成一个DeviceGuard调用，该调用将确保kernel代码在与第一个张量参数（或者如果函数接受张量列表，则是第一个张量列表参数中的第一个张量）的设备相匹配的设备上运行。在大多数情况下，这意味着内核作者不必担心设置设备。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;然而，在某些情况下，设置设备是不必要的，因为例如，你调用的函数已经管理了设备保护设置，或者你的函数根本不与任何设备交互。在这种情况下，可以通过在函数定义中添加device_guard: False来**禁用设备保护的代码生成**。<br>

## 1.7 device_check
```yaml
device_check: NoCheck
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;默认情况下，ATen代码生成会进行设备检查，以确保传递给kernel的**所有张量参数都在同一设备上**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;然而，在某些情况下，检查设备是不必要的，因为例如，你调用的函数允许在多个设备上工作。在这种情况下，可以通过在函数定义中添加device_check: NoCheck来禁用设备检查的代码生成。<br>

## 1.8 manual_kernel_registration
```yaml
manual_kernel_registration: True
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当设置了这个标志时，我们不会生成代码来自动将C++算子实现注册到TypeDefault（即catchAll分发关键字）与dispatcher。对于同一个op来说，同时拥有dispatch 部分和manual_kernel_registration: True是没有意义的。你可以在torch/csrc/autograd/VariableTypeManual.cpp中找到手动注册的部分。目前，将此字段设置为True的操作应该与tools/autograd/gen_variable_type.py中的MANUAL_CATCHALL相匹配（它可以是MANUAL_CATCHALL的超集，但我们没有这样的用例）。这个字段应该很少使用。<br>

## 1.9 use_const_ref_for_mutable_tensors
```yaml
use_const_ref_for_mutable_tensors: True
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当设置了这个标志后，我们将为那些底层数据可能会改变的张量生成const Tensor&（或类似）的参数，就像我们为其他张量所做的那样。以前，我们**生成的是Tensor &类型的参数**，这种参数有两个问题：1）它允许改变张量本身所引用的TensorImpl；2）它并不是必需的，因为**底层数据的改变并不需要通过修改张量引用来实现**。（这就像是当我们需要const T*（指向常量的指针）时(指向的内容不能该)，却错误地使用了T * const（常量指针）(指针本身不能改，但内容可改)。）

## 1.10 autogen
```yaml
- func: my_op_(Tensor(a!) self) -> Tensor(a!)
...
  autogen: my_op, my_op.out
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;autogen关键字用于指定代码生成系统应该为哪个原生函数生成实现。<br>
- 对于原生函数的inplace变体（op name以_结尾），我们将生成一个函数式变体和一个out=变体。
- 如果给出了函数式变体，我们将生成一个out=变体。
- 我们不支持为视图操作、绕过分发器的操作以及复合操作使用autogen。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们还会为生成的op生成kernel，这些内核仅仅是从基础操作中复制并返回结果。这些生成的内核可以在<gen-out>/aten/src/ATen/CompositeViewCopyKernels.cpp中找到。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同时请注意，对于添加到native_functions.yaml中的新操作符，如果它们满足上述要求，则应该包含autogen关键字，因为函数化依赖于它。我们将在代码生成过程中强制执行这一点。<br>


# 2 Writing an implementation in C++
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;原生函数的实现位于native/目录中的相应C++文件中（这些文件大致按topic组织，但除了cuda目录外，它们的组织没有语义上的意义，因为cuda目录是构建系统唯一知道如何构建.cu文件的地方）。要编写一个native函数，你只需要编写一个与ATen meta data中生成的头文件签名相匹配的C++实现（不需要头文件）。有许多简单的原生函数；你可以看看其中的一些来了解该怎么做。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然编写ATen函数主要是实现你想要的算法，但还有一些不那么明显的细节你也应该考虑。<br>

## 2.1 Will your function be automatically differentiable ?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果你正在编写一对函数foo和foo_backward，并且意图是foo_backward实现foo的导数，那么你的foo实现可能不是自动可微分的：它可能会使用像data_ptr()这样的函数，或者根据它是在CPU还是CUDA张量上操作而进行不同的分发。一旦你编写了这两个函数，你需要在tools/autograd/derivatives.yaml中编写一个条目来将它们关联起来。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;然而，在某些情况下，你可以在ATen中编写一个函数，并且它会自动被微分！当函数实现仅调用其他本身可微分的操作时，就可能出现这种情况。在这种情况下，你不需要在tools/autograd/derivatives.yaml中编写条目。<br>

## 2.2 Choosing the right dispatch keyword
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在C++中编写完原生函数后，选择native_functions.yaml中使用的分发关键字(dispatch keyword)至关重要，因为它为分发器提供了有关实现的后端和自动微分支持的信息。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是决定正确分发关键字的步骤：<br>

**step1：考虑inference 前向：你的内核是否适用于所有后端？** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**否:** 你可能为不同的后端提供了不同的内核，例如，在实现中使用了依赖于后端的逻辑，或者它是通过**DispatchStub**实现的。DispatchStub仅在你明确通过REGISTER_DISPATCH提供一个内核时才支持一个后端。通常，它只支持一些内置后端，如CPU、CUDA、QuantizedCPU等，而不支持外部后端，如XLA。请编写一个分发部分(dispatch)，枚举所有支持的后端，并将它们指向相应的实现。<br>
```yaml
dispatch:
  CPU: kernel_cpu
  CUDA: kernel_cuda
  QuantizedCPU: kernel_quantized_cpu
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;你已经完成了。现在这个操作将在CPU/CUDA/QuantizedCPU后端的推理中被调用！<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注意：为了支持训练，你需要在derivatives.yaml中编写一个公式，因为你的后端实现不支持自动微分。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**是：你可能在实现中调用了其他at::操作。请转到步骤2。**

**step2: 考虑训练，你的kernel是否支持autograd ？**
**是：** 换句话说，你提供了一个支持inference和自动微分的CompositeImplicitAutograd kernel。为了使用自动微分支持进行训练，只需**跳过添加分发dispatch部分**，你就完成了。这将使这个op能够正确地注册用于推理和训练。<br>
**是:**，但如果你仍然想提供一个数值稳定的梯度公式而不是使用自动微分，那么请编写（该公式）。
```yaml
dispatch:
  CompositeExplicitAutograd: kernel
```
*(注意：这里显示声明需要一个backward kernel)*<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;你已经完成了。这个操作将在所有后端的推理中被调用。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注意：为了支持训练，你需要添加一个自动微分的公式，否则在反向传播时调用具有requires_grad=True的张量时会出错。<br>

**否：此类操作主要使用的是_out样板代码，而其out版本没有定义导数公式。例如：




















