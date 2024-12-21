# 定义变量上方法的导数公式和Python签名.

# 术语
- 输出梯度（output gradient）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;指的是前向函数输出的梯度。输出梯度被用作后向函数的输入。grads是一个包含输出梯度的向量，而在本文件中的所有导数公式中，grad等同于grads[0]。<br>

- 输入梯度（input gradient）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;指的是forward functions输入的梯度。输入梯度是backward function的输出，对应于本文件中定义的导数公式中包含的输入名称。<br>

此外，每当我们谈论计算“梯度”时，我们实际上指的是使用给定的“输出梯度”作为向量来计算向量雅可比乘积。<br>

# 2 entry 解释
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每个条目包含：<br>
- name: 它指定了您正在为其**定义导数**的ATen函数名称以及参数规范。
- optional dispatch: 它可以用于指定每个autograd分发键的导数。如果未指定此条目，则梯度条目将被视为默认梯度（即为每个backward dispatch key注册）。（请参阅_test_autograd_multiple_dispatch以了解如何为不同的分发键注册单独的导数的示例。）允许的分发键列表（除了表示Autograd别名键的“Default”之外）可以在torchgen/model.py:AUTOGRAD_KEYS中找到。<br>
- 一个或多个梯度条目，将可微分的输入名称映射到指定如何计算其梯度的公式。请注意，单个梯度条目可以通过指定键“input1, input2”（请参阅atan2的示例）来为**多个输入名称**指定梯度公式。<br>
- 一个参数可以被标记为“非可微分”（non_differentiable）。<br>
- optional output_differentiability : 值为一个与前向函数输出数量相同长度的列表。该列表应仅包含布尔值，指定每个output是否可微分。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果对于一个返回多个元素但使用grad而不是grads[idx]的函数没有指定**output_differentiability**，那么除了第一个输出之外的所有输出都将被标记为非可微分的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果所有输出都不可微分，您也可以将该函数名称添加到gen_variable_type.py中的DONT_REQUIRE_DERIVATIVE列表中。<br>

# 3 Tensor和TensorList 参数有两种情况：
## 3.1 该参数可微分
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;即可能存在关于该参数的梯度。您应该：
- 指定该梯度的公式
- 或者指定not_implemented("function_name")作为公式，表示此功能尚未实现（但将来可能会实现，用户可以在问题中请求此功能）
## 3.2 参数不可微分的
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因为数据类型非浮点或者该参数对应的function不可微分，您应该：<br>
- 不要为该参数指定任何公式;
- 或者显示指定该参数是不可微分的，在这种情况下，我们相信您保证此参数永远不会设置为requires_grad=True，并且如果设置了，它将被默默地忽略。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果一个函数有非原地（out-of-place）和原地（in-place）两种变体，那么inplace variable的导数定义是可选的。它将默认采用非原地变体的定义。请注意，**带有_out后缀的变体永远是不可微分的**。<br>

# 4 Gradient expression
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;梯度表达式是标准的C++表达式，对ATen变量(variables)进行操作。**在梯度表达式中，以下变量/函数是可用的(in scope)**：<br>
1. **grad** <br>
- 输出的梯度，在python中常写作grad_output, 我们将对其进行左乘(**计算weight梯度时在乘号左边**);
- 当一个函数有多个可微分的输出时，您可以使用'grads'来引用每个输出的梯度，例如，'grads[0]'，'grads[1]';
- 当一个函数返回多个可微分的且已命名的输出时，您可以使用'grad_{name}'来引用每个输出的梯度，例如，'grad_x'，'grad_y';
- 当一个函数返回**一个**可微分的输出（即第一个输出）以及**一些**不可微分的输出时，您必须使用'**grad**'来引用这个可微分输出的梯度（这种情况在我们的代码生成中有特殊处理）;
- 请注意，"output_differentiability" entry（见上文）可以修改可微分输出的数量;
- 在一个可微分函数的导数集中(不同varaiable的导数)，不允许混合使用“grad”、“grads”和“grad_{name}”。对于该可微分函数，您必须保持一致;

2. **Declarations** : torch/share/ATen/Declarations.yaml <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;任何输入参数，无论是张量还是非张量，包括仅在Declarations.yaml中出现的参数名称，例如'output'。<br>

3. **result** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**评估ATen原生函数声明的正向表达式的结果**。如果正向表达式输出一个元组，则使用"resultX" 来访问第X个条目。<br>

4. **grad_input_mask**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个std::array<bool, n>类型，用于指定哪些输入的梯度是实际需要的。例如，在条目input0, input1: foo(grad_input_mask)中，grad_input_mask是一个大小为二的数组，其中grad_input_mask[0]为true表示input0需要梯度，grad_input_mask[1]为true表示input1需要梯度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果你的函数为一系列张量计算梯度，那么grad_input_mask将只为这个列表有一个条目，用于指定列表中是否有零个或至少一个张量需要梯度。如果我们想要支持更精细的信号传递，我们将需要一个不是std::array类型的替代变量。<br>

5. **retain_variables**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个布尔值，如果用户指定了在之后可能**再次进行反向传播**时应保留已保存的变量，则该值为true。这允许一种优化：如果我们知道变量不会被保留，就可以销毁已保存的缓冲区。例如，_cudnn_rnn就使用了这种优化。<br>

6. **wrap_opt_if**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;是一个接受两个参数的函数，第一个参数是一个张量变量，第二个参数是一个布尔条件，用于指示**是否应在图中保存该变量**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这个函数的返回类型是std::optional<Tensor>，当条件评估为false时，返回::std::nullopt；否则，返回被std::optional<Tensor>包装的variable。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如，wrap_opt_if(var_0, grad_input_mask[1] || grad_input_mask[2]) 的意思是，只要第二个（grad_input_mask[1]）或第三个（grad_input_mask[2]）参数需要梯度，就保存 var_0。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这个表达式的另一种解释是，var_0 在第二个或第三个参数的反向计算中是必需的。注意：在条件表达式中不支持使用 var_i.requires_grad()，请改用 grad_input_mask[i]。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注意(NOTE)：wrap_opt_if 可用于**避免在多输出反向公式中保存冗余变量**。有关此问题的更多详细信息，请参阅 https://github.com/pytorch/pytorch/issues/97575。<br>

# 5 FunctionsManual : torch/csrc/autograd/FunctionsManual.cpp
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果你需要一个复杂的表达式，例如包含局部变量，那么可以在torch/csrc/autograd/FunctionsManual.cpp中编写一个_backward函数，并从这里调用它。顺便提一下，建议阅读 [这个链接](https://github.com/zdevito/ATen/issues/163) ; 它描述了一个在从Python移植到C++的反向传播过程中可能发生的重要风险。<br>

# 6 Double Backwards
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;双重反向梯度表达式可能会有些令人困惑; 最重要的是要记住以下两点：<br>
1. 你需要为每一个输入定义一个导数公式，包括那些被命名为类似'grad_output'的输入;
2. 以及（2）要相乘的梯度总是被称为'grad'（尽管它实际上是一个梯度的梯度）。

# 7 result 对应的公式
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;您还可以通过为返回值（如果未指定名称，则通常为“result”）定义一个公式来**添加前向导数定义**。这个公式**与反向公式的工作方式相同**，并且高级实现也应该放在FunctionsManual(torch/csrc/autograd/FunctionsManual.cpp)文件中。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这个公式应该使用参数“foo_p”的（原始）值、其前向梯度“foo_t”以及函数的结果“result”来计算一个单一的雅可比向量积。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;请注意，前向导数可以在以下两种情况下**自动生成**：<br>
- 如果你的函数是线性的（而非仿射或多线性的），你可以通过在公式中使用字符串"auto_linear"来指定这一点。
- 如果你的函数是逐元素应用的（并且只有一个输入），你可以通过在公式中使用字符串"auto_element_wise"来指定这一点。

# 8 TensorList
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;请注意，为了避免**解包开销**，以TensorList作为输入的函数将始终调用其**前向导数公式**。此函数负责检查是否需要执行任何计算，并且在没有任务需要执行时应返回一个未定义的Tensor。您可以查看“cat_forward”以获取完整示例。<br>

# 9 虚假梯度定义
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注意：这里有一些梯度定义是虚假的(bogus)（使用zeros_like实现）。这些梯度（希望）不会被我们的前端使用。您必须检查前端代码；搜索OpName.apply以查看它是否仍在使用旧的(legacy) Python风格API。<br>

# 10 Returning views
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;存在以下情况：<br>
- 如果一个函数不返回任何视图（view），那么它可以具有任意的输出;
- 如果一个函数至少返回一个Tensor，并且这个Tensor是其某个输入的可微视图（differentiable view）：
- - 如果只有一个可微输出，那么这个Tensor被标记为可微视图。（例如，别名或转置操作）;
- - 如果有多个可微输出，那么默认情况下，所有这些视图都被标记为可微视图，并且在创建时设置allow_rebase_history=false。这意味着对这些视图进行的任何就地（in-place）操作都会引发错误。（例如，unbind操作）.
 
# 11 关于未定义输出梯度的说明：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所有反向函数都必须支持所有未定义输出梯度Tensor的组合，即grad[i].defined() == false的情况。根据您的导数公式所使用的输入和输出梯度的数量，代码生成可能会根据以下三种情况自动添加一定程度的未定义梯度支持：<br>

## 11.1 1个输入梯度和1个输出梯度：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;会自动添加完整的未定义梯度支持，因此您无需考虑此问题，除非代码生成中存在错误。<br>

## 11.2 1个输入梯度和多个输出梯度：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;只有在所有输出梯度都未定义的情况下，才会自动添加未定义梯度支持。如果只有部分输出梯度未定义，则需要您明确添加对此类情况的支持。<br>

## 11.3 多个输入梯度：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;没有自动支持，因此您需要自行添加。<br>

## 11.4 注意事项
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果您的导数公式使用了多个输出梯度，那么通常更建议在反向函数本身（如果您正在使用一个的话）中添加对未定义梯度的支持，而不是在这个文件中的导数公式里添加。<br>

## 11.5 undefined tensor
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;未定义的Tensor是通过默认构造函数at::Tensor()创建的。它是一种表示填充了零的Tensor的高效方式(不是真填充0了)，因为这个Tensor不包含任何大小信息，也没有分配存储数据。但是，因此也无法对它们执行Tensor操作。所以，您的反向函数应该将未定义的输出梯度视为零，并且需要将其作为一个特殊情况来处理。<br>

## 11.6 所有输出梯度都未定义
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果所有输出梯度都是未定义的，那么反向函数**返回未定义的输入梯度**应该是正确的。由于我们使用了链式法则，输出梯度等于零应该导致输入梯度也等于零，除非存在一些罕见的特殊情况。<br>

## 11.7 部分输出梯度未定义
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果部分输出梯度是未定义的，那么反向函数返回未定义的输入梯度**可能是可以接受的**——这取决于具体的函数，因此您需要自行确定。如果对于给定的输入梯度，返回未定义的Tensor是正确的，那么从逻辑上讲，返回一个全为零的已定义梯度也是正确的，但这并不是首选，因为这样会**降低效率**。<br>

# 12 命名的一致性
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**NB: The parameter names here MUST be consistent with the parameter names in native_functions.yaml** <br>

# 13 生成的反向传播函数声明和定义
- */pytorch/torch/csrc/autograd/generated/Functions.h
- */pytorch/torch/csrc/autograd/generated/Functions.cpp




