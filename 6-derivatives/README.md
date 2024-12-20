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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果对于一个返回多个元素但使用grad而不是grads[idx]的函数没有指定output_differentiability，那么除了第一个输出之外的所有输出都将被标记为非可微分的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果所有输出都不可微分，您也可以将该函数名称添加到gen_variable_type.py中的DONT_REQUIRE_DERIVATIVE列表中。<br>

# 3 Tensor和TensorList 参数有两种情况：
## 3.1 该参数可微分
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;即可能存在关于该参数的梯度。您应该：
- 指定该梯度的公式
- 或者指定not_implemented("function_name")作为公式，表示此功能尚未实现（但将来可能会实现，用户可以在问题中请求此功能）
## 3.2 参数不可微分的
因为数据类型非浮点或者该参数对应的function不可微分，您应该：
- 不要为该参数指定任何公式;
- 或者显示指定该参数是不可微分的，在这种情况下，我们相信您保证此参数永远不会设置为requires_grad=True，并且如果设置了，它将被默默地忽略。

如果一个函数有非原地（out-of-place）和原地（in-place）两种变体，那么原地变体的导数定义是可选的。它将默认采用非原地变体的定义。请注意，**带有_out后缀的变体永远是不可微分的**。

# 4 Gradient expression
梯度表达式是标准的C++表达式，对ATen变量(variables)进行操作。在梯度表达式中，以下变量/函数是可用的：<br>
- grad : 输出的梯度，在python中常写作grad_output, 我们将对其进行左乘;
- 
