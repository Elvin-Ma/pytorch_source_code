# pytorch_source_code
pytorch source code introduce and analysis

# 缩写
- TH = Torch
- THC = Torch Cuda
- THS = Torch Sparse(now defunct)
- THNN = Torch Neural Network(now defunct)
- THCS = Torch Cuda Spase(now defunct)

referece: [pytorch README.md]()

# 架构
```python
├── android: 用于构建android应用程序
├── aten: A Tensor Library，pytorch的C++ tensor 库,实现了tensor的运算 (不支持自动梯度)，大部分op都可以在此找到，Aten的op实现分为两种;1. native op：op的现代C++实现 2. legacy op(TH, THC, THCUNN):原torch遗留下来的op的传统C语言实现,不要在里面添加新的op,这部分正在慢慢转移到native中
│  └── conda:包含了conda构建使用的脚本以及配置文件
│  └── src：op的实现
│     └──ATen：op的现代C++实现
│       └──core：Aten的核心功能，目前正在迁移到c10文件夹中
│       └──native：op的现代实现，在这里添加你想要添加的新的op
│         └──cpu：不是op的实际cpu实现，但是包含了特殊指令集的实现，比如AVX
│         └──cuda：op的cuda实现
│         └──sparse：COO的稀疏tensor的op的cpu和cuda的实现
│         └──mkl mkldnn miopen cudnn：绑定到具体后端的op的实现
│     └──TH：TorcH,CPU tensor的实现。TH,THC,THCUNN是原torch遗留下来的op的C语言实现
│       └──generic：TH,THC,THCUNN三者的generic中包含了操作的实际实现，这些文件会被编译N次
│       └──…
│     └──THC：GPU tensor的实现
│       └──generic
│       └──…
│     └──THCUNN：GPU版底层神经网络实现
│       └──generic
│     └──THH：THC在hip上的实现
│     └──THHUNN：THCUNN在hip上的实现
│  └── tools:包含了一些测试脚本
├── benchmarks:包含了一些pytorch特性的benchmark测试
├── binaries
├── c10:核心Tensor库，从ATen/core迁移过来，只包含最基本的功能，可以运行在服务端和移动端，适用于对二进制文件大小有要求的环境中
│  └── benchmark
│  └── core：CPU的核心库
│  └── cuda：cuda的核心库
│  └── hip：HIP的构建文件,该文件夹不包含实际的文件，这些文件通过tools/amd_build中的HIPIFY脚本拷贝过来
│  └── macros:包含了一些自定义的宏
│  └── test：测试程序
│  └── util：通用的工具库
├── caffe2: caffe2
│  └── core：caffe2的核心文件，包括tensor,workspace,blobs等
│  └── operators：caffe2的op
│  └── python：caffe2的python绑定
│  └── …
├── cmake:用于构建pytorch的cmake文件
├── docker:用于在docker中使用pytorch
├── docs:pytorch相关的说明文档
├── ios:pytroch的ios模块，用于构建ios应用程序
├── modules
├── scripts:包含了一些有用的工具,比如build_android.sh用于创建android上的pytorch库
├── submodules
├── test:pytorch的python前端的python单元测试
│  └── test_torch.py:pytorch功能的基本测试
│  └── test_autograd.py:非神经网络自动微分的测试
│  └── test_nn.py :神经网络op以及自动微分的测试
│  └── test_jit.py: JIT 编译器和TorchScript的测试
│  └──…
│  └──cpp:pytorch的C++前端的C++单元测试
│  └──onnx:onnx输出功能的测试，同时使用pytorch和caffe2
├── third_party:pytorch依赖的第三方库
├── tools:pytorch库的代码生成脚本
│  └── autograd: 生成自微分相关函数的工具
│  └──…
│  └──shared
├── torch:实际的pytorch库。包含了我们平时import的python模块，所有不在csrc文件夹中的是一个python模块，符合pytorch的python前端模块结构
│  └── csrc: pytorch库的C++实现，包含1. C++前端的实现 2.绑定代码(python bindings)。csrc中每个独立文件与python模块对应。python绑定文件的规范列表请查看setup.py,通常他们以python_为前缀
│     └──api:pytorch的C++前端接口
│     └──autograd：自动微分反向模式的实现
│     └──cuda
│     └──distributed：pytorch的分布式训练
│     └──generic
│     └──jit: TorchScript JIT的编译器和前端
│     └──…
│     └──utils
│     └──Module.cpp:构建Python扩展模块
│  └── nn:torch.nn模块
│  └── optim:torch.optim模块
│  └── distributed:torch. distributed模块
```
