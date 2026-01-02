# 1 python 侧 Ops 模块

- _Ops 类，是 PyTorch 中 torch.ops 模块的底层实现载体，核心作用是为 PyTorch 提供**算子直接使用, 自定义算子的管理、注册和加载入口**

- 懒加载(lazy loading)机制，无需提前定义所有可能的算子命名空间，降低内存占用，同时让用户自定义算子的命名空间具备统一、简洁的访问形式;

- **负责ops模块的注册以及子命名空间的储存与路由**

```python
# - */pytorch/torch/_ops.py

class _Ops(types.ModuleType):
    '''
    types.ModuleType 是 Python 中所有模块对象的底层类型，
    _Ops 继承它意味着 _Ops 实例可以具备 Python 标准模块的行为和属性，
    这也是 torch.ops 能以「模块形式」被使用（如 import torch.ops）的核心原因。
    '''

    # 显式指定模块对应的文件标识，符合 Python 模块的规范，用于标识模块的来源文件
    __file__ = "_ops.py"

    def __init__(self):
        # 创建一个名为 torch.ops 的模块对象，奠定 torch.ops 作为模块的基础
        super().__init__("torch.ops")
        # 记录所有已成功加载的共享库路径，方便用户后续查询哪些自定义算子库已被加载
        self.loaded_libraries = set()
        # 高阶算子命名空间（_HigherOrderNamespace 实例），用于管理 PyTorch 内置的高阶算子，与用户自定义算子无直接关联
        self.higher_order = _HigherOrderNamespace()
        # 用于存储动态创建的算子命名空间名称,支持后续的迭代功能, 调用的时候才会惰加载 : _ops.aten._assert_async.msg
        # ['aten', 'profiler', 'inductor', 'quantized', 'prim', 'prims', 'rngprims', 'debugprims', 'mkldnn', 'mkl', 'onednn', 'load_libraries']
        self._dir = []

    # 动态创建算子命名空间
    def __getattr__(self, name: str) -> _OpNamespace:
        # Here we are creating `torch.ops.my_namespace`
        # 动态创建自定义算子的命名空间（_OpNamespace 实例），为后续存放该命名空间下的自定义算子做准备
        # _OpNamespace 是 PyTorch 用于封装单个算子命名空间的类，负责管理该命名空间下的所有算子
        # 将创建的命名空间绑定为 _Ops 实例的属性，避免下次访问该同名命名空间时重复创建(重复进入__getattr__)
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        # 添加动态创建的命名空间名称到 _dir 列表中，用于后续迭代查询/遍历所有已创建的命名空间
        self._dir.append(name)
        return namespace

    def __iter__(self) -> Iterator[str]:
        '''
        让 _Ops 实例（即 torch.ops 模块）支持迭代遍历，
        可以通过 for 循环等方式获取所有动态创建的自定义算子命名空间名称
        '''
        return iter(self._dir)

    def import_module(self, module):
        """
        Imports a Python module that has torch.library registrations.

        Generally, to extend PyTorch with custom operators, a user will
        create a Python module whose import triggers registration of
        the custom operators via a torch.ops.load_library call or a call
        to one or more torch.library.* APIs.

        It is unexpected for Python modules to have side effects, so some
        linters and formatters will complain. Use this API to import Python
        modules that contain these torch.library side effects.

        Args:
            module (str): The name of the Python module to import

        """
        # 动态导入模块 - 可以在运行时根据字符串名称导入模块
        # 提供导入机制的编程接口 - 让开发者能够控制和自定义导入过程
        importlib.import_module(module)

    def load_library(self, path):
        """
        从指定路径加载共享库（编译后的自定义算子库），执行库中的全局初始化代码，
        将自定义算子注册到 PyTorch JIT 运行时，使其可以被 torch.ops 调用;

        自动执行共享库中所有的 C++ 全局 / 静态变量初始化代码、以及全局构造函数;

        自定义算子的「注册逻辑」，正是被开发者封装在这份全局初始化代码中（
        通常通过 PyTorch 提供的 C++ 宏定义TORCH_LIBRARY_IMPL简化实现-动态库自己注册），
        这是整个「自动绑定」的触发前提 —— 无需手动调用注册函数，加载库即触发注册;

        注册的地址是 PyTorch 内置 C++ 全局注册表.

        具体逻辑为:
        - 加载共享库 → 执行 C++ 全局初始化 → 算子信息存入 C++ 全局注册表；
        - 首次访问 torch.ops.ns.op → 触发 Python 层两级 __getattr__（_Ops → _OpNamespace）；
        - _OpNamespace 跨层查询 C++ 注册表 → 创建并缓存 Python 算子包装对象；
        - 返回可调用对象，完成最终绑定，后续访问直接读取缓存。

        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        # 解析输入路径，处理相对路径、平台相关路径别名等问题，返回可直接加载的绝对路径
        path = _utils_internal.resolve_library_path(path)
        # dl_open_guard 是 PyTorch 提供的共享库安全加载保护机制，
        # 用于处理跨平台加载冲突、库依赖解析等潜在问题，保证加载过程的稳定性
        with dl_open_guard():
            # Import the shared library into the process, thus running its
            # static (global) initialization code in order to register custom
            # operators with the JIT.
            try:
                # 加载共享库并执行其全局初始化代码 ——
                # 而自定义算子的注册逻辑（如绑定到 torch.ops 命名空间），正是封装在这份全局初始化代码中
                ctypes.CDLL(path)
            except Exception as e:
                raise OSError(f"Could not load this library: {path}") from e
        # 将解析后的有效路径添加到 self.loaded_libraries 集合中，
        # 完成库加载记录，方便用户后续查询已加载的库列表
        self.loaded_libraries.add(path)

# The ops "namespace"
ops: _Ops = _Ops()
```

# 2 从命名空间到算子

## 2.1 从ops 命名空间动态获取单个算子

```python
# 注释：说明 torch.fn 与 torch.ops.aten.fn 两种调用形式的「算子解析机制」存在差异
# Resolution of torch.fn is different from torch.ops.aten.fn
# 注释：补充 torch.fn 的解析逻辑：使用 Python 原生参数解析器（argparser）
# torch.fn uses the Python argparser, matches with the
# 注释：承接上一行，torch.fn 会匹配对应的算子模式（schema，即参数/返回值定义规范），并调用方法的「未装箱版本」
# 注释：补充：unboxed version 指不封装原始数据，直接操作，效率较高
# appropriate schema, and calls into the unboxed version of the method
# 注释：说明 torch.ops.aten.fn 的解析机制：通过 JIT（即时编译）中定义的逻辑完成
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT.
# 注释：补充 JIT 解析细节：先创建该算子所有重载版本（overloads）的栈结构，再尝试匹配
# JIT creates a stack of all the overloads and then tries to match the
# 注释：承接上一行，JIT 会在运行时匹配正确的重载版本，且始终调用方法的「装箱版本」
# 注释：补充：boxed version 指数据会被封装为特定对象后处理，兼容性强但有轻微性能开销
# correct one at runtime and always calls into the boxed version of the method
# 注释：说明 Autograd（自动求导）代码生成器的作用：创建 VariableType（变量类型）、TracerType（追踪器类型）
# Autograd codegen creates VariableType, TracerType,
# 注释：承接上一行，Autograd 代码生成器还会创建原地操作（inplace）/视图（view）类型，以及对应的 Python 绑定（让 C++ 算子可被 Python 调用）
# inplace or view type and python bindings.
# 注释：说明 Aten 代码生成器的作用：为 Tensor 类生成对应的张量操作方法（Aten 是 PyTorch 核心算子库）
# Aten codegen generates tensor methods for the tensor class.

# 注释：说明 _OpNamespace 继承 ModuleType 的原因：Torch Script 仅支持对「模块（module）」进行属性查找
# _OpNamespace is a subclass of ModuleType because the torch script
# 注释：承接上一行，为了让 torch.ops.foo.bar() 能在 Torch Script 中正常工作，需要确保 ops 和 foo 都是模块类型
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# 注释：承接上一行，明确核心需求：保证 ops 和 foo 为模块类型，兼容 Torch Script 语法
# to work from script, we need to ensure ops and foo are modules

# 定义 _OpNamespace 类，继承自 types.ModuleType（Python 中模块对象的原生类型）
# 作用：实现算子命名空间管理，支持将 C++ 实现的算子动态绑定到 Python 环境中
class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """
    # 为 _OpNamespace 类设置 __file__ 类属性，值为 "torch.ops"
    # 作用：模拟 Python 模块的 __file__ 属性（标识模块文件路径），兼容 Torch Script 的模块属性查找逻辑
    __file__ = "torch.ops"

    # 定义类的构造方法，用于初始化 _OpNamespace 实例
    # 参数 name：字符串类型，命名空间名称（如 "my_namespace"）；返回值：None
    def __init__(self, name: str) -> None:
        # 调用父类（types.ModuleType）的构造方法，初始化模块名称
        # 传入模块名称 "torch.ops." + name，确保符合 torch.ops 命名空间规范
        super().__init__("torch.ops." + name)
        # 将命名空间名称绑定为实例属性，方便后续方法中调用
        self.name = name
        # 初始化实例属性 _dir，为字符串空列表
        # 作用：缓存该命名空间下已绑定的算子名称，支持后续迭代操作
        self._dir: list[str] = []

    # 定义 __iter__ 魔法方法，实现迭代器协议
    # 作用：让 _OpNamespace 实例支持 for 迭代，返回已绑定的算子名称列表迭代器
    # 返回值：字符串类型的迭代器（Iterator[str]）
    def __iter__(self) -> Iterator[str]:
        # 返回 self._dir 列表的迭代器，迭代时依次返回已绑定的算子名称
        return iter(self._dir)

    # 定义 __getattr__ 魔法方法，核心：动态获取实例上不存在的属性（未绑定的算子）
    # 参数 op_name：字符串类型，要获取的算子名称（如 "my_op"）
    # 返回值：OpOverloadPacket 类型（算子重载包，封装算子及其所有重载版本）
    def __getattr__(self, op_name: str) -> OpOverloadPacket:
        # 判断要获取的属性是否是 Torch Script 内部特殊属性（__origin__/__self__）
        if op_name in ("__origin__", "__self__"):
            # 若是特殊属性，抛出 AttributeError 异常，禁止访问
            raise AttributeError(
                f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'"
            )

        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        # 注释：获取 `my_namespace::my_op` 格式的算子（若存在），同时检查重载版本，多版本会抛出异常
        # 提取当前实例的命名空间名称，赋值给局部变量，简化后续代码
        namespace_name = self.name
        # 拼接生成算子的全限定名称，格式「命名空间::算子名」（PyTorch 算子标准命名格式）
        qualified_op_name = f"{namespace_name}::{op_name}"
        # 拼接生成算子对应的 Python 模块名称，兼容模块管理逻辑
        module_name = self.__module__ + "." + namespace_name

        # 开启 try 异常捕获块，捕获获取算子过程中可能抛出的 RuntimeError
        try:
            # 调用 _get_packet 工具函数，获取算子对象和重载版本列表
            op, overload_names = _get_packet(qualified_op_name, module_name)
            # 判断是否成功获取算子对象（op 不为 None）
            if op is None:
                # 若未获取到，抛出 AttributeError 异常，提示算子不存在
                raise AttributeError(
                    f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
                )
        # 捕获 try 块中的 RuntimeError 异常（通常是算子重载版本异常）
        except RuntimeError as e:
            # Turn this into AttributeError so getattr(obj, key, default)
            # works (this is called by TorchScript with __origin__)
            # 注释：将 RuntimeError 转为 AttributeError，兼容 getattr 函数（Torch Script 会调用）
            # 抛出 AttributeError 异常，替代原始 RuntimeError
            raise AttributeError(
                f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
            ) from e  # 保留原始异常上下文，方便调试追溯问题

        # 为获取到的算子对象设置 __module__ 属性，关联对应的 Python 模块, 方便debug
        op.__module__ = module_name
        # 创建 OpOverloadPacket 实例（算子重载包），封装算子相关信息
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        # 为算子重载包设置 __module__ 属性，兼容模块属性查找逻辑
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # cache the opoverloadpacket to ensure that each op corresponds to
        # a unique OpOverloadPacket object
        # 注释：缓存算子重载包，确保每个算子对应唯一的 OpOverloadPacket 实例，避免重复创建
        # 使用 setattr 函数，将算子重载包绑定为当前实例的属性（属性名=算子名）
        # 后续访问该算子时，直接从实例属性获取，无需再次查找（实现缓存）
        setattr(self, op_name, opoverloadpacket)
        # 将算子名称添加到 self._dir 列表，更新已绑定算子缓存，支持迭代
        self._dir.append(op_name)
        # 返回创建并缓存完成的算子重载包，供调用者执行算子调用
        return opoverloadpacket
```

## 2.2 从c++ 获取函数

- 注意: overload_names 可获取算子重载版本列表

```python
# 定义工具函数 _get_packet，核心：从 PyTorch C++ 层获取算子对象和重载版本列表
# 参数 qualname：算子全限定名称（如 "my_namespace::my_op"）
# 参数 op_module：算子对应的 Python 模块名称
def _get_packet(qualname, op_module):
    # 调用 PyTorch C++ 层绑定的函数，传入算子全限定名称，获取算子对象和重载版本列表
    op, overload_names = torch._C._jit_get_operation(qualname)
    # 判断是否成功获取算子对象（op 不为 None）
    if op is not None:
        # let the script frontend know that op is identical to the builtin op
        # with qualified_op_name
        # 注释：让 Torch Script 前端识别该算子为对应全限定名称的内置算子
        # 注册算子为 Torch Script 内置算子，确保 Torch Script 能正确处理
        torch.jit._builtins._register_builtin(op, qualname)
        # 为算子对象设置 __module__ 属性，关联对应的 Python 模块
        op.__module__ = op_module
    # 返回获取到的算子对象和重载版本列表，供调用者使用
    return op, overload_names

# 定义工具函数 _refresh_packet，核心：刷新已存在的算子重载包（更新算子对象和重载版本）
# 场景：算子 C++ 实现更新后，刷新 Python 层缓存
# 参数 packet：需要刷新的 OpOverloadPacket 实例
def _refresh_packet(packet):
    # 调用 _get_packet 函数，传入 packet 中的算子信息，重新获取最新的算子对象和重载版本
    op, overload_names = _get_packet(packet._qualified_op_name, packet._op.__module__)
    # 判断是否成功获取最新算子对象，若失败则抛出断言异常
    if op is None:
        raise AssertionError(f"failed to get packet for {packet._qualified_op_name}")
    # 将最新算子对象赋值给 packet 实例，更新缓存
    packet._op = op
    # 将最新重载版本列表赋值给 packet 实例，更新缓存
    packet._overload_names = overload_names
```

# 3 算子重载工具箱：OpOverloadPacket

- 该类不对应某个具体的算子实现，而是**封装了「基础未解析算子」**和该算子的**所有重载版本信息**的「数据包 / 容器」。用户无法直接通过它调用特定重载的算子，需要通过「属性查询」（如 .default、.float32）获取具体的 OpOverload（普通算子）或 TorchBindOpOverload（torchbind 算子）对象，才能调用对应重载的逻辑。

- OpOverloadPacket 是一个「算子重载工具箱」，里面装着各种规格（重载版本）的工具（OpOverload），你需要先`从工具箱里拿出具体工具`，才能干活；`也可以直接用工具箱（默认调用核心重载）`。

- self._op 由 torch._C._jit_get_operation() 返回，JIT 会在 C++ 端进行参数类型匹配，自动选择正确的重载;

- python 端只是统计重载版本的信息；

-- __getattr__ : 按需查询并创建具体重载算子实例，缓存后返回，避免重复查询 C++, default 也不是默认加载的，则要走到这里;

**重载逻辑** <br>
```sh
第一步：首次访问 torch.ops.aten.linear.default 时，触发 __getattr__
此时 torch.ops.aten.linear 实例上并没有 default 这个属性，Python 会自动调用 OpOverloadPacket 类的 __getattr__ 魔法方法，传入 key="default"。
第二步：框架内部自动处理 default 重载的查询与转换
__getattr__ 方法内部会自动将 Python 层友好的 default 转换为 C++ 层识别的空字符串（use_key = "" if key == "default" else key），无需用户手动适配。
第三步：自动从 C++ 层查询重载算子并创建实例
调用 torch._C._get_operation_overload 从 PyTorch C++ 核心层查询 aten::linear 的 default 重载，自动创建对应的 OpOverload 实例。
第四步：自动缓存，后续访问直接复用
框架会通过 setattr(self, key, overload) 将 default 重载实例绑定为 torch.ops.aten.linear 的属性，同时存入 _dir 缓存列表，后续再访问 torch.ops.aten.linear.default 时，直接从缓存中获取，无需重复查询 C++ 层。
```

**源码实现** <br>

```python

# 注释：OpOverloadPacket 类的核心定位：封装「基础未解析算子」，不对应某个具体的算子实现
# 注释：该类仅作为「算子重载版本的容器/数据包」，用户需要通过「属性查询」（如 .default、._scalar）
# 注释：才能获取到可直接调用的 OpOverload（普通算子）或 TorchBindOpOverload（torchbind 算子）实例
# OpOverloadPacket class contains pointer to a base unresolved operator that doesn't correspond to a specific operator
# You can obtain an OpOverload object through attribute query.
class OpOverloadPacket:
    # 构造方法：初始化算子重载包的核心元数据，接收4个关键参数
    # 参数说明：
    #   qualified_op_name：算子全限定名（格式："命名空间::算子名"，如 "aten::add"）
    #   op_name：算子简短名称（如 "add"）
    #   op：底层未解析的基础算子对象（来自 PyTorch C++ 层，无具体重载信息）
    #   overload_names：该算子的所有重载版本名称列表（空字符串对应 default 重载）
    def __init__(self, qualified_op_name, op_name, op, overload_names):
        # These attributes are accessible on the object through the properties
        # defined below but are immutable
        # 注释：以下属性均为「只读不可变」，外部可通过属性方法访问，不可直接修改，保证实例唯一性
        # 核心属性：存储算子全限定名，用于后续查询 C++ 层算子信息
        self._qualified_op_name = qualified_op_name
        # 符合 Python 可调用对象规范：存储算子简短名称，外部可通过 .__name__ 访问
        self.__name__ = op_name
        # 核心属性：存储底层 C++ 基础算子对象，是后续调用和重载查询的核心
        self._op = op
        # 核心属性：存储该算子的所有重载版本名称列表，用于后续生成 schema 和重载查询
        self._overload_names = overload_names
        # 缓存属性：存储已通过属性查询获取到的具体重载算子名称，支持迭代和去重
        self._dir = []
        # 标记属性：判断该算子是否包含 torchbind 重载（torchbind 用于将 C++ 类绑定到 Python 环境）
        # 注释：通过遍历所有重载的 schema，判断是否包含 ScriptObject 类型参数（torchbind 算子的特征）
        self._has_torchbind_op_overload = any(
            _has_script_object_arg(schema) for schema in self._schemas.values()
        )

    # 魔法方法：__deepcopy__ 深拷贝方法，无任何实际操作，直接返回当前实例自身
    # 注释：设计原因：OpOverloadPacket 实例是「不可变且唯一」的（一个算子对应一个唯一实例）
    # 注释：无需创建新的拷贝对象，避免冗余，同时保证实例的唯一性
    # it's a no-op since OpOverloadPacket object is immutable and must be unique for a given op.
    def __deepcopy__(self, memo=None):
        return self

    # 魔法方法：__repr__ 官方格式化输出方法，用于调试时查看实例核心信息
    # 注释：返回格式为 <OpOverloadPacket(op='命名空间.算子名')>，直观展示算子归属
    # 注释：通过 split("::") 拆分全限定名，解构为「命名空间」和「算子名」
    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    # 魔法方法：__hash__ 哈希方法，用于将实例作为字典键、放入集合等哈希相关操作
    # 注释：基于底层 _op 对象的哈希值生成，保证同一个算子的 OpOverloadPacket 实例哈希值唯一
    def __hash__(self):
        return hash(self._op)

    # 魔法方法：__str__ 字符串格式化输出方法，返回简洁的「命名空间.算子名」格式
    # 注释：比 __repr__ 更简洁，用于普通打印和字符串拼接场景
    def __str__(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    # 只读属性：op，返回底层基础算子对象 _op
    # 注释：使用 @property 装饰器实现「只读」，外部可访问（packet.op）但无法修改，保证实例稳定性
    @property
    def op(self):
        return self._op

    # 只读属性：_schemas，返回「重载名称: 算子schema」的字典
    # 注释：schema 是算子的核心规范，包含参数类型、返回值类型、参数约束等信息
    # 注释：通过 PyTorch C++ 层接口 _get_schema 查询每个重载的 schema，构建字典返回
    # 注释：外部可访问（packet._schemas）但无法修改，用于查看算子元数据，辅助参数校验
    @property
    def _schemas(self):
        return {
            overload_name: torch._C._get_schema(self._qualified_op_name, overload_name)
            for overload_name in self._overload_names
        }

    # 核心魔法方法：__getattr__ 动态获取实例上不存在的属性（即具体的算子重载版本）
    # 注释：核心逻辑：按需查询并创建具体重载算子实例，缓存后返回，避免重复查询 C++ 层
    def __getattr__(self, key):
        # 1. 特殊处理 __file__ 属性，直接返回 "torch.ops"
        # 注释：兼容 Python 模块规范，让算子重载包看起来更像标准 Python 模块对象
        if key == "__file__":
            return "torch.ops"

        # 2. 处理双下划线开头的特殊内置属性（如 __name__、__call__、__hash__）
        # ensure that query for dunder attributes that does not exist on
        # opoverloadpacket but instead exists on the self._op object does not unnecessarily call
        # `_get_operation_overload` (which is an expensive operation).
        # This is done to prevent any potential slowdown. This list can be extended
        # if there exists other attributes like `__name__` that only exist on self._op and not on the
        # opoverloadpacket.
        # This is ok since we are guaranteed that an overload name for an aten op can't start with '__'
        # 注释：优化点：将这类属性查询转发给底层 _op 对象，避免调用昂贵的 C++ 重载查询接口，提升性能
        # 注释：PyTorch 规范：aten 算子的重载名称不会以 "__" 开头，因此该判断安全无冲突
        try:
            if key.startswith("__"):
                return getattr(self._op, key)
        except AttributeError:
            # for consistency because it seems weird to
            # throw an attribute error with a message containing
            # an object name different from the one the attribute
            # query was performed on.
            # 注释：异常格式化处理：保证报错信息对应当前 OpOverloadPacket 实例，提升报错友好性
            # 注释：隐藏底层 _op 对象的细节，避免用户困惑
            raise AttributeError(
                f"'{str(self)}' can't have an overload name beginning with '__' and the "
                f"underlying op {str(self._op)} has no attribute {key} either."
            ) from None

        # 3. 处理普通重载名称查询，获取具体的 OpOverload/TorchBindOpOverload 实例
        try:
            # This is ok since we are guaranteed that an overload name for an aten op can't be 'default'
            # 注释：PyTorch 规范：aten 算子的重载名称不会是 "default"，因此该转换安全
            # 注释：将 Python 层友好的 "default" 转换为 C++ 层识别的空字符串，适配底层查询接口
            use_key = "" if key == "default" else key
            # TODO: disallow access to overloads registered by JIT
            # 注释：调用 PyTorch C++ 层接口，查询对应重载名称的算子详细信息（包含 op 对象、dk 标签、tags）
            op_dk_tags = torch._C._get_operation_overload(
                self._qualified_op_name, use_key
            )
            # 若查询结果为 None，说明该重载名称不存在，抛出 AttributeError 异常
            if op_dk_tags is None:
                raise AttributeError(
                    f"The underlying op of '{str(self)}' has no overload name '{key}'"
                )

            # 解析 C++ 层返回的重载算子信息，拆分为三个独立变量
            op_, op_dk_, tags = op_dk_tags
            # 查询该重载算子对应的 schema 信息，用于后续创建重载实例
            schema = torch._C._get_schema(self._qualified_op_name, use_key)
            # 根据是否包含 ScriptObject 参数，创建对应的重载算子实例
            # 注释：普通算子创建 OpOverload 实例，torchbind 算子创建 TorchBindOpOverload 实例
            overload = (
                OpOverload(self, op_, op_dk_, schema, tags)
                if not _has_script_object_arg(schema)
                else TorchBindOpOverload(self, op_, op_dk_, schema, tags)
            )
            # 缓存优化：将创建好的重载实例绑定为当前 OpOverloadPacket 实例的属性
            # 注释：后续访问该重载时，直接从属性获取，无需重复查询 C++ 层，提升性能
            setattr(self, key, overload)
            # 将该重载名称添加到 _dir 缓存列表，支持后续迭代和查看
            self._dir.append(key)
            # 返回创建并缓存完成的具体重载算子实例，供用户调用执行
            return overload
        except RuntimeError:
            # 异常转换：将 C++ 层返回的 RuntimeError 转为 Python 标准的 AttributeError
            # 注释：提升报错友好性，明确提示用户重载名称不存在
            raise AttributeError(
                f"The underlying op of '{str(self)}' has no overload name '{key}'"
            ) from None

    # 魔法方法：__iter__ 实现迭代器协议，让 OpOverloadPacket 实例支持 for 循环迭代
    # 注释：返回 _dir 缓存列表的迭代器，迭代结果为已获取的具体重载算子名称
    def __iter__(self):
        return iter(self._dir)

    # 魔法方法：__call__ 支持直接调用 OpOverloadPacket 实例（如 torch.ops.aten.add()）
    # 注释：/ 是 Python 3.8+ 引入的「位置-only参数」标记，用于分隔位置参数和关键字参数
    # 注释：设计原因：避免和 aten 算子的关键字参数 "self" 命名冲突，保证所有 aten 算子都可通过关键字调用
    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.
    def __call__(self, /, *args, **kwargs):
        # overloading __call__ to ensure torch.ops.foo.bar()
        # is still callable from JIT
        # We save the function ptr as the `op` attribute on
        # OpOverloadPacket to access it here.
        # 注释：重载 __call__ 方法的核心目的：让 torch.ops.foo.bar() 可直接调用，兼容 JIT 静态编译

        # Directly calling OverloadPacket goes into C++, which will check
        # the schema and cause an error for torchbind op when inputs consist of FakeScriptObject so we
        # intercept it here and call TorchBindOpverload instead.
        # 注释：特殊处理：torchbind 算子 + FakeScriptObject 输入，需要在 Python 层进行分发处理
        # 注释：避免直接调用 C++ 层导致报错，保证 torchbind 算子的兼容性
        if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
            return _call_overload_packet_from_python(self, args, kwargs)
        # 默认情况：直接调用底层 _op 对象，将参数转发给 C++ 层，等价于调用 default 重载算子
        return self._op(*args, **(kwargs or {}))

    # 辅助方法：overloads()，返回该算子的所有重载版本名称列表（格式化后）
    # 注释：将 C++ 层的空字符串重载名称转为 Python 层友好的 "default"，提升用户体验
    # TODO: use this to make a __dir__
    def overloads(self):
        return [n if n else "default" for n in self._overload_names] # "" 空对应default

```

# 4 算子的重载版本

## 4.1 基类 OperatorBase

- OperatorBase 是 OpOverload（C++ ATen 算子）和 HigherOrderOperator（Python 专属高阶算子）的基类;
  - 子类 1：OpOverload：对应 C++ 实现的 ATen 核心算子（之前详细讲解过）；
  - 子类 2：HigherOrderOperator：对应仅在 Python 层实现、无法被 TorchScript 序列化的高阶算子.
- 核心职责: 是「`统一封装算子的分发缓存、Python 内核注册等基础能力」，为子类提供通用的底层支撑」`;
- 覆盖了「性能优化」「内核自定义」「模式扩展」「高阶变换」四大场景，为子类提供了全面的基础支撑.

- 核心能力：提供「分发缓存管理」「Python 内核自定义注册」「dispatch 模式扩展」等通用能力，是 PyTorch Python 层调度机制的基础支撑；
- 设计价值：解耦「算子具体实现」和「算子调度 / 注册的公共逻辑」，让子类只需关注自身特有功能，无需重复实现通用的缓存、注册逻辑.

- py_impl：实现灵活的 **Python 自定义注册**, 是 OperatorBase 类的核心方法，提供「装饰器风格」的注册接口，支持为不同类型的目标（DispatchKey、模式、functorch 变换）注册自定义 Python 实现.

- **py_impl 和 torch.library.impl 算子注册的不同** <br>
  - 访问路径相同：两者最终都通过 torch.ops.namespace.op_name 访问
  - 内部存储不同：torch.library.impl 存储在库系统中，py_impl 存储在算子对象的 py_kernels 字典中
  - 功能不同：torch.library.impl 用于创建新算子，py_impl 用于覆盖现有算子的特定调度键实现

```python
class OperatorBase:
    def __init__(self):
        # 1. 分发缓存字典：预处理「DispatchKey → 实际实现」的映射，优化调度性能
        # 存储结构：key=DispatchKey（如 AutogradCPU、CUDA），value=DispatchKey 或 Python 可调用对象
        # 核心作用：
        #   - 缓存预处理结果，避免每次调度都重复计算映射关系，提升效率（尤其是 C++ 层直接读取，速度更快）；
        #   - 支持别名映射（如 AutogradCPU → Autograd），复用通用内核实现；
        #   - 既支持 C++ 内核（存储 DispatchKey），也支持 Python 自定义实现（存储可调用对象）。
        self._dispatch_cache: Dict[
            DispatchKey, Union[DispatchKey, Callable[..., Any]]
        ] = {}

        # 2. Python 内核注册字典：为指定 DispatchKey 注册自定义 Python 实现，覆盖默认 C++ 内核
        # 存储结构：key=DispatchKey，value=Python 可调用对象（自定义内核逻辑）
        # 核心作用：
        #   - 允许用户在 Python 层自定义算子逻辑，无需修改 C++ 代码；
        #   - 不破坏原有 C++ 内核注册，仅在指定 DispatchKey 下覆盖行为，灵活扩展；
        #   - 是 PyTorch Python 调度器的核心实现，支撑自定义 dispatch 逻辑。
        self.py_kernels: Dict[DispatchKey, Callable[..., Any]] = {}

        # 3. Python 模式注册字典：为指定 TorchDispatchMode 注册自定义行为，扩展 dispatch mode
        # 存储结构：key=TorchDispatchMode/张量子类，value=Python 可调用对象
        # 核心场景：主要用于 ProxyTensorMode（代理张量模式，如静态图追踪、形状校验）
        # 核心作用：将 dispatch 机制从「DispatchKey」扩展到「Python 模式」，支持更灵活的行为定制。
        self.python_key_table: Dict[
            Union[Type[TorchDispatchMode], Type[torch.Tensor]], Callable[..., Any]
        ] = {}

        # 4. Functorch 变换注册字典：为 functorch 变换注册自定义行为，支撑高阶张量变换
        # 存储结构：key=functorch 变换类型，value=Python 可调用对象
        # 核心场景：仅作用于 HigherOrderOperator（Python 高阶算子），如自动微分、张量重排等 functorch 变换。
        self.functorch_table = {}

    # has_kernel_for_dispatch_key：查询是否为指定 DispatchKey 注册了 Python 内核
    # 参数 k：DispatchKey（如 Autograd、CUDA）
    # 返回值：布尔值（True=已注册，False=未注册）
    # 核心作用：快速判断某个 DispatchKey 是否有自定义 Python 实现，避免无效查询。
    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    # has_kernel_for_any_dispatch_key：查询是否为指定 DispatchKey 集合中的任意一个注册了 Python 内核
    # 参数 ks：DispatchKey 集合
    # 返回值：布尔值（True=存在已注册的 DispatchKey，False=均未注册）
    # 核心逻辑：遍历 py_kernels，排除别名 Key，判断是否与目标集合有交集
    # 核心作用：批量查询 DispatchKey 注册状态，提升框架内部逻辑判断效率。
    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False

    def py_impl(self, k: Any) -> Callable[[_F], _F]:
        # 定义内部装饰器函数，接收并返回被装饰的 Python 可调用对象（保持装饰器规范）
        def inner(fn: _F) -> _F:
            # 分支 1：处理 TorchDispatchMode 或 torch.Tensor 子类的注册
            if inspect.isclass(k) and (
                issubclass(k, TorchDispatchMode) or issubclass(k, torch.Tensor)
            ):
                # 确保该模式/子类未被注册过，避免重复覆盖
                assert k not in self.python_key_table
                # TODO(voz): Should we replace setting DispatchKey.Python entirely with setting mode keys?
                # 将自定义实现注册到 python_key_table 字典
                self.python_key_table[k] = fn
                # 清除分发缓存：注册新行为后，原有缓存失效，强制下次重新计算映射
                self._dispatch_cache.clear()
                # 返回被装饰的函数，保持装饰器语法正确性
                return fn

            # 分支 2：处理 functorch 变换类型的注册
            if isinstance(k, torch._C._functorch.TransformType):
                # 确保该变换类型未被注册过，避免重复覆盖
                assert k not in self.functorch_table
                # 将自定义实现注册到 functorch_table 字典
                self.functorch_table[k] = fn
                # 返回被装饰的函数，保持装饰器语法正确性
                return fn

            # 分支 3：处理 DispatchKey 的 Python 内核注册
            # 校验 k 是合法的 DispatchKey
            assert isinstance(k, DispatchKey)
            # 禁止直接注册 DispatchKey.Python，引导用户注册具体模式
            assert (
                k != DispatchKey.Python
            ), "Please register a mode for the torch._C.DispatchKey.Python key instead."

            # 若该 DispatchKey 已注册过 Python 内核，抛出运行时异常，避免意外覆盖
            if k in self.py_kernels:
                raise RuntimeError(
                    f"Trying to override a python impl for {k} on operator {self.name()}"
                )
            # 将自定义 Python 内核注册到 py_kernels 字典
            self.py_kernels[k] = fn
            # 清除分发缓存：注册新内核后，原有缓存失效，强制下次重新计算映射
            self._dispatch_cache.clear()
            # 返回被装饰的函数，保持装饰器语法正确性
            return fn

        # 返回内部装饰器函数，支持 @op.py_impl(...) 语法
        return inner

    def py_functionalize_impl(self, fn: _F) -> _F:
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI as _CppFunctionalizeAPI,
            FunctorchFunctionalizeAPI as _FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI as _PythonFunctionalizeAPI,
        )

        # 1. 构建三种功能化(functional, out, inplace)变体的包装函数，统一调用用户传入的核心逻辑 fn
        # 变体 1：对应 DispatchKey.Functionalize（C++ 层功能化内核）
        def functionalize_dk_fn(*args, **kwargs):
            return fn(_CppFunctionalizeAPI(), *args, **kwargs)

        # 变体 2：对应 FunctionalTensorMode（Python 层功能化模式）
        def functionalize_dispatch_mode_fn(mode, *args, **kwargs):
            return fn(_PythonFunctionalizeAPI(mode), *args, **kwargs)

        # 变体 3：对应 functorch.TransformType.Functionalize（functorch 功能化变换）
        def functionalize_functorch_fn(interpreter, *args, **kwargs):
            return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)

        # 2. 批量调用 py_impl 方法，完成三种变体的注册
        self.py_impl(DispatchKey.Functionalize)(functionalize_dk_fn)
        self.py_impl(torch._subclasses.functional_tensor.FunctionalTensorMode)(
            functionalize_dispatch_mode_fn
        )
        self.py_impl(torch._C._functorch.TransformType.Functionalize)(
            functionalize_functorch_fn
        )

        # 3. 返回用户传入的核心逻辑 fn，保持装饰器语法正确性
        return fn
```


## 4.2 子类 OpOverload
- OpOverload 是对应某个具体算子**重载版本**的可调用对象（继承自 OperatorBase），不同于 OpOverloadPacket（重载容器，不对应具体实现），它是用户执行算子计算的「**最终落地载体**」。
- 每个 OpOverload 实例持有：① 具体重载算子的`C++ 实现指针`；② 父类 OpOverloadPacket 引用；③ 算子 schema、分发信息(dispatch info)等元数据。
- `只能通过 OpOverloadPacket 的属性查询（如 .default、._scalar）获取`，无法手动直接创建，保证实例的唯一性和规范性。
- __call__：默认执行，触发 PyTorch 标准 dispatch 机制，适合绝大多数用户场景；
- redispatch：自定义执行，手动指定 DispatchKey，适合框架开发者、自定义算子扩展等高级场景;

```python
class OpOverload(OperatorBase):
    def __init__(self, overloadpacket, op, op_dk, schema, tags):
    # 1. 调用父类（OperatorBase）构造方法，继承底层算子基础能力（如分发缓存、Python 内核管理）
    super().__init__()
    # 2. 核心执行属性：存储该重载算子的 C++ 底层具体实现，是 __call__ 方法执行的核心
    self._op = op
    # 3. 核心分发属性：存储分发相关的算子入口，用于 decompose、redispatch 等高级功能
    self._op_dk = op_dk # dispatch key?
    # 4. 核心元数据属性：存储该重载算子的 schema 规范（参数类型、返回值、别名信息等），是后续所有查询的基础
    self._schema = schema
    # 5. 关联属性：存储父类 OpOverloadPacket 实例，保持与重载包的关联，方便回溯算子全量信息
    self._overloadpacket = overloadpacket
    # 6. 标签属性：存储算子标签（如 "inplace"、"view"），标记算子特性，辅助框架内部判断
    self._tags = tags
    # 7. 格式化属性：将 C++ 层空字符串重载名转为 Python 层友好的 "default"，提升用户体验
    self._overloadname = (
        "default" if schema.overload_name == "" else schema.overload_name
    )
    # 8. 全限定名属性：构建算子的完整名称（如 "aten::linear.default"），用于内部注册和查询
    self._name = self._schema.name
    if schema.overload_name:
        self._name += "." + schema.overload_name
    # 9. 符合 Python 规范：构建 __name__ 属性（如 "linear.default"），兼容 Python 可调用对象规范
    self.__name__ = f"{self._schema.name.split('::')[1]}.{self._overloadname}"
    # 10. 兼容属性：设置 __module__ 和底层 op 的 __module__，保持与父类重载包一致，呼应之前讲解的 __module__ 作用
    self.__module__ = overloadpacket.__module__
    op.__module__ = overloadpacket.__module__
    # 11. 符合 Python 规范：构建 __qualname__ 属性，标识对象的限定名称，兼容模块和反射逻辑
    self.__qualname__ = self._name
    # 12. 注解属性：初始化空注解字典，兼容 Python 类型注解规范，方便后续扩展
    self.__annotations__ = {}
    # 13. 懒加载属性：延迟初始化 OperatorHandle（仅在需要时创建），避免不必要的 C++ 层查询，优化性能
    # 注释：并非所有 OpOverload 都需要 OperatorHandle（如 TorchScript 算子），懒加载更高效
    self._lazy_handle = None

    # 14. 标记属性：判断该算子是否在 Python 层通过 Library.def 定义，用于区分 Python/C++ 算子来源
    self._defined_in_python = self.__qualname__ in torch.library._builtin_ops

    # 15. 核心判断：计算 is_view 属性（判断该算子是否是「视图算子」）
    # 视图算子（如 torch.view、torch.slice）：不修改原张量数据，仅返回原张量的新视图，不占用额外内存
    # 非视图算子（如 torch.add、torch.linear）：修改/创建新张量数据，占用额外内存
    is_write = None
    for a in self._schema.arguments:
        if a.alias_info is None:
            continue
        if is_write is None:
            is_write = a.alias_info.is_write  # 提取参数是否为「写入模式」（修改原数据）
        else:
            # 保守判断：混合可变/不可变别名输入，视为非视图算子
            is_write = a.alias_info.is_write or is_write
    # 视图算子判定条件：存在别名信息，且所有参数均非「写入模式」（不修改原张量数据）
    self.is_view = is_write is not None and not is_write

    ...

    @property
    def tags(self):
        '''
        返回算子标签，方便外部判断算子特性, pytorch 的tag:
        class Tag(Enum):
            core = 0
            cudagraph_unsafe = 1
            data_dependent_output = 2
            dynamic_output_shape = 3
            flexible_layout = 4
            generated = 5
            inplace_view = 6
            maybe_aliasing_or_mutating = 7
            needs_contiguous_strides = 8
            needs_exact_strides = 9
            needs_fixed_stride_order = 10
            nondeterministic_bitwise = 11
            nondeterministic_seeded = 12
            pointwise = 13
            pt2_compliant_tag = 14
            reduction = 15
            view_copy = 16
        '''
        return self._tags

    # 懒加载属性：按需创建 OperatorHandle，优化性能,
    # _handle 懒加载是关键优化，避免初始化时不必要的 C++ 层交互，
    # 仅在需要重分发时才创建，提升框架启动和算子初始化性能
    @property
    def _handle(self):
        # 若未初始化，调用 C++ 层接口查询并创建 OperatorHandle
        if self._lazy_handle is None:
            self._lazy_handle = torch._C._dispatch_find_schema_or_throw(
                self._schema.name, self._schema.overload_name
            )
        # 返回已创建的 OperatorHandle，用于 redispatch 重分发方法
        return self._lazy_handle

    # __call__：魔法方法，让实例可直接调用（如 linear_default(x, weight, bias)）
    # / 是位置-only参数标记，避免和 aten 算子的关键字参数 "self" 冲突，保证所有算子可通过关键字调用
    def __call__(self, /, *args, **kwargs):
        # 直接将参数转发给底层 C++ 算子实现 _op，执行具体的重载计算逻辑
        # 这是用户调用具体重载算子的最终入口，触发默认的 dispatch 机制
        return self._op(*args, **kwargs)

    # redispatch：高级方法，实现「自定义重分发」，手动指定 DispatchKey 集合
    # 核心场景：框架内部自定义算子分发逻辑，跳过默认的 dispatch 优先级，强制使用指定的内核实现
    def redispatch(self, /, keyset, *args, **kwargs):
        # 借助懒加载的 _handle（OperatorHandle），调用 C++ 层的 boxed 重分发接口
        return self._handle.redispatch_boxed(keyset, *args, **kwargs)

    # 查询内核是否支持指定 DispatchKey（如 CPU、CUDA、Autograd）
    def has_kernel_for_dispatch_key(self, k):
        # 先查询父类（OperatorBase）的 Python 内核，再查询 C++ 层内核，保证查询全面性
        return super().has_kernel_for_dispatch_key(
            k
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    # 查询内核是否支持指定 DispatchKey 集合中的任意一个
    def has_kernel_for_any_dispatch_key(self, ks):
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(
            self.name(), ks
        ) or super().has_kernel_for_any_dispatch_key(ks)

    # 判断该算子是否可分解为「复合基础算子」（如 CompositeImplicitAutograd 系列）
    def _can_decompose(self):
        dk = DispatchKey.CompositeImplicitAutograd
        return dk in self.py_kernels or torch._C._dispatch_has_kernel_for_dispatch_key(
            self.name(), dk
        )

    # 将该算子分解为复合基础算子，方便调试和自定义扩展
    # 核心场景：复杂算子拆解为简单基础算子（如 matmul 拆解为 mm + broadcast），便于跟踪计算过程
    def decompose(self, *args, **kwargs):
        dk = DispatchKey.CompositeImplicitAutograd
        # 优先使用 Python 层内核分解
        if dk in self.py_kernels:
            return self.py_kernels[dk](*args, **kwargs)
        # 再使用 C++ 层内核分解
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        # 无法分解时返回 NotImplemented
        else:
            return NotImplemented

    # _uncache_dispatch：清除指定 DispatchKey 的分发缓存
    # 作用：算子更新后，强制下次重新计算分发逻辑，避免使用过期缓存
    def _uncache_dispatch(self, key):
        self._dispatch_cache.pop(key, None)

    # _get_dispatch：核心实现 Python 层分发逻辑预处理，缓存分发处理器（handler）
    # 作用：优化后续分发性能，避免重复计算，处理 Python、PreDispatch、Functionalize 等特殊 DispatchKey
    # 细节：缓存符合条件的 handler，后续分发直接复用，同时处理 dispatch mode 等高级特性
    def _get_dispatch(self, key):
        # 省略具体内部逻辑，核心作用总结：
        # 1. 预处理 DispatchKey，解析最终要使用的内核键；
        # 2. 缓存分发处理器，提升后续调用性能；
        # 3. 处理特殊分发场景（如 Python 模式、Functionalize 优化）；
        # 4. 为框架的高级分发功能（如 dispatch mode）提供支撑
```

## 4.3 子类 HigherOrderOperator

HigherOrderOperator **只能在 Python 层面注册**，不支持 C++ 层面的直接注册，其设计初衷就是为了解决 Python 层高阶算子的分发和扩展问题。

HigherOrderOperator 是 PyTorch 为 ** 高阶算子（Higher Order Operator, HOP）** 设计的抽象基类，其使用场景均围绕「需要灵活分发、兼容 Python 层动态机制的复杂算子」展开，核心场景包括：<br>

- 1. 实现 FuncTorch 相关高阶功能:
  - FuncTorch 是 PyTorch 用于高阶自动微分、JIT 追踪等功能的核心模块，HigherOrderOperator 是 FuncTorch 高阶算子的基础载体。
- 2. 自定义支持 TorchDispatchMode 的高阶算子
  - 当你需要实现「兼容 PyTorch 分发模式上下文（如 no_dispatch()、自定义 TorchDispatchMode）」的复杂算子时，HigherOrderOperator 是最优选择。
- 3. 扩展 Tensor 子类的专属高阶运算
  - 当你自定义了 PyTorch Tensor 子类（如带额外元信息的自定义 Tensor），且需要为其实现专属高阶算子时，HigherOrderOperator 是标准方案。
- 4. 构建无全局命名冲突的复杂组合算子
  - 当你需要实现「由多个基础 PyTorch 算子组合而成的复杂算子」，且希望避免污染全局 torch.ops 命名空间时，HigherOrderOperator 是规范选择。
- 5. 构建无全局命名冲突的复杂组合算子
  - 当你需要实现「由多个基础 PyTorch 算子组合而成的复杂算子」，且希望避免污染全局 torch.ops 命名空间时，HigherOrderOperator 是规范选择。

```python
import abc
import torch
from typing import Any, Callable, TypeVar

# 类型变量定义，匹配原代码中的 _F 类型（表示任意可调用函数）
_F = TypeVar('_F', bound=Callable)

# 全局占位符（原代码中存在的全局变量/方法，保持代码完整性）
_higher_order_ops = {}  # 存储所有高阶算子实例的全局字典
_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = set()  # 默认透传分发键集合
DispatchKey = torch._C.DispatchKey  # 分发键类型别名
dispatch_functorch = lambda self, args, kwargs: None  # FuncTorch 分发占位符
resolve_key = lambda self, dispatch_key: dispatch_key  # 分发键解析占位符
_to_flat_tuple = lambda args, kwargs: ()  # 参数扁平化占位符
_compute_keyset = lambda args, kwargs, non_fallthrough_keys: torch._C._dispatch_keyset_full()  # 分发键集合计算占位符
_len_torch_dispatch_stack_pre_dispatch = lambda: 0  # 预分发栈长度查询占位符

class OperatorBase:
    """PyTorch 算子基础类（占位符，保持继承关系完整性）"""
    def __init__(self):
        self._dispatch_cache = {}
        self.py_kernels = {}
        self.python_key_table = {}

    def py_impl(self, k: Any) -> Callable[[_F], _F]:
        def decorator(func: _F) -> _F:
            self.py_kernels[k] = func
            return func
        return decorator

###########################################################################
# 全局注释版 HigherOrderOperator 类
###########################################################################
class HigherOrderOperator(OperatorBase, abc.ABC):
    """
    PyTorch 高阶算子（Higher Order Operator, HOP）抽象基类。
    核心定位：
    1.  为 Python 层高阶算子提供统一的注册、分发、缓存规范，避免全局命名冲突；
    2.  支持 TorchDispatchMode、Tensor 子类、PreDispatch 等 Python 层动态分发场景；
    3.  所有实例自动注册到 `torch.ops.higher_order.{name}` 命名空间，禁止污染全局 `torch.ops`。

    使用约束：
    - 禁止直接实例化，必须通过子类继承并实现抽象方法 `__call__`；
    - 仅支持 Python 层内核注册，不支持 C++ 层直接注册；
    - 核心分发逻辑在 `dispatch` 方法中实现，无需子类手动重写（特殊场景除外）。

    关键特性：
    - 维护「非透传分发键集合」，控制哪些分发键需要执行当前算子逻辑；
    - 仅缓存非 PreDispatch 分发键对应的内核，避免动态模式下的缓存冲突；
    - 兼容 `no_dispatch()` 上下文，尊重 Python 层分发模式的优先级。
    """

    def __init__(self, name: str, *, cacheable: bool = False):
        """
        构造方法：初始化高阶算子实例，完成命名空间配置、全局注册、分发键初始化。

        参数：
            name: str - 算子唯一标识名称，最终会暴露为 `torch.ops.higher_order.{name}`；
            cacheable: bool (关键字-only) - 是否支持内核缓存，默认 False，PreDispatch 键始终不缓存。

        异常：
            RuntimeError - 直接实例化当前抽象类时抛出，要求必须子类继承。
        """
        # 调用父类 OperatorBase 构造方法，初始化缓存、内核字典等基础属性
        super().__init__()

        # 合法性校验：禁止直接实例化抽象基类，强制子类继承实现具体功能
        if type(self) is HigherOrderOperator:
            raise RuntimeError(
                "Direct instantiation of HigherOrderOperator is not allowed. Please subclass it."
            )

        # 保存算子核心属性：名称、缓存状态
        self._name = name  # 算子唯一名称（私有属性，通过 name() 方法暴露）
        self._cacheable = cacheable  # 缓存开关（私有属性，通过 cacheable() 方法暴露）

        # 命名空间配置：让算子被识别为 `torch.ops.higher_order` 下的成员
        self.__name__ = name  # 覆盖实例名称，避免 _OPNamespace 抛出异常
        self._ns = "higher_order"  # 算子所属命名空间标识
        self.__module__ = "torch.ops.higher_order"  # 覆盖实例所属模块，符合 PyTorch 算子规范

        # 全局注册：将当前算子实例存入全局高阶算子字典，便于后续查找和管理
        _higher_order_ops[name] = self

        # 初始化「非透传分发键集合」：默认包含所有 PyTorch 分发键
        # 非透传键：表示该分发键需要执行当前算子的逻辑，不向下透传
        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        # 处理默认透传分发键：将预设的默认透传键从非透传集合中移除
        # 透传键：表示该分发键跳过当前算子，直接向下透传到下一级分发逻辑
        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        """
        注意：必须注册 PreDispatch 键实现
        原因：高阶算子常使用 aot-dispatch 追踪张量突变，在预分发阶段（PreDispatch）
        若不注册专属实现，嵌套追踪时会检测到 PreDispatch 键仍处于活跃状态，导致逻辑冲突。
        预分发阶段的核心处理是重定向到下一级分发键，且仅在无活跃模式时安全执行。
        """

    def py_impl(self, k: Any) -> Callable[[_F], _F]:
        """
        Python 层内核注册装饰器生成方法（核心注册接口）。
        功能：
        1.  接收分发键 k（通常为 DispatchKey 类型），生成一个内核注册装饰器；
        2.  若 k 是有效 DispatchKey 且未在非透传集合中，自动将其添加到非透传集合；
        3.  调用父类方法完成实际的 Python 内核注册，将内核函数存入 py_kernels 字典。

        参数：
            k: Any - 通常为 DispatchKey 类型，指定内核对应的分发键。

        返回：
            Callable[[_F], _F] - 内核注册装饰器，用于装饰具体的 Python 内核函数。
        """
        # 若 k 是有效 DispatchKey 且未在非透传集合中，更新非透传集合
        if isinstance(k, DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)

        # 调用父类 OperatorBase 的 py_impl 方法，完成内核注册的核心逻辑
        return super().py_impl(k)


    def cacheable(self) -> bool:
        """
        查询方法：返回算子是否支持内核缓存的状态。

        返回：
            bool - 初始化时传入的 cacheable 参数值，PreDispatch 键不受该状态影响（始终不缓存）。
        """
        return self._cacheable

    def fallthrough(self, dispatch_key: DispatchKey) -> None:
        """
        透传键配置方法：将指定分发键标记为「fallthrough」，从非透传集合中移除。

        参数：
            dispatch_key: DispatchKey - 需要标记为透传的分发键。
        """
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def dispatch(self, /, dispatch_key: DispatchKey, *args: Any, **kwargs: Any) -> Any:
        """
        核心分发方法：根据传入的分发键，路由到对应的 Python 内核执行。
        执行优先级（从高到低）：
        1.  优先查询分发缓存（非 PreDispatch 键）；
        2.  处理 FuncTorch 动态层前端模式；
        3.  处理 Python 层分发（TorchDispatchMode → Tensor 子类）；
        4.  处理 PreDispatch 预分发模式；
        5.  解析最终分发键，执行对应 Python 内核并缓存（非 PreDispatch 键）。

        参数：
            dispatch_key: DispatchKey（位置-only）- 分发键，指定当前的执行模式/上下文；
            *args: Any - 算子执行所需的位置参数；
            **kwargs: Any - 算子执行所需的关键字参数。

        返回：
            Any - 算子内核执行的结果。

        异常：
            NotImplementedError - 未找到对应分发键的注册内核时抛出；
            TypeError - Python 层多分发所有处理器均返回 NotImplemented 时抛出。
        """
        # 延迟导入：避免循环导入，仅在分发时加载所需模块
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        # 分支 1：优先从分发缓存中获取内核（提升重复调用效率）
        if dispatch_key in self._dispatch_cache:
            kernel = self._dispatch_cache[dispatch_key]
            # 断言：缓存的内核不能是 DispatchKey 类型（禁止 C++ 内核缓存）
            assert not isinstance(kernel, DispatchKey)
            return kernel(*args, **kwargs)

        # 分支 2：处理 FuncTorch 动态层前端模式（专属分发逻辑）
        if dispatch_key == DispatchKey.FuncTorchDynamicLayerFrontMode:
            return dispatch_functorch(self, args, kwargs)

        # 分支 3：处理 Python 层分发（核心：多路径分发，兼容用户模式和 Tensor 子类）
        if dispatch_key == DispatchKey.Python:
            # 辅助函数：检测张量是否包含 Python 分发键（是否需要重载）
            def has_python_key(tensor: torch.Tensor) -> bool:
                return torch._C._dispatch_keys(tensor).has("Python")

            # 辅助函数：检查单个参数是否需要重载，并收集到列表中
            overloaded_args_list = []
            def check_overloaded(arg: Any) -> None:
                if isinstance(arg, torch.Tensor) and has_python_key(arg):
                    overloaded_args_list.append(arg)

            # 遍历所有位置参数和关键字参数，收集需要重载的张量
            for arg in (*args, *kwargs.values()):
                check_overloaded(arg)
                # 递归处理列表/元组类型的嵌套参数
                if isinstance(arg, (list, tuple)):
                    for a in arg:
                        check_overloaded(a)
            overloaded_args = tuple(overloaded_args_list)

            # 子步骤 3.1：分发给当前活跃的 TorchDispatchMode（用户自定义模式）
            curr_mode = _get_current_dispatch_mode()
            if curr_mode is not None:
                from torch.utils._python_dispatch import _pop_mode_temporarily

                if type(curr_mode) in self.python_key_table:
                    # 查找当前模式对应的注册处理器
                    handler = self.python_key_table[type(curr_mode)]
                    # 临时弹出模式上下文，执行处理器逻辑
                    with _pop_mode_temporarily() as mode:
                        # 采用「自然调用约定」：(mode, *args, **kwargs)
                        result = handler(mode, *args, **kwargs)
                else:
                    # 未找到对应模式的注册处理器，抛出未实现异常
                    raise NotImplementedError(
                        f"There was no rule registered for HOP {self._name} and mode {curr_mode}. "
                        f"We recommend filing an issue."
                    )
                # 若处理器返回有效结果，直接返回（终止后续分发逻辑）
                if result is not NotImplemented:
                    return result

            # 子步骤 3.2：分发给需要重载的 Tensor 子类
            for arg in overloaded_args:
                subclass_type = type(arg)
                # 跳过禁用了 __torch_dispatch__ 的子类
                if subclass_type.__torch_dispatch__ == torch._C._disabled_torch_dispatch_impl:
                    continue
                if subclass_type in self.python_key_table:
                    # 查找 Tensor 子类对应的注册处理器
                    handler = self.python_key_table[subclass_type]
                    # 采用「自然调用约定」：(*args, **kwargs)
                    result = handler(*args, **kwargs)
                else:
                    # 未找到对应子类的注册处理器，抛出未实现异常
                    raise NotImplementedError(
                        f"There was no rule registered for HOP {self._name} and subclass {subclass_type}. "
                        f"We recommend filing an issue."
                    )
                # 若处理器返回有效结果，直接返回（终止后续分发逻辑）
                if result is not NotImplemented:
                    return result

            # 子步骤 3.3：所有处理器均返回 NotImplemented，分发失败
            raise TypeError(
                f"Multiple dispatch failed for {self._name}. There was no registered that "
                f"did not return NotImplemented. Use HOP.py_impl to register some. "
                f"Tried mode: {curr_mode}) and subclasses: "
                f"{[type(a) for a in overloaded_args]}"
            )

        # 分支 4：处理 PreDispatch 预分发模式（避免嵌套追踪冲突）
        functionality_key = torch._C._to_functionality_key(dispatch_key)
        if functionality_key == DispatchKey.PreDispatch:
            from torch.utils._python_dispatch import _pop_mode_temporarily

            # 校验：预分发栈非空，且尊重 no_dispatch() 上下文（排除 Python 键）
            if (_len_torch_dispatch_stack_pre_dispatch() > 0) and not torch._C._dispatch_tls_is_dispatch_key_excluded(
                DispatchKey.Python
            ):
                # 断言：预分发模式下必须有活跃的模式，且已注册对应处理器
                curr_mode = _get_current_dispatch_mode()
                assert curr_mode is not None, (
                    "Illegal invocation of dispatch on torch._C.DispatchKey.PreDispatch without a mode."
                )
                assert type(curr_mode) in self.python_key_table, (
                    f"Current active mode {curr_mode} not registered"
                )

                # 查找预分发模式对应的处理器，临时弹出模式上下文执行
                handler = self.python_key_table[type(curr_mode)]
                with _pop_mode_temporarily(functionality_key) as mode:
                    return handler(mode, *args, **kwargs)

        # 分支 5：解析最终分发键，执行对应 Python 内核（核心执行逻辑）
        final_key = resolve_key(self, dispatch_key)

        # 校验：是否注册了对应最终分发键的 Python 内核
        if final_key not in self.py_kernels:
            raise NotImplementedError(
                f"could not find kernel for HigherOrderOperator {self._name} "
                f"at dispatch key {final_key} (resolved from {dispatch_key})"
            )

        # 缓存：仅缓存非 PreDispatch 键的内核（PreDispatch 行为依赖动态模式，无法缓存）
        if dispatch_key != DispatchKey.PreDispatch:
            self._dispatch_cache[dispatch_key] = self.py_kernels[final_key]

        # 取出内核并执行，断言内核非 DispatchKey 类型（禁止执行 C++ 内核）
        kernel = self.py_kernels[final_key]
        assert not isinstance(kernel, DispatchKey)
        return kernel(*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        """
        抽象调用方法：算子的对外调用入口（子类必须实现）。
        核心逻辑：
        1.  禁用 Dynamo 追踪（Dynamo 已提前追踪高阶算子逻辑，无需重复追踪）；
        2.  扁平化参数，校验是否存在需要覆盖的 torch 函数；
        3.  计算最高优先级分发键，调用 dispatch 方法完成核心分发执行。

        参数：
            *args: Any（位置-only）- 算子执行所需的位置参数；
            **kwargs: Any - 算子执行所需的关键字参数。

        返回：
            Any - 算子执行的最终结果。

        注意：
            - 使用位置-only 参数（/），避免与自定义算子的 `self` 参数命名冲突；
            - 子类实现时可扩展参数校验、结果后处理等逻辑，但需保留核心分发流程。
        """
        # 延迟导入：仅在调用时加载 Dynamo 禁用工具
        from torch._dynamo import disable

        # 禁用 Dynamo 追踪：避免重复追踪高阶算子内部逻辑
        @disable
        def wrapper() -> Any:
            # 步骤 1：扁平化参数，便于后续 torch 函数覆盖校验
            flat_args = _to_flat_tuple(args, kwargs)

            # 步骤 2：校验并处理 torch 函数覆盖
            if torch.overrides.has_torch_function(flat_args):
                return torch.overrides.handle_torch_function(
                    self, flat_args, *args, **kwargs
                )

            # 步骤 3：计算分发键集合，获取最高优先级分发键
            dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
            highest_priority_key = dispatch_key_set.highestPriorityTypeId()

            # 步骤 4：调用核心分发方法，执行算子逻辑并返回结果
            return self.dispatch(highest_priority_key, *args, **kwargs)

        # 执行包装函数，返回最终结果
        return wrapper()

    ...
```

## 4.4 TorchBindOpOverload

- **Fallthrough**
  - 定义：当一个 dispatch key 被标记为 fallthrough 时，它意味着该 dispatch key 不提供实际的实现，而是"直通"到其他 dispatch key 的实现
  - 作用：允许特定 dispatch key 作为透明层，将请求传递给其他 dispatch key 处理
  - 检查方式：通过 torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough() 检查某个 dispatch key 是否被标记为 fallthrough
  - 代码体现：在 TorchBindOpOverload._fallthrough_keys() 方法中，会检查某个 key 是否是 fallthrough：

- **Fallback**
  - 定义：当没有找到特定 dispatch key 的实现时，系统会回退到默认的或通用的实现
  - 作用：提供一个**备用**实现，确保操作总能被执行
  - 机制：通常涉及后端回退机制，允许在特定后端不可用时使用通用实现(eg.cpu 实现)
  - 代码体现：在 resolve_key 函数中可以看到后端回退逻辑：

- **torch.ScriptObject**
  核心设计初衷是 解决「Python 自定义类无法与 PyTorch C++ 生态高效、安全交互」的问题，构建 Python 自定义类与 PyTorch C++ 内核、Torch Script、模型导出 / 编译等全生态的「统一桥梁」。

> example:

```python
import torch
import torch.utils.cpp_extension

# 1. 定义一个简单的 Python 自定义类（数据处理器，实现数据归一化）
class DataNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean  # 均值
        self.std = std    # 标准差

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化计算：(x - 均值) / 标准差"""
        return (x - self.mean) / self.std

# 2. 通过 TorchBind 机制，将自定义类包装为 torch.ScriptObject（核心步骤）
# 注：TorchBind 注册需配合简单 C++ 绑定（此处简化，聚焦 Python 端效果）
# 实际使用中，需通过 torch.library 或 cpp_extension 完成 C++ 绑定，最终得到 ScriptObject 实例
def create_script_object_normalizer(mean: float, std: float) -> torch.ScriptObject:
    # 模拟 TorchBind 包装过程：返回一个 ScriptObject 实例（真实场景中由 C++ 绑定返回）
    normalizer = DataNormalizer(mean, std)
    # 包装为 ScriptObject，使其能被 PyTorch C++ 调度器识别
    return torch._C._wrap_script_object(normalizer)  # 底层 API，用于演示包装效果
```

-  **FakeScriptObject（模拟 / 占位对象）**
  - 本质：仅存在于 Python 端的轻量级模拟对象 / 占位对象，是 torch.ScriptObject 的「纯 Python 镜像」，无任何 C++ 端对应实现，C++ 调度器无法识别它。
  - 适用场景：模型追踪（tracing）、导出（export）、编译（compile）等需要跳过 C++ 调度的场景。

```python
# 注释：说明 TorchBindOpOverload 类的适用场景：
# 1. 属于 PyTorch TorchBind 自定义操作
# 2. 该操作至少有一个重载模式的输入包含 torch.ScriptObject（自定义类实例）
# 注释：说明该类的核心行为：当输入中包含 FakeScriptObject 时，跳过 C++ 调度器，完全在 Python 端完成调度

# 定义 TorchBindOpOverload 类，继承自 OpOverload（PyTorch 操作重载基类，提供基础调度能力）
class TorchBindOpOverload(OpOverload):
    # 定义方法 _fallthrough_keys，返回值为 DispatchKey 列表（DispatchKey 是 PyTorch 调度场景标识）
    # 该方法的核心作用：获取当前操作的「有效穿透调度键」列表，fallthrough key 不执行具体逻辑，仅向下传递调度权
    def _fallthrough_keys(self) -> List[DispatchKey]:
        # TODO: we should be calling the fallback for these, but a fallthrough is almost close
        # enough to the fallback in most cases that we care about.

        # 定义默认穿透调度键列表，包含 PyTorch 核心调度场景
        _DEFAULT_FALLTHROUGH_KEYS = [
            DispatchKey.Autograd,        # 自动求导调度场景
            DispatchKey.AutogradCPU,     # CPU 设备自动求导调度场景
            DispatchKey.AutogradCUDA,    # CUDA 设备自动求导调度场景
            DispatchKey.ADInplaceOrView, # 原地操作/视图自动求导调度场景
            DispatchKey.BackendSelect,   # 设备后端选择调度场景
            DispatchKey.PythonTLSSnapshot, # Python TLS 快照调度场景
            DispatchKey.PythonDispatcher, # Python 调度器核心场景
        ]

        # 定义内部辅助函数：判断某个调度键是否应使用「穿透逻辑fallthrough」而非「回退逻辑fallback」
        def _may_use_fallthrough_instead_of_fallback(key: DispatchKey):
            # 第一步：检查当前操作（self.name() 获取操作名）是否为该调度键注册了内核（具体实现逻辑）
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), key):
                # 若注册了内核，返回该内核是否为「穿透内核fullthrough」（仅传递调度权，无具体实现）
                return torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                    self.name(), key
                )

            # 第二步：若未注册内核，返回以下复合条件（满足其一即符合穿透要求）
            # 条件1：该调度键无对应的 Python 内核实现
            # 条件2：该调度键的 Python 内核就是穿透内核
            return (
                key not in self.py_kernels
                or self.py_kernels[key] is torch.library.fallthrough_kernel
            )

        # 列表推导式：遍历默认穿透键列表，仅保留满足辅助函数判断条件的键
        # 最终返回「有效穿透调度键」列表，用于后续 Python 端调度
        return [
            key
            for key in _DEFAULT_FALLTHROUGH_KEYS
            if _may_use_fallthrough_instead_of_fallback(key)
        ]

    # 装饰器：将该方法标记为上下文管理器，支持 with 语句调用
    # 作用：实现「进入上下文时注册、退出上下文时清理」的原子操作，避免残留影响
    @contextlib.contextmanager
    # 方法作用：临时将当前 TorchBind 操作注册为「有副作用操作」，退出上下文时自动取消注册
    def _register_as_effectful_op_temporarily(self):
        # 延迟导入：从 PyTorch 高阶操作副作用模块导入所需组件
        # _EffectType：副作用类型枚举；_register_effectful_op：副作用操作注册函数；SIDE_EFFECTS：已注册副作用操作字典
        from torch._higher_order_ops.effects import (
            _EffectType,
            _register_effectful_op,
            SIDE_EFFECTS,
        )

        # try-finally 块：保证无论上下文内代码是否报错，最终都能执行清理操作
        try:
            # 检查当前操作是否已在副作用字典中，避免重复注册
            if self not in SIDE_EFFECTS:
                # 将当前操作注册为「有序副作用操作」（_EffectType.ORDERED）
                # 保证操作执行顺序不被打乱，对模型追踪/编译至关重要
                _register_effectful_op(self, _EffectType.ORDERED)
            # 上下文管理器核心：暂停方法执行，进入 with 包裹的代码块
            # 代码块执行完成后，恢复此处继续执行 finally 块
            yield
        finally:
            # 退出上下文时：检查当前操作是否仍在副作用字典中
            if self in SIDE_EFFECTS:
                # 从副作用字典中删除当前操作，完成清理，恢复操作原始状态
                del SIDE_EFFECTS[self]

    # 注释：说明 / 的作用：位置仅参数标记，避免与 aten 操作的 self 参数命名冲突
    # 保证所有 aten 操作都能通过关键字参数正常调用
    # Use positional-only argument to avoid naming collision with aten ops arguments
    # that are named "self". This way, all the aten ops can be called by kwargs.

    # 定义 __call__ 方法：类实例被直接调用时的入口方法（如 op(*args) 会触发该方法）
    # /：标记后续参数为位置仅参数，不能以关键字形式传递（此处用于避免命名冲突）
    # *args：接收所有可变位置输入参数
    # **kwargs：接收所有可变关键字输入参数
    def __call__(self, /, *args, **kwargs):
        # 调用外部函数：判断输入参数中是否包含 FakeScriptObject
        # 若是，返回 True，需要跳过 C++ 调度器，执行纯 Python 端调度
        if _must_dispatch_in_python(args, kwargs):
            # 注释：解释纯 Python 调度的原因和注意事项
            # 1. C++ 调度器无法识别 FakeScriptObject，必须在 Python 端调度
            # 2. 仅临时注册副作用操作，避免影响操作的正常急切执行
            # 3. 不在构造函数中全局注册，避免影响 autograd.profiler 等操作
            # When any inputs are FakeScriptObject, we need to
            # skip c++ dispatcher and dispatch in python through _get_dispatch of python_dispatcher
            # because C++ dispatcher will check the schema and cannot recognize FakeScriptObject.
            #
            # Note:
            # 1. We only register the torchbind op temporarily as effectful op because we only want
            #    the effect token functionalization logic to be applied during tracing. Otherwise, the behavior
            #    of the eagerly executing the op might change after tracing.
            # 2. We don't want to register the op as effectful for all torchbind ops in ctor because this might
            #    cause unexpected behavior for some autograd.profiler ops e.g. profiler._record_function_exit._RecordFunction.

            # 进入上下文管理器：临时将操作注册为有副作用操作
            with self._register_as_effectful_op_temporarily():
                # 调用 Python 端调度核心方法，传入输入参数和有效穿透键列表
                # 返回调度执行结果
                return self._dispatch_in_python(args, kwargs, self._fallthrough_keys())

        # 若输入中无 FakeScriptObject：直接调用底层 C++ 操作内核（self._op）
        # 按 PyTorch 正常调度逻辑执行，无需 Python 端特殊处理
        return self._op(*args, **kwargs)

    # 方法作用：纯 Python 端调度的核心逻辑，实现调度键计算、处理器获取、穿透递归和最终执行
    # args/kwargs：操作的输入参数
    # fallthrough_keys：有效穿透调度键列表
    def _dispatch_in_python(self, args, kwargs, fallthrough_keys):
        # 第一步：初始化完整的 PyTorch 调度键集合
        non_fallthrough_keys = torch._C._dispatch_keyset_full()

        # 第二步：遍历穿透键列表，从完整调度键集合中移除所有穿透键
        # 最终 non_fallthrough_keys 仅保留「需要执行具体实现的非穿透调度键」
        for key in fallthrough_keys:
            non_fallthrough_keys = non_fallthrough_keys.remove(key)

        # 第三步：根据输入参数和非穿透键集合，计算当前场景的有效调度键集合
        dispatch_key_set = _compute_keyset(args, kwargs, non_fallthrough_keys)

        # 第四步：从有效调度键集合中获取「最高优先级调度键」（PyTorch 调度遵循高优先级优先执行）
        dispatch_key = dispatch_key_set.highestPriorityTypeId()

        # 第五步：获取调度处理器（带缓存优化，避免重复获取带来的性能开销）
        # self._dispatch_cache：调度处理器缓存字典，存储已获取过的处理器
        handler = (
            # 若调度键不在缓存中：调用 _get_dispatch 获取处理器
            self._get_dispatch(dispatch_key)
            if dispatch_key not in self._dispatch_cache
            # 若调度键在缓存中：直接从缓存读取，提升性能
            else self._dispatch_cache[dispatch_key]
        )

        # 第六步：处理处理器为 DispatchKey 类型的场景（说明该处理器是穿透逻辑，无具体实现）
        if isinstance(handler, DispatchKey):
            # 注释：穿透键可通过 torch.library.impl 运行时注册，需要添加到穿透列表并重新调度
            # fallthrough keys can be registered at runtime via torch.library.impl
            # so need to add it to fallthrough_keys and re-dispatch.

            # 检查该调度键对应的内核是否为穿透内核
            if torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                self.name(), dispatch_key
            ):
                # 递归调用自身：将该键添加到穿透列表，重新执行 Python 端调度（向下传递调度权）
                return self._dispatch_in_python(
                    args, kwargs, fallthrough_keys + [dispatch_key]
                )

            # 若不是穿透内核：抛出运行时异常，详细说明错误场景和排查建议
            raise RuntimeError(
                f"Torchbind op {self} received a FakeScriptObject input when dispatching {handler}."
                f" but no python implementation is found."
                f" Please file an issue on this when you encounter this error."
                f" This error can happen when you export or compile the model."
                f" It can still happpen even if a C++ implementation for {dispatch_key}. "
                f" has been registered. That's because FakeScriptObject purely lives in python and cannot work "
                f" with a C++ implementation."
            )

        # 断言：确保处理器是可调用对象（函数/方法），进行类型校验，避免后续调用报错
        # type: ignore[arg-type]：忽略类型检查工具的参数类型提示（兼容动态获取的处理器）
        assert isinstance(handler, Callable)  # type: ignore[arg-type]

        # 第七步：调用可调用处理器，传入输入参数，执行具体逻辑并返回结果
        # 完成纯 Python 端的调度和执行
        return handler(*args, **kwargs)
```

# 5 py_impl 和 torch.library.impl 函数注册的区别

- HigherOrderOperator.py_impl：
  - 高阶算子（HOP）专属、Python 层唯一、绑定实例的内核注册方法，仅服务于 HigherOrderOperator 子类实例，核心目标是支撑高阶算子的 Python 层动态分发（兼容 TorchDispatchMode、PreDispatch 等），无跨层面、多后端支持能力。
- torch.library.impl：
  - PyTorch 全局通用、双层面支持、绑定算子库的内核注册 API，是 PyTorch 算子生态的核心扩展接口，核心目标是构建可扩展的普通算子 / 自定义算子库，支持多后端、多分发键、Python/C++ 双层面注册。

# 6 example

```python
import torch

# 定义参数与构造张量（与上面案例一致，省略重复代码）
batch_size = 4
in_features = 12
out_features = 6
input_tensor = torch.randn(batch_size, in_features)
weight_tensor = torch.randn(out_features, in_features)

# 核心调用：bias=None 关闭偏置
output_tensor_no_bias = torch.ops.aten.linear(input_tensor, weight_tensor, None)

# 查看结果
print("=" * 60)
print("无偏置输出张量形状：", output_tensor_no_bias.shape)
print("无偏置输出张量内容：\n", output_tensor_no_bias)
```