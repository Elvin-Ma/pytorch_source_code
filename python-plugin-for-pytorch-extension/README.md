# 1 pytorch out-of-tree device extensions

在 PyTorch 源码中，"Leverage the Python plugin mechanism to load out-of-the-tree device extensions" 指的是利用 Python 的动态插件机制（如 setuptools entry points）加载独立于 PyTorch 主源码树（out-of-the-tree）的硬件设备扩展（如自定义 GPU、NPU 等）。这种设计允许第三方开发者无需修改 PyTorch 源码即可添加对新硬件的支持。

# 2 在第三方硬件中使用 setuptools Entry Points 注册插件

第三方扩展在 setup.py 中声明 Entry Point，将扩展模块注册到 PyTorch 的特定命名空间（如 torch.device）：

```python
from setuptools import setup

setup(
    name="custom_device_extension",
    entry_points={
        "torch.device": [
            # 格式: "设备名 = 模块路径:初始化函数"
            "custom_npu = custom_npu_module:init_custom_npu"
        ]
    },
    ...
)
```

# 3. PyTorch 启动时动态加载扩展

- [torch/__init__.py](/root/projects/pytorch/torch/__init__.py)

```python
def _import_device_backends():
    """
    Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
    See this RFC: https://github.com/pytorch/pytorch/issues/122468
    """
    from importlib.metadata import entry_points

    group_name = "torch.backends"
    if sys.version_info < (3, 10):
        backend_extensions = entry_points().get(group_name, ())
    else:
        backend_extensions = entry_points(group=group_name)

    for backend_extension in backend_extensions:
        try:
            # Load the extension
            entrypoint = backend_extension.load()
            # Call the entrypoint
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the backend extension: {backend_extension.name}. "
                f"You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0."
            ) from err
```

# 4 entry_points
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;entry_points 是 Python 的 importlib.metadata 模块中提供的一种机制，用于在**不同 Python 项目**之间定义和发现可扩展的接口。它主要用于查找和加载**通过 Python 包注册的功能点**（例如插件、命令行工具等）。

**具体作用** <br>
- 插件系统支持：<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;它允许开发者**定义一个“入口点”（entry point）**，其他包可以通过这个入口点**动态发现并加载特定功能**。常见于一些框架（如 setuptools 插件系统、Pytest 插件等）中，用来支持第三方扩展。

- 动态发现模块功能：<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可以`根据名称或组名来查询已注册的功能`，这些功能通常`指向某个模块中的类或函数`。

example：通过 entry_points(group='myapp.plugins') 可以找到`所有`注册为 myapp.plugins 组的入口点。
调用注册的功能：

加载后可以直接调用相关功能，实现**模块化和解耦架构**。

- [importlib](/usr/lib/python3.10/importlib/metadata/__init__.py)

**module = import_module(match.group('module'))**

```python
class EntryPoint(
        collections.namedtuple('EntryPointBase', 'name value group')):
    """An entry point as defined by Python packaging conventions.

    See `the packaging docs on entry points
    <https://packaging.python.org/specifications/entry-points/>`_
    for more information.

    >>> ep = EntryPoint(
    ...     name=None, group=None, value='package.module:attr [extra1, extra2]')
    >>> ep.module
    'package.module'
    >>> ep.attr
    'attr'
    >>> ep.extras
    ['extra1', 'extra2']
    """

    pattern = re.compile(
        r'(?P<module>[\w.]+)\s*'
        r'(:\s*(?P<attr>[\w.]+)\s*)?'
        r'((?P<extras>\[.*\])\s*)?$'
    )
    """
    A regular expression describing the syntax for an entry point,
    which might look like:

        - module
        - package.module
        - package.module:attribute
        - package.module:object.attribute
        - package.module:attr [extra1, extra2]

    Other combinations are possible as well.

    The expression is lenient about whitespace around the ':',
    following the attr, and following any extras.
    """

    dist: Optional['Distribution'] = None

    def load(self):
        """Load the entry point from its definition. If only a module
        is indicated by the value, return that module. Otherwise,
        return the named object.
        """
        match = self.pattern.match(self.value)
        module = import_module(match.group('module'))
        attrs = filter(None, (match.group('attr') or '').split('.'))
        return functools.reduce(getattr, attrs, module)

    @property
    def module(self):
        match = self.pattern.match(self.value)
        return match.group('module')

    @property
    def attr(self):
        match = self.pattern.match(self.value)
        return match.group('attr')

    @property
    def extras(self):
        match = self.pattern.match(self.value)
        return re.findall(r'\w+', match.group('extras') or '')

    def _for(self, dist):
        self.dist = dist
        return self

    def __iter__(self):
        """
        Supply iter so one may construct dicts of EntryPoints by name.
        """
        msg = (
            "Construction of dict of EntryPoints is deprecated in "
            "favor of EntryPoints."
        )
        warnings.warn(msg, DeprecationWarning)
        return iter((self.name, self))

    def __reduce__(self):
        return (
            self.__class__,
            (self.name, self.value, self.group),
        )

    def matches(self, **params):
        """
        EntryPoint matches the given parameters.

        >>> ep = EntryPoint(group='foo', name='bar', value='bing:bong [extra1, extra2]')
        >>> ep.matches(group='foo')
        True
        >>> ep.matches(name='bar', value='bing:bong [extra1, extra2]')
        True
        >>> ep.matches(group='foo', name='other')
        False
        >>> ep.matches()
        True
        >>> ep.matches(extras=['extra1', 'extra2'])
        True
        >>> ep.matches(module='bing')
        True
        >>> ep.matches(attr='bong')
        True
        """
        attrs = (getattr(self, param) for param in params)
        return all(map(operator.eq, params.values(), attrs))

```

# 5 entry_points 注册需要特定的key



# 6 example: pytorch 自身的entry_points

**torchrun 启动会走到torch.distributed.run.main()**

```python
    entry_points = {
        "console_scripts": [
            "torchrun = torch.distributed.run:main",
        ],
        "torchrun.logs_specs": [
            "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
        ],
    }
```

- torchrun 代码

```python
#!/usr/bin/python
# EASY-INSTALL-ENTRY-SCRIPT: 'torch','console_scripts','torchrun'
import re
import sys

# for compatibility with easy_install; see #2198
__requires__ = 'torch'

try:
    from importlib.metadata import distribution
except ImportError:
    try:
        from importlib_metadata import distribution
    except ImportError:
        from pkg_resources import load_entry_point


def importlib_load_entry_point(spec, group, name):
    dist_name, _, _ = spec.partition('==')
    matches = (
        entry_point
        for entry_point in distribution(dist_name).entry_points
        if entry_point.group == group and entry_point.name == name
    )
    return next(matches).load()


globals().setdefault('load_entry_point', importlib_load_entry_point)


if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(load_entry_point('torch', 'console_scripts', 'torchrun')())
```