# 如何构建 PyTorch 库及其各个组件
- PyTorch 项目中，BUILD.bazel 文件用于定义如何构建 PyTorch 库及其各个组件。
- [BUILD.bazel](https://github.com/pytorch/pytorch/blob/main/BUILD.bazel)
**(应增加详细解释)**

# 1 BUILD.bazel 中对torch._C的声明
```python
pybind_extension(
    name = "functorch/_C",
    copts=[
        "-DTORCH_EXTENSION_NAME=_C"
    ],
    srcs = [
        "functorch/csrc/init_dim_only.cpp",
    ],
    deps = [
        ":functorch",
        ":torch_python",
        ":aten_nvrtc",
    ],
)
```

# 2 torch._C 模块定义
- [stub.c](https://github.com/pytorch/pytorch/blob/main/torch/csrc/stub.c)

```c
#include <Python.h>

extern PyObject* initModule(void);

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  return initModule();
}
```

# 3 initModule()

