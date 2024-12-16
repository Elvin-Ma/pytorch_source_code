# pytorch 算子分类汇总
- [参考自native_functions.yaml](https://github.com/pytorch/pytorch/tree/v2.5.0/aten/src/ATen/native/README.md)

&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;ATen "native" 函数是向 ATen 添加operators和functions的现代机制。Native 函数在 native_functions.yaml 中**声明**，并在此目录中的一个 cpp 文件中定义其实现。<br>

&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;&nbasp;与所有ATen方法/函数一样，原生函数在ATen的C++和Python API中均可用。在C++中，它们可以作为Tensor上的方法（如t.mymeth()）或ATen命名空间中的函数（如at::myfunc()）来使用。在PyTorch中，它们可以作为Variable上的方法或torch._C._FunctionBase上的函数来使用。（用户有责任将这些函数重新导出到更面向用户的模块中。）<br>
