# Op Dispatch Stack
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在注册算子的时候，pytorch除了会自动注册底层实现意外，还会注册一些其它dispatchkey的kernel，比如自动微分，Tracer等。<br>

在编译生成的文件VariableType_3.cpp中，有自动微分的注册: <br>

```c++
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
m.impl("max_unpool2d",
       TORCH_FN(VariableType::max_unpool2d)
);
}
```

