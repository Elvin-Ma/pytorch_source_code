# 1 通过torch.dist 通信算子调度机制
## 1.1 python 侧通信算子（comm ops）调用接口

- [**distributed_c10d.py**](https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py)

```python
# torch/distributed/distributed_c10d.py
def broadcast(tensor, src, group=None, async_op=False):
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=None, async_op=False)
def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)
def all_gather(tensor_list, tensor, group=None, async_op=False)
def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None, async_op)
def gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
def all_to_all_single(output,input, output_split_sizes=None,
                  input_split_sizes=None, group=None, async_op=False)
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None)
```

> 注意：这些python侧算子内通过**ProcessGroup的方法**来调度具体的通信算子

```python
# torch/distributed/distributed_c10d.py
@_exception_logger
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    _check_single_tensor(tensor, "tensor")
    if _rank_not_in_group(group):
        _warn_not_in_group("all_reduce")
        return

    if tensor.is_complex():
        if not supports_complex(op):
            raise ValueError(f"all_reduce does not support {op} on complex tensors")
        tensor = torch.view_as_real(tensor)

    opts = AllreduceOptions()
    opts.reduceOp = op
    if group is None:
        group = _get_default_group()

    if group in _world.pg_coalesce_state.keys():
        # We are in coalescing context, do not issue single operation, just append a collective representation
        coll = _CollOp(all_reduce, tensor, None, op, None)
        _world.pg_coalesce_state[group].append(coll)
        if async_op:
            return _IllegalWork()
        else:
            return None

    work = group.allreduce([tensor], opts)

    if async_op:
        return work
    else:
        work.wait()

```

## 1.2 通过pybind 绑定到 ProcessGroup 里的算子

- [**distributed/c10d/init.cpp**](torch/csrc/distributed/c10d/init.cpp)

```c++
auto processGroup =
      py::class_<
          ::c10d::ProcessGroup,
          c10::intrusive_ptr<::c10d::ProcessGroup>,
          ::c10d::PyProcessGroup>(module, "ProcessGroup",
          R"(A ProcessGroup is a communication primitive that allows for
          collective operations across a group of processes.

          This is a base class that provides the interface for all
          ProcessGroups. It is not meant to be used directly, but rather
          extended by subclasses.)")
          .def(
              py::init<int, int>(),
              py::arg("rank"),
              py::arg("size"),
              R"(Create a new ProcessGroup instance.)")
          .def(
              py::init<
                  const c10::intrusive_ptr<::c10d::Store>&,
                  int,
                  int>(),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Create a new ProcessGroup instance.)")
          .def("rank", &::c10d::ProcessGroup::getRank, R"(Get the rank of this process group.)")
          .def("size", &::c10d::ProcessGroup::getSize, R"(Get the size of this process group.)")
          .def("name", &::c10d::ProcessGroup::getBackendName, R"(Get the name of this process group.)")
          .def("_id", &::c10d::ProcessGroup::getID)
          .def(
              "_backend_id",
              &::c10d::ProcessGroup::getBackendID,
              py::arg("backend_type"))
          .def(
              "broadcast",
              &::c10d::ProcessGroup::broadcast,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Broadcasts the tensor to all processes in the process group.

              See :func:`torch.distributed.broadcast for more details.)")
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> tensors = {x};
                return self->broadcast(tensors, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Broadcasts the tensor to all processes in the process group.

              See :func:`torch.distributed.broadcast` for more details.)")
          .def(
              "allreduce",
              &::c10d::ProcessGroup::allreduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& xs,
                 const ::c10d::ReduceOp& op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                return self->allreduce(xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")

          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 const ::c10d::ReduceOp& op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                std::vector<at::Tensor> xs = {x};
                return self->allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")
          .def(
              "allreduce_coalesced",
              &::c10d::ProcessGroup::allreduce_coalesced,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")

          .def(
              "reduce",
              &::c10d::ProcessGroup::reduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.reduce` for more details.)")

          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank,
                 const ::c10d::ReduceOp& op) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return self->reduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.reduce` for more details.)")
          .def(
              "allgather",
              &::c10d::ProcessGroup::allgather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return self->allgather(
                    outputs, inputs, ::c10d::AllgatherOptions());
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather: for more details.)")
          .def(
              "_allgather_base",
              &::c10d::ProcessGroup::_allgather_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather_coalesced",
              &::c10d::ProcessGroup::allgather_coalesced,
              py::arg("output_lists"),
              py::arg("input_list"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "allgather_into_tensor_coalesced",
              &::c10d::ProcessGroup::allgather_into_tensor_coalesced,
              py::arg("outputs"),
              py::arg("inputs"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "gather",
              &::c10d::ProcessGroup::gather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Gathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.gather` for more details.)")

          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return self->gather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Gathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.gather` for more details.)")
          .def(
              "scatter",
              &::c10d::ProcessGroup::scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.scatter` for more details.)")
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> inputs = {input};
                std::vector<at::Tensor> outputs = {output};
                return self->scatter(outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensors"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.scatter` for more details.)")
          .def(
              "reduce_scatter",
              &::c10d::ProcessGroup::reduce_scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces and scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.reduce_scatter` for more details.)")

          ...

```

## 1.3 在ProcessGroup中还要完成一次算子的分发

- [**ProcessGroup collective method**](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroup.hpp)

**注意：ProcessGroup 统一入口中的通信方法，先调用dispatcher中的collective op, 再用collective op 调用call** <br>

```python
virtual c10::intrusive_ptr<Work> allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts = AllreduceOptions()) {

  # 从dispater 中获取collective op 然后再执行
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("c10d::allreduce_", "")
          .typed<
              std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                  at::TensorList,
                  const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                  const c10::intrusive_ptr<::c10d::ReduceOp>&,
                  const std::optional<at::Tensor>& sparse_indices,
                  int64_t)>();

  return std::get<1>(op.call(
      tensors,
      c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
      c10::make_intrusive<ReduceOp>(opts.reduceOp),
      opts.sparseIndices,
      opts.timeout.count()));
}
```

**op.call 会最终调度到Dispatcher.h 里的call 来进行查表操作：** <br>

```c++
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
#ifndef NDEBUG
  DispatchTraceNestingGuard debug_guard;
  if (show_dispatch_trace()) {
    detail::_print_dispatch_trace("[call]", toString(op.operator_name()), dispatchKeySet);
  }
#endif
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  auto step_callbacks = at::getStepCallbacksUnlessEmpty(at::RecordScope::FUNCTION);
  if (C10_UNLIKELY(step_callbacks.has_value() && op.operatorDef_->op.isObserved())) {
    return callWithDispatchKeySlowPath<Return, Args...>(op, *step_callbacks, dispatchKeySet, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING

#ifdef FBCODE_CAFFE2
  if(profilingOperatorEvents()) {
    struct FireOpRAII {
       FireOpRAII(at::RecordFunction::schema_ref_t schema_ref) : schema_ref_(schema_ref) {
           fireOpStartUSDT(schema_ref);
        }
       ~FireOpRAII() { fireOpEndUSDT(schema_ref_); }
       at::RecordFunction::schema_ref_t schema_ref_;
    } event(op.schema());
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  } else {
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
  }
#else
    return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
#endif // FBCODE_CAFFE2
}
```

> **然而: c10d::allreduce_ 在哪里声明、实现和的呢？**

## 1.4 c10d 里通信算子的注册和impl

- [torch/csrc/**distributed**/c10d/Ops.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Ops.cpp)

### 1.4.1 通信算子注册到： 注册到 c10d, 命名空间下
```c++
// 2nd level expansion
// FUNC: op name
// DEV: device
#define REGISTER_C10D_OP1(FUNC, DEV) \
  TORCH_LIBRARY_IMPL(c10d, DEV, m) { \
    m.impl(#FUNC, FUNC##DEV);        \
  }

// 1st level expansion
#define REGISTER_C10D_OP(FUNC)  \
  REGISTER_C10D_OP1(FUNC, CPU)  \
  REGISTER_C10D_OP1(FUNC, CUDA) \
  REGISTER_C10D_OP1(FUNC, PrivateUse1)

// Now we start to register ops with the three device keys

REGISTER_C10D_OP(send)
REGISTER_C10D_OP(recv_)
REGISTER_C10D_OP(recv_any_source_)
REGISTER_C10D_OP(reduce_)
REGISTER_C10D_OP(broadcast_)
REGISTER_C10D_OP(allreduce_)
REGISTER_C10D_OP(allreduce_coalesced_)
REGISTER_C10D_OP(allgather_)
REGISTER_C10D_OP(_allgather_base_)
REGISTER_C10D_OP(allgather_coalesced_)
REGISTER_C10D_OP(allgather_into_tensor_coalesced_)
REGISTER_C10D_OP(reduce_scatter_)
REGISTER_C10D_OP(_reduce_scatter_base_)
REGISTER_C10D_OP(reduce_scatter_tensor_coalesced_)
REGISTER_C10D_OP(gather_)
REGISTER_C10D_OP(scatter_)
REGISTER_C10D_OP(alltoall_)
REGISTER_C10D_OP(alltoall_base_)
REGISTER_C10D_OP(barrier)
```

## 1.4.2 通信算子的具体impl
```c++
namespace c10d {
namespace ops {
// Return input tensors as output tensors to make inplace allreduce look like
// a functional API, so that make_fx can correctly build the dependencies in
// the graph later.
#define IMPL_ALLREDUCE(DEV)                                                   \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>               \
      allreduce_##DEV(                                                        \
          at::TensorList tensors,                                             \
          const c10::intrusive_ptr<ProcessGroup>& process_group,              \
          const c10::intrusive_ptr<ReduceOp>& reduce_op,                      \
          const std::optional<at::Tensor>& sparse_indices,                    \
          int64_t timeout) {                                                  \
    auto tensor_vec = tensors.vec();                                          \
    auto work = process_group->getBackend(c10::DeviceType::DEV) -> allreduce( \
        tensor_vec,                                                           \
        AllreduceOptions{                                                     \
            *reduce_op.get(), std::chrono::milliseconds(timeout)});           \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(     \
        std::move(tensor_vec), work);                                         \
  }

IMPL_ALLREDUCE(CPU)
IMPL_ALLREDUCE(CUDA)
IMPL_ALLREDUCE(PrivateUse1)
} // namespace ops
} // namespace c10d
```

**重点: 可以看出，最终通信算子调度到了具体实现后端(eg. ProcessGroupNCCL)的通信方法里。** <br>

## 1.5 ProcessGoupNCCL 具体实现

- [csrc/distributed/c10d/ProcessGroupNCCL.cpp](torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)

**最终调度到[ncclAllReduce](third_party/nccl/nccl/src/collectives.cc)**

```c++
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    at::Tensor& tensor,
    const char* profilingTitle,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  check_gpu_single_tensor(tensor);

  if (intraNodeComm_ != nullptr && opts.reduceOp == ReduceOp::SUM) {
    using namespace intra_node_comm;
    auto algo = intraNodeComm_->selectAllReduceAlgo(tensor);
    if (algo != intra_node_comm::AllReduceAlgo::NONE) {
      intraNodeComm_->allReduce(tensor, algo);
      return c10::make_intrusive<IntraNodeCommWork>();
    }
  }
  TORCH_CHECK(
      !isFloat8Type(tensor.scalar_type()),
      "Float8 dtypes are not currenlty supported for NCCL reductions");
  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart, // globalRankStart
      globalRankStride, // globalRankStride
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensor, "nccl:all_reduce", opts);
}
```

# 2 如何区分不同类型的通信组

通过DeviceType 或 BackendType 得到不同的通信后端：ProcessGroupNCCL, ProcessGroupGloo, ProcessGroupMPI, ProcessGroupUCC, ProcessGroupXCL, ProcessGroupCustom等。

- [BackendType](/torch/csrc/distributed/c10d/ProcessGroup.hpp)

```c++
enum BackendType : uint8_t {
  UNDEFINED = 0,
  GLOO = 1,
  NCCL = 2,
  UCC = 3,
  MPI = 4,
  XCCL = 5,
  CUSTOM = 6,
};
```

- [DeviceType](c10/core/DeviceType.h)

```c++
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1, // CUDA.
  MKLDNN = 2, // Reserved for explicit MKLDNN
  OPENGL = 3, // OpenGL
  OPENCL = 4, // OpenCL
  IDEEP = 5, // IDEEP.
  HIP = 6, // AMD HIP
  FPGA = 7, // FPGA
  MAIA = 8, // ONNX Runtime / Microsoft
  XLA = 9, // XLA / TPU
  Vulkan = 10, // Vulkan
  Metal = 11, // Metal
  XPU = 12, // XPU
  MPS = 13, // MPS
  Meta = 14, // Meta (tensors with no data)
  HPU = 15, // HPU / HABANA
  VE = 16, // SX-Aurora / NEC
  Lazy = 17, // Lazy Tensors
  IPU = 18, // Graphcore IPU
  MTIA = 19, // Meta training and inference devices
  PrivateUse1 = 20, // PrivateUse1 device
  // NB: If you add more devices:
  //  - Change the implementations of DeviceTypeName and isValidDeviceType
  //    in DeviceType.cpp
  //  - Change the number below
  COMPILE_TIME_MAX_DEVICE_TYPES = 21,
};

```

- 根据DeviceType 或 BackendType 获取不同的后端

- [ProcesssGroup.hpp](torch/csrc/distributed/c10d/ProcessGroup.hpp)
```c++
  c10::intrusive_ptr<Backend> getBackend(BackendType backendType) const {
    TORCH_CHECK(
        backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end(),
        "Could not find backend type ",
        backendType,
        ".");
    return backendTypeToBackend_.at(backendType);
  }
```

- [ProcessGroup.cpp](torch/csrc/distributed/c10d/ProcessGroup.cpp)

```c++
c10::intrusive_ptr<Backend> ProcessGroup::getBackend(
    c10::DeviceType deviceType) {
  // If there is a backend associated with this device type then return it
  if (deviceTypeToBackend_.find(deviceType) != deviceTypeToBackend_.end()) {
    return deviceTypeToBackend_.at(deviceType);
  }

  // Get the backend type associated with the device
  ProcessGroup::BackendType backendType{ProcessGroup::BackendType::UNDEFINED};
  try {
    backendType = deviceTypeToBackendType_.at(deviceType);
  } catch (const std::out_of_range& e) {
    TORCH_CHECK(
        false, "No backend type associated with device type ", deviceType);
  }

  // Check if the backend has already been initialized
  if (backendTypeToBackend_.find(backendType) != backendTypeToBackend_.end()) {
    auto backend = backendTypeToBackend_.at(backendType);
    deviceTypeToBackend_[deviceType] = backend;
    return backend;
  }

  TORCH_CHECK(
      false,
      "Could not retrieve or create the backend ",
      backendType,
      " for device type ",
      deviceType);
}
```

一个processgroup的实例的 backend 在python 测被注册：

```python
def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    backend_options=None,
    timeout=None,
    pg_tag=None,
    device_id=None,
    group_desc=None,
):
  prefix_store = PrefixStore(f"{group_name}/", store)
    # The backend for PG will be set later based on what's inside BackendConfig
    # and timeout are set in each backend's option.
    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
  pg._register_backend(torch.device(device), backend_type, backend_class)
  ...
```

# 3 注册backend + 注册通信后端(_register_backend) + 注册通信组(_register_process_group)

## 3.1 distributed_c10d.py : backend

**用于根据 string 快速自动化创建通信组.**

```python
class Backend(str):
    """
    An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """

    UNDEFINED = "undefined"
    GLOO = "gloo"
    NCCL = "nccl"
    UCC = "ucc"
    MPI = "mpi"
    XCCL = "xccl"

    _BackendPlugin = namedtuple("_BackendPlugin", ["creator_fn", "extended_api"])

    _plugins: Dict[str, _BackendPlugin] = {}

    backend_list = [UNDEFINED, GLOO, NCCL, XCCL, UCC, MPI]

    # 3rd-party devices can register the default backend support here
    default_device_backend_map: Dict[str, str] = {
        "cpu": GLOO,
        "cuda": NCCL,
        "xpu": XCCL,
    }

    backend_capability: Dict[str, List[str]] = {
        GLOO: ["cpu", "cuda"],
        NCCL: ["cuda"],
        XCCL: ["xpu"],
        UCC: ["cpu", "cuda"],
        MPI: ["cpu", "cuda"],
    }

    backend_type_map: Dict[str, ProcessGroup.BackendType] = {
        UNDEFINED: ProcessGroup.BackendType.UNDEFINED,
        GLOO: ProcessGroup.BackendType.GLOO,
        NCCL: ProcessGroup.BackendType.NCCL,
        XCCL: ProcessGroup.BackendType.XCCL,
        UCC: ProcessGroup.BackendType.UCC,
        MPI: ProcessGroup.BackendType.MPI,
    }

    def __new__(cls, name: str):
        """Create and return a new instance of the class."""
        if not isinstance(name, str):
            raise ValueError("Backend constructor parameter must be string-ish")
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)

        if value == Backend.UNDEFINED:
            value = name.lower()
        return value

    @classmethod
    def register_backend(
        cls,
        name,
        func,
        extended_api=False,
        devices: Optional[Union[str, List[str]]] = None,
    ) -> None:
    ...
```

## 3.2 注册通信组backend

- python 侧

```python
pg._register_backend(torch.device(device), backend_type, backend_class)
```

- python-cpp binding

```c++
  auto processGroup =
      py::class_<
          ::c10d::ProcessGroup,
          c10::intrusive_ptr<::c10d::ProcessGroup>,
          ::c10d::PyProcessGroup>(module, "ProcessGroup",
          R"(A ProcessGroup is a communication primitive that allows for
          collective operations across a group of processes.

          This is a base class that provides the interface for all
          ProcessGroups. It is not meant to be used directly, but rather
          extended by subclasses.)")
      .def(
          "_register_backend",
          [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
              const c10::Device& device,
              const ::c10d::ProcessGroup::BackendType& backendType,
              const std::optional<c10::intrusive_ptr<::c10d::Backend>>&
                  backend) {
            self->setBackend(device.type(), backendType, backend);
          },
          py::arg("device"),
          py::arg("backend_type"),
          py::arg("backend") =
              std::optional<c10::intrusive_ptr<::c10d::Backend>>(),
          py::call_guard<py::gil_scoped_release>())
```

- setBackend

```c++
  void setBackend(
      c10::DeviceType deviceType,
      BackendType backendType,
      const std::optional<c10::intrusive_ptr<Backend>>& backend) {
    // TODO: should we add these entries after the backend setting succeeds?
    deviceTypeToBackendType_[deviceType] = backendType;
    deviceTypes_.insert(deviceType);
    // if the backendType is already set then reuse it for this device
    if (backendTypeToBackend_.find(backendType) !=
        backendTypeToBackend_.end()) {
      auto existingBackend = backendTypeToBackend_.at(backendType);
      deviceTypeToBackend_[deviceType] = existingBackend;
      TORCH_CHECK(
          existingBackend->getBoundDeviceId() ==
          (*backend)->getBoundDeviceId());
    } else {
      // check if backend has value
      if (backend.has_value()) {
        deviceTypeToBackend_[deviceType] = backend.value();
        backendTypeToBackend_[backendType] = backend.value();
        (*backend)->setBoundDeviceId(bound_device_id_);
      }
    }
  }
```

## 3.3 process_group 注册
**可根据名称快速找到对应ProcessGoroup.**

- python 侧

```python
_register_process_group(group_name, pg)
```

- c++ 侧
```c++
namespace c10d {

static bool thread_isolation_mode = false;
static GroupRegistry process_registry;

void set_thread_isolation_mode(bool enable) {
  thread_isolation_mode = enable;
}

bool get_thread_isolation_mode() {
  return thread_isolation_mode;
}

void register_process_group(
    const std::string& group_name,
    const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
  if (thread_isolation_mode) {
    RankLocal<::GroupRegistry>::get().register_group(group_name, group);
  } else {
    process_registry.register_group(group_name, group);
  }
}

c10::intrusive_ptr<c10d::ProcessGroup> resolve_process_group(
    const std::string& group_name) {
  if (thread_isolation_mode) {
    return RankLocal<::GroupRegistry>::get().resolve_group(group_name);
  } else {
    return process_registry.resolve_group(group_name);
  }
}
} // namespace c10d
```

- 使用

```c++
def _get_group_size_by_name(group_name: str) -> int:
    group = _resolve_process_group(group_name)
    return group.size()
```



# 4 直接访问ProcessGroupNCCL

ProcessGroupNCCL 从**backend**实例获取，并赋予一些ProcessGroupNCCL的属性.

- backend 的实例化及相关属性bind

```c++
  auto backend =
      py::class_<::c10d::Backend, c10::intrusive_ptr<::c10d::Backend>>(
          module, "Backend")
          .def("rank", &::c10d::Backend::getRank)
          .def("size", &::c10d::Backend::getSize)
          .def("name", &::c10d::Backend::getBackendName)
          .def_property_readonly(
              "supports_splitting",
              &::c10d::Backend::supportsSplitting,
              "(test whether the backend supports splitting)")
          .def(
              "broadcast",
              &::c10d::Backend::broadcast,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 int rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return self->broadcast(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              &::c10d::Backend::allreduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& xs,
                 const ::c10d::ReduceOp& op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                return self->allreduce(xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 const ::c10d::ReduceOp& op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                std::vector<at::Tensor> xs = {x};
                return self->allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce_coalesced",
              &::c10d::Backend::allreduce_coalesced,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce",
              &::c10d::Backend::reduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 int rootRank,
                 const ::c10d::ReduceOp& op) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return self->reduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather",
              &::c10d::Backend::allgather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_allgather_base",
              &::c10d::Backend::_allgather_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return self->allgather(
                    outputs, inputs, ::c10d::AllgatherOptions());
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather_coalesced",
              &::c10d::Backend::allgather_coalesced,
              py::arg("output_lists"),
              py::arg("input_list"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "gather",
              &::c10d::Backend::gather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return self->gather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "scatter",
              &::c10d::Backend::scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> inputs = {input};
                std::vector<at::Tensor> outputs = {output};
                return self->scatter(outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensors"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce_scatter",
              &::c10d::Backend::reduce_scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 const ::c10d::ReduceOp& op) {
                std::vector<at::Tensor> outputs = {output};
                std::vector<std::vector<at::Tensor>> inputs = {input};
                ::c10d::ReduceScatterOptions opts;
                opts.reduceOp = op;
                return self->reduce_scatter(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_reduce_scatter_base",
              &::c10d::Backend::_reduce_scatter_base,
              py::arg("outputTensor"),
              py::arg("inputTensor"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall_base",
              &::c10d::Backend::alltoall_base,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall_base",
              [](::c10d::Backend& self,
                 at::Tensor& output,
                 at::Tensor& input,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                return self.alltoall_base(
                    output,
                    input,
                    outputSplitSizes,
                    inputSplitSizes,
                    ::c10d::AllToAllOptions());
              },
              py::arg("output"),
              py::arg("input"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall",
              &::c10d::Backend::alltoall,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "send",
              &::c10d::Backend::send,
              py::arg("tensors"),
              py::arg("dstRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "recv",
              &::c10d::Backend::recv,
              py::arg("tensors"),
              py::arg("srcRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "recv_anysource",
              &::c10d::Backend::recvAnysource,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "barrier",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 const ::c10d::BarrierOptions& opts) {
                return self->barrier(opts);
              },
              py::arg("opts") = ::c10d::BarrierOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_set_sequence_number_for_group",
              &::c10d::Backend::setSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_sequence_number_for_group",
              &::c10d::Backend::getSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "monitored_barrier",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 const std::chrono::milliseconds& timeout,
                 bool waitAllRanks) {
                ::c10d::BarrierOptions opts;
                opts.timeout = timeout;
                return self->monitoredBarrier(opts, waitAllRanks);
              },
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::arg("wait_all_ranks") = false,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "eager_connect_single_device",
              &::c10d::Backend::eagerConnectSingleDevice,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_backend_name",
              &::c10d::Backend::getBackendName,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_start_coalescing",
              &::c10d::Backend::startCoalescing,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_end_coalescing",
              &::c10d::Backend::endCoalescing,
              py::call_guard<py::gil_scoped_release>());
```

- ProcessGroupNCCL的实例化及相关属性bind

```c++
#ifdef USE_C10D_NCCL
  auto processGroupNCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupNCCL>(
          module, "ProcessGroupNCCL", backend)
          .def(
              py::init<
                  const c10::intrusive_ptr<::c10d::Store>&,
                  int,
                  int,
                  c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>>(),
              py::call_guard<py::gil_scoped_release>(),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("options"),
              R"(Create a new ProcessGroupNCCL instance.)")
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                auto options = ::c10d::ProcessGroupNCCL::Options::create();
                options->is_high_priority_stream = false;
                options->timeout = timeout;
                return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = ::c10d::kProcessGroupNCCLDefaultTimeout,
              py::call_guard<py::gil_scoped_release>(),
              R"(Create a new ProcessGroupNCCL instance.)")
          .def(
              "_shutdown",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                return self->shutdown();
              },
              py::call_guard<py::gil_scoped_release>())
          .def("_group_start", &::c10d::ProcessGroupNCCL::groupStart)
          .def("_group_end", &::c10d::ProcessGroupNCCL::groupEnd)
          .def(
              "comm_split_count",
              &::c10d::ProcessGroupNCCL::getCommSplitCounter)
          .def(
              "_set_default_timeout",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 std::chrono::milliseconds timeout) {
                self->getOptions()->timeout = timeout;
              },
              py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_add_ephemeral_timeout",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 const std::chrono::milliseconds& timeout) {
                self->addEphemeralTimeout(timeout);
              },
              py::arg("timeout"))
          .def(
              "_verify_work_timeout",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 const c10::intrusive_ptr<::c10d::Work>& work,
                 const std::chrono::milliseconds& timeout) {
                return self->verifyWorkTimeoutForTest(work, timeout);
              },
              py::arg("work"),
              py::arg("timeout"))
          .def_property_readonly(
              "options",
              &::c10d::ProcessGroupNCCL::getOptions,
              R"(Return the options used to create this ProcessGroupNCCL instance.)")
          .def_property_readonly(
              "uid", &::c10d::ProcessGroupNCCL::getUid, R"(Return the uid.)")
          .def_property(
              "bound_device_id",
              &::c10d::ProcessGroupNCCL::getBoundDeviceId,
              &::c10d::ProcessGroupNCCL::setBoundDeviceId,
              R"(Return the bound device id.)")
          .def(
              "perform_nocolor_split",
              &::c10d::ProcessGroupNCCL::performNocolorSplit)
          .def("register_mem_pool", &::c10d::ProcessGroupNCCL::registerMemPool)
          .def(
              "deregister_mem_pool",
              &::c10d::ProcessGroupNCCL::deregisterMemPool)
          .def(
              "abort",
              &::c10d::ProcessGroupNCCL::abort,
              py::call_guard<py::gil_scoped_release>(),
              R"(Abort the process group.)")
          .def(
              "_is_initialized",
              &::c10d::ProcessGroupNCCL::isInitialized,
              py::call_guard<py::gil_scoped_release>());
```

- 直接调度 ProcessGroupNCCL allreduce 的调用栈
```c++
```shell
#0  c10d::ProcessGroupNCCL::allreduce (this=0x5ea0f30, tensors=std::vector of length 1, capacity 1 = {...}, opts=...)
    at /root/projects/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3835
#1  0x00007ffff5f2c66b in <lambda(const c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> >&, std::vector<at::Tensor, std::allocator<at::Tensor> >&, const c10d::ReduceOp&)>::operator()(const c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> > &, std::vector<at::Tensor, std::allocator<at::Tensor> > &, const c10d::ReduceOp &) const (__closure=0x4a0b058, self=...,
    xs=std::vector of length 1, capacity 1 = {...}, op=...) at /root/projects/pytorch/torch/csrc/distributed/c10d/init.cpp:2488
#2  0x00007ffff5f76ab3 in pybind11::detail::argument_loader<c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> >&, c10d::ReduceOp const&>::call_impl<c10::intrusive_ptr<c10d::Work>, torch::distributed::c10d::(anonymous namespace)::c10d_init(PyObject*, PyObject*)::<lambda(const c10::intrusive_ptr<c10d::Backend>&, std::vector<at::Tensor>&, const c10d::ReduceOp&)>&, 0, 1, 2, pybind11::gil_scoped_release>(<
lambda(const c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> >&, std::vector<at::Tensor, std::allocator<at::Tensor> >&, const c10d::ReduceOp&)> &, std::index_sequence, pybind11::gil_scoped_release &&) (this=0x7fffffffd860, f=...)
    at /root/projects/pytorch/third_party/pybind11/include/pybind11/cast.h:1631
#3  0x00007ffff5f6f3de in pybind11::detail::argument_loader<c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> > const&, std::vector<at::Tensor, std::allocator<at::Tensor> >&, c10d::ReduceOp const&>::call<c10::intrusive_ptr<c10d::Work>, pybind11::gil_scoped_release, torch::distributed::c10d::(anonymous namespace)::c10d_init(PyObject*, PyObject*)::<lambda(const c10::intrusive_ptr<c10d::Backend>&, std::vector<at::Tensor>&, const c10d::ReduceOp&)>&>(<lambda(const c
10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> >&, std::vector<at::Tensor, std::allocator<at::Tensor> >&, const c10d::ReduceOp&)> &) (this=0x7fffffffd860, f=...) at /root/projects/pytorch/third_party/pybind11/include/pybind11/cast.h:1600
#4  0x00007ffff5f61b3e in <lambda(pybind11::detail::function_call&)>::operator()(pybind11::detail::function_call &) const (this=0x0, call=...)
    at /root/projects/pytorch/third_party/pybind11/include/pybind11/pybind11.h:278
#5  0x00007ffff5f61c1d in <lambda(pybind11::detail::function_call&)>::_FUN(pybind11::detail::function_call &) ()
    at /root/projects/pytorch/third_party/pybind11/include/pybind11/pybind11.h:249
#6  0x00007ffff4cdb3f3 in pybind11::cpp_function::dispatcher (self=0x7ffed0739170, args_in=0x7ffff79ff200, kwargs_in=0x7ffecea81380)
    at /root/projects/pytorch/third_party/pybind11/include/pybind11/pybind11.h:1002
#7  0x000000000054c584 in cfunction_call (func=0x7ffed0740270, args=0x7ffff79ff200, kwargs=0x7ffecea81380)
    at /usr/local/src/conda/python-3.12.8/Objects/methodobject.c:537
#8  0x000000000051da5b in _PyObject_MakeTpCall (tstate=0x9c0e50 <_PyRuntime+458992>, callable=0x7ffed0740270, args=<optimized out>, nargs=<optimized out>,
    keywords=0x7ffff79d39d0) at /usr/local/src/conda/python-3.12.8/Objects/call.c:240
#9  0x0000000000528303 in _PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ffff7ad0080, throwflag=<optimized out>) at Python/bytecodes.c:2715
#10 0x00000000005e469e in PyEval_EvalCode (co=<optimized out>, globals=0x7ffff7c2d380, locals=<optimized out>)
    at /usr/local/src/conda/python-3.12.8/Python/ceval.c:578
#11 0x000000000060aae7 in run_eval_code_obj (tstate=0x9c0e50 <_PyRuntime+458992>, co=0x7ffff79e0800, globals=0x7ffff7c2d380, locals=0x7ffff7c2d380)
    at /usr/local/src/conda/python-3.12.8/Python/pythonrun.c:1722
#12 0x0000000000605cc7 in run_mod (mod=<optimized out>, filename=0x7ffff7b67cf0, globals=0x7ffff7c2d380, locals=0x7ffff7c2d380, flags=0x7fffffffe120,
    arena=0x7ffff7b53cb0) at /usr/local/src/conda/python-3.12.8/Python/pythonrun.c:1743
--Type <RET> for more, q to quit, c to continue without paging--
#13 0x000000000061e022 in pyrun_file (fp=fp@entry=0x9c3490, filename=filename@entry=0x7ffff7b67cf0, start=start@entry=257, globals=globals@entry=0x7ffff7c2d380,
    locals=locals@entry=0x7ffff7c2d380, closeit=closeit@entry=1, flags=0x7fffffffe120) at /usr/local/src/conda/python-3.12.8/Python/pythonrun.c:1643
#14 0x000000000061d960 in _PyRun_SimpleFileObject (fp=0x9c3490, filename=0x7ffff7b67cf0, closeit=1, flags=0x7fffffffe120)
    at /usr/local/src/conda/python-3.12.8/Python/pythonrun.c:433
#15 0x000000000061d753 in _PyRun_AnyFileObject (fp=0x9c3490, filename=0x7ffff7b67cf0, closeit=1, flags=0x7fffffffe120)
    at /usr/local/src/conda/python-3.12.8/Python/pythonrun.c:78
#16 0x00000000006167e3 in pymain_run_file_obj (skip_source_first_line=0, filename=0x7ffff7b67cf0, program_name=0x7ffff7b67db0)
    at /usr/local/src/conda/python-3.12.8/Modules/main.c:360
#17 pymain_run_file (config=0x963a30 <_PyRuntime+77008>) at /usr/local/src/conda/python-3.12.8/Modules/main.c:379
#18 pymain_run_python (exitcode=0x7fffffffe0f4) at /usr/local/src/conda/python-3.12.8/Modules/main.c:633
#19 Py_RunMain () at /usr/local/src/conda/python-3.12.8/Modules/main.c:713
#20 0x00000000005cfa89 in Py_BytesMain (argc=<optimized out>, argv=<optimized out>) at /usr/local/src/conda/python-3.12.8/Modules/main.c:767
#21 0x00007ffff7cb9d90 in __libc_start_call_main (main=main@entry=0x5cf9c0 <main>, argc=argc@entry=2, argv=argv@entry=0x7fffffffe378)
    at ../sysdeps/nptl/libc_start_call_main.h:58
#22 0x00007ffff7cb9e40 in __libc_start_main_impl (main=0x5cf9c0 <main>, argc=2, argv=0x7fffffffe378, init=<optimized out>, fini=<optimized out>,
    rtld_fini=<optimized out>, stack_end=0x7fffffffe368) at ../csu/libc-start.c:392
#23 0x00000000005cf8b9 in _start ()
```

# 5 _c10d_functional 实验性接口调度机制
- [python 侧 _functional_collectioves.py接口](https://github.com/pytorch/torch/distributed/_functional_collectives.py)

**该接口调度的是这里注册的算子：** <br>
- [Functional.cpp](https://github.com/pytorch/torch/csrc/distributed/c10d/Functional.cpp)

```c++
TORCH_LIBRARY(_c10d_functional, m) {
  m.def(
      "all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce),
      {at::Tag::pt2_compliant_tag});

  m.def(
      "all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)",
      torch::dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, ::all_reduce_),
      {at::Tag::pt2_compliant_tag});

  ...
```

**这里的算子其实还是会通过ProcessGroup来调度具体的通信算子** <br>
```c++
std::vector<at::Tensor> all_gather_into_tensor_coalesced(
    std::vector<at::Tensor> inputs,
    int64_t group_size,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::string group_name) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& tensor : inputs) {
    outputs.push_back(allocate_all_gather_output(tensor, group_size));
  }

  auto group = c10d::resolve_process_group(group_name);
  auto work = group->allgather_into_tensor_coalesced(outputs, inputs);
  for (const auto& tensor : outputs) {
    c10d::register_work(tensor, work);
  }
  return outputs;
}
```

# 6 collective op kernel 具体实现
## 6.1 cuda kernel
- [https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/intra_node_comm.cu]
```
# /root/mtn/pytorch/torch/csrc/distributed/c10d/intra_node_comm.cu
at::Tensor IntraNodeComm::allReduce(
    const at::Tensor& input,
    AllReduceAlgo algo) {
  // Report usage for testing purposes.
  // We don't care about overflowing.
  ++usageCounter;
  auto stream = at::cuda::getCurrentCUDAStream();
  c10::cuda::CUDACachingAllocator::recordStream(
      input.storage().data_ptr(), stream);
  switch (algo) {
    case AllReduceAlgo::ONE_SHOT:
      return oneShotAllReduce(input, stream);
    case AllReduceAlgo::TWO_SHOT:
      return twoShotAllReduce(input, stream);
    case AllReduceAlgo::HCM:
      return hybridCubeMeshAllReduce(input, stream);
    default:
      C10_THROW_ERROR(ValueError, "IntraNodeComm: invalid algo");
  }
}
```

## 6.2 调用NCCL 通信库
- [ProcessGroupNCCL.cpp](https://github1s.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)

```python
# torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp =
            getNcclReduceOp(opts.reduceOp, input, ncclDataType, comm);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}
```

# 7 其它相关组件

## 7.1 ProcessGroup-Store-NCCLComm
- 进程组实例会**保留对存储Store的引用**，因为在构造函数运行后很长时间内可能还会使用存储。<br>
- 构造函数不会创建任何 MCCL 通信器（MCCL communicators）。**单个 MCCL 通信器只能用于特定的设备集合，因此它们是在执行集体操作（collective operation）时按需创建的**。<br>
- 如果之后执行另一个集体操作，针对不同的设备集合，进程组会创建另一个 MCCL 通信器。这些 MCCL 通信器会**被缓存并在可能的情况下重用**。<br>

## 7.2 创建多个进程组
- 如果需要创建多个进程组，每个进程组可能有不同的 rank（进程编号）和 size（进程数量），可以通过为每个进程组传递一个新的存储实例来实现。<br>
- 如果只有一个存储对象，可以使用 c10d::PrefixStore 来派生作用域实例。这也是 torch.distributed Python API 的做法。<br>

## 7.3 Work 作用
- [Work 类](torch/csrc/distributed/c10d/Work.hpp)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 PyTorch 的 csrc/distributed/c10d/Work.hpp 文件中，Work 类是与分布式训练中的异步操作和任务管理相关的核心组件。Work 类的主要作用是封装和管理分布式训练中的异步工作单元，这些工作单元可能涉及跨多个计算节点的数据传输、梯度同步或其他通信操作。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是 Work 类在 PyTorch 分布式训练中的一些关键作用：<br>

- 异步操作封装：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work 类用于封装分布式训练中的异步操作。这些操作可能涉及网络通信、数据同步或模型参数的更新等。通过 Work 类，开发者可以提交异步任务并在任务完成时获取结果，而无需阻塞主线程。<br>

- 任务管理与调度：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work 类提供了任务管理和调度的功能。它允许开发者跟踪异步任务的执行状态，包括任务是否已启动、是否已完成以及是否遇到错误。这有助于开发者在分布式训练中更有效地管理资源和任务，确保训练过程的顺利进行。<br>

- 错误处理与恢复：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ork 类还提供了错误处理和恢复机制。当异步任务遇到错误时，Work 类可以捕获这些错误并向开发者报告。开发者可以根据错误信息采取相应的恢复措施，以确保训练的连续性和稳定性。<br>

- 性能优化：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过 Work 类，开发者可以优化分布式训练的性能。例如，他们可以利用 Work 类提供的接口来重叠计算和通信操作，从而减少训练过程中的等待时间。此外，Work 类还可以与 PyTorch 中的其他性能优化技术（如梯度压缩、混合精度训练等）结合使用，以进一步提高训练效率。<br>

- 跨节点通信：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work 类在跨节点通信中发挥着关键作用。它允许不同计算节点上的进程相互通信和同步数据，这是分布式训练中的核心功能之一。通过 Work 类，开发者可以实现高效的跨节点通信，从而加速训练过程并提高模型的收敛速度。需要注意的是，Work 类的具体实现和使用方式可能因 PyTorch 的版本和分布式训练的后端（如 NCCL、Gloo 或 MPI）而有所不同。因此，开发者在使用 Work 类时需要参考 PyTorch 的官方文档和分布式训练的相关指南，以确保正确理解和使用这一功能。<br>

总的来说，Work 类在 PyTorch 分布式训练中扮演着重要角色，它封装和管理异步操作，提供任务管理和调度功能，支持错误处理和恢复机制，并有助于优化训练性能和实现跨节点通信。<br>

# 7.4 Future
- [Future 类](pytorch/aten/src/ATen/core/ivalue_inl.h)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 /home/mtn_torch/pytorch/aten/src/ATen/core/ivalue_inl.h 文件中，Future 是一个重要的类，它主要用于表示**异步计算**的结果。这个 Future 类是 IValue 类的一个扩展或特化，用于**封装异步操作完成后的返回值**。以下是 Future 在这个上下文中的主要作用：

- 封装异步结果：<br>
Future 对象用于存储异步操作完成后的结果。这允许程序在异步操作进行时继续执行其他任务，而无需等待结果。一旦异步操作完成，Future 对象将持有该操作的结果，并允许调用者通过适当的接口检索这个结果。<br>

- 提供非阻塞接口：<br>
通过 Future，调用者可以查询异步操作的状态（例如，是否已完成、是否出错）以及获取操作的结果。调用者可以选择阻塞等待结果（如果结果尚未可用），或者继续执行其他任务并在稍后检查 Future 的状态。<br>

- 支持链式操作和回调：<br>
Future 通常支持链式操作，允许调用者将多个异步操作链接在一起，形成一个执行链。此外，Future 还可能支持回调机制，允许调用者注册在异步操作完成时执行的回调函数。<br>

- 错误处理：<br>
Future 提供了错误处理机制，允许调用者在尝试获取结果时捕获和处理可能发生的异常。这使得异步编程更加健壮，因为调用者可以优雅地处理错误情况，而不是让程序崩溃。<br>

- 跨线程和跨进程通信：<br>
在多线程或分布式环境中，Future 可以作为线程间或进程间通信的一种机制。一个线程或进程可以执行异步操作并返回一个 Future 对象给另一个线程或进程，后者可以在适当的时候查询或等待这个 Future 对象的结果。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 PyTorch 的实际应用中，Future 通常与异步执行的任务一起使用，如网络请求、数据库查询、大规模数据处理或复杂的计算任务（如神经网络的前向传播和反向传播）。通过 Future，开发者可以构建出具有高性能和良好用户体验的应用程序，因为程序可以在等待异步操作完成时继续执行其他任务。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的是，Future 的具体实现和使用方式可能因 PyTorch 的版本和构建配置而有所不同。因此，开发者在使用 Future 时需要参考 PyTorch 的官方文档和源代码，以确保正确理解和使用这个类。<br>

## 7.5 Store
### 7.5.1 TCPStore
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在PyTorch分布式系统中，TCPStore主要用于**进程间的通信和初始化分布式进程组**。TCPStore作为分布式键值存储(KVStore)，允许进程之间共享信息，这在分布式训练中非常重要。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体来说，在PyTorch分布式训练中，TCPStore可能用于设置和存储以下类型的数据：<br>
- 进程初始化信息：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在分布式训练中，每个进程需要知道如何联系其他进程。TCPStore可以存储这些信息，如主进程的IP地址（MASTER_ADDR）和端口号（MASTER_PORT），以及参与训练的进程总数（world_size）和当前进程的等级（rank）。这些信息是初始化分布式进程组所必需的。<br>

- 模型参数和梯度：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然TCPStore不是直接用于存储模型参数和梯度的（这些通常存储在GPU或其他设备的内存中），但在某些情况下，TCPStore可能用于在进程之间传递参数或梯度的更新信息。然而，在PyTorch的分布式数据并行（DDP）中，梯度的同步通常是通过更高效的通信机制（如AllReduce）来实现的。<br>

- 训练状态信息：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TCPStore还可以用于存储训练过程中的状态信息，如当前迭代次数、学习率等。这些信息可以在进程之间共享，以确保所有进程都使用相同的训练设置。<br>

- 同步信号：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在分布式训练中，进程之间需要同步以确保它们按照相同的步骤进行训练。TCPStore可以用于存储同步信号，以指示所有进程都已准备好进入下一个训练阶段。<br>

- 其他辅助信息：<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;根据具体的应用场景，TCPStore还可以存储其他类型的辅助信息，如日志信息、调试信息等。这些信息有助于监控和调试分布式训练过程。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的是，TCPStore只是PyTorch分布式系统中的一个组件，它与其他组件（如后端通信机制、进程组等）一起工作，以实现高效的分布式训练。在使用TCPStore时，需要确保所有进程都能够访问它，并且需要仔细处理竞争条件和同步问题。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此外，PyTorch还提供了其他初始化方法（如环境变量初始化、共享文件系统初始化等），这些方法可以根据具体的应用场景和需求来选择。在选择初始化方法时，需要考虑系统的可用性、可靠性和性能等因素。<br>

### 7.5.2 FileStore
- 作用：FileStore是一个基于文件系统的分布式存储实现，它使用文件来存储键值对。它允许在多个进程之间共享数据，而无需通过网络进行传输。<br>
- 特点：FileStore的优点是简单且易于实现，因为它依赖于现有的文件系统。然而，它的性能可能受到文件系统I/O性能的限制。它适用于对性能要求不高的分布式应用场景。<br>
- 用法：在使用FileStore时，需要指定一个文件路径来存储键值对。多个进程可以访问同一个文件路径来共享数据。<br>

# 8 相关类的实现总结
## 8.1 Store UML

![Store UML](images/store.jpg)

## 8.2 c++ process group

![process group](images/process-group.jpg)

## 8.3 create process group

![creat process group](images/process-group-create.jpg)
