# 1. Stream 的用法

```python
import torch

# 创建两个Stream
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# 在Stream s1中执行操作
with torch.cuda.stream(s1):
    # 异步操作，如数据拷贝或计算
    y = torch.randn(1000, 1000, device='cuda')
    y = y * 2  # 此操作在s1中执行

# 在Stream s2中执行另一操作
with torch.cuda.stream(s2):
    z = torch.randn(1000, 1000, device='cuda')
    z = z + 1  # 此操作在s2中执行

# 等待所有Stream完成
torch.cuda.synchronize()
```

# 2. Cuda Event 的用法

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Event 用于记录GPU操作的**时间点**或**同步Stream**。

**用途:**
- 测量操作耗时（如核函数执行时间）。
- 跨Stream同步（例如，Stream2等待Stream1的某个操作完成）。

**注意：**
- 延迟初始化：在事件被记录或导出到另一个进程时才会被延迟初始化；
- 相同device上的stream才能 record 当前设备的event;
- 其它设备上的streams 可以等待当前设备上的event;
- 跨进程通信：Inter-Process Communication Handle;
- event.synchronize()：等待特定流上的事件完成;
- torch.cuda.synchronize()：等待所有Stream完成；
- event.wait(stream2) 是让 stream2 等待 stream1 中的 event 而非相反;
- event.record(stream1) 在 stream1 中标记一个时间点, 只是一个标记，不会阻塞流的执行;
- event.wait(stream2)：让 stream2 等待 stream1 中标记的时间点(**record 前提交的操作**);
- event.wait 可以精确控制同步位置；
- stream.wait_stream(other_stream)：同步到另一个流的 全部已提交操作;
- event.wait(stream) 仅在流内部生效，不阻塞cpu线程；
- event.synchronize()： 阻塞当前cpu线程，直到事件event完成；
- torch.cuda.synchronize() 或 stream.synchronize() 和 event.synchronize() 会阻塞当前CPU.


## 2.1 跨进程通信：IPC
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IPC Handle（Inter-Process Communication Handle） 是一种用于**跨进程通信的机制**。在 CUDA 的上下文中，IPC Handle 提供了一种方式，使得`一个进程创建的 GPU 资源（如 Event 或 Stream）可以被其他进程共享和使用`。通过 IPC Handle，不同进程可以协调 GPU 上的操作，实现高效的同步与资源共享.<br>

具体来说，IPC Handle 是一个轻量级的引用，指向底层的 CUDA 资源（如 cudaEvent_t 或 cudaStream_t）.

IPC Handle `只能在相同设备上使用`。例如，如果 Event 是在 cuda:0 上创建的，那么其他进程也必须在 cuda:0 上重建该 Event.

使用 IPC Handle 的进程不会自动管理底层资源的生命周期。用户需要确保**原始进程中的资源**在其他进程使用期间保持有效(其它进程不会管理原始进程中的资源)。

```python
# 进程 A
event = torch.cuda.Event(interprocess=True)  # 必须设置 interprocess=True
event.record()  # 记录事件
ipc_handle = event.ipc_handle()  # 导出 IPC Handle

# 进程 B
device = torch.device('cuda:0')  # 确保与原始进程相同的设备
new_event = torch.cuda.Event.from_ipc_handle(device, ipc_handle)


# 进程 B 可以等待进程A 中的Event 完成:
new_event.wait(torch.cuda.current_stream())
```


## 2.1 Event python api
```python
class Event(torch._C._CudaEventBase):
    r"""Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)

    .. _CUDA Event Documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super().__new__(
            cls,
            enable_timing=enable_timing, # 是否记录时间
            blocking=blocking,           # 是否阻塞
            interprocess=interprocess,   # 在不同进程间共享event
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super().from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Record the event in a given stream.

        Uses ``torch.cuda.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        super().record(stream)

    def wait(self, stream=None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Use ``torch.cuda.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
            `CUDA Event documentation`_ for more info.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        super().wait(stream)

    def query(self):
        r"""Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super().query()

    def elapsed_time(self, end_event):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
        return super().elapsed_time(end_event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cudaEventSynchronize()``: see
            `CUDA Event documentation`_ for more info.
        """
        super().synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
        return super().ipc_handle()

    @property
    def _as_parameter_(self):
        ...

    def __repr__(self) -> str:
        ...

```

## 2.1 测量耗时
```python
# 创建Event
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 记录事件到当前Stream
start_event.record()

# 执行操作
x = torch.randn(1000, 1000, device='cuda')
y = x * x

# 记录结束事件
end_event.record()

# 等待事件完成
end_event.synchronize()

# 计算耗时（毫秒）
elapsed_time = start_event.elapsed_time(end_event)
print(f"Time: {elapsed_time} ms")
```

## 2.2 基于event 的同步

**原理**
```bash
Stream1: [操作1] --> [操作2] --> [event.record()]
                          |
                          | (event.wait 插入到 Stream2)
                          v
Stream2: [操作A] --> [event.wait()] --> [操作B] (操作B 等待 Stream1 的 event 完成)
```

**示例代码**

```python
import torch

# 创建流和事件
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
event = torch.cuda.Event()

# Stream1 中执行操作并记录事件
with torch.cuda.stream(stream1):
    a = torch.randn(1000, device='cuda')  # 操作1
    event.record()                        # 记录事件（操作1完成）
    b = torch.randn(1000, device='cuda')  # 操作2（在事件记录之后）

# Stream2 等待事件完成（仅等待操作1）
with torch.cuda.stream(stream2):
    event.wait()                          # stream2 等待 stream1 中的事件完成（操作1完成）
    c = a + 1                             # 安全使用 a（操作1已完，操作2可能未完成）

# 显式同步所有流
torch.cuda.synchronize()
```

## 2.3 stream.wait_stream() vs event.wait()

| 特性                 | `event.wait()`                  | `stream.wait_stream()`          |
|----------------------|---------------------------------|----------------------------------|
| **同步粒度**         | 细粒度（同步到事件记录点）       | 粗粒度（同步到流的全部已提交操作） |
| **依赖范围**         | 仅事件前的操作                  | 另一个流的所有已提交操作          |
| **灵活性**           | 高（可指定任意中间点）           | 低（必须等待整个流）              |
| **典型场景**         | 部分结果依赖（如流水线并行）     | 全依赖（如数据完整加载后计算）    |
| **是否阻塞CPU**         | 否（异步）                       | 否                           |

## 2.4 wait 和 synchronize 的区别

| 特性               | `event.wait()`                       | `event.synchronize()`                |
|--------------------|--------------------------------------|--------------------------------------|
| **作用对象**       | 插入到流中，影响该流后续操作         | 阻塞当前 CPU 线程                    |
| **同步范围**       | 仅针对某个流                         | 全局同步（所有流）                   |
| **阻塞性**         | 非阻塞 CPU                           | 阻塞 CPU                             |
| **典型用途**       | 流间依赖控制                         | 强制同步、计时、调试                 |

# 3 cuda Strem

**torch.cuda.Stream**

>**Stream 里的操作都是用转为用Event 来实现的**. <br>

```python
class Stream(torch._C._CudaStreamBase, _StreamBase):
    r"""Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.  See :ref:`cuda-semantics` for
    details.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.cuda.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event) -> None:
        r"""Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.cuda.Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
           `CUDA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA Stream documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
        event.wait(self)

    def wait_stream(self, stream) -> None:
        r"""Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Record an event.

        Args:
            event (torch.cuda.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA Stream documentation`_ for more info.
        """
        super().synchronize()
```
