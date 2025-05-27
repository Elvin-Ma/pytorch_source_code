import torch
import torch.distributed as dist
from datetime import timedelta
from typing  import override
import datetime


def test_prefix():
    class MyStore(dist.TCPStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.store = store

        @override
        def set(self, key, value):
            print(f"Intercepted set: key={key}, value={value}")
            super().set(key, value)
            # self.store.set(key, value)


    # 初始化主 Store
    backend_store = MyStore("127.0.0.1", 1234, world_size=1, is_master=True)

    # 创建带前缀的 PrefixStore
    prefix = "task_a"
    task_store = dist.PrefixStore(prefix, backend_store)


    # dist.init_process_group(
    #     backend='nccl',
    #     store=task_store,
    #     rank=0,
    #     world_size=1
    # )

    #pg = dist.new_group(backend="nccl")

    pg = dist.ProcessGroupNCCL(task_store, 0, 1, datetime.timedelta(seconds=10),)

    data = torch.randn(3,4).cuda()

    work = pg.allreduce([data])
    work.wait()

    # # 写入数据（实际键为 "task_a/data"）
    # task_store.set("data", "5dx")

    # # 从底层 Store 读取（需知道前缀）
    # value = backend_store.get("task_a/data")

    # print(backend_store.keys())

def test_preocess_group():
    import torch.distributed as c10d
    store = c10d.TCPStore("127.0.0.1", 45678, world_size=1, is_master=True)

    prefix_store = c10d.PrefixStore("prefix", store)
    process_group = c10d.ProcessGroupNCCL(prefix_store, 0, 1)
    process_group.allreduce(torch.rand(10).cuda(0))

if __name__ == "__main__":
    # test_prefix()
    test_preocess_group()
    print(f"run prefix_demo.py successfully !!!")
