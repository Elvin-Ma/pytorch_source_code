# 0 Torch Distributed Elastic
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Makes distributed PyTorch fault-tolerant and elastic.<br>

# 1 torchrun (Elastic Launch)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torch.distributed.launch 的 superSet(超级).<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torchrun 提供了比 torch.distributed.launch 更全面的功能，并额外包括以下功能：<br>
- worker故障能够优雅地(gracefully )处理(handled)，通过重启所有工作节点(restart all workers)来应对。
- 工作节点的RANK和WORLD_SIZE会自动分配。
- 节点数量允许在最小(minimum size)和最大规模(maxmum size)之间变化（具有弹性）。

**Note**<br>
*torchrun 是一个 Python 控制台脚本，它对应于在 setup.py 的 entry_points 配置中声明的主模块 torch.distributed.run。它等同于执行 python -m torch.distributed.run。* <br>

# 2 Transitioning from torch.distributed.launch to torchrun
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torchrun 支持与 torch.distributed.launch 相同的参数，但不再支持现已弃用的 --use-env 参数。要从 torch.distributed.launch 迁移到 torchrun，请按照以下步骤操作：<br>

1. 启动指令修改： 如果您的训练脚本已经通过读取 LOCAL_RANK 环境变量来获取本地工作节点的排名（local_rank），那么您只需简单地省略掉 --use-env 标志（因为 torchrun 已经不再支持这个标志），例如：<br>

```shell
# use torch.distributed.launch
python -m torch.distributed.launch --use-env train_script.py


# use torchrun
torchrun train_script.py
```

2. 训练脚本修改：如果您的训练脚本通过命令行参数 --local-rank 来读取本地工作节点的排名，请修改您的**训练脚本**，使其从 LOCAL_RANK 环境变量中读取，如下所示的代码片段演示了这一点：<br>

```python
# use torch.distributed.launch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()

local_rank = args.local_rank

# use torchrun
import os
local_rank = int(os.environ["LOCAL_RANK"])
```

# 3 Note
- 在版本2.0.0中的更改：启动器(launcher)将向您的脚本传递 --local-rank=<rank> 参数。从 PyTorch 2.0.0 开始，建议使用带短横线的 --local-rank，而不是**之前使用的带下划线**的 --local_rank。<br>
- 为了向后兼容，用户可能需要在其参数解析代码中处理这两种情况。这意味着在参数解析器中需要同时包含 "--local-rank" 和 "--local_rank"。如果只提供了 "--local_rank"，启动器将触发错误：“error: unrecognized arguments: –local-rank=<rank>”。对于仅支持 PyTorch 2.0.0+ 的训练代码，仅包含 "--local-rank" 应该就足够了。<br>
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int)
args = parser.parse_args()
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上述更改足以实现从 torch.distributed.launch 到 torchrun 的迁移。要利用 torchrun 的新功能，如弹性、容错性和错误报告，请参阅：<br>
- 训练脚本：要了解如何编写符合 torchrun 要求的**训练脚本**，请[参阅更多信息](https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script)。
- 本页其余部分：要了解 torchrun 的更多功能，请参阅本页的其余部分。<br>

# 4 Usage
## 4.1 Single-node multi-worker
```shell
torchrun
    --standalone # 独立的
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

## 4.2 Stacked single-node multi-worker
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;要在同一主机上运行多个单节点、多工作器的实例instances（separate job），我们需要确保每个instance（work）设置在不同的端口(port)上，以避免端口冲突port conflict（或者更糟糕的是，两个作业被合并为一个作业）。为此，您需要使用 --rdzv-backend=c10d 运行，并通过设置 --rdzv-endpoint=localhost:$PORT_k 来**指定不同的端口**。对于 --nodes=1 的情况，通常更方便的是让 torchrun 自动选择一个空闲的随机端口，而不是每次运行都手动分配不同的端口。<br>

```shell
torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0 # 0 表示自动选择窗口
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

## 4.3 Fault-tolerant(固定数量的工作进程，无弹性扩展，可容忍3次故障)

```shell
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;HOST_NODE_ADDR，格式为<主机>[:<端口>]（例如，node1.example.com:29400），用于指定应实例化并托管C10d rendezvous后端的节点和端口。它可以是您训练集群中的任何节点，但理想情况下，您**应选择一个带宽较高的节点**。<br>

**NOTE:**<br>
*If no port number is specified HOST_NODE_ADDR defaults to 29400.* <br>

## 4.4 Elastic(min=1, max=4 弹性扩展，可容忍3次故障)

```shell
torchrun
    --nnodes=1:4
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID   # 确保不同的训练作业不会相互干扰
    --rdzv-backend=c10d # 使用PyTorch的C10d（分布式通信库）作为rendezvous后端
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

# 5 Note on rendezvous backend
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于多节点训练，您需要指定：<br>
- --rdzv-id：一个唯一的作业ID（由参与该作业的所有节点共享）
- --rdzv-backend：torch.distributed.elastic.rendezvous.RendezvousHandler的一个实现
- --rdzv-endpoint：集合后端运行的端点；通常格式为主机:端口。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目前，开箱即用支持c10d（推荐）、etcd-v2和etcd（旧版）rendezvous后端。要使用etcd-v2或etcd，请设置一个启用了v2 API的etcd服务器（例如，使用--enable-v2参数）。<br>

**WARNING:**<br>
*etcd-v2和etcd集合后端使用etcd API v2。您必须在etcd服务器上启用v2 API。我们的测试使用的是etcd v3.4.3。* <br>
*对于基于etcd的集合，我们建议使用etcd-v2而不是etcd，它们在功能上是等价的，但etcd-v2采用了修订后的实现。etcd已处于维护模式，并将在未来的版本中被移除。* <br>

# 6 Train script
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果您的训练脚本可以与 torch.distributed.launch 一起使用，那么它将继续与 torchrun 一起使用，但有以下不同之处：<br>

1. 无需手动传递 RANK、WORLD_SIZE、MASTER_ADDR 和 MASTER_PORT。<br>
2. 可以提供 rdzv_backend 和 rdzv_endpoint。对于大多数用户来说，这将设置为 c10d（参见集合点）。默认的 rdzv_backend 创建一个非弹性的集合点，其中 rdzv_endpoint 保存主地址。<br>
3. 确保您的脚本中包含 load_checkpoint(path) 和 save_checkpoint(path) 逻辑。当任意数量的工作进程失败时，我们会使用相同的程序参数重启所有工作进程，因此您可能会丢失到最近检查点之前的进度（参见弹性启动）。<br>
4. 已移除 use_env 标志。如果您之前是通过解析 --local-rank 选项来获取本地排名的，那么现在需要从环境变量 LOCAL_RANK 中获取本地排名（例如 int(os.environ["LOCAL_RANK"])）。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是一个训练脚本的说明性示例，该脚本在每个训练周期进行检查点保存，因此，在发生故障时，最坏情况下损失的进度是一个完整的训练周期。<br>

```python
def main():
     args = parse_args(sys.argv[1:])
     state = load_checkpoint(args.checkpoint_path)
     initialize(state)

     # torch.distributed.run ensures that this will work
     # by exporting all the env vars needed to initialize the process group
     torch.distributed.init_process_group(backend=args.backend)

     for i in range(state.epoch, state.total_num_epochs)
          for batch in iter(state.dataset)
              train(batch, state.model)

          state.epoch += 1
          save_checkpoint(state)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;要查看符合 torchelastic 要求的训练脚本的具体示例，请访问我们的[示例页面](https://github.com/pytorch/elastic/tree/master/examples)。<br>

# 7 术语
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下是这些术语的中文翻译：<br>
- Node - 一个物理实例或容器；对应于作业管理器工作的单元。
- Worker - 在分布式训练 上下文 中的工作单元。
- WorkerGroup - 执行相同功能的工作进程集合（例如，训练器）。
- LocalWorkerGroup - 在同一节点上运行的、属于工作进程组的一个工作进程子集(Subset)。
- RANK - 工作进程在工作进程组中的排名。
- WORLD_SIZE - 工作进程组中的总工作进程数。
- LOCAL_RANK - 工作进程在本地工作进程组中的排名。
- LOCAL_WORLD_SIZE - 本地工作进程组的大小。
- rdzv_id - 用户定义的ID，用于唯一标识一个作业的工作进程组。每个节点使用这个ID加入特定的工作进程组。
- rdzv_backend - 集合的后端（例如，c10d）。这通常是一个强一致性的键值存储。
- rdzv_endpoint - 集合后端端点；通常格式为<主机>:<端口>。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个节点运行LOCAL_WORLD_SIZE个工作进程，这些工作进程组成一个本地工作进程组(LocalWorkerGroup)。作业中所有节点上的本地工作进程组的集合构成工作进程组(WorkerGroup)。<br>

# 8 Environment Variables
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以下环境变量在您的脚本中可用：<br>

- LOCAL_RANK - 本地排名。
- RANK - 全局排名。
- GROUP_RANK - 工作进程组的排名。一个介于0和max_nnodes之间的数字。**当每个节点运行一个工作进程组时，这是节点的排名**。
- ROLE_RANK - **具有相同角色的所有工作进程中该工作进程的排名。工作进程的角色在WorkerSpec中指定**。
- LOCAL_WORLD_SIZE - 本地世界大小（例如，本地运行的工作进程数）；等于在torchrun上指定的--nproc-per-node。
- WORLD_SIZE - 世界大小（作业中的总工作进程数）。
- ROLE_WORLD_SIZE - 以WorkerSpec中指定的**相同角色**启动的工作进程总数。
- MASTER_ADDR - 运行rank为0的工作进程的主机的完全限定域名（FQDN）；用于初始化Torch分布式后端。
- MASTER_PORT - MASTER_ADDR上的端口，可用于**托管C10d TCP存储**。
- TORCHELASTIC_RESTART_COUNT - 到目前为止的工作进程组重启次数。
- TORCHELASTIC_MAX_RESTARTS - 配置的最大重启次数。
- TORCHELASTIC_RUN_ID - 等于集合run_id（例如，**unuque job ID**）。
- PYTHON_EXEC - Python 解释器的可执行文件路径。如果提供，Python用户脚本将使用PYTHON_EXEC的值作为可执行文件。默认情况下使用sys.executable。

# 9 部署(Deployment)
- （对于C10d后端不需要）启动rendezvous后端服务器并获取端点（作为--rdzv-endpoint传递给启动脚本）。

- 单节点多工作进程：在主机上启动launcher，以启动创建并监控本地工作进程组的代理进程。

- 多节点多工作进程：**在所有参与训练的节点上，使用相同的参数**启动launcher。

# 10 故障模式(Failure Modes)
- 工作进程故障(Worker Failure)：对于一个包含n个工作进程的训练作业，如果k（k<=n）个工作进程发生故障，所有工作进程都将停止，并根据最大重启次数（max_restarts）进行重启。
- 代理故障(Agent Failuer)：代理故障会导致本地工作进程组(LocalWorkerGroup)故障。作业管理器可以选择**让整个作业失败（集群语义）** 或**尝试替换节点**。代理支持这两种行为。
- 节点故障(Node Failure)：与代理故障相同。

# 11 节点变更(Membership Changes)
- Node离开departure（缩容scale-down）：代理(agent)会收到节点离开的通知，所有**现有的**工作进程都会停止，然后形成一个新的工作进程组，并且所有工作进程都会以新的RANK和WORLD_SIZE启动。<br>
- Node 加入（扩容scale-up）：新节点被加入到作业中，所有现有的工作进程**都会停止**，然后形成一个新的工作进程组，并且所有工作进程都会以**新的**RANK和WORLD_SIZE启动。<br>

# 12 重要通知(important notices)
1. 此工具和多进程分布式（单节点或多节点）GPU训练**目前仅在使用NCCL分布式后端时才能达到最佳性能**。因此，对于GPU训练，建议使用NCCL后端。<br>

2. 本模块为您提供了初始化Torch进程组所需的环境变量，您无需手动传递RANK。要在您的训练脚本中初始化进程组，只需运行：<br>

```python
import torch.distributed as dist
dist.init_process_group(backend="gloo|nccl")
```

3. 在您的训练程序中，您可以选择使用常规的分布式函数，或者使用 torch.nn.parallel.DistributedDataParallel() 模块。如果您的训练程序使用GPU进行训练，并且您希望使用 torch.nn.parallel.DistributedDataParallel() 模块，以下是配置方法。<br>
```python
local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;请确保 device_ids 参数设置为您的代码将操作的唯一GPU设备ID。这**通常是进程的本地排名**。换句话说，为了使用这个工具，device_ids 需要设置为 [int(os.environ("LOCAL_RANK"))]，并且 output_device 需要设置为 int(os.environ("LOCAL_RANK"))。<br>

4. 在发生故障或成员变更时，所有存活的工作进程会立即被终止。请确保对您的进度进行检查点保存。检查点的频率应取决于您的工作对丢失工作的容忍度。

5. 此模块仅支持同质的 LOCAL_WORLD_SIZE。即，假设所有节点运行的本地工作进程数量（每个角色）是相同的。

6. RANK 是不稳定的。在重启之间，节点上的本地工作进程可能会被分配与之前不同的排名范围。切勿对排名的稳定性或 RANK 与 LOCAL_RANK 之间的任何相关性做出硬编码假设。

7. 当使用弹性扩展（min_size!=max_size）时，不要对 WORLD_SIZE 做出硬编码假设，因为**允许节点离开和加入时**，WORLD_SIZE 可能会发生变化。

8. 建议您的脚本采用以下结构：<br>
```python
def main():
  load_checkpoint(checkpoint_path)
  initialize()
  train()

def train():
  for batch in iter(dataset):
    train_step(batch)

    if should_checkpoint:
      save_checkpoint(checkpoint_path)
```

9. （建议）在工作进程出错时，此工具将总结错误的详细信息（例如时间、排名、主机、进程ID、堆栈跟踪等）。在每个节点上，按时间戳判断，第一个错误会被启发式地报告为“Root Cause” error。要在错误摘要输出中包含堆栈跟踪，您必须按照以下示例所示，为您训练脚本中的主要入口点函数**添加装饰器**。如果不添加装饰器，则摘要中将不包含异常的堆栈跟踪，而仅包含退出代码。有关torchelastic错误处理的详细信息，请参阅：https://pytorch.org/docs/stable/elastic/errors.html。<br>

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # do train
    pass

if __name__ == "__main__":
    main()
```
