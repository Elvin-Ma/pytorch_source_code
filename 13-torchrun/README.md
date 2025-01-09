# 0 torchrun
- /root/miniconda3/envs/pytorch2.5/bin/torchrun

```python
#!/root/miniconda3/envs/pytorch2.5/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from torch.distributed.run import main
if __name__ == '__main__':
    # sys.argv = ['/root/miniconda3/envs/pytorch2.5/bin/torchrun', '--nproc_per_node=2', '--rdzv_backend', 'c10d', '--rdzv_endpoint=localhost:0', '--local-ranks-filter', '0', '--role', 'rank', '--tee', '3', 'train.py', '--job.config_file', './train_configs/llama2_7b.toml']
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

# 1 torch.distributed.run
- torchrun 多进程启动流程如下：<br>
![torchrun](images/torchrun.png)

# 2 torchrun 指令
- -h, --help：<br>
  显示帮助消息并退出.<br>
- --nnodes NNODES：<br>
  节点数，或节点范围的格式<minimum_nodes>:<maximum_nodes>;<br>
- --nproc_per_node/--nproc-per-node NPROC_PER_NODE：<br>
  每个节点的工作进程数；支持的值：[auto, cpu, gpu, int];<br>
- --rdzv_backend/rdzv-backend RDZV_BACKEND：<br>
  集合（Rendezvous）后端;<br>
- --rdzv_endpoint/rdzv-endpoint RDZV_ENDPOINT：<br>
  集合后端端点；通常格式为<host>:<port>;<br>
- --rdzv_id/rdzv-id RDZV_ID：<br>
  用户定义的group id;<br>
- --rdzv-conf/rdzv_conf RDZV_CONF：<br>
  额外的集合配置（<key1>=<value1>,<key2>=<value2>,...）;<br>
- --standalone：<br>
  启动一个**本地独立的集合后端**，该后端由一个空闲端口上的C10d TCP存储表示。这对于启动单节点多工作进程作业很有用。如果指定了该选项，则会**自动分配--rdzv-backend、--rdzv-endpoint和--rdzv-id，并忽略任何显式设置的值**;<br>
- --max_restarts/max-restarts MAX_RESTARTS：<br>
  在失败之前工作进程组的**最大重启次数**;<br>
- --monitor_interval/monitor-interval MONITOR_INTERVAL：<br>
  监控工作进程状态的间隔（秒）;<br>
- --start_method/start-method {spawn,fork,forkserver}：<br>
  创建工作进程时使用的多进程启动方法;<br>
- --role ROLE：<br>
  用户定义的工作进程workers角色;<br>
- -m, --module：<br>
  将每个进程更改为将启动脚本解释为Python模块，以与python -m相同的行为执行;<br>
- --no_python/no-python：<br>
  不在训练脚本前添加python——直接执行它。当脚本不是Python脚本时很有用;<br>
- --run_path/run-path：<br>
  使用runpy.run_path在同一解释器中运行训练脚本。脚本必须提供为绝对路径(例如/abs/path/script.py),这优先于--no-python;<br>
- --log_dir/log-dir LOG_DIR：<br>
  用于日志文件的基础目录（例如/var/log/torch/elastic）。对于多次运行，将重复使用相同的目录(使用**rdzv_id作为前缀**创建一个唯一的作业级子目录);<br>
- -r REDIRECTS, --redirects REDIRECTS：<br>
  将标准流重定向到日志目录中的日志文件（例如[-r 3]将所有工作进程的stdout+stderr重定向，[-r 0:1,1:2]将本地等级0的stdout和等级1的stderr重定向）;<br>
- -t TEE, --tee TEE：<br>
  将标准流复制到 **日志文件(a log file)** 并输出到控制台(请参阅--redirects以了解格式);<br>
- --local_ranks_filter/local-ranks-filter LOCAL_RANKS_FILTER：<br>
  仅在控制台中显示指定等级的日志（例如[--local_ranks_filter=0,1,2]将仅显示等级0、1和2的日志）。这仅适用于stdout和stderr，不适用于通过--redirect或--tee保存的日志文件;<br>
- --node_rank/node-rank NODE_RANK：<br>
  多节点分布式训练中的node rank;<br>
- --master_addr/master-addr MASTER_ADDR：<br>
  主节点（rank0）的地址，仅用于static rendezvous。它可以是rank0的**IP地址或主机名**, 对于单节点多进程训练，--master-addr可以简单地是127.0.0.1；IPv6应该有模式[0:0:0:0:0:0:0:1];<br>
- --master_port/ master-port MASTER_PORT：<br>
  主节点（rank0）上用于分布式训练期间通信的端口。它仅用于static rendezvous;<br>
- --local_addr/local-addr LOCAL_ADDR：<br>
  本地节点的地址。**如果指定，将使用给定的地址进行连接。否则，将查找本地节点地址。** 否则，它将默认为本地机器的FQDN(完全限定域名);<br>
- --logs_specs/logs-specs LOGS_SPECS：<br>
  torchrun.logs_specs组入口点名称，值必须是LogsSpecs类型。可用于覆盖自定义日志行为。-h, --help：显示帮助消息并退出。<br>

# 3 Pytorch Elastic Trainer (PET)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PyTorch Elastic Trainer (PET) 提供了一个可以用**容错**和**弹性**方式跨集群来训练模型的框架。PET 通过两种方式提供这些功能：<br>
- 容错: 当 PyTorch worker 进程抛出某类可重试错误时，它会被 PET 捕获并**重试训练过程**。<br>
- 弹性：只要worker的数量维持在开始工作时指定的范围内，新worker就可以随时离开或加入到现有训练job的进程池。当成员发生变化时，所有worker会**重新集合（re-rendezvous）** 以**建立一个新的进程组**，并从以前的良好状态之中恢复训练。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了与 PET 集成，PyTorch 用户需要对其训练逻辑进行以下更改：<br>
- 用户需要使 PET 来控制他们的训练循环。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本质上，用户提供了一个“内部训练”循环，该循环被 PET 包裹在一个可重试的循环中。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PET循环是可重试的循环，其负责**建立或重新建立过程组**，以及将用户的训练恢复到良好状态。<br>
- 在新worker加入进程池时，用户需要指定**状态是什么(weight, optimizer state ...)** 以及如何把状态施加到一个新worker之上。<br>

# 4 架构概述
## 4.1 Agent
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每个节点有一个独立的elastic-agent(LocalElasticAgent)。每个代理进程只负责管理该节点的一组本地工作进程，并与本作业(job)其他节点上的弹性代理(elastic)一起协调来确定进程组(ProcessGroup)成员身份的变化。具体如下图所示：<br>

![torchelastic diagram](https://raw.githubusercontent.com/pytorch/elastic/master/design/torchelastic/0.2.0/torchelastic_diagram.jpg)

## 4.2 成员变更
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当一个工作进程失败时，管理它的弹性代理(elastic-agent)会杀死该节点上的所有worker，然后与其他代理建立一个集合操作（rendezvous），并使用新的集合信息来**重启worker**。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;但是，当代理以非零错误代码退出时(错误或异常, 0 是正常退出)，应该由**上层调度模块（例如 Kubernetes）来重新启动代理**（同理，此代理将重新启动它负责的所有worker）。相同的恢复机制也适用于节点级故障。编排工具（诸如 Kubernetes ）会调度作业以便job可以使用最小数目的代理副本运行，然后每个代理将依次编排用户的训练脚本。<br>

![torchelastic agent diagram](https://github.com/pytorch/elastic/raw/master/design/torchelastic/0.2.0/torchelastic_agent_diagram.jpg)

# 5 问题及解决思路
- 需要一个节点/进程之间彼此发现的机制<br>
当成员发生变化时，所有worker会重新集合（re-rendezvous）以建立一个新的进程组。rendezvous就是这个发现机制。<br>
- 如何处理成员变更<br>
当一个工作进程失败时，管理它的弹性代理会杀死该节点上的所有worker，然后与其他代理建立一个集合操作（rendezvous），并使用新的集合信息来重启worker。但是，当代理以非零错误代码退出时，应该由上层调度模块（例如 Kubernetes）来重新启动代理（同理，此代理将重新启动它负责的所有worker）。<br>
- 如何捕获单个进程训练失败，如何在单个节点上管理所有训练进程<br>
每个代理进程只负责管理该节点的一组本地工作进程，并与本作业其他节点上的弹性代理一起协调来确定进程组成员身份的变化。
- 如何与现有训练代码集成。<br>
应用程序只需让其入口点或main函数与PyTorch distributed launcher兼容 。





