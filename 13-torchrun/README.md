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
- -h, --help：显示帮助消息并退出.
- --nnodes NNODES：节点数，或节点范围的格式<minimum_nodes>:<maximum_nodes>;
- --nproc_per_node/--nproc-per-node NPROC_PER_NODE：每个节点的工作进程数；支持的值：[auto, cpu, gpu, int];
- --rdzv_backend/rdzv-backend RDZV_BACKEND：集合（Rendezvous）后端;
- --rdzv_endpoint/rdzv-endpoint RDZV_ENDPOINT：集合后端端点；通常格式为<host>:<port>;
- --rdzv_id/rdzv-id RDZV_ID：用户定义的group id;
- --rdzv-conf/rdzv_conf RDZV_CONF：额外的集合配置（<key1>=<value1>,<key2>=<value2>,...）;
- --standalone：启动一个**本地独立的集合后端**，该后端由一个空闲端口上的C10d TCP存储表示。这对于启动单节点多工作进程作业很有用。如果指定了该选项，则会**自动分配--rdzv-backend、--rdzv-endpoint和--rdzv-id，并忽略任何显式设置的值**;
- --max_restarts/max-restarts MAX_RESTARTS：在失败之前工作进程组的**最大重启次数**;
- --monitor_interval/monitor-interval MONITOR_INTERVAL：监控工作进程状态的间隔（秒）;
- --start_method/start-method {spawn,fork,forkserver}：创建工作进程时使用的多进程启动方法;
- --role ROLE：用户定义的工作进程workers角色;
- -m, --module：将每个进程更改为将启动脚本解释为Python模块，以与python -m相同的行为执行;
- --no_python/no-python：不在训练脚本前添加python——直接执行它。当脚本不是Python脚本时很有用;
- --run_path/run-path：使用runpy.run_path在同一解释器中运行训练脚本。脚本必须提供为绝对路径(例如/abs/path/script.py),这优先于--no-python;
- --log_dir/log-dir LOG_DIR：用于日志文件的基础目录（例如/var/log/torch/elastic）。对于多次运行，将重复使用相同的目录(使用**rdzv_id作为前缀**创建一个唯一的作业级子目录);
- -r REDIRECTS, --redirects REDIRECTS：将标准流重定向到日志目录中的日志文件（例如[-r 3]将所有工作进程的stdout+stderr重定向，[-r 0:1,1:2]将本地等级0的stdout和等级1的stderr重定向）;
- -t TEE, --tee TEE：将标准流复制到 **日志文件(a log file)** 并输出到控制台(请参阅--redirects以了解格式);
- --local_ranks_filter/local-ranks-filter LOCAL_RANKS_FILTER：仅在控制台中显示指定等级的日志（例如[--local_ranks_filter=0,1,2]将仅显示等级0、1和2的日志）。这仅适用于stdout和stderr，不适用于通过--redirect或--tee保存的日志文件;
- --node_rank/node-rank NODE_RANK：多节点分布式训练中的node rank;
- --master_addr/master-addr MASTER_ADDR：主节点（rank0）的地址，仅用于static rendezvous。它可以是rank0的**IP地址或主机名**, 对于单节点多进程训练，--master-addr可以简单地是127.0.0.1；IPv6应该有模式[0:0:0:0:0:0:0:1];
- --master_port/ master-port MASTER_PORT：主节点（rank0）上用于分布式训练期间通信的端口。它仅用于static rendezvous;
- --local_addr/local-addr LOCAL_ADDR：本地节点的地址。**如果指定，将使用给定的地址进行连接。否则，将查找本地节点地址。** 否则，它将默认为本地机器的FQDN(完全限定域名);
- --logs_specs/logs-specs LOGS_SPECS：torchrun.logs_specs组入口点名称，值必须是LogsSpecs类型。可用于覆盖自定义日志行为。-h, --help：显示帮助消息并退出。
