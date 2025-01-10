# 1 torch-elastic 背景
TE 最重要的是 Agent 和 Rendezvous 这两个概念。<br>
- Agent是运行在单节点上的独立后台进程，可以认为是 worker manager 或者 process supervisor，其负责启动worker，监控 worker 运行，捕获woker异常，通过 rendezvous 实现 worker 间的相互发现，当有成员变动时候负责基于 rendezvous 进行变更同步。
为了实现弹性训练，需要有一个节点/进程之间彼此发现的机制。<br>
- rendezvous就是这个发现机制或者说同步组件。当系统启动或者成员变更时候，所有worker会（重新）集合（rendezvous）以建立一个新的进程组。<br>

## 1.1 功能分离
TE 是围绕在 Rendezvous 基础之上的多个elastic agent构成，这是一种功能分离，让我们对比一下看看。<br>

**Agent 偏重具体节点上的逻辑。** <br>
- Agent 负责具体业务逻辑相关操作，比如启动进程执行用户程序，监控用户程序运行情况，如果有异常就通知 Rendezvous。<br>
- Agent 是一个 worker manager，负责启动/管理 workers 进程，组成一个 worker group，监控 workers 运行状态，捕获失效 workers，如果有故障/新加入worker，则重启 worker group。<br>
- Agent负责维护 WORLD_SIZE 以及 RANK 信息。用户不需要再手动提供，Agent会自动处理这些。<br>
- Agent 是具体节点上的后台进程，是独立个体。Agent自己无法实现整体上的弹性训练，所以需要一个机制来完成 worker 之间的相互发现，变更同步等等（WORLD_SIZE 和 RANK 这些信息其实也需要多个节点同步才能确定），这就是下面的 Rendezvous 概念。<br>

**Rendezvous 负责集群逻辑，保证节点之间对于""有哪些节点参与训练"达成强一致共识.** <br>
- 每一个 Agent 内部包括一个 **Rendezvous handler**，这些 handler 总体上构成了一个 **Rendezvous 集群**，从而构成了一个 **Agent 集群**。<br>
- Rendezvous 完成之后，会创建一个**共享键值存储(shared key-value store)**, 这个store实现了一个torch.distributed.Store API。此**存储仅由已完成Rendezvous的成员共享**, 它旨在让Torch Distributed Elastic在**初始化作业**过程之中**交换控制和数据信息**。<br>
- Rendezvous 负责**在每个agent之上维护当前 group 所有相关信息**。每个 agent 之上有一个 rendezvous，**它们会互相通信，总体维护一套信息**，这些信息存储在上面提到的Store 之中。<br>
- Rendezvous 负责集群逻辑相关，比如**新加入节点，移除节点，分配rank**等等。<br>

## 1.2 Rendezvous 概述
在 Torch Distributed Elastic 上下文之中，我们使用 rendezvous 这个术语来特指一个特定功能：一个结合了**对等发现(peer discovery)** 的分布式同步（distributed synchronization）原语。<br>

Rendezvous 被Torch Distributed Elastic用来收集一个训练job的参与者（节点），这样，参与者们可以**商议**得到参与者列表和每个参与者的**角色**，也可以对训练**何时开始/恢复做出一致的集体决定**。<br>

**Rendezvous 把功能分割解耦，业务逻辑被抽象成为一系列算子**，比如 _RendevzousJoinOp。而 Rendezvous 内部维护了一套**状态机**，由算子决定下一步操作。比如 _RendezvousOpExecutor 来执行各种算子，依据**算子结果得到下一步应该执行的 Action**，从而对本身进行操作。<br>

比如在 _DistributedRendezvousOpExecutor 之中，如果发现了当前 action 是 ADD_TO_WAIT_LIST，会执行 _add_to_wait_list，进而调用 self._state.wait_list.add(self._node). <br>

```python
if action == _Action.FINISH:
    continue

    if action == _Action.ERROR_CLOSED:
        raise RendezvousClosedError

    if action == _Action.ERROR_TIMEOUT:
        raise RendezvousTimeoutError

    if action == _Action.SYNC:
        # Delay the execution by one second to avoid overloading the
        # backend if we are asked to poll for state changes.
        _delay(seconds=1)
    else:
        if action == _Action.KEEP_ALIVE:
            self._keep_alive()
        elif action == _Action.ADD_TO_PARTICIPANTS:
            self._add_to_participants()
        elif action == _Action.ADD_TO_WAIT_LIST:
            self._add_to_wait_list()
        elif action == _Action.ADD_TO_REDUNDANCY_LIST:
            self._add_to_redundancy_list()
        elif action == _Action.REMOVE_FROM_PARTICIPANTS:
            self._remove_from_participants()
        elif action == _Action.REMOVE_FROM_WAIT_LIST:
            self._remove_from_wait_list()
        elif action == _Action.REMOVE_FROM_REDUNDANCY_LIST:
            self._remove_from_redundancy_list()
            # update deadline since the node may participate in rendezvous process
            if update_deadline:
                deadline = update_deadline(self._settings.timeout.join)
        elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
            self._mark_rendezvous_complete()
        elif action == _Action.MARK_RENDEZVOUS_CLOSED:
            self._mark_rendezvous_closed()

        # Attempt to sync our changes back to other nodes.
        self._state_holder.mark_dirty()
```

# 2 Agent 总体逻辑
## 2.1 功能
Elastic agent 是 torchelastic 的控制台（control plane），**Elastic agent是一个独立进程，负责启动和管理底层 worker 进程**，代理具体负责：<br>
- 与PyTorch原生分布式协同工作：使每个worker都能获得所有需要的信息，以便成功调用**torch.distributed.init_process_group()**; <br>
- 容错：监控每个worker，当出现错误或者异常时能及时**终止所有worker并重启它们**; <br>
- 弹性：对成员更改作出反应，并**使用新的成员来重启所有workers**; <br>

## 2.2 工作基础
Torchelastic agent 和 用户 worker 依据故障切换契约来工作：<br>
- TE（torchelastic）希望用户worker以**5分钟为误差**完成工作。<br>
- 设计**DDP**应用程序时，**最好让所有worker都失败，而不只是一个worker失败**。<br>
- TE不会在代理之间同步重启次数(重启次数谁管谁的)。<br>
- TE **re-rendezvous不会减少重启次数(不会减少剩余重启次数)**。<br>
- 当单个代理完成其工作（成功或失败）时，它将**关闭rendezvous**。如果其他代理仍有worker在工作，他们将被**终止**。<br>
- 基于上述情况，如果至少有一个代理完成了任务，则缩容(scale down)**不起作用**。<br>
- 当代理检测到Scale up时，它不会减少 "max_restarts" (属于正常restart, 不会降低重启次数)。<br>
- Torchelast agent 之间通过etcd或者类似后端来保持协同工作。<br>

*注释：etcd是一个高可用的分布式键值存储系统，主要用于存储配置信息、服务发现、协调以及其他需要高度可用性的场景。* <br>

## 2.4 agent 有多种配置方式

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;单的agent部署在每个节点上，并与本地进程协同工作。更高级的agent可以远程启动和管理workers。Agent可以做到彻底的去中心化，与其他agents（管理同一个job的workers）进行沟通协调做出一个集体性决策，决策是基于其管理的 workers 情况来完成。对于如何配置，源码中也给出了示例，如果在GPU上启动训练一个拥有 8 个 trainer（每GPU一个trainer）的 job，我们可以做如下可能的配置: <br>

```python
1. Use 8 x single GPU instances, place an agent per instance, managing 1 worker per agent.
2. Use 4 x double GPU instances, place an agent per instance, managing 2 workers per agent.
3. Use 2 x quad GPU instances, place an agent per instance, managing 4 workers per agent.
4. Use 1 x 8 GPU instance, place an agent per instance, managing 8 workers per agent.
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;基类ElasticAgent 是一个 Abstract Class，真正运行的代理都需要由此派生。从 ElasticAgent 的注释可知，代理进程负责管理一个或多个worker process。工作进程被**假定为常规分布式PyTorch脚本**。当worker进程由代理创建时，代理Agent将为worker进程提供必要的信息，以便正确初始化torch进程组。部署时，**精确的拓扑**和 agent-to-worker 比率取决于代理的具体实现和用户作业放置偏好。<br>

```python
class ElasticAgent(abc.ABC):
    """
    Agent process responsible for managing one or more worker processes.
    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. 

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        log.exception(f"worker 0 failed with exit code : {failure.exit_code}")
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """
        Runs the agent, retrying the worker group on failures up to
        ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """
        Returns:
            The ``WorkerGroup`` for the given ``role``.
            Note that the worker group is a mutable object and hence in a
            multi-threaded/process environment it may change state.
            Implementors are encouraged (but not required) to return
            a defensive read-only copy.
        """
        raise NotImplementedError()
```

- SimpleElasticAgent 实现了基类的部分函数，其目的是为了方便扩展新代理的实现。
- LocalElasticAgent 派生了SimpleElasticAgent ，是目前弹性训练最终使用的代理，主要用于在本地进行操作，负责管理单机上所有的worker进程。

# 3 Worker
Worker 类代表了一个worker实例，我们上文介绍了WorkerSpec，Worker 就是依据 WorkerSpec 构建出来的，其重点成员变量如下：<br>

- id（任意）：唯一标识一个worker，具体是由ElasticAgent的特定实现来解释，对于本地代理，它可以是worker的pid（int），对于远程代理，它可以被编码为``host:port（string）`。<br>
- local_rank ：worker的local rank。<br>
- global_rank：worker的global rank。<br>
- role_rank：具有相同角色的所有worker的rank。<br>
- world_size：全局worker数量。<br>
- role_world_size：具有相同角色的worker数量。<br>





