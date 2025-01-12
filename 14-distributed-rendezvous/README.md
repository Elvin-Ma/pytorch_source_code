# 1 官方文档
**Below is a state diagram describing how rendezvous works:** <br>
![rendezvous_state_diagram](https://pytorch.org/docs/stable/_images/etcd_rdzv_diagram.png)

## 1.1 总体背景
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TE 是围绕在 Rendezvous 基础之上的多个elastic agent构成，这是一种功能分离，让我们对比一下看看。<br>

- Agent 偏重具体节点上的逻辑:<br>
1. Agent 负责具体业务逻辑相关操作，比如启动进程执行用户程序，监控用户程序运行情况，如果有异常就通知 Rendezvous。<br>
2. Agent 是一个 worker manager，负责启动/管理 workers 进程，组成一个 worker group，监控 workers 运行状态，捕获失效 workers，如果有故障/新加入worker，则重启 worker group。<br>
3. Agent负责维护 WORLD_SIZE 以及 RANK 信息。用户不需要再手动提供，Agent会自动处理这些。<br>
4. Agent 是具体节点上的后台进程，是独立个体。Agent自己无法实现整体上的弹性训练，所以需要一个机制来完成 worker 之间的相互发现，变更同步等等（WORLD_SIZE 和 RANK 这些信息其实也需要多个节点同步才能确定），这就是下面的 Rendezvous 概念。<br>

- Rendezvous 负责集群逻辑，保证节点之间对于""有哪些节点参与训练"达成强一致共识。<br>
1. 每一个 Agent 内部包括一个 Rendezvous handler，这些 handler 总体上构成了一个 Rendezvous 集群，从而构成了一个 Agent 集群。<br>
2. Rendezvous 完成之后，会创建一个共享键值存储（shared key-value store），这个store实现了一个torch.distributed.Store API。此存储仅3. 由已完成Rendezvous的成员共享，它旨在让Torch Distributed Elastic在初始化作业过程之中交换控制和数据信息。<br>
4. Rendezvous 负责在每个agent之上维护当前 group 所有相关信息。每个 agent 之上有一个 rendezvous，它们会互相通信，总体维护一套信息，这些信息存储在上面提到的Store 之中。<br>
5. Rendezvous 负责集群逻辑相关，比如新加入节点，移除节点，分配rank等等。<br>

## 1.2 基本概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其可以理解为一个分布式治理过程：Rendezvous 被Torch Distributed Elastic用来收集一个训练job的参与者（节点），这样，参与者们可以商议得到参与者列表和每个参与者的角色，也可以对训练何时开始/恢复做出一致的集体决定。即，通过 rendezvous，系统对参与者达成共识，给每一个参与者分配 rank，local rank，通知 world size等等，当需要弹性伸缩或者出现故障时候，就会重新进行 rendezvous 操作。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了实现弹性训练，需要有一个节点/进程之间彼此发现的机制。在TorchElastic中，rendezvous就是这个发现机制或者说同步组件，其被用来作为对等发现的分布式同步（治理）机制，用于同步、收集各个worker的信息，包括节点列表、各节点worker角色等，然后各个Agent才能共同决定训练的开始、结束、恢复等。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Torch Distributed Elastic的上下文中，我们使用术语**rendezvous(汇合点)**来指代一种特定的功能，该功能将分布式同步原语(distributed synchronization primitive)与对等点发现(peer discovery)结合在一起。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic使用rendezvous来收集训练作业（即节点）的参与者(participants)，使它们都能就相同的参与者列表和各自的角色达成一致，并就何时可以开始/恢复训练做出一致性(consistent)的集体决策。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic的rendezvous提供了以下关键功能：<br>

![figure1](https://img2020.cnblogs.com/blog/1850883/202112/1850883-20211227100606877-261831018.jpg)

## 1.3 Barrier
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;执行rendezvous的节点都会阻塞(block)，直到rendezvous被视为完成——这发生在至少有最小总数的节点加入了rendevous barrier（针对同一作业）时。这也意味着barrier的大小并不一定是固定的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在达到最小节点数后，还会有一个额外的短暂等待时间--这是为了确保rendezvous不会“太快”完成（这可能会排除那些几乎同时尝试加入的额外节点）。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果屏障(barrier)处聚集(rendezvous)的节点数达到了最大值，则集合立即完成。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此外，还有一个总体(overall)超时(timeout)设置，如果在超时时间内未达到最小节点数，则集合(rendezvous)操作失败。这是一个简单的故障安全机制，用于在资源管理器出现问题时帮助**释放部分分配的作业资源**，且此失败被视为不可重试的(non-retryable)。<br>

## 1.4 Exclusivity(独占性)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个简单的分布式barrier并不足够(sufficient)，因为我们还需要确保**在任何给定时间（针对给定任务）只存在一个节点组**。换句话说，新的节点（即晚加入的节点）不能够形成同一个任务的并行独立工作节点组。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic rendezvous确保 : 如果一个节点组已经完成集合rendezvous（并且因此可能已经开始训练），那么尝试rendezous的其他“迟到”节点将只能宣布自己处于等待状态，并且必须等到（之前已完成的）**现有rendezvous被销毁后才能继续**。<br>

## 1.5 Consistency(一致性)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当rendezvous完成时，其所有成员将就任务成员身份(job membership)以及各自在任务中的角色达成一致。这个角色通过一个称为 **"秩(rank)"的整数**来表示，其值在0和world size之间。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的是，秩并不是稳定的，即**同一个节点在下一次（重新）rendezvous时可能会被分配不同的秩**。<br>

## 1.6 Fault Tolerance(容错性)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 的rendezvous机制旨在容忍集合(rendezvous)过程中的节点故障(node failures)。如果在加入集合(rendezvous)和集合(rendezvous)完成之间某个进程崩溃(或失去网络连接等)，那么将自动与剩余的健康节点重新进行集合(**re-rendezvous**)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个节点也可能在已完成rendezvous(或已被其他节点观察到已完成rendezvous)之后发生故障————这种情况将由 **Torch Distributed Elastic 的训练循环（train_loop）来处理（在这种情况下，它也会触发重新集合re-rendezvous）**。<br>

## 1.7 Shared key-value store
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当rendezvous完成时，会创建并返回一个共享的键值存储(KVStore)。这个存储实现了 torch.distributed.Store API(参见[分布式通信文档](https://pytorch.org/docs/stable/distributed.html)). <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**这个存储(Store)仅由完成rendezvous的成员共享**。它旨在供 Torch Distributed Elastic 使用，以交换初始化worker control和数据平面(initialize job control and data-planes)所需的信息. <br>

## 1.8 Waiting workers and rendezvous closing
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 的rendezvous handler object提供了额外的功能，这些功能从技术上讲并不属于rendezvous过程的一部分：<br>
- 查询有多少工作节点在barrier处到达较晚，这些节点**可以参与下一次集合rendezvous**。<br>
- 设置集合rendezvous为关闭状态，以通知所有节点不要参与下一次集合。<br>

## 1.9 DynamicRendezvousHandler
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 提供了 DynamicRendezvousHandler 类，该类实现了上述描述的集合机制。它是一个**后端无关**的类型，在构造时期望指定一个特定的 RendezvousBackend 实例。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch 分布式用户可以实现自己的后端类型，或者使用 PyTorch 提供的以下实现之一: <br>

- C10dRendezvousBackend：使用 C10d 存储（默认是 TCPStore）作为rendezvous后端。使用 C10d store的主要优点是，它不需要第三方依赖（如 etcd）来建立rendezvous。<br>

- EtcdRendezvousBackend：取代了旧的 EtcdRendezvousHandler 类。将 EtcdRendezvousBackend 实例传递给 DynamicRendezvousHandler 在功能上等同于实例化一个 EtcdRendezvousHandler。<br>

**example** <br>
```python
 store = TCPStore("localhost")
 backend = C10dRendezvousBackend(store, "my_run_id")
 rdzv_handler = DynamicRendezvousHandler.from_backend(
     run_id="my_run_id",
     store=store,
     backend=backend,
     min_nodes=2,
     max_nodes=4
 )
```

# 2 静态结构
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; elastic 内部另有一套 Rendezvous，和 distributed 原有的 Rendezvous 那套不一样，别搞混了。**distributed 原有的 Rendezvous 就是一套简单的 KV 存储**。elastic Rendezvous 则要复杂得多。这里主要描述elastic 的 Rendezvous<br>

## 2.1 RendezvousParameters <br>
RendezvousParameters 是构建RendezvousHandler所需参数。<br>
- backend ：后端名称。<br>
- endpoint ：端点，格式是 <hostname>[:<port>]。<br>
- run_id : rendezvous 的 id。<br>
- min_nodes ：rendezvous 的**最小节点数目**。<br>
- max_nodes ：rendezvous 的**最大节点数目**。<br>
- kwargs ：后端的附加参数。<br>

## 2.2 RendezvousSettings
RendezvousSettings 类用来存储rendezvous的配置。可以理解为静态元信息。<br>
- run_id : rendezvous 的 id. <br>
- min_nodes ：rendezvous 的最小节点数目. <br>
- max_nodes ：rendezvous 的最大节点数目. <br>
- timeout ：超时时间. <br>
- keep_alive_interval ：节点在发送心跳之间等待的时间量. <br>
- keep_alive_max_attempt ： 心跳的最大重试次数. <br>

## 2.3 _RendezvousState
_RendezvousState 是rendezvous的状态。是动态信息，每一个 node 都会维护一个本地 state。<br>

- round：Rendezvous的当前轮次. <br>
- complete：一个布尔值，指示rendezvous当前一轮是否完成了。<br>
- deadline：截止时间，如果如果当前轮次一直在等待节点加入，如果这个参数设置了，就是**等待的截至时间**。<br>
- closed：一个布尔值，指示rendezvous是否结束了。<br>
- participants：字典结构，存放参与者和它们对应ranks。<br>
- wait_list：set结构，存放**等待参与下一轮rendezvous**操作的一组节点. <br>
- last_heartbeats：字典，包含每个节点上次心跳时间. <br>

## 2.4 _NodeDesc 节点
_NodeDesc 是rendezvous的一个节点。<br>
- fqdn：节点的完全限定域名（FQDN）。<br>
- pid：运行集合点处理程序（rendezvous handler）的**进程ID**。<br>
- local_id：进程范围内唯一的ID。<br>

```python
@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        return f"{self.addr}_{self.pid}_{self.local_id}"
```

## 2.5 backend
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 PyTorch 之中，backend 概念指的是当前进程要使用的通信后端，一般来说，支持的通信后端有 gloo，mpi，nccl 。建议用 nccl。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在弹性训练这里, **DynamicRendezvousHandler**需要我们在构建时候指定后端(RendezvousBackend)。用户可以自己实现后端，或者使用如下PyTorch附带实现之一: <br>

- **C10dRendezvousBackend**，其使用 C10d 存储(默认是**TCPStore**) 作为 rendezvous backend，其优势是不需要依赖第三方，比如etcd，来构建一个rendezvous 。<br>
- **EtcdRendezvousBackend**，其使用EtcdRendezvousHandler，EtcdRendezvousBackend 等类来基于 etcd 完成。<br>

EtcdRendezvousBackend 必须依赖 ETCD，需要安装一个 ETCD集群，所以推荐使用 c10d 后端，其易用性更好。我们接下来就主要介绍 c10d 后端。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C10d 后端主要基于一个 **TCPStore**，通过 TCP 进行同步。我们在之前文章中介绍过 TCPStore，**TCPStore 是基于 TCP 的分布式键值存储(KVStore)实现**（类似于 Redis）。是一个典型的**client-server架构**，服务器存储/保存数据，而存储客户端可以通过 TCP 连接到服务器存储并执行诸如set()插入键值对、get()检索键值对等操作。<br>

**example** <br>
```python
store = TCPStore("localhost")

backend = C10dRendezvousBackend(store, "my_run_id")

rdzv_handler = DynamicRendezvousHandler.from_backend(
    run_id="my_run_id",
    store=store,
    backend=backend,
    min_nodes=2,
    max_nodes=4
)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所以，对于 c10d 后端来说，在其中一个agent之上会运行**TCPStore Master**，其负责**监听端口**，提供API，Rendezvous 的各种**同步操作**，都是由各个agent连接到这个**中心化**的 TCPStore Master，在其上完成。<br>

**TCPStore 的创建** <br>
```python
def _create_tcp_store(params: RendezvousParameters) -> TCPStore:
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=29400)

    cfg_is_host = params.get_as_bool("is_host")
    # If the user has explicitly specified whether our process should host the
    # the store, respect it.
    if cfg_is_host is not None: # 如果配置了，就使用
        is_host = cfg_is_host
    # Otherwise try to determine whether we are the host based on our hostname
    # and IP address.
    else: # 动态看看本机是不是host
        is_host = _matches_machine_hostname(host)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # In specific cases we attempt to instantiate the store twice. For details
    # see the explanation in the except clause below.
    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
                use_libuv=params.use_libuv,
            )

            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)

            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # If we heuristically inferred the value of is_host as True and our
            # first attempt to instantiate the TCP store has failed, try it one
            # more time with is_host set to False. As an edge case there can be
            # more than one process that is part of the same rendezvous on this
            # machine and only one of them will eventually host the store.

            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc

    return store  # type: ignore[possibly-undefined]
```

**C10dRendezvousBackend** <br>
C10dRendezvousBackend 其核心就是一个 Store，用来存储相关信息，通过 set_state 和 get_state 来对 store 进行读写.<br>

## 2.6 StateHolder
### 2.6.1 _RendezvousStateHolder
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这个类的作用是**保存与其他节点同步(SYNC)的rendezvous状态**，但是需要一个派生类来完成功能。<br>

### 2.6.2 _BackendRendezvousStateHolder
_BackendRendezvousStateHolder 继承了 _RendezvousStateHolder。其 sync 就是**调用内部的后端, 对 store 进行读写**。<br>

### 2.6.3 如何使用
_DistributedRendezvousOpExecutor 中会使用_BackendRendezvousStateHolder. <br>

```python
def run(
    self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float
) -> None:
    """See base class."""
    action = None

    while action != _Action.FINISH:
        # Reads or writes the latest rendezvous state shared by all nodes in
        # the rendezvous. Note that our local changes might get overridden
        # by another node if that node synced its changes before us.
        
        has_set = self._state_holder.sync()  # 这里要同步各种状态，因为最新状态在 rendezvous。

        self._state = self._state_holder.state # 得到最新的状态
        ctx = _RendezvousContext(self._node, self._state, self._settings)

        # Determine the next action to take based on the current state of
        # the rendezvous.
        action = state_handler(ctx, deadline) 

        # 省略部分代码

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
            elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                self._remove_from_participants()
            elif action == _Action.REMOVE_FROM_WAIT_LIST:
                self._remove_from_wait_list()
            elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                self._mark_rendezvous_complete()
            elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                self._mark_rendezvous_closed()

            # Attempt to sync our changes back to other nodes.
            self._state_holder.mark_dirty() # 再次同步，把自己状态同步给其他节点
```

## 2.7 目前为止的逻辑
```python
                                                                       +
+-------------------------------+                                      |                                        +-------------------------------+
| _BackendRendezvousStateHolder |                                      |                                        | _BackendRendezvousStateHolder |
|                               |     +-------------------+            |           +--------------------+       |                               |
|             _settings +-----------> | RendezvousSettings|            |           | RendezvousSettings | <----------+ _settings                |
|                               |     +-------------------+            |           +--------------------+       |                               |
|                               |     +-------------------+            |           +--------------------+       |                               |
|             _state +--------------> | _RendezvousState  |            |           | _RendezvousState   | <----------+ _state                   |
|                               |     |                   |            |           |                    |       |                               |
|                               |     +-------------------+            |           +--------------------+       |                               |
|                               |                                      |                                        |                               |
|                               |     +-----------------------+        +           +----------------------+     |                               |
|             _backend +------------> | C10dRendezvousBackend |                    | C10dRendezvousBackend| <-------+  _backend                 |
|                               |     |                       |    +---------+     |                      |     |                               |
|                               |     |             _store +-----> |TCPStore | <---------+ _store         |     |                               |
|                               |     |                       |    |         |     |                      |     |                               |
|                               |     +-----------------------+    +---------+     +----------------------+     |                               |
|                               |                                                                               |                               |
|                               |         ^                            +                       ^                |                               |
|                               |         |                            |                       |                |                               |
|                               |         |                            |                       |                |                               |
|             sync +----------------------+                            |                       +---------------------+  sync                    |
|                               |   set_state                          |                         set_state      |                               |
+-------------------------------+                                      +                                        +-------------------------------+
```

# 3 动态逻辑








