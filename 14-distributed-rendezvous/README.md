# 1 官方文档
![rendezvous_state_diagram](https://pytorch.org/docs/stable/_images/etcd_rdzv_diagram.png)

## 1.1 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Torch Distributed Elastic的上下文中，我们使用术语**rendezvous(汇合点)**来指代一种特定的功能，该功能将分布式同步原语(distributed synchronization primitive)与对等点发现(peer discovery)结合在一起。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic使用rendezvous来收集训练作业（即节点）的参与者(participants)，使它们都能就相同的参与者列表和各自的角色达成一致，并就何时可以开始/恢复训练做出一致性(consistent)的集体决策。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic的rendezvous提供了以下关键功能：<br>

## 1.2 Barrier
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;执行rendezvous的节点都会阻塞(block)，直到rendezvous被视为完成——这发生在至少有最小总数的节点加入了rendevous barrier（针对同一作业）时。这也意味着barrier的大小并不一定是固定的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在达到最小节点数后，还会有一个额外的短暂等待时间--这是为了确保rendezvous不会“太快”完成（这可能会排除那些几乎同时尝试加入的额外节点）。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果屏障(barrier)处聚集(rendezvous)的节点数达到了最大值，则集合立即完成。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此外，还有一个总体(overall)超时(timeout)设置，如果在超时时间内未达到最小节点数，则集合(rendezvous)操作失败。这是一个简单的故障安全机制，用于在资源管理器出现问题时帮助**释放部分分配的作业资源**，且此失败被视为不可重试的(non-retryable)。<br>

## 1.3 Exclusivity(独占性)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个简单的分布式barrier并不足够(sufficient)，因为我们还需要确保**在任何给定时间（针对给定任务）只存在一个节点组**。换句话说，新的节点（即晚加入的节点）不能够形成同一个任务的并行独立工作节点组。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic rendezvous确保 : 如果一个节点组已经完成集合rendezvous（并且因此可能已经开始训练），那么尝试rendezous的其他“迟到”节点将只能宣布自己处于等待状态，并且必须等到（之前已完成的）现有rendezvous被销毁后才能继续。<br>

## 1.4 Consistency(一致性)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当rendezvous完成时，其所有成员将就任务成员身份(job membership)以及各自在任务中的角色达成一致。这个角色通过一个称为“秩”（rank）的整数来表示，其值在0和世界大小（world size）之间。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的是，秩并不是稳定的，即**同一个节点在下一次（重新）rendezvous时可能会被分配不同的秩**。<br>

## 1.5 Fault Tolerance(容错性)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 的rendezvous机制旨在容忍集合(rendezvous)过程中的节点故障(node failures)。如果在加入集合和集合完成之间某个进程崩溃（或失去网络连接等），那么将自动与剩余的健康节点重新进行集合(re-rendezvous)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个节点也可能在已完成集合（或已被其他节点观察到已完成集合）之后发生故障————这种情况将由 **Torch Distributed Elastic 的训练循环（train_loop）来处理（在这种情况下，它也会触发重新集合re-rendezvous）**。<br>

## 1.6 Shared key-value store
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当rendezvous完成时，会创建并返回一个共享的键值存储。这个存储实现了 torch.distributed.Store API(参见[分布式通信文档](https://pytorch.org/docs/stable/distributed.html)). <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这个存储(Store)仅由完成集合rendezvous的成员共享。它旨在供 Torch Distributed Elastic 使用，以交换初始化作业控制和数据平面(initialize job control and data-planes)所需的信息. <br>

## 1.7 Waiting workers and rendezvous closing
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 的集合rendezvous handler object提供了额外的功能，这些功能从技术上讲并不属于集合过程的一部分：<br>

- 查询有多少工作节点在barrier处到达较晚，这些节点可以参与下一次集合rendezvous。<br>
- 设置集合rendezvous为关闭状态，以通知所有节点不要参与下一次集合。<br>


## 1.8 DynamicRendezvousHandler
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic 提供了 DynamicRendezvousHandler 类，该类实现了上述描述的集合机制。它是一个后端无关的类型，在构造时期望指定一个特定的 RendezvousBackend 实例。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch 分布式用户可以实现自己的后端类型，或者使用 PyTorch 提供的以下实现之一：

- C10dRendezvousBackend：使用 C10d 存储（默认是 TCPStore）作为rendezvous后端。使用 C10d store的主要优点是，它不需要第三方依赖（如 etcd）来建立rendezvous。<br>

- EtcdRendezvousBackend：取代了旧的 EtcdRendezvousHandler 类。将 EtcdRendezvousBackend 实例传递给 DynamicRendezvousHandler 在功能上等同于实例化一个 EtcdRendezvousHandler。<br>

**Below is a state diagram describing how rendezvous works:**<br>



