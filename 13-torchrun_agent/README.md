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






