# 0 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Torch Distributed Elastic的上下文中，我们使用术语**rendezvous(汇合点)**来指代一种特定的功能，该功能将分布式同步原语(distributed synchronization primitive)与对等点发现(peer discovery)结合在一起。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic使用rendezvous来收集训练作业（即节点）的参与者(participants)，使它们都能就相同的参与者列表和各自的角色达成一致，并就何时可以开始/恢复训练做出一致性(consistent)的集体决策。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic的rendezvous提供了以下关键功能：<br>

# 1 Barrier
执行集合操作的节点都会阻塞，直到集合被视为完成——这发生在至少有最小总数的节点加入了集合屏障（针对同一作业）时。这也意味着屏障的大小并不一定是固定的。

在达到最小节点数后，还会有一个额外的短暂等待时间——这是为了确保集合不会“太快”完成（这可能会排除那些几乎同时尝试加入的额外节点）。

如果屏障处聚集的节点数达到了最大值，则集合立即完成。

此外，还有一个总体超时设置，如果在超时时间内未达到最小节点数，则集合操作失败。这是一个简单的故障安全机制，用于在资源管理器出现问题时帮助释放部分分配的作业资源，且此失败被视为不可重试的。
