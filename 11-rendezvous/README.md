# 0 概念
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Torch Distributed Elastic的上下文中，我们使用术语**rendezvous(汇合点)**来指代一种特定的功能，该功能将分布式同步原语(distributed synchronization primitive)与对等点发现(peer discovery)结合在一起。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic使用rendezvous来收集训练作业（即节点）的参与者(participants)，使它们都能就相同的参与者列表和各自的角色达成一致，并就何时可以开始/恢复训练做出一致性(consistent)的集体决策。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Torch Distributed Elastic的rendezvous提供了以下关键功能：<br>

# 1 Barrier
