## 核心功能

这个文件实现了 MoE dispatch 的预处理 kernel，作用是**统计 token 分布**：在知道每个 token 要发送给哪些 expert 之后，计算出：

1. 每个 expert 会收到多少个 token
2. 每个 rank 会收到多少个 token
3. 每个 RDMA rank 会收到多少个 token（可选，用于跨节点通信）
4. 每个 token 会被发送到哪些 rank（bool 矩阵）

## 输入输出

**输入：**
- `topk_idx` [num_tokens, num_topk]：每个 token 选中的 expert ID 列表（DeepSeek 通常 topk=8）

**输出：**
- `num_tokens_per_expert` [num_experts]：每个 expert 收到的 token 数
- `num_tokens_per_rank` [num_ranks]：每个 rank 收到的 token 数（**去重**：同一 token 被一个 rank 的多个 expert 选中只计数一次）
- `num_tokens_per_rdma_rank` [num_rdma_ranks]：每个 RDMA rank 收到的 token 数（可选）
- `is_token_in_rank` [num_tokens, num_ranks]：token i 是否被发送到 rank j

## 核心设计

### 两阶段 Block 分工

Kernel 使用 **分段 block 策略**：

- **前段 block**（sm_id 较小）：统计 expert 分布
  - 每个 block 负责 `kNumExpertsPerSM=4` 个 expert
  - 例如：256 个 expert 需要 64 个 block

- **后段 block**（sm_id 较大）：统计 rank 分布  
  - 每个 block 负责 `kNumRanksPerSM=8` 个 rank
  - 例如：32 个 rank 需要 4 个 block

### 避免全局 Atomic 的优化

**问题：** 如果每个线程直接 `atomicAdd(num_tokens_per_expert[expert_idx], 1)`，会导致严重的全局内存竞争。

**解决方案：** 两级归约

1. **线程级统计**：每个线程维护独立计数器 `num_tokens_per_expert_per_thread[thread_id][local_expert_id]`
2. **Shared memory reduce**：256 个线程的计数器在 shared memory 中求和
3. **一次性写回**：每个 expert 只写一次全局内存

这样把 256×N 次全局 atomic 变成了 N 次普通写入（N 为该 block 负责的 expert 数）。

## 具体流程

### 前段 Block：统计 Expert

```
1. 初始化 shared memory 计数器为 0
2. 每个线程 stride 遍历所有 token：
   - 读取 token i 的 topk expert 列表
   - 如果某个 expert 在本 block 负责范围内，本线程计数器 +1
3. __syncthreads()
4. Reduce：对每个 expert，将 256 个线程的计数器求和
5. 写回全局内存 num_tokens_per_expert
```

### 后段 Block：统计 Rank

```
1. 初始化 shared memory 计数器为 0
2. 每个线程 stride 遍历所有 token：
   - 读取 token i 的 topk expert 列表
   - 根据 expert ID 计算对应的 rank ID（expert 均匀分布到 rank）
   - 对每个 rank 去重标记：is_in_rank[rank_local_id] = 1
   - 同样处理 RDMA rank
3. 写入 is_token_in_rank[i, rank_j]
4. __syncthreads()
5. Reduce：对每个 rank/rdma_rank 求和
6. 写回全局内存
```

### 关键去重逻辑

**为什么需要去重：** 一个 token 可能选中同一个 rank 的多个 expert（比如 rank 0 负责 expert 0-7，token 同时选中 expert 0 和 expert 1），但**通信时只需要发送一次这个 token**。

**实现：** 
- 用局部数组 `is_in_rank[kNumRanksPerSM]` 标记当前 token 是否在某个 rank
- 最后 `num_tokens_per_rank += (is_in_rank[j] > 0)` 确保每个 token 对每个 rank 最多计数一次

## 性能调优参数

- `kNumThreads = 256`：每个 block 的线程数
- `kNumExpertsPerSM = 4`：前段 block 每个负责 4 个 expert
- `kNumRanksPerSM = 8`：后段 block 每个负责 8 个 rank
- `NUM_MAX_NVL_PEERS = 8`：NVLink 节点内通信的最大 peer 数（影响 RDMA rank 分组）

## RDMA Rank 概念

- **Rank**：节点内的 GPU（NVLink 通信）
- **RDMA Rank**：跨节点的通信单元，`NUM_MAX_NVL_PEERS=8` 个 rank 合并为 1 个 RDMA rank
- 例如：32 个 rank = 4 个 RDMA rank（每个负责 8 个 rank 的跨节点通信）

---

**其他内容：**
- 无 ROCm 相关代码
- 无 Backward 实现（这是 forward-only 的统计 kernel）
