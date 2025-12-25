# Layout Kernel Q&A

## Q1: 调用一次 get_dispatch_layout 最多会用几个 SM？

看 `csrc/kernels/layout.cu:165` 的计算逻辑：

```cpp
int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
              (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
```

其中 `kNumExpertsPerSM = 4`，`kNumRanksPerSM = 8` (line 164)。

**答案：**

使用的 SM 数（实际上是 block 数）= **⌈num_experts / 4⌉ + ⌈num_ranks / 8⌉**

拆解：
- **前段 block**：⌈num_experts / 4⌉ 个，每个负责统计 4 个 expert
- **后段 block**：⌈num_ranks / 8⌉ 个，每个负责统计 8 个 rank

**典型场景估算：**
- 假设 `num_experts = 256`, `num_ranks = 32`
  - 前段：256 / 4 = 64 个 block
  - 后段：32 / 8 = 4 个 block
  - 总共：**68 个 SM**

- 假设 `num_experts = 128`, `num_ranks = 16`
  - 前段：128 / 4 = 32 个 block
  - 后段：16 / 8 = 2 个 block
  - 总共：**34 个 SM**

**关键设计：**
- 这个 kernel 故意分成两部分，前段统计 expert，后段统计 rank
- 前段 block 执行完直接 return (line 68)，不会继续执行后段逻辑
- 这种设计避免了单个 kernel 需要大量 shared memory 同时处理两种统计

从代码看没有硬编码的上限，最大 SM 使用量取决于实际的 `num_experts` 和 `num_ranks` 参数。

---

## Q2: Layout 的前段和后段分别是干什么的？为什么要分前段和后段？

### 前段 Block：统计 Expert 维度

**目标**：计算 `num_tokens_per_expert[num_experts]`

**逻辑**（line 38-68）：
```
对于每个 token:
  对于它选中的 num_topk 个 expert:
    num_tokens_per_expert[expert_id]++
```

**特点**：简单累加，不需要去重

---

### 后段 Block：统计 Rank 维度

**目标**：计算 `num_tokens_per_rank[num_ranks]`, `num_tokens_per_rdma_rank[...]`, `is_token_in_rank[num_tokens, num_ranks]`

**逻辑**（line 89-149）：
```
对于每个 token:
  去重标记：这个 token 选中的 topk expert 分别属于哪些 rank
  对于每个涉及的 rank:
    num_tokens_per_rank[rank_id]++  // 注意：每个 token 对每个 rank 只计数一次
    is_token_in_rank[token_id, rank_id] = true
```

**关键差异**：需要**去重**！

看 line 105-122 的逻辑：
```cpp
int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
for (int j = 0; j < num_topk; ++j) {
    expert_idx = ...;
    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
    is_in_rank[rank_idx]++;  // 可能重复标记
}
// 去重：只要 is_in_rank[j] > 0 就计数 1
num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
```

---

### 为什么要分前段和后段？

#### 1. 统计语义不同

- **Expert 维度**：一个 token 可以被多个 expert 处理 → 每个 expert 独立计数
- **Rank 维度**：一个 token 只会被发送到某个 rank 一次，即使它选中该 rank 上的多个 expert → 需要去重

**举例**：
```
假设：
- num_experts = 8, num_ranks = 2
- Expert 0-3 在 Rank 0，Expert 4-7 在 Rank 1
- Token 0 选中 [Expert 1, Expert 2] (都在 Rank 0)

统计结果：
  num_tokens_per_expert[1] = 1
  num_tokens_per_expert[2] = 1
  num_tokens_per_rank[0] = 1  ← 注意：只计数一次，不是 2
```

#### 2. Shared Memory 优化

如果合并成一个 kernel，需要同时分配：
```cpp
__shared__ int num_tokens_per_expert_per_thread[256][4];   // 4KB
__shared__ int num_tokens_per_rank_per_thread[256][8];     // 8KB
__shared__ int num_tokens_per_rdma_rank_per_thread[256][1]; // 1KB
```

分开后每个 block 只用其中一组，节省 shared memory，提高 occupancy。

#### 3. 并行度匹配

- Expert 数量多（256+）→ 需要更多 block
- Rank 数量少（8-32）→ 需要较少 block

分开可以让每部分用最合适的并行度。

---

### 核心设计思路

这个 kernel 本质上是在做 **MoE dispatch 的预处理**：

1. **Expert 统计**：告诉每个 expert "你要处理多少个 token"（用于分配 buffer）
2. **Rank 统计**：告诉每个 rank "你要接收多少个 token"（用于通信 buffer）+ 标记哪些 token 需要发给哪个 rank

前段统计的是**计算维度**（expert），后段统计的是**通信维度**（rank），两者的语义和去重需求完全不同，所以分成两部分独立并行处理。

**这样设计的优势**：
- ✅ 逻辑清晰，避免复杂的条件分支
- ✅ Shared memory 利用率高
- ✅ 并行度可以独立优化

## Q3: get_dispatch_layout的实现在前段 SM 中没有做合并访存，也可以模版化 num_topk，有没有收益呢？

合并访存的版本：
```cuda
for (int i = thread_id; i < num_tokens * num_topk; i += kNumThreads) {         
    int token_id = i / num_topk;                                               
    int topk_id = i % num_topk;      
    auto shifted_topk_idx = topk_idx + token_id * num_topk;                                          
    if (token_id < num_tokens){
        expert_idx = static_cast<int>(shifted_topk_idx[topk_id]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
            ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
    }                                             
}
```

TBD