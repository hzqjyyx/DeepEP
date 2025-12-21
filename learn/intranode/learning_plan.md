# DeepEP Intranode 两周深度学习计划

> 目标：完全理解 Intranode 模式的设计思想和实现细节

---

## 代码规模评估

```
Python层:    buffer.py (704行，重点关注 dispatch/combine 方法)
C++ Binding: deep_ep.cpp (1893行)
Layout:      layout.cu (153行)
Intranode:   intranode.cu (1103行)
  - notify_dispatch: ~150行
  - dispatch kernel: ~410行
  - combine kernel:  ~410行
工具代码:    buffer.cuh (131行) + utils.cuh (640行) + api.cuh (350行)
测试:        test_intranode.py (311行)

核心需要精读: ~1800行
辅助理解代码: ~1200行
```

---

## 无 GPU 环境的学习策略

如果没有 8 卡 NVLink 环境：

1. **纯读代码 + 画图 trace**：本计划的核心方法，不依赖运行
2. **跳过编译和运行测试**：Day 1 的环境准备改为理解编译流程
3. **重点在理解，不在调试**：所有"验证"改为"纸上推演"
4. **可选：单卡模拟**：用 `num_ranks=1` 理解数据流，但无法验证通信

---

## Week 1: 从数据流动到基础通信

### Day 1: 概念建模 + 测试用例

**上午：核心概念理解（3小时）**

1. MoE dispatch/combine 的语义
   - Dispatch = scatter：把 token 按 expert ID 分发到不同 rank
   - Combine = reduce：把同一个 token 从各 expert 收集回来，加权求和

2. 画出 8-rank 的数据流图
   - 假设 4096 tokens, 256 experts, top-8
   - 每个 rank 有 32 个 local experts
   - 一个 token 可能选中多个 rank 的 experts

3. NVLink 拓扑理解
   - 单节点 8 卡通过 NVLink 全连接（~900GB/s 总带宽）
   - CUDA IPC：一个 GPU 可以直接访问另一个 GPU 的显存
   - 理解为什么需要 "shared buffer"

**下午：精读测试用例（3小时）**

精读 `tests/test_intranode.py`，理解测试逻辑：

1. **测试数据构造**：找到 `topk_idx` 的生成方式
   - 随机选择 expert IDs
   - 理解 `num_tokens`, `num_experts`, `num_topk` 的关系

2. **CPU Reference 实现**：找到 `get_dispatch_layout` 的 Python reference
   - 这是理解 layout 语义的最佳入口
   - 重点理解 `is_token_in_rank` 矩阵的含义

3. **核心验证函数**：理解 `check_data` 函数
   - 它验证了什么？
   - 为什么能用这种方式验证正确性？

**练习**：在纸上画 2-rank dispatch
- Rank 0 有 2 个 token，Rank 1 有 3 个 token
- 每个 token 选中的 experts 可能跨 rank
- 标注每个 token 最终去了哪个 rank

---

### Day 2: Layout 计算 —— 第一个 CUDA Kernel

**上午：精读 `layout.cu`（3小时）**

打开 `csrc/kernels/layout.cu`，逐行理解 `get_dispatch_layout` kernel：

1. **Kernel 参数**
   - 输入：`topk_idx` - 每个 token 选中的 expert IDs
   - 输出：`num_tokens_per_rank`, `num_tokens_per_expert`, `is_token_in_rank`

2. **Shared Memory 统计**
   - 找到 `num_tokens_per_expert_per_thread` 的定义
   - 理解为什么需要 per-thread 统计？（避免原子操作）
   - 理解 reduce 求和的过程

3. **Rank 统计**
   - 找到 `is_in_rank` 数组的使用
   - 根据 expert ID 计算 rank ID：`rank_id = expert_id / num_experts_per_rank`
   - `is_token_in_rank` 是 `[num_tokens, num_ranks]` 的 bool 矩阵

**下午：验证理解（2小时）**

写一个 Python 版本的 `get_dispatch_layout`：

```python
def cpu_get_dispatch_layout(topk_idx, num_experts, num_ranks):
    num_tokens, num_topk = topk_idx.shape
    num_experts_per_rank = num_experts // num_ranks

    num_tokens_per_rank = torch.zeros(num_ranks, dtype=torch.int)
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int)
    is_token_in_rank = torch.zeros(num_tokens, num_ranks, dtype=torch.bool)

    for token_id in range(num_tokens):
        for k in range(num_topk):
            expert_id = topk_idx[token_id, k].item()
            rank_id = expert_id // num_experts_per_rank

            num_tokens_per_expert[expert_id] += 1
            if not is_token_in_rank[token_id, rank_id]:
                is_token_in_rank[token_id, rank_id] = True
                num_tokens_per_rank[rank_id] += 1

    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank
```

对比测试代码的 reference 实现，验证你的理解。

**关键问题**：
- `is_token_in_rank` 为什么是 bool 而不是 int？（一个 token 发给某 rank 一次就够）
- 为什么需要单独输出 `num_tokens_per_expert`？（后续 MoE 计算需要）
- Shared memory 的作用是什么？（避免 global memory 原子操作）

---

### Day 3: Buffer 初始化 + 核心数据结构

**上午：理解 CUDA IPC（2小时）**

1. 精读 `csrc/deep_ep.cpp` 的 Buffer 构造函数
   - 找到 `barrier_signal`, `buffer_ptrs` 等元数据的分配
   - 找到 IPC handle 的获取（`get_mem_handle`）

2. 精读 `deep_ep/buffer.py` 的 `__init__` 方法
   - 找到所有 rank 交换 device_id 的逻辑
   - 找到所有 rank 交换 IPC handle 的逻辑
   - 找到 `sync` 方法的调用（打开远程 buffer）

**下午：精读 `buffer.cuh`（2小时）**

这是核心数据结构定义，精读 `csrc/kernels/buffer.cuh`：

1. **Buffer 指针数组**
   - `buffer_ptrs`: 指向各 rank 的 NVL buffer
   - `barrier_signal_ptrs`: 指向各 rank 的 barrier signal

2. **Dispatch Handle**
   - 找到 `DispatchHandle` 或类似结构的定义
   - 理解 handle 缓存了什么信息

**下午：画内存布局图（1小时）**

假设 2-rank，画出以下内存布局：

```
Rank 0 的 buffer (NVLink 可见):
+------------------+
| NVL buffer       | <- 实际数据传输区域
| (num_nvl_bytes)  |
+------------------+
| barrier_signals  | <- 用于跨 GPU 同步
+------------------+
| buffer_ptrs      | <- 指向所有 rank 的 buffer 基地址
+------------------+

Rank 1 可以通过 buffer_ptrs[0] 直接读写 Rank 0 的 NVL buffer
（这就是 NVLink 的魔力）
```

**关键输出**：
- `buffer_ptrs[i]` 存储的是 Rank i 的 NVL buffer 基地址
- `barrier_signal_ptrs[i]` 用来做跨 GPU 的 barrier 同步
- 需要在 GPU 上存储 `buffer_ptrs_gpu` 是因为 kernel 需要访问

---

### Day 4: Notify Dispatch + Barrier 机制

**上午：理解 Barrier（2小时）**

精读 `csrc/kernels/utils.cuh` 的 `barrier_block` 函数：

1. 找到 `memory_fence()` 调用 - 为什么需要？
2. 找到 `atomicAdd_system` / `atomicSub_system` - system scope 原子操作
3. 找到 `ld_volatile_global` - 和普通 load 有什么区别？

理解 CUDA 内存模型的基础：
- System-scope vs GPU-scope：跨 GPU 可见性
- 为什么 volatile load 不够？（编译器优化）

**下午：Notify Dispatch Kernel（3小时）**

精读 `csrc/kernels/intranode.cu` 的 `notify_dispatch` kernel：

1. **SM 0 的工作**（找到 `if (sm_id == 0)` 分支）
   - 第一次 Barrier：等待所有 rank 准备好
   - 写统计信息到共享 buffer
   - 第二次 Barrier：等待所有 rank 写完
   - 计算 prefix sum，返回 CPU 需要的总数

2. **其他 SM 的工作**（找到 `else` 分支）
   - 每个 SM 负责一个目标 rank
   - 统计发给该 rank 的 token 数，按 channel 分组
   - 计算 channel 的 prefix sum

**关键数据结构**：

```cpp
rank_prefix_matrix[i][j]: 前 i 个 rank 发给 rank j 的累计 token 数
channel_prefix_matrix[i][c]: 前 c 个 channel 发给 rank i 的累计 token 数
```

**关键问题**：
1. 为什么需要 prefix sum 而不是直接用 count？（快速定位写入位置）
2. CPU 为什么要等 `moe_recv_counter_mapped` 的信号？（需要知道接收多少 token）

---

### Day 5: API 层 + Launch Wrapper

**上午：精读 `api.cuh`（3小时）**

精读 `csrc/kernels/api.cuh`，理解 kernel launch 的封装：

1. **SWITCH_RANKS 宏**
   - 找到 `SWITCH_RANKS` 的定义
   - 理解如何根据 `num_ranks` 选择模板实例化
   - 为什么需要编译时确定 `num_ranks`？（性能优化）

2. **Launch 配置**
   - 找到 `SETUP_LAUNCH_CONFIG` 宏
   - 理解 `num_sms` 参数如何影响 grid 大小
   - 理解 `num_threads` 参数的选择

3. **Dispatch 的完整调用链**
   ```
   buffer.dispatch() [Python]
     -> deep_ep_cpp.dispatch() [C++ binding]
       -> dispatch() [api.cuh - launch wrapper]
         -> dispatch<kNumRanks, kNumThreads, ...>() [intranode.cu - kernel]
   ```

**下午：理解 Channel 抽象（2小时）**

在 `api.cuh` 和 `intranode.cu` 中找到 channel 相关逻辑：

1. **Channel 数量**
   - `num_channels = num_sms / 2`（一半 sender，一半 receiver）
   - 每个 channel 是一个独立的传输通道

2. **Channel 任务分配**
   - 找到 `get_channel_task_range` 或类似函数
   - 理解如何把 tokens 分配到各 channel

**关键问题**：
- 为什么需要 Channel 抽象？（并行度 + 负载均衡）
- Channel 数量如何影响性能？

---

### Day 6-7: 周末复盘 + 预习

**Day 6：总结 Week 1（3小时）**

1. 画出完整的调用链
   ```
   buffer.dispatch()
     -> get_dispatch_layout()  [layout kernel]
     -> notify_dispatch()       [同步 + prefix sum]
     -> dispatch kernel         [实际数据传输]
     -> CPU 读取接收计数
     -> 返回 recv_x, handle
   ```

2. 回答核心问题：
   - CUDA IPC 如何让一个 GPU 访问另一个 GPU 的显存？
   - Barrier 如何实现跨 GPU 同步？
   - Prefix sum 如何帮助快速定位数据？

3. 整理笔记，标记不理解的地方

**Day 7：预习 Dispatch Kernel（3小时）**

快速浏览 `intranode.cu` 的 `dispatch` kernel：

1. **识别整体结构**
   - Sender SM：偶数编号的 block
   - Receiver SM：奇数编号的 block
   - 找到 `if (sm_id % 2 == 0)` 或类似分支

2. **识别 Queue 机制**
   - 找到 `head_idx`, `tail_idx` 相关变量
   - 理解为什么需要 queue（流量控制）

3. **不要求理解细节**，只需看出结构

---

## Week 2: 核心通信 Kernel 深度剖析

### Day 8: Dispatch Kernel —— Buffer 布局和 Queue 机制

**全天精读 Dispatch Kernel 的准备工作（5小时）**

1. **Buffer 布局**（在 dispatch kernel 开头找到）

   ```cpp
   // 每个 channel 有一个队列
   channel_start_offset   // 发送窗口起始
   channel_end_offset     // 发送窗口结束
   channel_head_idx       // 队列头（receiver 更新）
   channel_tail_idx       // 队列尾（sender 更新）

   // 实际数据 buffer
   channel_x_buffers      // Token 数据
   channel_src_idx_buffers    // Token 原始索引
   channel_topk_idx_buffers   // Expert IDs
   ```

2. **Queue 状态变化**

   画出 Queue 的状态变化图：
   - 初始状态：head=0, tail=0
   - Sender 写入 3 个 tokens：tail=3
   - Receiver 读取 2 个 tokens：head=2
   - 如何判断队列空？`head == tail`
   - 如何判断队列满？`tail - head >= buffer_size`

3. **Sender/Receiver SM 分工**

   找到 SM 分配逻辑：
   - 偶数 SM 做 sender
   - 奇数 SM 做 receiver
   - 每对 sender/receiver 负责一个 channel

---

### Day 9: Dispatch Kernel —— Sender 逻辑

**上午：Sender 核心逻辑（3小时）**

在 dispatch kernel 中找到 sender 部分（`sm_id % 2 == 0` 分支）：

1. **流量控制**
   - 找到检查队列剩余空间的逻辑
   - 理解 `num_max_send_tokens` 的作用（最坏情况估计）
   - 为什么要悲观估计而不是精确计算？（减少同步开销）

2. **数据写入**
   - 找到 `send_head` 的保存（combine 时需要）
   - 找到 `UNROLLED_WARP_COPY` 宏的使用
   - 理解如何转换 expert ID 为 local expert ID

3. **队列推进**
   - 找到 `st_release_sys_global` 更新 `tail_idx`
   - 为什么用 release 语义？（保证数据写入对 receiver 可见）

**下午：Sender 细节（2小时）**

1. **Warp-level 操作**
   - 理解 `UNROLLED_WARP_COPY` 宏如何展开
   - 为什么用 warp-level copy 而不是 thread-level？

2. **负数编码**
   - 找到 start/end offset 的负数编码
   - 理解为什么用负数表示特殊状态

---

### Day 10: Dispatch Kernel —— Receiver 逻辑

**上午：Receiver 核心逻辑（3小时）**

在 dispatch kernel 中找到 receiver 部分：

1. **等待数据**
   - 找到读取 `channel_start/end_offset` 的逻辑
   - 找到等待 `tail_idx` 更新的循环

2. **拉取数据**
   - 找到 `ld_acquire_sys_global(channel_tail_idx)`
   - 为什么用 acquire 语义？（保证看到 sender 写入的数据）

3. **拷贝循环**
   - 找到从远程 buffer 拷贝到本地的逻辑
   - 找到更新 `head_idx` 的代码

**下午：Trace Queue 操作（2小时）**

手动 trace 一次完整的 dispatch：
- 假设 rank 0 有 3 个 tokens 发给 rank 1
- Channel buffer 大小 = 2
- 画出每个 cycle 的 head/tail 变化
- 标注何时 sender 阻塞，何时 receiver 阻塞

---

### Day 11: Combine Kernel —— Sender

**全天精读 Combine 的 Sender（5小时）**

找到 `combine` kernel 的 sender 部分：

1. **与 Dispatch Sender 的对比**
   - Dispatch 发送：原始数据 + expert ID
   - Combine 发送：处理后的数据（从 expert 输出）

2. **任务分配**
   - 找到根据 `rank_prefix_matrix` 计算 offset 的逻辑
   - 找到根据 `channel_prefix_matrix` 计算范围的逻辑

3. **数据发送**
   - 为什么只发送 `src_idx`？（告诉对方数据放回哪里）
   - 对比 dispatch 的 sender，有何不同？

---

### Day 12: Combine Kernel —— Receiver + Reduce

**上午：Combine Receiver（3小时）**

找到 combine kernel 的 receiver 部分：

1. **Queue Head 管理**
   - 找到更新 head 的 warp
   - 为什么需要 per-warp 的 head？

2. **等待数据**
   - 找到读取 `send_head`（dispatch 时保存的）
   - 找到等待对应 slot 数据到达的逻辑

**下午：Reduce 逻辑（3小时）**

找到核心 reduce 部分：

1. **核心 Reduce**
   - 找到读取 bias/weight 的逻辑
   - 找到从多个 rank 的 buffer 读数据的逻辑
   - 找到 FP32 累加 + 转回 BF16 的逻辑

2. **TMA 优化**（如果使用 SM90）
   - 找到 `tma_load_1d` / `tma_store_1d` 的使用
   - 理解 multi-stage pipeline 的作用
   - `tma_store_wait<kNumStages - 1>` 的作用？

---

### Day 13: 特殊情况和工具函数

**上午：Cached 模式（2小时）**

理解 `cached_notify_dispatch` 和 cached dispatch：

1. 找到 cached 相关的 kernel
2. 为什么需要 cached 模式？（重复 dispatch 相同 layout）
3. Handle 里缓存了什么？
4. 什么时候不能用 cached 模式？

**下午：精读 `utils.cuh`（3小时）**

1. **Acquire/Release 语义**
   - 找到 `st_relaxed_sys_global` vs `st_release_sys_global`
   - 找到 `ld_acquire_sys_global` vs `ld_volatile_global`
   - 画出各种 fence 的可见性范围

2. **"Undefined Behavior" PTX**
   - 找到 `LD_NC_FUNC = "ld.global.nc.L1::no_allocate.L2::256B"`
   - 为什么技术上是 UB？（见 README Undefined-behavior 部分）
   - 为什么实际 work？

3. **Warp 操作**
   - `elect_one_sync()` - 选出 warp leader
   - `warp_reduce_sum` - warp 内 reduce
   - `UNROLLED_WARP_COPY` 宏展开

---

### Day 14: 总结 + 实践练习

**上午：完整代码 walkthrough（3小时）**

从 Python 入口到 CUDA kernel，完整 trace 一次：

1. `buffer.dispatch(x, topk_idx, ...)` [Python]
2. `get_dispatch_layout` kernel - 计算 layout
3. `notify_dispatch` kernel - 同步，计算 prefix sum
4. `dispatch` kernel - sender/receiver 传输数据
5. CPU 读取 `moe_recv_counter_mapped`
6. 返回 `recv_x`, `handle`

**下午：回答核心问题（3小时）**

1. **设计问题**
   - 为什么用 Queue 而不是直接拷贝？
   - Channel 数量如何影响性能？
   - Prefix sum 的优势是什么？

2. **实现问题**
   - Sender/Receiver 如何避免 race condition？
   - Barrier 的实现原理？
   - TMA vs 普通 copy 的性能差异？

3. **调优问题**
   - `num_sms` 参数如何选择？
   - `nvl_chunk_size` 的含义？
   - 为什么有 auto-tuning？

**可选实验（如有 GPU 环境）**

1. 修改 `num_sms` 参数，观察性能变化
2. 修改 `nvl_chunk_size`，理解其作用
3. 禁用 TMA（`DISABLE_SM90_FEATURES=1`），对比性能

---

## 学习资源

### 必读文档
1. CUDA Programming Guide - Memory Model 章节
2. PTX ISA 文档 - 内存操作指令（ld/st 的 scope 和 ordering）
3. 本项目 README 的 Undefined-behavior 部分

### 调试技巧
1. 在 kernel 里加 `printf`（小心性能影响）
2. 使用 `cudaDeviceSynchronize()` + `cudaGetLastError()`
3. 画状态机图（特别是 Queue 的状态变化）

### 常见陷阱
1. 忽略内存序（memory ordering）
2. 混淆 `num_ranks` 和 `num_nvl_ranks`
3. 没理解 Channel 的范围划分
4. 忽略 FP8 的 scale 传输

---

## 成功标准

完成两周学习后，你应该能够：

**理论层面**
- [ ] 解释 dispatch/combine 的完整数据流
- [ ] 说明 Queue 机制如何避免 race condition
- [ ] 理解 Acquire/Release 内存语义的作用
- [ ] 解释 Prefix Sum 如何加速通信
- [ ] 画出 Dispatch Kernel 的 SM 分工图（哪些 SM 做 sender，哪些做 receiver）

**代码层面**
- [ ] 独立读懂任意一个 intranode kernel
- [ ] 解释 `SWITCH_RANKS` 宏如何展开
- [ ] 修改 `num_sms` 并预测性能影响
- [ ] 用 Python 实现简化版的 dispatch 逻辑
- [ ] 调试基本的通信错误（如 timeout）

**实践层面**
- [ ] 在测试中添加新的参数组合
- [ ] 修改 kernel 参数并重新编译
- [ ] 分析 auto-tuning 的结果
- [ ] 能够解释某段代码为什么这样写（而不是那样写）

---

## 下一步

掌握 Intranode 后，你可以：
1. 学习 Internode 模式（增加 RDMA 通信）
2. 学习 Low-latency 模式（CUDA Graph 兼容）
3. 深入 NVSHMEM 库的使用
4. 研究其他 MoE 通信库（如 Megatron-LM）

祝学习顺利！
