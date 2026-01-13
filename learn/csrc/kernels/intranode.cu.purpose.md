# intranode.cu 核心功能解析

这个文件实现了基于 NVLink 的节点内 (intranode) MoE all-to-all 通信，包含三组主要操作。

## 核心设计思想

### 生产者-消费者模型

所有通信都基于**循环队列**实现：
- **head 指针**：消费者（接收方）控制，表示"已处理到哪"
- **tail 指针**：生产者（发送方）控制，表示"已写入到哪"
- **流控**：发送方检查 `tail - head < 队列容量` 才能继续写入

### 偶数/奇数 SM 分离

dispatch 和 combine kernel 都使用 `num_sms` 个 SM：
- **偶数 SM (sm_id % 2 == 0)**：发送方
- **奇数 SM (sm_id % 2 == 1)**：接收方
- **channel = sm_id / 2**：每对 SM 负责一个通信 channel

这种分离避免了单 SM 内的发送/接收死锁。

### 数据存储位置

**缓冲区始终存储在接收方的内存中**：
- dispatch: rank A 发送 → 写入 `buffer_ptrs[B]` → rank B 接收
- combine: rank B 发送 → 写入 `buffer_ptrs[A]` → rank A 接收

---

## 1. notify_dispatch - 元数据交换

在实际数据传输前，各 rank 需要交换"要发送多少 token"的信息。

### 标准版 `notify_dispatch` (L11-159)

**Grid 配置**: `(1 + num_ranks)` 个 SM

**SM 0 (协调器)**:
1. 第一次 barrier：确保所有 rank 就绪
2. 每个 rank 将本地统计写入共享 buffer：
   - `per_rank_buffer[rank][dst_rank]` = 本 rank 发往 dst_rank 的 token 数
   - `per_expert_buffer[rank][expert]` = 本 rank 发往该 expert 的 token 数
3. 第二次 barrier：等待所有 rank 写入完成
4. 列向前缀和：计算每个 rank 累计接收的 token 数
5. 写回 CPU：通过 mapped memory 返回本 rank 的接收总数
6. 拷贝前缀和矩阵、清空队列元数据
7. 第三次 barrier

**SM 1~num_ranks (分发统计器)**:
- 每个 SM 负责一个 `dst_rank = sm_id - 1`
- 并行统计每个 channel 中发往该 rank 的 token 数
- 计算 channel 前缀和，供后续 dispatch 使用

### 缓存版 `cached_notify_dispatch` (L161-195)

当 token 分布已知（如 CUDA Graph 场景）时使用的简化版本：
- 跳过统计和前缀和计算
- 直接拷贝预先计算好的 `rank_prefix_matrix`
- 只需两次 barrier

---

## 2. dispatch - 数据分发

将 token 从源 rank 分发到各 expert 所在的 rank。

### 核心流程 (L197-610)

**Grid 配置**: `num_sms` 个 SM（偶数发送，奇数接收）

#### 发送方 (偶数 SM)

线程分组：`kNumThreads / kNumRanks` 个线程负责一个目标 rank

1. **通知接收方**：写入 `channel_start_offset` 和 `channel_end_offset`
   - 使用 `-value - 1` 编码，区分"值为0"和"未初始化"

2. **分批发送 token**：
   ```
   for token_idx in [token_start, token_end):
       # 外层: 检查队列空间，批量更新 tail
       while 队列有空间:
           # 内层: 发送单个 token
           if token 选中了该 rank:
               拷贝 x, src_idx, topk_idx, topk_weights, x_scales
       st_release_sys_global(tail, new_tail)  # 通知接收方
   ```

3. **topk_idx 转换**：全局 expert ID → 接收 rank 的本地 expert ID

#### 接收方 (奇数 SM)

1. **等待偏移量**：轮询 `channel_start_offset` 和 `channel_end_offset`
2. **接收数据**：
   ```
   while 还有 token 要接收:
       cached_tail = ld_acquire_sys_global(tail)  # 获取最新 tail
       拷贝 [cached_head, cached_tail) 范围的数据
       st_relaxed_sys_global(head, cached_head)   # 释放空间
   ```

3. **TMA 优化 (SM90)**：使用 Tensor Memory Accelerator 加速大块数据拷贝

---

## 3. combine - 结果聚合

将 expert 处理后的结果聚合回原始 token 位置，是 dispatch 的逆操作。

### 核心流程 (L691-1099)

**Grid 配置**: `num_sms` 个 SM（偶数发送，奇数接收）

#### 发送方 (偶数 SM)

与 dispatch 类似，但：
- 数据源是 expert 的输出
- 根据 `src_idx` 确定每个 token 要发回哪个原始位置

#### 接收方 (奇数 SM)

**关键区别**：同一个 token 可能从多个 rank 接收数据，需要 **reduce（求和）**

线程分工：
- **Warp 0**：head updater，监控所有 worker warp 的进度，更新全局 head
- **Warp 1~N**：worker，并行处理 token 的数据 reduce

```
// Worker warp 处理逻辑
for token_idx in [start + warp_id - 1, end) step (num_warps - 1):
    expected_head[lane] = send_head[token][lane]  # 每个 lane 对应一个源 rank
    等待所有需要的 rank 数据到达 (tail >= expected_head)

    result = bias_0 + bias_1
    for each rank with valid data:
        result += recv_buffer[rank][slot]

    write result to recv_x[token]
    update warp_channel_head_idx (通知 warp 0)
```

### 依赖：cached_notify_combine (L612-689)

combine 前的准备工作：
- 清空队列元数据
- **填充 send_head 的缺失值**：对于未发送到某 rank 的 token，用后续有效 head 的负数编码填充

---

## 缓冲区布局

### dispatch 使用的布局

存储在**接收方** (`buffer_ptrs[recv_rank]`) 中，跳过 `rank_prefix_matrix` 后：

```
偏移 0:                                 [rank_prefix_matrix: R×R int]
偏移 R×R×4:                             [channel_start_offset: C×R int]
                                        [channel_end_offset: C×R int]
                                        [channel_head_idx: C×R int]
                                        [channel_tail_idx: C×R int]
                                        [x_buffers: C×R×B×H int4]
                                        [src_idx_buffers: C×R×B int]
                                        [topk_idx_buffers: C×R×B×K topk_idx_t]
                                        [topk_weights_buffers: C×R×B×K float]
                                        [x_scales_buffers: C×R×B×S float]
```

- R: num_ranks
- C: num_channels
- B: num_recv_buffer_tokens (循环队列容量)
- H: hidden_int4
- K: num_topk
- S: num_scales

### combine 使用的布局

存储在**接收方** (`buffer_ptrs[recv_rank]`) 中，**从偏移 0 开始**（无 rank_prefix_matrix）：

```
偏移 0:                                 [channel_head_idx: C×R int]
                                        [channel_tail_idx: C×R int]
                                        [x_buffers: C×R×B×H int4]
                                        [src_idx_buffers: C×R×B int]
                                        [topk_weights_buffers: C×R×B×K float]
```

注意：combine 不需要 `start/end_offset`（数据量已知）和 `topk_idx/x_scales`（combine 只需要聚合 x 和 weights）。

---

## 关键工具类和宏

### Buffer\<T\> 类

**重要行为**：构造函数会移动传入的指针！

```cpp
Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
    total_bytes = num_elems * sizeof(T);
    ptr = gbl_ptr + offset * sizeof(T);  // 返回偏移后的指针
    gbl_ptr += total_bytes;              // 移动原指针！
}
```

这是一个简易的"线性分配器"，连续调用会自动分配连续的内存区域。

### 内存序语义

- `st_release_sys_global`: 发送方写 tail，确保之前的数据写入对接收方可见
- `ld_acquire_sys_global`: 接收方读 tail，确保读到的数据是完整的
- `st_relaxed_sys_global`: 接收方写 head，释放空间（不需要强同步）
- `ld_volatile_global`: 轮询等待时使用，防止编译器缓存优化

### 其他

- `SWITCH_RANKS`: 编译期生成不同 rank 数的模板特化
- `UNROLLED_WARP_COPY`: 循环展开的 warp 级数据拷贝
- `barrier_block<kNumRanks>`: 多 rank 全局屏障同步
- `get_channel_task_range`: 计算 channel 负责的 token 范围

---

## 关键设计细节

### 负数编码技巧

用于区分"值为 0"和"未初始化"：
```cpp
// 编码: value → -(value + 1)
st_relaxed_sys_global(ptr, -value - 1);

// 解码: encoded → -encoded - 1
value = -ld_volatile_global(ptr) - 1;
```

例如：0 → -1, 1 → -2, 10 → -11

### send_head 编码

dispatch 写入的 `send_head[token][rank]`：
- **非负值**：该 token 在循环队列中的位置
- **-1**：该 token 未发送到该 rank

`cached_notify_combine` 会填充 -1 的位置，改为 `-(next_valid_head + 1)`，方便 combine 接收方解析。

### 超时检测

所有轮询循环都有 `NUM_TIMEOUT_CYCLES` 超时保护，防止死锁时无限等待。

---

## 调用关系

```
dispatch 流程:
  notify_dispatch()           // 或 cached_notify_dispatch()
  └→ dispatch()               // 实际数据传输

combine 流程:
  cached_notify_combine()     // 清理 + 填充 send_head
  └→ combine()                // 实际数据传输 + reduce
```
