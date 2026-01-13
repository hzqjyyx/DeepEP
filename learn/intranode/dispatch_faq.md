# DeepEP Intranode Dispatch FAQ

## 基础概念

### Q1: thread_id 是单个 GPU 内的变量，为什么代码里按全部 rank 来分组？

这是**软件层面的分组**，不是硬件限制。

每个 SM 有 768 个线程，代码把它们按 rank 数量分组：

```
假设 8 个 rank：
    Thread 0~95:    负责 Rank 0
    Thread 96~191:  负责 Rank 1
    Thread 192~287: 负责 Rank 2
    ...
    Thread 672~767: 负责 Rank 7
```

这样做的原因是：一个 Sender SM 需要同时往 8 个不同的 rank 发送数据。把线程分成 8 组，每组负责往一个目标 rank 发送，可以并行处理。

---

### Q2: Buffer 内部布局是什么意思？每个字段是 int 还是指针？

Buffer 是一块**连续的 GPU Global Memory（HBM）**，不是结构体。各个字段顺序存放在这块内存中，通过指针偏移访问：

```
buffer_ptrs[rank] ──►┌─────────────────────────┐
                     │ rank_prefix_matrix      │  ← int 数组
                     ├─────────────────────────┤
                     │ channel_start_offset    │  ← int 数组
                     ├─────────────────────────┤
                     │ channel_end_offset      │  ← int 数组
                     ├─────────────────────────┤
                     │ channel_head_idx        │  ← int 数组
                     ├─────────────────────────┤
                     │ channel_tail_idx        │  ← int 数组
                     ├─────────────────────────┤
                     │ channel_x_buffers       │  ← int4 数组（token 数据）
                     ├─────────────────────────┤
                     │ channel_src_idx         │  ← int 数组
                     ├─────────────────────────┤
                     │ ...                     │
                     └─────────────────────────┘
```

代码中的 `Buffer<T>` 类是一个工具，帮助计算每个字段的偏移量并顺序分配内存。

---

### Q3: Buffer 放在哪里？是 SM 的 Shared Memory 吗？

**不是 Shared Memory，是 Global Memory（HBM）**。

两者的关键区别：

| 特性 | Shared Memory | Global Memory (HBM) |
|-----|---------------|---------------------|
| 可见范围 | 只有本 SM 能看到 | 本 GPU 所有 SM 能看到 |
| 跨 GPU 访问 | 不可以 | 可以（通过 NVLink + CUDA IPC） |
| 容量 | 几十 KB | 几十 GB |

DeepEP 的 Buffer 放在 Global Memory，这样其他 GPU 可以通过 NVLink 直接写入。这是整个跨 GPU 通信的基础。

---

## 设计原理

### Q4: 为什么需要 Receiver？Sender 直接写到输出 recv_x 不行吗？

不行，有两个原因：

**原因 1：写入位置需要全局协调**

```
8 个 rank 同时往 Rank 0 发数据，如果 Sender 直接写 recv_x：

    Rank 1 的 Channel 0 想写 recv_x[???]  ← 写到哪？
    Rank 2 的 Channel 0 想写 recv_x[???]  ← 写到哪？
    Rank 3 的 Channel 1 想写 recv_x[???]  ← 写到哪？
    ...

每个 Sender 需要知道其他所有 rank、所有 channel 发了多少数据，
这需要 O(ranks × channels) 的全局同步，代价太高。

用中间 buffer 的好处：
    - 每个 (channel, src_rank) 有独立的 buffer 区域
    - Sender 只管往自己的区域写，不用协调
    - Receiver 在本地，知道所有信息，可以高效计算写入位置
```

**原因 2：流量控制**

```
循环队列提供天然的流量控制：
    - buffer 大小有限，Sender 写满会等待
    - Receiver 控制消费速度
    - 不会出现写覆盖未读数据的情况

如果直接写 recv_x，没有这种 back-pressure 机制。
```

所以 Receiver 不只是"拷贝"，而是**按正确顺序归并多个来源的数据**。

---

### Q5: Channel 是什么含义？

Channel 是把 token 分组并行传输的抽象。

```
假设 4096 个 token，4 个 channel：
    Channel 0: 负责 token 0~1023
    Channel 1: 负责 token 1024~2047
    Channel 2: 负责 token 2048~3071
    Channel 3: 负责 token 3072~4095

每个 channel 由一对 SM 负责：
    SM 0 (Sender)  + SM 1 (Receiver)  → Channel 0
    SM 2 (Sender)  + SM 3 (Receiver)  → Channel 1
    SM 4 (Sender)  + SM 5 (Receiver)  → Channel 2
    SM 6 (Sender)  + SM 7 (Receiver)  → Channel 3
```

每个 channel 有独立的 head/tail 指针和 buffer 区域，可以完全并行工作，提高吞吐量。

Channel 数量 = `num_sms / 2`，默认 `num_sms = 20`，所以默认有 10 个 channel。

---

### Q6: channel_start_offset 和 token_start_idx 有什么区别？

这是两个不同层面的概念：

```
token_start_idx / token_end_idx：
    本 channel 负责处理哪些 token（输入侧）
    例：Channel 1 负责 token 1024~2047

channel_start_offset / channel_end_offset：
    发给某个 rank 的数据，在目标 rank 的接收 buffer 里从哪开始（输出侧）
    例：Channel 1 发给 Rank 3 的数据，从 Rank 3 接收 buffer 的第 500 位开始
```

为什么需要 `channel_start_offset`？因为多个 channel 同时往同一个 rank 发数据，需要告诉 Receiver 每个 channel 的数据从哪个位置开始。

---

### Q7: 本地 token 为什么也要"发送"？不能直接在原地计算 FFN 吗？

不能，因为 **MoE 需要按 expert 分组计算，不是按 token 顺序计算**。

```
原始数据布局（按 token 顺序）：
┌───────────────────────────────────────────────────────┐
│ token 0 │ token 1 │ token 2 │ token 3 │ ...          │
│ (exp 0,2)│ (exp 1,3)│ (exp 0,1)│ (exp 2,3)│           │
└───────────────────────────────────────────────────────┘

MoE FFN 需要的布局（按 expert 分组）：
┌───────────────────────────────────────────────────────┐
│ Expert 0 的所有 token │ Expert 1 的所有 token │ ...   │
│ token 0, token 2      │ token 1, token 2      │       │
└───────────────────────────────────────────────────────┘
```

如果不重排数据：

1. **无法 batch 计算**：同一个 expert 处理的 token 在内存中不连续，每个 token 单独计算效率极低
2. **跨 rank 的 expert 根本在别的 GPU 上**：必须把数据传过去才能计算

所以 dispatch 的本质是 **scatter + reorder**：把 token 发送到正确的 GPU，并按 expert 分组排列。

---

## 高级话题

### Q8: 同一 GPU 内的 Sender 和 Receiver 之间的通信，能用 Distributed Shared Memory (DSMEM) 加速吗？

理论上可以。

同一 GPU 上的数据流：

```
Sender SM 0                    Receiver SM 1
(Channel 0)                    (Channel 0)
     │                              ▲
     │ 写入                         │ 读取
     ▼                              │
┌────────────────────────────────────┐
│  本 rank 的 Buffer (Channel 0)     │  ← Global Memory (HBM)
└────────────────────────────────────┘
```

本地 Sender 写的数据，本地 Receiver 要读取。这部分数据流可以用 DSMEM 直接在 SM 之间传递，省掉 HBM 的写入和读取。

但 DeepEP 没有做这个优化，可能的原因：
1. DSMEM 需要 Thread Block Cluster，增加代码复杂度
2. 需要区分本地和远程两条路径
3. 跨 GPU 通信才是主要瓶颈

---

### Q9: 远程数据能直接写到 DSMEM 吗？

不能。**NVLink 只能访问 Global Memory 地址空间**。

```
远程 GPU 通过 NVLink 写数据：
    GPU 1 → NVLink → GPU 0 的 ???

可以写到：
    ✓ GPU 0 的 Global Memory (HBM)
    ✗ GPU 0 的 Shared Memory（没有暴露给远程 GPU）
    ✗ GPU 0 的 DSMEM（只在同一 GPU 内的 SM 间共享）
```

所以远程数据必须先落到 HBM，然后 Receiver 从 HBM 读取。在这个路径上加一层 DSMEM 搬运没有收益（每个数据只读一次）。

---

### Q10: DeepEP 用了 Hopper 的哪些特性？

主要用了 **TMA (Tensor Memory Accelerator)**：

```cuda
// intranode.cu 中的 TMA 使用
tma_load_1d(tma_buffer, src, tma_mbarrier, bytes);
tma_store_1d(tma_buffer, dst, bytes, false);
```

TMA 是 Hopper 的异步内存拷贝单元，可以在不占用 SM 计算资源的情况下进行内存传输。

没有使用的 Hopper 特性：
- Distributed Shared Memory (DSMEM)
- Thread Block Cluster

---

## 配置相关

### Q11: Channel 数量是多少？可以调整吗？

```python
# 默认配置
num_sms = 20  # 参与 dispatch 的 SM 数量
num_channels = num_sms / 2 = 10  # channel 数量

# 可以通过 API 调整
Buffer.set_num_sms(new_value)  # 必须是偶数
```

不同的 rank 数量有推荐配置，可以通过 `Buffer.get_dispatch_config(num_ranks)` 获取。

---

### Q12: 循环队列的 buffer 大小是多少？

`num_recv_buffer_tokens` 参数控制，通过 `get_dispatch_config()` 获取推荐值：

```python
# 8 ranks 的默认配置
Config(num_sms=20, ..., num_recv_buffer_tokens=256, ...)
```

Buffer 太小会导致频繁的流量控制等待，太大会浪费显存。

---

## 调试相关

### Q13: 如何判断通信是否出现问题？

代码中有超时检测机制：

```cuda
// intranode.cu
if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf("DeepEP timeout for dispatch senders/receivers, rank %d, ...\n", rank, ...);
    trap();
}
```

如果看到 timeout 错误，可能的原因：
1. 某个 rank 的 kernel 没有启动
2. Barrier 同步失败
3. NVLink 通信问题

---

### Q14: 内存序（memory ordering）为什么重要？

```cuda
// Sender 写完数据后更新 tail
st_release_sys_global(channel_tail_idx, tail);

// Receiver 读取 tail
tail = ld_acquire_sys_global(channel_tail_idx);
```

Release-Acquire 配对保证：
- Sender 在 release 之前写入的数据
- Receiver 在 acquire 之后一定能看到

如果不用这种同步，可能出现：Receiver 看到 tail 更新了，但数据还没写完（可见性问题）。

`sys` 后缀表示 system scope，保证跨 GPU 可见。这是 NVLink 通信正确性的关键。
