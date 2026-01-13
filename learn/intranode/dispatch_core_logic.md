# DeepEP Intranode Dispatch 核心逻辑

## 概述

DeepEP 的 Intranode Dispatch 实现了单节点内多 GPU 之间的 MoE token 分发。核心目标是：**把每个 GPU 上的 token 发送到拥有对应 expert 的 GPU 上，并按 expert 分组排列，以便高效进行 batch 计算**。

### 为什么需要 Dispatch？

MoE 模型中，每个 token 会选择 top-k 个 expert 进行计算。但 expert 分布在不同的 GPU 上，所以需要：
1. 把 token 发送到拥有目标 expert 的 GPU
2. 在目标 GPU 上按 expert 分组排列数据
3. 这样同一个 expert 处理的所有 token 在内存中连续，可以高效 batch 计算

### 核心设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      Intranode Dispatch 设计                     │
├─────────────────────────────────────────────────────────────────┤
│  1. 通信介质：NVLink + CUDA IPC                                  │
│     - 每个 GPU 分配一块 Global Memory 作为共享 Buffer            │
│     - 其他 GPU 通过 NVLink 可以直接读写这块 Buffer               │
│                                                                 │
│  2. 数据传输：循环队列                                           │
│     - 用 head/tail 指针协调 Sender 和 Receiver                  │
│     - 天然的流量控制，避免写覆盖未读数据                          │
│                                                                 │
│  3. 并行度：Channel 分组                                         │
│     - 把 token 分成多个 Channel 并行传输                         │
│     - 每个 Channel 由一对 Sender/Receiver SM 负责                │
│                                                                 │
│  4. SM 分工：Sender/Receiver 分离                                │
│     - 偶数 SM 做 Sender（写远程 Buffer）                         │
│     - 奇数 SM 做 Receiver（从本地 Buffer 读取并写入输出）         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 调用链

```
buffer.dispatch()                    [Python: deep_ep/buffer.py]
    │
    ├─► get_dispatch_layout()        [计算 token 分布，哪些 token 去哪些 rank]
    │
    ├─► notify_dispatch()            [GPU 间同步，计算 prefix sum]
    │
    ├─► dispatch()                   [实际数据传输]
    │
    └─► CPU 读取 moe_recv_counter    [获取接收到的 token 数量]
```

---

## 核心数据结构

### Buffer 内存布局

每个 GPU 分配一块共享 Buffer，内部顺序存放多个字段：

```
buffer_ptrs[rank] 指向的内存区域：

┌──────────────────────────────────────────────────────────────┐
│ rank_prefix_matrix                                           │
│   大小：num_ranks × num_ranks × sizeof(int)                  │
│   含义：rank_prefix_matrix[i][j] = 前 i 个 rank 发给 rank j  │
│         的累计 token 数                                       │
├──────────────────────────────────────────────────────────────┤
│ channel_start_offset                                         │
│   大小：num_channels × num_ranks × sizeof(int)               │
│   含义：每个 (channel, src_rank) 的起始偏移                   │
├──────────────────────────────────────────────────────────────┤
│ channel_end_offset                                           │
│   大小：num_channels × num_ranks × sizeof(int)               │
│   含义：每个 (channel, src_rank) 的结束偏移                   │
├──────────────────────────────────────────────────────────────┤
│ channel_head_idx                                             │
│   大小：num_channels × num_ranks × sizeof(int)               │
│   含义：循环队列的 head 指针（Receiver 更新）                  │
├──────────────────────────────────────────────────────────────┤
│ channel_tail_idx                                             │
│   大小：num_channels × num_ranks × sizeof(int)               │
│   含义：循环队列的 tail 指针（Sender 更新）                    │
├──────────────────────────────────────────────────────────────┤
│ channel_x_buffers                                            │
│   大小：num_channels × num_ranks × buffer_size × hidden      │
│   含义：token 数据的循环缓冲区                                │
├──────────────────────────────────────────────────────────────┤
│ channel_src_idx_buffers                                      │
│   含义：源 token 索引（combine 时需要知道数据来自哪个 token） │
├──────────────────────────────────────────────────────────────┤
│ channel_topk_idx_buffers                                     │
│   含义：转换后的 local expert ID                              │
├──────────────────────────────────────────────────────────────┤
│ channel_topk_weights_buffers                                 │
│   含义：expert 权重                                          │
├──────────────────────────────────────────────────────────────┤
│ channel_x_scales_buffers                                     │
│   含义：FP8 的 scale 值                                      │
└──────────────────────────────────────────────────────────────┘
```

### 循环队列机制

```
循环队列状态：

    head                tail
      ↓                  ↓
┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │ A │ B │ C │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┴───┘
      ↑───────────↑
       Receiver 还没读的数据

操作：
    - Sender 写入后：tail++
    - Receiver 读完后：head++

状态判断：
    - 队列空：head == tail
    - 队列满：tail - head >= buffer_size

流量控制：
    - Sender 检测到队列满时等待
    - Receiver 读取后更新 head，释放空间
```

---

## Dispatch Kernel 核心逻辑

### SM 分工

```
num_sms 个 SM 参与 dispatch：

    SM 0  (偶数) → Sender,   Channel 0
    SM 1  (奇数) → Receiver, Channel 0
    SM 2  (偶数) → Sender,   Channel 1
    SM 3  (奇数) → Receiver, Channel 1
    ...

每个 Channel 独立工作，互不干扰
```

### 线程分组

```
每个 SM 有 768 个线程，按 rank 分组：

假设 8 个 rank：
    Thread 0~95:    负责 Rank 0
    Thread 96~191:  负责 Rank 1
    Thread 192~287: 负责 Rank 2
    ...
    Thread 672~767: 负责 Rank 7

Sender SM：每组线程负责往对应 rank 发送数据
Receiver SM：每组线程负责接收来自对应 rank 的数据
```

### Sender 逻辑

```cuda
// ===== Sender 核心逻辑 =====

// 1. 告诉 Receiver 这个 channel 要发多少 token
//    用负数编码区分 "还没写" 和 "值为0"
channel_start_offset = -(起始偏移) - 1;
channel_end_offset = -(结束偏移) - 1;

// 2. 获取本 channel 负责的 token 范围
get_channel_task_range(num_tokens, num_channels, channel_id,
                       &token_start_idx, &token_end_idx);

int tail = 0;  // 队列尾指针

// 3. 遍历所有 token
for (token_idx = token_start_idx; token_idx < token_end_idx; ) {

    // 3a. 流量控制：等待 Receiver 腾出空间
    while (buffer_size - (tail - head) < num_max_send_tokens) {
        head = ld_volatile_global(channel_head_idx);  // 轮询 head
    }

    // 3b. 发送一批 token
    for (chunk = 0; chunk < num_max_send_tokens && token_idx < end; ) {

        // 记录 send_head（combine 时需要）
        send_head[token_idx][responsible_rank] = tail;

        // 检查这个 token 是否需要发给这个 rank
        if (!is_token_in_rank[token_idx][responsible_rank]) {
            token_idx++;
            continue;
        }

        // 计算写入位置（循环队列取模）
        int slot = tail % buffer_size;

        // 写入远程 GPU 的 buffer（通过 NVLink）
        channel_x_buffers[slot] = x[token_idx];
        channel_src_idx[slot] = token_idx;
        channel_topk_idx[slot] = convert_to_local_expert_id(topk_idx[token_idx]);
        channel_topk_weights[slot] = topk_weights[token_idx];

        tail++;
        chunk++;
        token_idx++;
    }

    // 3c. 更新 tail 指针，通知 Receiver 有新数据
    //     使用 release 语义保证数据写入对 Receiver 可见
    barrier_within_rank();
    st_release_sys_global(channel_tail_idx, tail);
}
```

### Receiver 逻辑

```cuda
// ===== Receiver 核心逻辑 =====

// 1. 等待 Sender 告诉我要接收多少
while ((start_offset = ld_volatile(channel_start_offset)) == 0);
while ((end_offset = ld_volatile(channel_end_offset)) == 0);
start_offset = -start_offset - 1;  // 解码
end_offset = -end_offset - 1;
num_tokens_to_recv = end_offset - start_offset;

// 2. 计算写入输出 buffer 的偏移
int output_offset = rank_prefix_matrix[...] + start_offset;

int head = 0;  // 队列头指针

// 3. 循环接收数据
while (num_tokens_to_recv > 0) {

    // 3a. 等待 Sender 写入数据
    while (head == tail) {
        tail = ld_acquire_sys_global(channel_tail_idx);  // 轮询 tail
    }

    // 3b. 从 buffer 拷贝到输出
    int num_ready = tail - head;
    for (i = 0; i < num_ready; i++) {
        int slot = (head + i) % buffer_size;

        recv_x[output_offset + i] = channel_x_buffers[slot];
        recv_src_idx[output_offset + i] = channel_src_idx[slot];
        recv_topk_idx[output_offset + i] = channel_topk_idx[slot];
        recv_topk_weights[output_offset + i] = channel_topk_weights[slot];
    }

    // 3c. 更新 head 指针，告诉 Sender 我读完了
    head += num_ready;
    output_offset += num_ready;
    barrier_within_rank();
    st_relaxed_sys_global(channel_head_idx, head);

    num_tokens_to_recv -= num_ready;
}
```

### 同步机制

| 操作 | 语义 | 用途 |
|-----|------|------|
| `st_release_sys_global(tail)` | Release | Sender 写完数据后更新 tail，保证数据可见 |
| `ld_acquire_sys_global(tail)` | Acquire | Receiver 读 tail 时保证能看到 Sender 写的数据 |
| `ld_volatile_global(head)` | Volatile | Sender 轮询 head，不被编译器优化掉 |
| `st_relaxed_sys_global(head)` | Relaxed | Receiver 更新 head，不需要强同步 |
| `bar.sync` | Barrier | 同一 rank 的线程组内同步 |

`sys` 后缀表示 **system scope**，保证跨 GPU 可见性。

---

## 完整示例

### 场景设定

```
- 2 个 GPU：Rank 0, Rank 1
- 每个 GPU 有 4 个 token
- 4 个 expert：Rank 0 拥有 Expert 0,1；Rank 1 拥有 Expert 2,3
- top-2：每个 token 选择 2 个 expert
- 2 个 Channel
```

### 初始数据

```
Rank 0 的 token：                    Rank 1 的 token：
┌─────────────────────────┐          ┌─────────────────────────┐
│ token 0: 选 expert 0, 2 │          │ token 4: 选 expert 1, 3 │
│ token 1: 选 expert 1, 3 │          │ token 5: 选 expert 0, 2 │
│ token 2: 选 expert 0, 1 │          │ token 6: 选 expert 2, 3 │
│ token 3: 选 expert 2, 3 │          │ token 7: 选 expert 0, 1 │
└─────────────────────────┘          └─────────────────────────┘

Expert 分布：
    Rank 0 拥有 Expert 0, 1
    Rank 1 拥有 Expert 2, 3
```

### Step 1: 计算 is_token_in_rank

```
is_token_in_rank[token][rank] = token 是否需要发给 rank

根据 expert 所在的 rank 判断：
    expert 0,1 在 Rank 0
    expert 2,3 在 Rank 1

              Rank 0    Rank 1
token 0:        ✓         ✓       (expert 0 在 R0, expert 2 在 R1)
token 1:        ✓         ✓       (expert 1 在 R0, expert 3 在 R1)
token 2:        ✓         ✗       (expert 0,1 都在 R0)
token 3:        ✗         ✓       (expert 2,3 都在 R1)
token 4:        ✓         ✓       (expert 1 在 R0, expert 3 在 R1)
token 5:        ✓         ✓       (expert 0 在 R0, expert 2 在 R1)
token 6:        ✗         ✓       (expert 2,3 都在 R1)
token 7:        ✓         ✗       (expert 0,1 都在 R0)
```

### Step 2: 计算统计信息

```
num_tokens_per_rank（从各 rank 发出）：

从 Rank 0 发出：
    → Rank 0: token 0,1,2 = 3 个
    → Rank 1: token 0,1,3 = 3 个

从 Rank 1 发出：
    → Rank 0: token 4,5,7 = 3 个
    → Rank 1: token 4,5,6 = 3 个

rank_prefix_matrix（累计值）：
                    目标 Rank 0    目标 Rank 1
来自 Rank 0 累计:        3              3
来自 Rank 0+1 累计:      6              6

Rank 0 总共接收 6 个 token
Rank 1 总共接收 6 个 token
```

### Step 3: Channel 任务划分

```
假设 2 个 Channel，每个 rank 有 4 个 token：

Rank 0:
    Channel 0: 负责 token 0, 1
    Channel 1: 负责 token 2, 3

Rank 1:
    Channel 0: 负责 token 4, 5
    Channel 1: 负责 token 6, 7
```

### Step 4: Dispatch 数据流

```
以 Rank 0 的 Channel 0 为例（负责 token 0, 1）：

Sender SM 0 的工作：
    处理 token 0（选 expert 0, 2）：
        - expert 0 在 Rank 0 → 线程组 0 写入 Rank 0 的 buffer
        - expert 2 在 Rank 1 → 线程组 1 写入 Rank 1 的 buffer（通过 NVLink）

    处理 token 1（选 expert 1, 3）：
        - expert 1 在 Rank 0 → 线程组 0 写入 Rank 0 的 buffer
        - expert 3 在 Rank 1 → 线程组 1 写入 Rank 1 的 buffer（通过 NVLink）

Receiver SM 1 的工作：
    从 Rank 0 的 buffer 读取数据
    这个 buffer 被本地 Sender 和远程 Sender（来自 Rank 1）同时写入
    Receiver 把数据写入 recv_x
```

### Step 5: 数据流动时间线

```
以 Rank 0 的 Sender 发送 token 0 到 Rank 1 为例：

T=0  Sender 写 channel_start_offset = -0-1 = -1
     Sender 写 channel_end_offset = -2-1 = -3（Channel 0 发 2 个 token 给 R1）

T=1  Sender 检查 head，发现 buffer 有空间

T=2  Sender 写入 token 0 的数据到 Rank 1 的 buffer
     ┌──────────────────────────────────────────┐
     │ Rank 1 的 buffer (Channel 0, from R0)    │
     │                                          │
     │ slot 0: token 0 的数据 ← 刚写入          │
     │ slot 1: (空)                             │
     │                                          │
     │ tail = 1                                 │
     └──────────────────────────────────────────┘

T=3  Sender 写入 token 1 的数据，tail = 2

T=4  Sender 调用 st_release_sys_global(tail, 2)

T=5  Rank 1 的 Receiver 看到 tail=2
     开始从 buffer 拷贝数据到 recv_x

T=6  Receiver 拷贝完成，更新 head=2
```

### Step 6: Dispatch 完成后

```
Rank 0 的 recv_x：                    Rank 1 的 recv_x：
┌─────────────────────────┐          ┌─────────────────────────┐
│ [0] token 0 (来自 R0)   │          │ [0] token 0 (来自 R0)   │
│ [1] token 1 (来自 R0)   │          │ [1] token 1 (来自 R0)   │
│ [2] token 2 (来自 R0)   │          │ [2] token 3 (来自 R0)   │
│ [3] token 4 (来自 R1)   │          │ [3] token 4 (来自 R1)   │
│ [4] token 5 (来自 R1)   │          │ [4] token 5 (来自 R1)   │
│ [5] token 7 (来自 R1)   │          │ [5] token 6 (来自 R1)   │
└─────────────────────────┘          └─────────────────────────┘

同时输出：
    recv_src_idx: 每个位置的原始 token ID
    recv_topk_idx: 转换后的 local expert ID（0 或 1）
    recv_topk_weights: 对应的权重
```

---

## 整体数据流图

```
                          Dispatch 整体流程

    Rank 0                                           Rank 1
    ┌─────────────────┐                              ┌─────────────────┐
    │ 本地 token      │                              │ 本地 token      │
    │ [0,1,2,3]       │                              │ [4,5,6,7]       │
    └────────┬────────┘                              └────────┬────────┘
             │                                                │
             ▼                                                ▼
    ┌─────────────────┐                              ┌─────────────────┐
    │ Sender SM       │                              │ Sender SM       │
    │ (偶数 SM)       │                              │ (偶数 SM)       │
    │                 │         NVLink               │                 │
    │ 写入 R1 buffer ─┼──────────────────────────────┼→ buffer        │
    │                 │                              │                 │
    │                 │         NVLink               │                 │
    │ buffer ←────────┼──────────────────────────────┼─ 写入 R0 buffer│
    └─────────────────┘                              └─────────────────┘
             │                                                │
             ▼                                                ▼
    ┌─────────────────┐                              ┌─────────────────┐
    │ Receiver SM     │                              │ Receiver SM     │
    │ (奇数 SM)       │                              │ (奇数 SM)       │
    │                 │                              │                 │
    │ 从 buffer 拷贝  │                              │ 从 buffer 拷贝  │
    │ 到 recv_x       │                              │ 到 recv_x       │
    └────────┬────────┘                              └────────┬────────┘
             │                                                │
             ▼                                                ▼
    ┌─────────────────┐                              ┌─────────────────┐
    │ recv_x          │                              │ recv_x          │
    │ 按 expert 分组  │                              │ 按 expert 分组  │
    │ 准备送去 MoE    │                              │ 准备送去 MoE    │
    └─────────────────┘                              └─────────────────┘
```

---

## 代码位置参考

| 文件 | 内容 | 关键行号 |
|-----|------|---------|
| `deep_ep/buffer.py` | Python dispatch 接口 | 322-402 |
| `csrc/kernels/intranode.cu` | notify_dispatch kernel | 11-113 |
| `csrc/kernels/intranode.cu` | dispatch kernel | 197-532 |
| `csrc/kernels/intranode.cu` | combine kernel | 691-1032 |
| `csrc/kernels/buffer.cuh` | Buffer 数据结构 | 8-131 |
| `csrc/kernels/utils.cuh` | 内存操作工具函数 | 全文件 |
