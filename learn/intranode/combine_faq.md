# DeepEP Intranode Combine FAQ

## 基础概念

### Q1: send_head 是怎么拿到的？是 Dispatch 算好留在显存里的吗？

是的，**Dispatch 阶段计算并写入显存**。

```cpp
// intranode.cu:344-346 (Dispatch Sender)
if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
    send_head[token_idx * kNumRanks + responsible_rank] =
        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;
```

Dispatch Sender 在发送每个 token 时，顺便记录"这个 token 被写入目标 rank buffer 的哪个 slot"。

但有个问题：Dispatch 只记录了发送成功的 token（`>= 0`），对于不发送的 token 记录 `-1`。Combine 需要知道"不从某个 rank 接收时应该等多少"，所以有个预处理 kernel `cached_notify_combine` 来填充这些 `-1`。

---

### Q2: cached_notify_combine 是什么？

**把 `-1` 转换为有意义的负数值**，让 Combine Receiver 能正确更新队列 head。

原理：倒序遍历 token，用"上一个有效 head"来填充 `-1` 位置：

```
原始 send_head (Dispatch 后):
    token 0: [head=5,  head=3,  head=-1]   // 发给 rank 0/1，不发给 rank 2
    token 1: [head=-1, head=4,  head=-1]   // 只发给 rank 1
    token 2: [head=6,  head=-1, head=2 ]   // 发给 rank 0/2

转换后 (cached_notify_combine 后):
    token 0: [head=5,  head=3,  head=-4]   // -1 → -(last_head) - 1 = -3-1 = -4
    token 1: [head=-6, head=4,  head=-5]   // 根据相邻 token 推算
    token 2: [head=6,  head=-5, head=2 ]
```

Combine Receiver 使用时：
```cpp
// 正数: 消费一个 slot，head = expected_head + 1
// 负数: 还原为 last_head，head = -expected_head - 1（不消费）
int new_head = (expected_head < 0) ? (-expected_head - 1) : (expected_head + 1);
```

这样 Receiver 就能正确更新队列 head，让 Sender 知道哪些 slot 可以复用。

---

### Q3: x_buffer 等字段既然是变长，那是怎么提前分配内存的？

**不是变长，是固定大小的环形队列**。

用户创建 Buffer 时指定总大小：
```python
Buffer(group, num_nvl_bytes=1024*1024*1024)  # 1GB
```

Kernel 启动时，根据 `hidden`、`num_topk`、`num_ranks` 等参数，**反推**这块内存能容纳多少 token slot：
```cpp
// 伪代码
bytes_per_slot = hidden_int4 * sizeof(int4) + sizeof(int) + num_topk * sizeof(float) + ...
num_recv_buffer_tokens = num_nvl_bytes / (num_channels * num_ranks * bytes_per_slot)
```

环形队列：`slot_idx = tail % num_recv_buffer_tokens`，满了就等 Receiver 消费后移动 head。

---

### Q4: 每个 rank 分配相同大小的 buffer 吗？

是的。每个 (channel, src_rank) 组合有一个固定大小的环形队列：

```
Buffer 布局:
┌─────────────────────────────────────────────────────┐
│ Channel 0                                           │
│ ├─ from Rank 0: [slot_0, slot_1, ..., slot_N-1]    │
│ ├─ from Rank 1: [slot_0, slot_1, ..., slot_N-1]    │
│ └─ ...                                             │
│ Channel 1                                           │
│ ├─ from Rank 0: [slot_0, slot_1, ..., slot_N-1]    │
│ └─ ...                                             │
└─────────────────────────────────────────────────────┘
```

不管实际数据量如何分布，每个队列的大小相同。这简化了实现，但可能造成空间浪费（如果某些 rank 之间传输量很不均衡）。

---

## 线程模型

### Q5: lane 和 CUDA 里的 thread 是一个概念吗？

是的，**lane 是 warp 内的 thread 编号**（0~31）。

```
Warp 0: [lane_0, lane_1, ..., lane_31] = [thread_0, thread_1, ..., thread_31]
Warp 1: [lane_0, lane_1, ..., lane_31] = [thread_32, thread_33, ..., thread_63]
...

lane_id = thread_id % 32
warp_id = thread_id / 32
```

---

### Q6: 8 个 rank 时，一个 warp 只有 8 个 lane 在工作？

**部分操作是这样，部分操作不是**：

| 操作 | 工作的 lane | 说明 |
|------|------------|------|
| 队列管理（等待 tail、更新 head） | lane 0~7 | 每个 lane 负责一个 rank |
| Reduce（累加数据） | lane 0~31 | 每个 lane 处理 hidden 的不同部分 |
| 写 topk_weights | lane 0~7 | 每个 lane 处理一个 topk position |

队列管理时 lane 8~31 确实闲置，这是用空间换代码简洁性的权衡。但 Reduce 阶段所有 lane 都在工作。

---

### Q7: while (__any_sync(...)) 是如何等待所有 rank 数据到达的？

```cpp
while (__any_sync(0xffffffff, tail[lane_id] <= expected_head && expected_head >= 0))
```

**`__any_sync` 是 warp 级同步原语**：
- 如果**任意一个 lane** 的条件为 true，返回 true（整个 warp 继续等待）
- 只有**所有 lane** 的条件都为 false，才返回 false（跳出循环）

```
假设 8 ranks，tail = [6, 4, 10, 5, ...]，expected_head = [5, 3, -4, 7, ...]

lane 0: tail[0]=6 > expected=5   → false ✓ (数据到了)
lane 1: tail[1]=4 > expected=3   → false ✓ (数据到了)
lane 2: expected=-4 < 0          → false ✓ (不需要等，负数表示不接收)
lane 3: tail[3]=5 <= expected=7  → true  ✗ (还没到)
...

__any_sync 结果: true（因为 lane 3 还在等）
→ 整个 warp 继续等待
```

---

### Q8: tail 是在哪里赋值的？

**Warp 0（队列管理器）持续从 sender 读取并更新到共享内存**：

```cpp
// intranode.cu:861 (Warp 0)
channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);
```

Worker warp 从共享内存 `tail[]` 读取，不直接访问 global memory。这样减少了跨 GPU 的内存访问次数。

---

## 设计原理

### Q9: Dispatch 到底留了什么数据给 Combine？

通过 Python 层的 `handle` 传递：

```python
handle = (
    rank_prefix_matrix,         # 每个 rank 发给每个 rank 的 token 数前缀和
    channel_prefix_matrix,      # 每个 (rank, channel) 的 token 数前缀和
    recv_channel_prefix_matrix, # 接收端的 channel 前缀和
    recv_src_idx,               # 接收到的 token 的原始索引
    is_token_in_rank,           # token 是否发给某 rank 的 mask
    send_head                   # 最关键：每个 token 在每个 rank 的 buffer slot 位置
)
```

Combine 主要用 **`send_head`** 来知道"从哪里读数据"。

---

### Q10: bias_0 和 bias_1 是什么？

**可选的输出偏置**，用于在 Combine 时直接加上 bias，避免额外的 kernel：

```cpp
// intranode.cu:948-952
int4 bias_0_value = bias_0 ? __ldg(bias_0 + token_idx * hidden_int4 + i) : make_int4(0,0,0,0);
int4 bias_1_value = bias_1 ? __ldg(bias_1 + token_idx * hidden_int4 + i) : make_int4(0,0,0,0);

// result = bias_0 + bias_1 + sum(来自各 rank 的数据)
```

Python 层可以传入：
```python
buffer.combine(x, handle, bias=(bias_0, bias_1))
```

---

### Q11: recv_x[token_idx] = result 是整块数据的拷贝吗？

不是整块拷贝，是 **逐 int4 写入**，warp 内 32 个 lane 并行：

```cpp
for (int i = lane_id; i < hidden_int4; i += 32) {
    // lane 0 写 i=0, 32, 64, ...
    // lane 1 写 i=1, 33, 65, ...
    // ...
    recv_x[token_idx * hidden_int4 + i] = out_int4;
}
```

一个 warp 32 个 lane 并行，覆盖整个 hidden 维度。这是 CUDA 的标准写法。

---

## 高级话题

### Q12: Dispatch、FFN、Combine 是三个独立的 kernel 吗？

是的，当前实现是三个独立 kernel：

```
Dispatch kernel → (sync) → FFN kernel → (sync) → Combine kernel
```

**low-latency 模式**提供了 hook 机制可以做粗粒度重叠：
```python
recv_x, ..., hook = buffer.low_latency_dispatch(..., return_recv_hook=True)
# dispatch 发起后，不等数据到达
do_ffn_computation()  # FFN 和 dispatch 的接收阶段重叠
hook()  # 显式等待数据到达
```

但没有做"算一个 expert 就发一个"的细粒度流水。

---

### Q13: 为什么不做 FFN 和 Combine 的细粒度流水？

理论上可以做：算完一批 expert 就发一批，不等全部算完。但有几个挑战：

1. **实现复杂度**：需要把 FFN 和 Combine Sender 融合成一个 persistent kernel
2. **同步开销**：细粒度同步有额外开销，可能抵消收益
3. **收益有限**：如果 FFN 计算时间 >> 通信时间，通信完全被隐藏，细粒度流水收益不大

实验分支（如 AntGroup-Opt）有一些相关优化探索。

---

### Q14: Sender 和 Receiver 之间如何避免死锁？

通过**循环队列的流量控制**：

```
Sender:
    while (tail - head >= buffer_size)  // 队列满，等待
        head = ld_volatile(head_ptr);
    write_data();
    tail++;
    st_release(tail_ptr, tail);

Receiver:
    while (tail <= expected_head)  // 数据没到，等待
        tail = ld_acquire(tail_ptr);
    read_data();
    head = expected_head + 1;
    st_relaxed(head_ptr, head);
```

只要 `buffer_size >= num_max_send_tokens`，就不会死锁：
- Sender 最多写 `num_max_send_tokens` 个就等待
- Receiver 一定能读到数据并释放空间

---

### Q15: 为什么 Receiver Warp 0 单独做队列管理？

**减少跨 GPU 内存访问**：

如果每个 worker warp 都直接读 `tail_ptr`：
- 每个 warp 每次循环都要跨 GPU 读取
- `num_warps × num_tokens × num_ranks` 次跨 GPU 访问

现在的设计：
- Warp 0 持续读取 `tail_ptr` 并更新到共享内存
- Worker warp 从共享内存读取
- 跨 GPU 访问大幅减少

同样，`head_ptr` 的更新也集中在 Warp 0，避免多个 warp 同时写同一个位置。

---

## 配置相关

### Q16: Combine 的 SM 数量可以调整吗？

可以，通过 `Config` 或 `Buffer.get_combine_config()` 获取推荐值：

```python
# 获取推荐配置
config = deep_ep.Buffer.get_combine_config(num_ranks)

# 或手动指定
config = deep_ep.Config(num_sms=20, nvl_chunk_size=8, nvl_buffer_size=256)

# 使用
buffer.combine(x, handle, config=config)
```

不同 rank 数量的最优配置不同，可以通过 tuning 找到最优值。

---

### Q17: 如何判断 Combine 出了问题？

代码中有超时检测机制：

```cpp
if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf("DeepEP timeout for combine senders/receivers, rank %d, ...\n", rank, ...);
    trap();
}
```

如果看到 timeout 错误，可能的原因：
1. `send_head` 数据不正确（Dispatch 和 Combine 使用的 handle 不匹配）
2. 某个 rank 的 kernel 没有启动
3. NVLink 通信问题
