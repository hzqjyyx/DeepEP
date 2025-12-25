# intranode.cu 核心功能解析

这个文件实现了基于 NVLink 的节点内(intranode) MoE 通信,包含三组主要操作:

## 1. notify_dispatch - 通知阶段的元数据交换

**两个版本:**

### 标准版 `notify_dispatch` (L11-159)
- **SM0 (控制块)**: 多 rank 间交换 token 分布信息
  - 每个 rank 将本地的 `num_tokens_per_rank` 和 `num_tokens_per_expert` 写入共享 buffer
  - Barrier 同步后,计算前缀和得到每个 rank 将接收的总 token 数
  - 将结果写回 CPU 可见的 mapped memory (`moe_recv_counter_mapped`)
  - 复制 rank prefix matrix 供后续使用
  - 清空通信队列元数据区域

- **SM1~SMN (工作块)**: 并行计算每个 channel 要发送的 token 数
  - 每个 SM 负责一个目标 rank
  - 多个 warp 并行遍历 token,统计发往该 rank 的数量
  - 计算 channel 级别的前缀和,为后续数据发送做准备

### 缓存版 `cached_notify_dispatch` (L161-195)
- 简化版本,用于已知 token 分布的场景
- 直接复制预先计算好的 `rank_prefix_matrix`
- 清空队列元数据
- 减少一次 barrier 轮次

## 2. dispatch - 实际数据分发

**核心机制** (L197-610):
- **双缓冲设计**: 偶数 SM 发送,奇数 SM 接收
- **循环队列**: 使用 head/tail 指针管理有限容量的 channel buffer

### 发送端 (is_sender = true)
1. **任务分配**: 按 channel 切分 token 范围
2. **流控**: 检查接收端队列剩余容量,避免溢出 (轮询 `channel_head_idx`)
3. **数据传输**:
   - 拷贝 `x` (激活值,int4 格式)
   - 拷贝 `src_idx` (源 token 索引)
   - 拷贝并变换 `topk_idx`/`topk_weights` (专家路由信息,需调整专家 ID 到本地编号)
   - 拷贝 `x_scales` (量化 scale)
4. **更新尾指针**: 使用 release 语义通知接收端

### 接收端 (is_sender = false)
1. **轮询偏移量**: 等待发送端写入 `channel_start/end_offset` (用负数编码避免零值混淆)
2. **轮询数据**: 等待 `channel_tail_idx` 更新
3. **批量拷贝**:
   - SM90: 使用 TMA (Tensor Memory Accelerator) 实现高效拷贝
   - SM80: 使用普通 global memory load/store
4. **更新头指针**: 释放队列空间供发送端复用

### 关键优化
- **Warp 级并行**: 同一 SM 内多个 warp 分摊同一 rank 的 token 处理
- **SM 级 barrier**: `bar.sync` 指令同步负责同一 rank 的所有线程
- **超时检测**: 防止死锁 (`NUM_TIMEOUT_CYCLES`)

## 3. combine - 结果聚合

**反向操作** (L691-1099):
- **发送端**: 将 all-to-all 后的结果按 channel 发回原 rank
- **接收端**: 
  1. **头指针更新线程** (warp 0): 监控所有 worker warp 的进度,更新队列头
  2. **Worker 线程** (warp 1~N): 
     - 根据 `send_head` 确定每个 token 需要从哪些 rank 接收数据
     - 从多个 channel buffer 读取数据并 reduce (加法累积)
     - 支持 bias 融合 (`bias_0 + bias_1 + sum(recv_data)`)
     - TMA pipeline 优化 (8-stage)

### 依赖关系处理
- `send_head` 记录了每个 token 在每个 rank 的队列位置
- 使用 `-value - 1` 编码未选中的 token (负数表示"跳过的前一个 token")
- `cached_notify_combine` 预处理填充这些负数编码

## 数据结构布局

所有 rank 共享的 buffer 按固定 layout 组织:
```
[Rank Prefix Matrix: R×R int]
[Channel Metadata: start/end/head/tail offset, C×R int each]
[Channel Data: C×R×B×H int4 for x]
[Channel Data: C×R×B int for src_idx]
[Channel Data: C×R×B×K topk_idx_t for topk_idx]
[Channel Data: C×R×B×K float for topk_weights]
[Channel Data: C×R×B×S float for x_scales]
```
- R: num_ranks
- C: num_channels
- B: num_recv_buffer_tokens (循环队列容量)
- H: hidden_int4
- K: num_topk
- S: num_scales

## 关键宏和模板

- `SWITCH_RANKS`: 编译期生成不同 rank 数的特化版本
- `LAUNCH_KERNEL`: 封装 kernel launch 配置
- `UNROLLED_WARP_COPY`: 循环展开的 warp 级拷贝
- `Buffer<T>`: 类型安全的 buffer 指针包装

---

**忽略内容速览:**
- Backward: 无(此文件仅包含前向通信)
- ROCm: 无相关代码
