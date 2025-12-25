## internode.cu 核心功能

这个文件实现 **跨节点 RDMA 通信**的 MoE dispatch/combine 操作，是 DeepEP 处理多机分布式场景的核心。

---

## 架构设计

### 双层通信拓扑

```
Node A (RDMA Rank 0)          Node B (RDMA Rank 1)
├─ GPU 0 (NVL Rank 0)         ├─ GPU 4 (NVL Rank 0)
├─ GPU 1 (NVL Rank 1)         ├─ GPU 5 (NVL Rank 1)
│   ...                       │   ...
└─ 节点内 NVLink 互联         └─ 节点内 NVLink 互联
        │                            │
        └────── RDMA/InfiniBand ─────┘
```

- **NVLink 层**：节点内 GPU 间高速通信（intranode.cu 处理）
- **RDMA 层**：跨节点通过 NVSHMEM/IBGDA 进行 RDMA Put/Get 操作

---

## 主要 Kernel

### 1. `notify_dispatch` - 分发前的全局同步

**职责：**
- SM 0 执行全局协调：
  - 等待所有 inflight RDMA 操作完成（`nvshmemi_ibgda_quiet`）
  - NVSHMEM 全局 barrier（`nvshmem_sync_all`）
  - 向所有 RDMA rank 广播 token 计数（通过 RDMA Put）
  - 同步 NVLink barrier
- 其他 SM：准备 NVL 缓冲区

**关键 RDMA 操作：**
```cpp
nvshmemi_ibgda_quiet(dst_rdma_rank, qp_id);  // 等待特定 QP 完成
nvshmemi_ibgda_put_nbi_warp(recv_ptr, send_ptr, bytes, dst_rank, qp_id);
```

---

### 2. `dispatch` - 核心数据分发

**Warp 角色分工（32 warps per block）：**
```
┌─ RDMA Sender (7 warps)
│  └─ 遍历所有 token，通过 RDMA Put 发送到远程节点
├─ RDMA Sender Coordinator (1 warp)  
│  └─ 管理 32-slot 发送窗口，避免接收端溢出
├─ RDMA & NVL Forwarder (NUM_MAX_NVL_PEERS warps)
│  └─ 从 RDMA buffer 读取，转发到本地 NVL rank
└─ NVL Receivers
   └─ 从本地 NVLink GPU 接收数据
```

**数据流：**
```
Token → 检查路由标志 (is_token_in_rank[64bit]) 
     → 等待远程 buffer 有空间 (tail - head < capacity)
     → 计算 slot_idx (ring buffer)
     → Warp-level copy:
        ├─ hidden_int4 (主要特征)
        ├─ x_scales (FP8 scale)
        ├─ SourceMeta (元数据: src_rdma_rank + NVL rank bitmap)
        ├─ topk_idx (expert 索引)
        └─ topk_weights (expert 权重)
     → RDMA Put NBI 发送 chunk
     → 原子操作更新远程 tail 计数器
```

**Ring Buffer 管理：**
- 每个 channel 有固定容量 `num_max_rdma_chunked_recv_tokens`
- Head/Tail 指针控制生产者-消费者同步
- 发送端维护 32-slot window 防止过载

---

### 3. `cached_notify` - Combine 前的清理

**3 个 SM 的分工：**
- SM 0: RDMA quiet + buffer 清零 + 全局 barrier
- SM 1: 反向遍历 RDMA head 数组，压缩负数编码的 offset
- SM 2+: 用 TMA 反向遍历 NVL head 数组并压缩

**负数编码压缩：**
```cpp
// 存储时: encode(-offset - 1)
// 读取时: -value - 1 = offset
```
节省内存，combine 时可快速定位起始位置。

---

### 4. `combine` - 聚合来自多源的数据

**反向流程（相对于 dispatch）：**
```
Expert 输出 (recv_x) → NVL Sender 发送到本地其他 GPU
                     ↓
                NVL & RDMA Forwarder
                ├─ 从 NUM_MAX_NVL_PEERS 个 NVL buffer 读取
                ├─ 对每个 token 执行 reduce-sum (combine 多个 expert 结果)
                └─ RDMA Put 发送到目标节点
                     ↓
                RDMA Receiver 接收远程数据
                     ↓
                输出: combined_x (最终合并结果)
```

**Combine 核心逻辑：**
```cpp
for (each token) {
    float combined_output = 0;
    for (each src_nvl_rank in NUM_MAX_NVL_PEERS) {
        if (is_token_in_nvl_rank[src_nvl_rank]) {
            data = read_from_nvl_buffer(src_nvl_rank, slot_idx);
            weight = topk_weights[src_nvl_rank];
            combined_output += data * weight * scale;
        }
    }
    output[token_idx] = combined_output;
}
```

---

## RDMA/NVSHMEM 关键机制

### IBGDA 操作（InfiniBand GDA）

```cpp
// 非阻塞 Put (主要数据传输)
nvshmemi_ibgda_put_nbi_warp<true>(
    recv_ptr,    // 远程地址
    send_ptr,    // 本地地址  
    bytes,       // 大小
    dst_rank,    // 目标 RDMA rank
    qp_id,       // Queue Pair ID
    lane_id      // Warp 内 lane
);

// 原子加 (更新计数器)
nvshmemi_ibgda_amo_nonfetch_add(
    dst_counter_addr, value, dst_rank, qp_id
);

// 等待操作完成
nvshmemi_ibgda_quiet(dst_rank, qp_id);
```

### 多 QP 设计
- 每个 channel 对应独立的 QP（Queue Pair）
- 支持异步并发传输
- Low-Latency 模式：每个 local expert 一个 QP

---

## 与 intranode.cu 的区别

| 维度 | Intranode (NVLink) | Internode (RDMA) |
|-----|-------------------|-----------------|
| **通信范围** | 单节点内 GPU | 跨节点 |
| **硬件** | NVLink | RDMA/InfiniBand |
| **Buffer 类型** | AsymBuffer | SymBuffer + AsymBuffer |
| **同步原语** | Barrier signals | NVSHMEM barriers + RDMA AMO |
| **数据传输** | 直接 memcpy/warp_copy | RDMA Put NBI + TMA |
| **Ring Buffer** | 无 | 有（异步流控） |
| **Forwarder** | 无 | 有（NVL ↔ RDMA 桥接） |
| **复杂度** | 简单直连 | 多跳路由 + 缓冲管理 |

---

## 性能优化

1. **TMA (Tensor Memory Accelerator)**
   - Hopper 架构的硬件加速器
   - 绕过 L1 cache，直接 global memory ↔ shared memory
   - 配合 mbarrier 实现异步 pipeline

2. **Warp-level Copy**
   - 32 threads 并行复制连续内存
   - 适合大块数据传输（hidden states）

3. **Window-based Flow Control**
   - 32-slot 发送窗口
   - 避免接收端 buffer 溢出

4. **Channel 并行**
   - 多个独立 QP/channel 同时传输
   - 充分利用 RDMA 带宽

5. **SourceMeta 压缩**
   - 64-bit 打包：8-bit src_rdma_rank + 8-bit nvl_rank_bitmap
   - 减少元数据开销

---

## 两种模式

### Normal Mode (训练/预填充)
- 全局同步：`nvshmem_sync_all()`
- 高吞吐量设计
- CPU 需等待 GPU signal 获取接收 token 数
- 支持 SM 控制（`set_num_sms()`）

### Low-Latency Mode (推理解码)
- 团队同步：`nvshmem_sync(rdma_team)`  
- CUDA Graph 兼容
- 每个 local expert 独立 QP
- 固定 buffer 大小（推荐 batch < 256）
- Hook-based receive 实现计算-通信 overlap

---

## 关键数据结构

```cpp
struct SourceMeta {
    int src_rdma_rank;              // 来源 RDMA rank (8-bit)
    int is_token_in_nvl_rank_bits;  // NVL rank bitmap (8-bit)
};

// Token layout in buffer
struct TokenData {
    int4 hidden[hidden_int4];       // 主要特征 (FP8/BF16)
    float x_scales[num_scales];     // 量化 scale
    SourceMeta src_meta;            // 路由元数据
    topk_idx_t topk_idx[k];         // Expert 索引
    float topk_weights[k];          // Expert 权重
};
```

---

## 文件特征总结

- **代码行数**：2377 行（不含 ROCm/Backward）
- **核心复杂度**：多层 warp 角色分工 + RDMA 异步管理
- **NVSHMEM SLA 依赖**：`ibgda_device.cuh` 中的底层 RDMA 操作
- **性能瓶颈**：RDMA 延迟 + 缓冲管理开销
- **优化空间**：queue-based buffer → fixed buffer（参考 issue #39）

---

**简单列出（ROCm & Backward）：**
- ROCm 相关：AMD GPU 平台的条件编译分支，主要在 TMA/mbarrier 替代实现
- Backward 相关：反向传播的 dispatch/combine，逻辑与前向对称但数据流反向
