# internode_ll.cu 核心功能分析

这是 DeepEP 的低延迟(low-latency)模式下的跨节点通信内核实现,专门为推理解码场景设计。与 normal 模式不同,这里完全基于 RDMA 实现纯 GPU-GPU 通信,避免 CPU 参与带来的延迟。

## 核心组件

### 1. Barrier 同步机制 (line 23-70)

自定义的全局 barrier,用于协调多个 rank 之间的同步:

- **Quiet QPs**: 通过 `nvshmemi_ibgda_quiet()` 确保所有 RDMA 传输完成
- **计数器更新**: 每个 rank 维护本地计数器 `sync_buffer_ptr[rank]`,通过 P2P 或 RDMA 更新远端计数器
- **超时处理**: 如果远端 rank 超过 `NUM_TIMEOUT_CYCLES` 未响应,将其标记到 `mask_buffer_ptr` 并继续执行,避免整体卡死
- **P2P 优化**: 优先使用 `nvshmemi_get_p2p_ptr()` 获取对端显存地址,直接写入(NVSwitch/NVLink);回退到 RDMA PUT

### 2. Dispatch 内核 (line 130-463)

将 tokens 分发到各 expert 所在的 rank,分为发送和接收两个阶段:

#### 发送阶段 (line 190-351)

**分工设计**:
- **前 N-1 个 warp**: FP8 转换 + RDMA 发送
  - 读取 `x` (BF16),计算 FP8 scale,转换并写入 `rdma_x` 缓冲区
  - 每 128 个通道为一组计算 amax 并 warp reduce,生成 per-channel scale
  - 通过 `nvshmemi_ibgda_put_nbi_warp()` 或 P2P 写入远端 `rdma_recv_x`
  - 使用 `UNROLLED_WARP_COPY` 宏优化小数据拷贝(隐藏 load latency)
  
- **最后一个 warp**: 统计 + 清理
  - SM0 的最后一个 warp 负责清理 `next_clean` 缓冲区(为下一次调用准备)
  - 每个 SM 统计本 SM 负责的 experts 接收到的 token 数量(通过 warp reduce)
  - 等待本地所有发送完成后,通过 RDMA atomic add 发送计数到远端 `rdma_recv_count`

**消息格式** (line 179-183):
```
[4B src_idx][12B reserved][hidden_bytes data][scale_bytes]
```

**关键优化**:
- `atomic_counter_per_expert`: 发送前分配 slot
- `atomic_finish_counter_per_expert`: 两阶段计数机制,确保所有 warp 完成后才发送 count
- `FINISHED_SUM_TAG`: 用于区分是否完成统计的标记值

#### 接收阶段 (line 354-462)

**等待策略**:
- Sub-warp 1 的 lane 0 负责轮询 `rdma_recv_count`,等待远端发来的负数计数(编码格式 `-num_tokens-1`)
- 超时则将 src_rank 加入 mask,继续处理其他数据

**数据整理**:
- 从 `rdma_recv_x` 拷贝到紧凑的 `packed_recv_x`
- 源信息写入 `packed_recv_src_info`
- 更新 `packed_recv_layout_range` 记录每个 rank 的 token 范围
- FP8 scales 重排为特定 layout (line 440-459),方便后续 combine kernel 读取

**Warp 同步**:
- 使用 `bar.sync` 指令在 warp group 内部同步,避免 `__syncthreads()` 开销

### 3. Combine 内核 (line 716-1139)

将 experts 输出的 tokens 按原始顺序组合回去,支持 LogFMT 压缩格式。

#### 发送阶段 (line 774-941)

**TMA Pipeline**:
- 3-stage pipeline: prefetch → encode → store
- 使用 `mbarrier` 实现异步拷贝和 pipeline 控制
- 每个 warp 独立处理,避免 block-level 同步

**LogFMT 编码** (line 558-639):
- 将 BF16 值映射到 log2 空间,量化为 10-bit 整数
- 每 256 bits 压缩为 160 bits (5/8 压缩率)
- 动态判断是否启用:检查 log_amax < 0 且满足动态范围要求
- Metadata: 每 128 通道记录 `(amax, amin)` 的 BF162 pair

**Zero-copy 优化**:
- 如果 `zero_copy=true` 且 P2P 可用,直接从 `rdma_send_x` buffer 读取数据(假设已预填充)
- 否则从 `x` 读取并写入 buffer

**同步机制**:
- `atomic_clean_flag`: 确保 `next_clean` 清理完成后才发送 finish flag
- 通过 RDMA atomic add 或 P2P 写 `rdma_recv_flag` 通知接收方

#### 接收阶段 (line 944-1138)

**分组设计**:
- 每组包含 `num_decode_warps` 个 reduction warp + 1 个 TMA load warp
- 最多 2 组并行,动态分配(基于 hidden size 和 block 大小)

**TMA Load Warp** (line 1034-1071):
- 读取 `topk_idx`,确定每个 token 的 expert 来源
- 检查 LogFMT metadata,判断是否压缩(`logfmt_check_amaxmin`)
- 计算 TMA 字节数(compressed vs uncompressed)
- 异步加载到共享内存 `tma_ld_buffers`,通过 `full_barriers` 通知

**Reduction Warps** (line 1072-1137):
- 等待 TMA 加载完成(`mbarrier_wait`)
- 根据 `cast_info` 确定数据格式和偏移
- 调用 `decode_and_accumulate`:
  - LogFMT: 解压 10-bit 值,还原到 FP32,乘以 weight 累加
  - BF16: 直接转 FP32,乘以 weight 累加
- 累加 `num_topk` 个 expert 的输出
- 写回共享内存 `tma_st_buffers`,TMA store 到 `combined_x`

**LogFMT 解码** (line 674-713):
- 从 packed 160-bit 数据中提取 10-bit 编码值和符号位
- 计算 `exp2((encoded - 1) * step + log_amin)` 还原原值
- 0 编码特殊处理为真零

### 4. 辅助功能

- **`clean_low_latency_buffer`** (line 73-127): 前后 barrier 保证缓冲区清理安全
- **`query_mask_buffer`** (line 1242-1257): 查询当前 masked ranks
- **`update_mask_buffer`** (line 1260-1273): 动态更新 mask
- **`clean_mask_buffer`** (line 1276-1288): 重置 mask buffer

## 设计特点

### 性能优化

1. **纯 GPU 通信**: 完全避免 CPU 参与,通过 RDMA atomic 和 P2P 实现计数同步
2. **异构 warp 分工**: 不同 warp 执行不同任务,最大化并行度
3. **细粒度同步**: 使用 warp-level barrier 和 elect_one_sync,避免全 block 同步
4. **TMA 加速**: Hopper 架构的 Tensor Memory Accelerator 用于异步拷贝
5. **P2P 快路径**: 优先使用本地 P2P 写入,回退到 RDMA

### 容错机制

- **Timeout 检测**: 基于 `clock64()` 的超时机制,避免无限等待
- **动态 masking**: 超时节点自动标记并跳过,不阻塞整体进度
- **Trap vs Mask**: 如果没有 mask buffer,直接 `trap()` 终止;否则继续执行

### 压缩支持

LogFMT 在线压缩(combine 发送阶段):
- **自适应启用**: 每 128 通道独立判断是否满足压缩条件
- **Warp-level 决策**: 通过 `warp_reduce_and` 同步压缩决策
- **元数据传输**: amax/amin 作为解压参数随数据发送

---

## 其他相关内容

- **ROCm 支持**: 代码中未包含 ROCm 相关条件编译
- **Backward 传播**: 该文件仅实现 forward pass,无 backward 逻辑
