这个文件是 DeepEP 的 CUDA 内核 API 头文件，定义了三大通信模块的内核启动接口：

## **Intranode（节点内 NVLink 通信）**
- `barrier()`: 节点内同步屏障
- `notify_dispatch()`: 通知接收端 tokens 到达的计数器更新
- `cached_notify_dispatch()`: 缓存版本，复用前次计算的 prefix sum
- `dispatch()`: 将 tokens 按 expert 分散到各 rank
- `cached_notify_combine()`: 通知发送端准备接收的计数器
- `combine()`: 从各 rank 聚合 tokens 回来

## **Internode（跨节点 RDMA 通信）**
- `get_source_meta_bytes()`: 查询源信息元数据大小
- `notify_dispatch()`: 跨节点通知（处理 RDMA + NVL 混合）
- `dispatch()`: RDMA/NVL 混合分散 tokens
- `cached_notify()`: 缓存版本的跨节点通知
- `combine()`: RDMA/NVL 混合聚合 tokens

## **Internode Low-Latency（推理低延迟模式）**
- `clean_low_latency_buffer()`: 清理 LL buffer
- `dispatch()`: 推理阶段的低延迟分散（CUDA Graph 兼容）
- `combine()`: 推理阶段的低延迟聚合（支持 zero-copy）
- `query_mask_buffer()`: 查询 rank mask 状态
- `update_mask_buffer()`: 更新 rank mask
- `clean_mask_buffer()`: 清理 mask buffer

**核心特征：**
- 支持 FP8/BF16 多精度（via `topk_idx_t`、`cudaDataType_t`）
- Token 级颗粒度控制（`is_token_in_rank` 布尔数组）
- SM 资源管理（`num_sms` 参数）
- 缓存优化版本复用 prefix sum 计算
- LL 模式支持 CUDA Graph + hook 重叠
