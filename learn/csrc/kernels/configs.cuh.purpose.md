根据文件内容，这是 DeepEP 项目的 CUDA 核心配置头文件，主要功能如下：

**系统常量定义**
- NVLink 对端数上限（`NUM_MAX_NVL_PEERS`）和 RDMA 对端数上限（`NUM_MAX_RDMA_PEERS`）
- GPU 工作空间大小（32MB）和本地专家数上限（1024）
- 缓冲区对齐要求（128字节）

**通信协议参数**
- 完成信号标签（`FINISHED_SUM_TAG`）
- 自旋等待纳秒数（500ns）
- 低延迟模式的收发阶段定义

**超时控制**
- 条件编译支持调试模式和正式模式下不同的超时时间
- 正式模式：100秒超时；调试模式：10秒超时

**编译器兼容性处理**
- CLion IDE 的 CUDA 索引支持（定义 `__CUDA_ARCH__` 和 `__CUDACC_RDC__`）
- 移除 PyTorch 的 FP16/BF16 限制（undefine 相关宏）

**精度和架构支持**
- 条件引入 FP8 支持（SM90 及以上），Ampere 架构降级处理
- 根据编译选项支持不同的 topk 索引位宽（32 或 64 位）

**外部库集成**
- CUDA 运行时和 BF16 库
- NVSHMEM 库（可选，用于多节点通信）
- InfiniBand 驱动库
