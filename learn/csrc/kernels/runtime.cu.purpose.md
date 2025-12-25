这个文件定义了 DeepEP 的运行时初始化和通信基础设施：

**Intranode 命名空间 (行 16-33):**
- `barrier()` 函数：实现 GPU 线程块级别的屏障同步
- 通过模板参数 `kNumRanks` 支持不同 rank 数量
- 使用 `SWITCH_RANKS` 宏在编译时展开不同配置的 kernel 启动

**Internode 命名空间 (行 35-96):**
- **NVSHMEM 初始化：**
  - `get_unique_id()`：获取 NVSHMEM 全局唯一 ID 用于跨节点同步
  - `init()`：初始化 NVSHMEM，创建 CPU RDMA team（当 `num_ranks > NUM_MAX_NVL_PEERS` 时）
  - 行 58-69：低延迟模式下按 `NUM_MAX_NVL_PEERS` 分组创建子 team

- **内存管理：**
  - `alloc()`：通过 NVSHMEM 分配对齐内存
  - `free()`：释放 NVSHMEM 内存

- **同步和清理：**
  - `barrier()`：全局屏障同步
  - `finalize()`：销毁 RDMA team 并清理 NVSHMEM

**关键设计点：**
- `cpu_rdma_team` 是全局变量，为低延迟模式下的 RDMA 通信创建独立 team
- 低延迟模式条件：`num_ranks > NUM_MAX_NVL_PEERS` 且 `num_ranks % NUM_MAX_NVL_PEERS == 0`
