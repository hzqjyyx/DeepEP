## Buffer 管理结构

### **Buffer**
基础的单缓冲区结构，用于简单的线性内存管理：
- 持有单个指针 `ptr` 和总字节数 `total_bytes`
- 构造时从全局指针池 `gbl_ptr` 中分配连续内存，并自动推进全局指针
- `operator[]` 提供随机访问
- `advance_also()` 用于跳过已分配的内存空间

### **AsymBuffer**
非对称缓冲区，用于 Per-SM 分配场景（如 dispatch 中的接收缓冲）：
- 支持单秩（`kNumRanks=1`）或多秩（`kNumRanks>1`）的指针数组
- 构造时按 SM 划分：每个 SM 获得独立的内存窗口
  - 内存布局：`per_channel_bytes = num_bytes * num_ranks`，总大小乘以 `num_sms`
  - 偏移计算：`per_channel_bytes * sm_id + num_bytes * offset`
- `advance()` 支持在同一 SM 内部跳过指定数量的元素
- `buffer(idx)` / `buffer_by(rank_idx, idx)` 提供不同的访问接口

### **SymBuffer**
对称缓冲区，用于 send/recv 配对场景（如 combine 中的双向通信）：
- 支持解耦模式（`kDecoupled=true`）：维护独立的 `send_ptr` 和 `recv_ptr`
  - 内存翻倍：`total_bytes = per_channel_bytes * num_sms * 2`
- 支持耦合模式（`kDecoupled=false`）：只用 `send_ptr`
- 构造时自动分离 send 和 recv 缓冲区窗口

---

**核心设计特点：**
- 📌 都是 Device-side 的 inline 结构，零运行时开销
- 📌 通过指针池引用计数思想管理内存分配
- 📌 支持 Per-SM 隔离，避免 SM 间内存竞争
- 📌 模板参数化支持多种精度和通信拓扑
