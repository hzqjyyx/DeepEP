看了这两个文件,主要功能是 **DeepEP 的 Python-C++ 绑定层和共享内存管理**。

## deep_ep.hpp - 核心数据结构定义

### SharedMemoryAllocator (L38-49)
GPU 共享内存分配器,支持两种底层实现:
- CUDA IPC (cudaIpcMemHandle_t) - 传统 IPC 机制
- CUDA Fabric API (CUmemFabricHandle) - 新的跨进程内存共享机制

提供统一接口: malloc/free/get_mem_handle/open_mem_handle/close_mem_handle

### Buffer 类 (L54-298)
DeepEP 的核心类,管理所有通信相关资源:

**缓冲区管理:**
- NVLink buffer (L63-65): `buffer_ptrs[8]` - 存储本地 GPU 可访问的所有 NVLink peer 的缓冲区指针
- RDMA buffer (L68-69): `rdma_buffer_ptr` - NVSHMEM 分配的远程可访问缓冲区
- Low-latency buffer (L59-60): 双缓冲机制,`low_latency_buffer_idx` 在 0/1 之间切换
- Shrink mode buffer (L72-74): mask/sync buffer 用于动态禁用某些 rank

**拓扑信息:**
- `rank/rdma_rank/nvl_rank`: 全局/RDMA组/NVLink组 内的 rank
- `num_ranks/num_rdma_ranks/num_nvl_ranks`: 对应的总数
- `ipc_handles[8]`: 存储所有 peer 的 IPC handle

**同步机制:**
- `comm_stream`: 专用通信流
- `barrier_signal_ptrs[8]`: 跨 rank barrier 的 signal 指针
- `moe_recv_counter*`: host-pinned 计数器,用于 CPU 轮询 GPU 接收进度(normal mode)
- `available/destroyed`: 生命周期标志

**核心 API (按执行顺序):**
1. `sync()` (L147): IPC handle 交换和 NVSHMEM 初始化
2. **Normal mode:**
   - `get_dispatch_layout()` (L153): 计算 token 分布
   - `intranode_dispatch()` (L171): NVLink 内节点 dispatch
   - `internode_dispatch()` (L217): RDMA 跨节点 dispatch
   - `intranode_combine()` (L188): NVLink combine
   - `internode_combine()` (L238): RDMA combine
3. **Low-latency mode:**
   - `low_latency_dispatch()` (L264): 返回 recv_hook 用于 overlap
   - `low_latency_combine()` (L276): 返回 recv_hook
   - `clean_low_latency_buffer()` (L255): 清理缓冲区
   - `get_next_low_latency_combine_buffer()` (L291): 获取下一个 combine buffer
   - `low_latency_update/query/clean_mask_buffer()` (L293-297): Shrink mode 掩码管理

## deep_ep.cpp - PyBind11 绑定实现

没看到完整代码,但从 hpp 推断主要是:
- 暴露 `Buffer` 类及其所有方法到 Python
- 处理 Python/C++ 类型转换 (torch::Tensor ↔ pybind11::object)
- 绑定 `shared_memory::SharedMemoryAllocator`
- 导出 `EventHandle` 等辅助类型

## 设计上的关键点

**IPC handle 交换流程 (L147 sync()):**
```
每个 rank:
1. malloc 本地 buffer
2. get_mem_handle 获取本地 handle
3. all_gather 收集所有 rank 的 handles
4. open_mem_handle 映射 peer buffers 到本地地址空间
→ 最终每个 rank 都能通过 buffer_ptrs[i] 访问 rank i 的 buffer
```

**为什么需要三个 counter (L102-111):**
- `moe_recv_counter`: normal mode 总接收 token 数,CPU 轮询它来确定何时可以读取
- `moe_recv_expert_counter`: expert 级别粒度的接收进度
- `moe_recv_rdma_counter`: RDMA 级别的接收进度
→ 多粒度计数器用于不同的同步场景

**explicitly_destroy (L90-92):**
有些使用场景需要手动调用 `destroy()`,有些依赖析构函数自动清理。这个标志控制析构函数行为,防止重复释放或提前释放。

**workspace (L99):**
临时工作空间,用于 kernel 内部的 scratch memory,避免反复分配。

---

**ROCm/Backward 相关:**
- ROCm: AMD GPU 支持(如果有的话,应该在 cpp 里有 `#ifdef USE_ROCM` 分支)
- Backward: MoE 反向传播时 dispatch/combine 方向相反,但这里只定义前向接口,反向应该复用或在 Python 层处理
