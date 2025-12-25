这个文件定义了 DeepEP 的 CUDA 核心库的编译配置：

**核心功能:**

- **定义编译函数** - `add_deep_ep_library()` 是个 CMake 函数，封装了所有 CUDA 库的通用编译选项
  - 启用位置独立代码 (PIC)
  - 设置 C++17 和 CUDA 17 标准
  - 启用 CUDA 可分离编译 (separable compilation)
  - 链接 NVSHMEM、CUDA 运行时库和 InfiniBand 驱动

- **编译五个核心库:**
  1. `runtime_cuda` - NVSHMEM 初始化和缓冲区分配
  2. `layout_cuda` - Token 分布计算
  3. `intranode_cuda` - NVLink 通信
  4. `internode_cuda` - RDMA 通信（正常模式）
  5. `internode_ll_cuda` - RDMA 通信（低延迟模式）

- **导出库列表** - 将所有编译好的库保存到 `EP_CUDA_LIBRARIES` 变量，供上层 CMakeLists.txt 链接
