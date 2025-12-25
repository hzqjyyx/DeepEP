这个文件定义了 DeepEP 的异常处理机制：

**核心功能：**

- **EPException 类** - 自定义异常类，继承自 `std::exception`，用于捕获和格式化 CUDA/GPU 错误信息
  - 构造函数接收 error name、file、line、error message
  - 格式化输出: `Failed: {name} error {file}:{line} '{error}'`

- **CUDA_CHECK 宏** - 包装 CUDA API 调用
  - 执行命令后检查返回值是否为 `cudaSuccess`
  - 失败时抛出 EPException，包含 CUDA error string

- **CU_CHECK 宏** - 包装 CUDA Driver API 调用
  - 检查返回值是否为 `CUDA_SUCCESS`
  - 失败时通过 `cuGetErrorString()` 获取错误描述并抛出异常

- **EP_HOST_ASSERT 宏** - Host 端断言
  - 条件失败时抛出 EPException

- **EP_DEVICE_ASSERT 宏** - Device 端断言
  - 条件失败时用 `printf` 打印错误信息
  - 执行 `asm("trap;")` 触发 GPU trap 停止执行

- **EP_STATIC_ASSERT 宏** - 编译时静态断言（可选定义）

**设计特点：**
- 统一的异常处理策略，便于调试
- Host 和 Device 分离处理（Host 抛异常，Device 用 trap）
- 宏定义允许重新定义，支持自定义实现
