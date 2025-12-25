# launch.cuh 核心功能解析

这个文件是 CUDA kernel 启动的基础设施层，解决两个核心问题：

## 1. 抹平 SM90/SM80 架构差异

提供统一的 kernel 启动接口，底层根据 `DISABLE_SM90_FEATURES` 宏自动切换实现：

**SM90 (Hopper) 路径**
- 使用 `cudaLaunchKernelEx` 新 API
- 启用 cooperative launch：允许跨 block 全局同步（grid.sync()）
- 配置 cluster dimension：2x1x1 或 1x1x1，让相邻 SM 通过 distributed shared memory 通信
- TMA 专用配置：显式设置最大动态共享内存（默认 48KB → 可达 228KB）

**SM80 (Ampere) 路径**
- 传统 `<<<>>>` 语法启动
- 不支持 cluster 和 TMA
- 纯粹的参数传递和错误检查

典型使用：
```cpp
SETUP_LAUNCH_CONFIG(num_sms, 256, stream);
SET_SHARED_MEMORY_FOR_TMA(my_kernel);  // SM90 专用
LAUNCH_KERNEL(&cfg, my_kernel, arg1, arg2);
```

## 2. 运行时参数 → 编译时模板参数

通过 switch-case 宏将运行时值映射到编译时常量，强制编译器为每种组合生成特化版本。

**核心动机**：GPU kernel 性能依赖编译时优化
- 循环展开需要固定次数
- 向量化加载需要对齐和固定大小
- 分支消除需要常量条件

**提供的分发宏**：

| 宏名 | 参数范围 | 用途 |
|-----|---------|------|
| `SWITCH_RANKS` | 2/4/8 | 节点内 NVLink 通信的 GPU 数量 |
| `SWITCH_RDMA_RANKS` | 2~20 | 跨节点 RDMA 通信的节点数量 |
| `SWITCH_TYPES` | CUDA_R_16BF | 数据类型（目前仅 bfloat16）|
| `SWITCH_HIDDEN` | 2048~8192 | Transformer 隐藏层维度 |

典型使用：
```cpp
// 定义 case 展开逻辑
#define LAUNCH_CASE(HIDDEN) \
    do { my_kernel<HIDDEN><<<...>>>(); return; } while(0)

// 分发到特化版本
SWITCH_HIDDEN(LAUNCH_CASE);  // 根据 hidden 变量选择 kernel<2048/3072/...>
```

## 为什么要这样设计

**问题**：DeepEP 的通信 kernel 性能极度敏感
- hidden size 决定循环次数，编译时展开可减少 30%+ 指令
- rank 数决定内存访问模式，常量化可启用向量化加载
- cluster 配置影响 distributed shared memory 拓扑

**权衡**：
- ✅ 牺牲编译时间和二进制大小（每种组合都生成代码）
- ✅ 换取运行时零开销分发 + 激进编译器优化
- ✅ 适合参数空间有限的场景（这里只有 8×10×1×8 = 640 种组合）

## 技术细节

**cooperative launch 的意义**
- 普通 kernel 只能 block 内同步（`__syncthreads()`）
- cooperative launch 允许 grid 全局同步（`grid.sync()`）
- DeepEP 的通信 kernel 需要多个 SM 协同完成 all-to-all 操作

**cluster dimension 的作用**
- 传统：block 只能用 shared memory，需要 global memory 中转跨 SM 通信
- cluster：同一 cluster 内的 block 可通过 distributed shared memory 直接通信
- 2x1x1 配置：每两个相邻 SM 形成 cluster，减少通信跳数

**TMA 共享内存限制**
- Hopper 默认动态共享内存上限 48KB
- TMA 传输大张量需要更大 buffer（DeepEP 用到 228KB）
- 必须在启动前显式提高限制，否则 kernel 启动失败

---

**其他提及内容**：
- ROCm 相关：无（文件纯 CUDA）
- Backward 相关：无（纯前向通信基础设施）
