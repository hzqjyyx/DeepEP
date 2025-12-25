# CUDA Q&A

## 什么是 clusterDim？

### CUDA 的执行层级演进

**传统模型（SM80 及之前）**

```
Grid（整个 kernel）
 └── Block（一组 thread，运行在单个 SM 上）
      └── Thread（最小执行单元）
```

Block 内的线程可以通过 **shared memory** 通信（~100 KB，延迟 ~20 cycles）。
但 Block 之间只能通过 **global memory** 通信（延迟 ~200-400 cycles），很慢。

**Hopper 引入 Cluster（SM90）**

```
Grid
 └── Cluster（一组相邻的 Block，运行在相邻的 SM 上）  ← 新层级
      └── Block
           └── Thread
```

同一 Cluster 内的 Block 可以通过 **Distributed Shared Memory (DSMEM)** 直接访问彼此的 shared memory，无需绕道 global memory。

### clusterDim 的作用

`clusterDim` 定义每个 Cluster 包含多少个 Block，是一个三维向量 `(x, y, z)`：

```
Cluster 中的 Block 数量 = clusterDim.x × clusterDim.y × clusterDim.z
```

**示例**

```cpp
clusterDim = (2, 2, 1)   // 每个 cluster 有 2×2×1 = 4 个 block
```

```
Grid 被划分为多个 Cluster：

┌─────────────────┐  ┌─────────────────┐
│   Cluster 0     │  │   Cluster 1     │
│ ┌─────┬─────┐   │  │ ┌─────┬─────┐   │
│ │ B0  │ B1  │   │  │ │ B4  │ B5  │   │
│ ├─────┼─────┤   │  │ ├─────┼─────┤   │
│ │ B2  │ B3  │   │  │ │ B6  │ B7  │   │
│ └─────┴─────┘   │  │ └─────┴─────┘   │
└─────────────────┘  └─────────────────┘
     ↑
  这 4 个 Block 可以互相访问对方的 shared memory
```

### 为什么需要 Cluster？

| 通信方式 | 延迟 | 适用场景 |
|---------|------|---------|
| Shared Memory（Block 内） | ~20 cycles | 同一 Block 内线程协作 |
| **DSMEM（Cluster 内）** | ~30 cycles | **相邻 Block 间协作** ← 新增 |
| Global Memory（任意 Block） | ~200-400 cycles | 全局数据交换 |

Cluster 填补了 "Block 内通信很快，Block 间通信很慢" 的鸿沟。

### 实际应用场景

1. **矩阵乘法**：相邻 Block 共享输入 tile，避免重复从 global memory 加载
2. **通信库（如 DeepEP）**：相邻 SM 协调数据搬运
3. **Reduce 操作**：Cluster 内先做局部 reduce，再汇总

### 限制

- Cluster 最大 16 个 Block（硬件限制）
- `clusterDim.x × clusterDim.y × clusterDim.z ≤ 16`
- Cluster 内的 Block 必须在物理上相邻的 SM 上运行

### DeepEP 中的用法

```cpp
// csrc/kernels/launch.cuh
attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);  // x 维度 2 个 block
attr[1].val.clusterDim.y = 1;                           // y 维度 1 个 block
attr[1].val.clusterDim.z = 1;                           // z 维度 1 个 block
```

DeepEP 只用 `clusterDim.x = 2`，因为通信是一维数据流（token 序列），只需让相邻 2 个 SM 通过 DSMEM 通信即可。
