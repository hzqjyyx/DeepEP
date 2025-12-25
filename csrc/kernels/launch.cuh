/**
 * @file launch.cuh
 * @brief CUDA kernel 启动配置宏和模板参数分发宏
 *
 * 本文件提供两类工具：
 * 1. Kernel 启动宏：封装 SM90 (Hopper) 的新特性（cooperative launch、cluster dimension）
 *    与 SM80 (Ampere) 的传统启动方式，通过 DISABLE_SM90_FEATURES 宏切换
 * 2. Switch 宏：将运行时参数（rank数、数据类型、hidden size）转换为编译时模板参数，
 *    用于实例化特定配置的 kernel
 *
 * CUDA 基础概念：
 *   - SM (Streaming Multiprocessor)：GPU 的计算单元，一个 GPU 有多个 SM
 *   - Thread：最小执行单元
 *   - Block：一组 thread，运行在同一个 SM 上，可通过 shared memory 通信
 *   - Grid：一组 block，组成完整的 kernel 执行
 *   - Cluster (SM90+)：一组相邻 SM，可通过 distributed shared memory 直接通信
 */

// 防止头文件被多次包含
#pragma once

// configs.cuh: 包含 kernel 配置常量（如 NUM_MAX_NVL_PEERS）
#include "configs.cuh"
// exception.cuh: 包含 EP_HOST_ASSERT、CUDA_CHECK、EPException 等错误处理工具
#include "exception.cuh"

/**
 * SETUP_LAUNCH_CONFIG - 初始化 kernel 启动配置
 *
 * SM90 版本：
 *   - 创建 cudaLaunchConfig_t 结构体
 *   - 启用 cooperative launch（允许 grid 内线程同步）
 *   - 设置 cluster dimension：若 SM 数为偶数则用 2x1x1 cluster，否则 1x1x1
 *     cluster 允许相邻 SM 间通过 distributed shared memory 通信
 *
 * SM80 版本：
 *   - 仅保存参数到局部变量，使用传统 <<<>>> 语法启动
 */
// #ifndef: 如果宏未定义则编译以下代码（防止重复定义）
#ifndef SETUP_LAUNCH_CONFIG
// 如果没有禁用 SM90 特性，使用新版 API
#ifndef DISABLE_SM90_FEATURES
// 宏定义，参数：num_sms=使用的SM数量, num_threads=每个block的线程数, stream=CUDA流
// 反斜杠 \ 表示宏定义跨行继续
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                       \
    /* cudaLaunchConfig_t: CUDA 12+ 的 kernel 启动配置结构体                     \
     * 成员依次为: gridDim(grid大小), blockDim(block大小), dynamicSmemBytes,     \
     *            stream(CUDA流), attrs(启动属性数组), numAttrs(属性数量)         \
     * 这里 num_sms 作为 gridDim，即启动 num_sms 个 block，每个 block 占一个 SM */  \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    /* cudaLaunchAttribute: 存储 kernel 启动的额外属性                            \
     * 这里声明一个长度为2的数组，用于存储两个属性 */                               \
    cudaLaunchAttribute attr[2];                                                \
    /* 第一个属性：启用 cooperative launch                                        \
     * cooperative launch 允许 grid 内所有线程进行全局同步（grid.sync()）          \
     * 普通 kernel 只能在 block 内同步（__syncthreads()）                         \
     * 这对于需要全局协调的通信操作很重要 */                                       \
    attr[0].id = cudaLaunchAttributeCooperative;                                \
    attr[0].val.cooperative = 1;  /* 1 表示启用 */                               \
    /* 第二个属性：设置 cluster dimension (Hopper SM90 新特性)                     \
     * Cluster 是介于 block 和 grid 之间的层级                                    \
     * 同一 cluster 内的 block 可以通过 distributed shared memory 直接通信        \
     * 比 global memory 通信更快 */                                               \
    attr[1].id = cudaLaunchAttributeClusterDimension;                           \
    /* clusterDim.x=2: 每个 cluster 包含 2 个 block（在 x 维度上）                 \
     * 如果 SM 数为奇数，则用 1（因为无法均分）                                    \
     * 这允许相邻两个 SM 上的 block 通过 distributed shared memory 通信 */        \
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      \
    attr[1].val.clusterDim.y = 1;  /* y 维度 cluster 大小为 1 */                  \
    attr[1].val.clusterDim.z = 1;  /* z 维度 cluster 大小为 1 */                  \
    /* 将属性数组绑定到配置结构体 */                                               \
    cfg.attrs = attr;                                                           \
    /* 告诉 CUDA 我们设置了 2 个属性 */                                            \
    cfg.numAttrs = 2
#else
// SM80 版本（或禁用 SM90 特性时）：使用传统启动方式
// 只是把参数保存到局部变量，供后面的 LAUNCH_KERNEL 宏使用
// 双下划线前缀 __ 用于避免与用户变量名冲突
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    int __num_sms = (sms);                        \
    int __num_threads = (threads);                \
    auto __stream = (stream)  /* auto 自动推导类型，这里是 cudaStream_t */
#endif
#endif

/**
 * LAUNCH_KERNEL - 启动 CUDA kernel
 *
 * SM90 版本：使用 cudaLaunchKernelEx 启动，支持 cooperative launch 和 cluster
 * SM80 版本：使用传统 <<<grid, block, smem, stream>>> 语法
 *
 * 注意：SM80 版本使用 SETUP_LAUNCH_CONFIG 中定义的 __num_sms, __num_threads, __stream
 */
#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
// cudaLaunchKernelEx: CUDA 12+ 的新 kernel 启动函数
// 参数: config=启动配置指针, kernel=kernel函数指针, ...=kernel参数
// CUDA_CHECK: 检查返回值，失败时抛出异常
// ##__VA_ARGS__: 可变参数，## 用于当没有额外参数时移除前面的逗号
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else
// SM80 版本：使用传统的 <<<>>> 语法启动 kernel
// 这是 CUDA C++ 的特殊语法，会被 nvcc 编译器处理
#define LAUNCH_KERNEL(config, kernel, ...)                                                 \
    /* do { ... } while(0) 是常见的宏技巧，让宏像单条语句一样使用                           \
     * 可以安全地在 if/else 等语句中使用，不会出现悬挂问题 */                               \
    do {                                                                                   \
        /* <<<gridDim, blockDim, sharedMem, stream>>>: CUDA kernel 启动语法                \
         * __num_sms: grid 中的 block 数量（每个 block 对应一个 SM）                       \
         * __num_threads: 每个 block 中的线程数                                            \
         * 0: 动态共享内存大小（这里不使用）                                                \
         * __stream: CUDA 流，用于异步执行和排序操作 */                                     \
        kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__);                    \
        /* cudaGetLastError(): 获取最后一个 CUDA 错误                                       \
         * kernel 启动是异步的，这里只能捕获启动时的错误（如参数错误）                       \
         * kernel 执行中的错误需要 cudaDeviceSynchronize() 后才能获取 */                    \
        cudaError_t e = cudaGetLastError();                                                \
        if (e != cudaSuccess) {                                                            \
            /* 创建异常对象，包含错误来源、文件名、行号、错误信息 */                         \
            EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
            /* 输出到标准错误流 */                                                          \
            fprintf(stderr, "%s\n", cuda_exception.what());                                \
            /* 抛出异常，中断执行 */                                                        \
            throw cuda_exception;                                                          \
        }                                                                                  \
    } while (0)
#endif
#endif

/**
 * SET_SHARED_MEMORY_FOR_TMA - 配置 TMA (Tensor Memory Accelerator) 所需的动态共享内存
 *
 * TMA 是 Hopper (SM90) 引入的硬件单元，可以高效地在 global memory 和 shared memory
 * 之间传输多维张量数据，无需 GPU 线程介入。
 *
 * SM90 版本：
 *   - 设置 kernel 的最大动态共享内存大小（默认限制较小，需要显式提高）
 *   - 在 launch config 中指定实际使用的共享内存字节数
 *   - 依赖外部定义的 smem_size 变量
 *
 * SM80 版本：空操作（SM80 不支持 TMA）
 */
#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                \
    /* cudaFuncSetAttribute: 设置 kernel 函数的属性                                                                       \
     * cudaFuncAttributeMaxDynamicSharedMemorySize: 允许 kernel 使用的最大动态共享内存                                     \
     * 默认限制是 48KB，但 Hopper 支持高达 228KB                                                                          \
     * 需要在启动前显式设置更大的值，否则超限会报错                                                                         \
     * smem_size: 外部定义的变量，表示需要的共享内存大小 */                                                                 \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
    /* 在启动配置中设置实际使用的动态共享内存字节数 */                                                                      \
    cfg.dynamicSmemBytes = smem_size;
#else
// SM80 版本：空操作，void() 是一个不做任何事的表达式
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

/*==============================================================================
 * 模板参数分发宏
 *
 * 以下 SWITCH_* 宏将运行时参数映射到编译时模板参数。
 * 这是 CUDA 常见的 "template instantiation" 模式：
 *   - 编译器为每种参数组合生成特化版本（编译时展开）
 *   - 运行时根据实际参数选择对应版本执行
 *   - 允许编译器针对具体参数做优化（展开循环、常量折叠等）
 *
 * 为什么需要这种模式？
 *   - GPU kernel 中循环展开、常量参数可以显著提升性能
 *   - 但这要求参数在编译时已知
 *   - 通过 switch 可以将运行时值"转换"为编译时常量
 *
 * 使用方式：
 *   #define CASE_MACRO(RANKS) do { my_kernel<RANKS>(...); return; } while(0)
 *   SWITCH_RANKS(CASE_MACRO);
 *
 * 注意：case_macro 内部必须 return 或 break，否则会 fall-through 到下一个 case
 * （这里故意没有 break，依赖 case_macro 内部的 return）
 *============================================================================*/

/**
 * SWITCH_RANKS - 根据节点内 rank 数量分发
 *
 * rank：分布式计算中的进程编号，这里每个 GPU 对应一个 rank
 * 支持的 rank 数：2, 4, 8（对应单节点内 2/4/8 GPU 的配置）
 * 用于 intranode（节点内）NVLink 通信 kernel
 */
#define SWITCH_RANKS(case_macro)                           \
    /* num_ranks: 外部定义的变量，表示参与通信的 rank 数量 */ \
    switch (num_ranks) {                                   \
        case 2:                                            \
            /* case_macro(2) 会展开为调用 kernel<2>(...) 的代码 */ \
            case_macro(2);                                 \
        case 4:                                            \
            case_macro(4);                                 \
        case 8:                                            \
            case_macro(8);                                 \
        default:                                           \
            /* 不支持的 rank 数，触发断言失败 */            \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    /* while(false) 配合 do-while 或单独使用都是安全的       \
     * 这里确保宏后面可以加分号而不产生空语句警告 */          \
    while (false)

/**
 * SWITCH_RDMA_RANKS - 根据 RDMA 节点数量分发
 *
 * RDMA (Remote Direct Memory Access): 允许网卡直接读写远程 GPU 内存
 * InfiniBand/RoCE 网络使用此技术实现低延迟跨节点通信
 *
 * 计算方式：num_ranks / NUM_MAX_NVL_PEERS
 *   - num_ranks: 总 rank 数（所有 GPU 数量）
 *   - NUM_MAX_NVL_PEERS: 每节点内的 GPU 数（NVLink 连接的 peer 数，通常为 8）
 *   - 相除得到节点数
 *
 * 支持的节点数：2, 3, 4, 6, 8, 12, 16, 18, 20
 * 用于 internode（跨节点）RDMA 通信 kernel
 */
#define SWITCH_RDMA_RANKS(case_macro)                           \
    switch (num_ranks / NUM_MAX_NVL_PEERS) {                    \
        case 2:                                                 \
            case_macro(2);                                      \
        case 3:                                                 \
            case_macro(3);                                      \
        case 4:                                                 \
            case_macro(4);                                      \
        case 6:                                                 \
            case_macro(6);                                      \
        case 8:                                                 \
            case_macro(8);                                      \
        case 12:                                                \
            case_macro(12);                                     \
        case 16:                                                \
            case_macro(16);                                     \
        case 18:                                                \
            case_macro(18);                                     \
        case 20:                                                \
            case_macro(20);                                     \
        default:                                                \
            EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    }                                                           \
    while (false)

/**
 * SWITCH_RANKS_WITH_DTYPE - 带数据类型的 rank 分发
 *
 * 与 SWITCH_RANKS 类似，但 case_macro 接收两个参数：(dtype, ranks)
 * 用于需要同时特化数据类型和 rank 数的 kernel
 *
 * 例如：kernel<nv_bfloat16, 8>(...) 表示使用 bfloat16 类型、8 个 rank
 */
#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)         \
    switch (num_ranks) {                                   \
        case 2:                                            \
            /* 展开为 kernel<dtype, 2>(...) */              \
            case_macro(dtype, 2);                          \
        case 4:                                            \
            case_macro(dtype, 4);                          \
        case 8:                                            \
            case_macro(dtype, 8);                          \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

/**
 * SWITCH_TYPES - 根据数据类型分发
 *
 * 将 CUDA 数据类型枚举 (cudaDataType) 映射到 C++ 类型
 * 目前仅支持：CUDA_R_16BF -> nv_bfloat16
 *
 * nv_bfloat16: NVIDIA 的 Brain Float 16 类型
 *   - 16位浮点数，1位符号 + 8位指数 + 7位尾数
 *   - 与 float32 有相同的指数范围（动态范围），但精度较低
 *   - 在深度学习训练中广泛使用，平衡了精度和性能
 *
 * CUDA_R_16BF: cudaDataType 枚举值，表示实数 bfloat16
 */
#define SWITCH_TYPES(case_macro)                          \
    /* type: 外部定义的 cudaDataType 变量 */               \
    switch (type) {                                       \
        case CUDA_R_16BF:                                 \
            /* 将枚举值映射为 C++ 类型 nv_bfloat16 */       \
            case_macro(nv_bfloat16);                      \
        default:                                          \
            EP_HOST_ASSERT(false and "Unsupported type"); \
    }                                                     \
    while (false)

/**
 * SWITCH_HIDDEN - 根据 hidden dimension 分发
 *
 * hidden dimension: Transformer 模型中的隐藏层维度
 * 这是 MoE (Mixture of Experts) 中 token 的特征向量长度
 *
 * 支持的 hidden size：2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192
 * 这些值对应不同模型的隐层维度：
 *   - 3072: GPT-OSS
 *   - 6144: Qwen3 Coder
 *   - 7168: DeepSeek 系列
 *
 * 为什么要编译时特化？
 *   - hidden size 决定了内存访问模式和循环次数
 *   - 编译时已知可以让编译器完全展开循环
 *   - 可以优化向量化加载（如 128-bit load 需要对齐和固定大小）
 *   - 减少运行时分支判断开销
 */
#define SWITCH_HIDDEN(case_macro)                           \
    /* hidden: 外部定义的变量，表示隐藏层维度 */             \
    switch (hidden) {                                       \
        case 2048:                                          \
            case_macro(2048);                               \
        case 2560:                                          \
            case_macro(2560);                               \
        case 3072:                                          \
            case_macro(3072); /* for gpt-oss */             \
        case 4096:                                          \
            case_macro(4096);                               \
        case 5120:                                          \
            case_macro(5120);                               \
        case 6144:                                          \
            case_macro(6144); /* For qwen3 coder */         \
        case 7168:                                          \
            case_macro(7168);                               \
        case 8192:                                          \
            case_macro(8192);                               \
        default:                                            \
            EP_HOST_ASSERT(false and "Unsupported hidden"); \
    }                                                       \
    while (false)
