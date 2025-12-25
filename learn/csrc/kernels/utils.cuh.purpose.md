# csrc/kernels/utils.cuh 功能分析

这是一个底层 CUDA 工具库,提供硬件级内存操作、同步原语和数学工具。核心是用 inline PTX 汇编绕过编译器优化,实现精确控制的内存语义。

## 核心功能模块

### 1. 内存操作原语 (Memory Operations)

**读操作家族:**
- `ld_nc_global<T>()` - 非缓存读取,用 `ld.global.nc.L1::no_allocate.L2::256B` PTX 指令
  - 避免污染 L1/L2 cache,适合一次性读取大块数据
  - 支持 uint8/int/int64/float/int2/int4 等类型
  
- `ld_na_relaxed<T>()` - 松散内存序的无缓存读取
  - L1 不分配,内存序最弱,用于性能敏感路径
  
- `ld_acquire_*()` - acquire 语义读取
  - `ld_acquire_sys_global()` - 系统级 acquire (跨 GPU 可见)
  - `ld_acquire_global()` - GPU 级 acquire
  - `ld_acquire_cta()` - CTA 级 acquire
  - 保证后续内存操作在读取完成后执行

- `ld_volatile_global<T>()` - volatile 读取
  - 强制从内存读取,不优化掉重复读

**写操作家族:**
- `st_na_global<T>()` - 非缓存写入,用 `st.global.L1::no_allocate` PTX
  - L1 不分配,减少缓存压力
  
- `st_na_relaxed<T>()` - 松散内存序的无缓存写入
  
- `st_na_release<T>()` - release 语义的无缓存写入
  - 保证之前的内存操作在写入前完成
  
- `st_release_sys_global()` / `st_release_cta()` - release 语义写入
  - 用于跨线程/跨 GPU 的内存同步

**原子操作:**
- `atomic_add_release_sys_global()` / `atomic_add_release_global()` - release 语义原子加
- `atomic_cas_cta_acquire()` - acquire 语义的 CAS (Compare-And-Swap)
- `atomic_exch_cta_release()` - release 语义的 exchange

### 2. 内存屏障 (Memory Fences)

```cpp
memory_fence()         // fence.acq_rel.sys - 系统级屏障(最强)
memory_fence_gpu()     // fence.acq_rel.gpu - GPU 级屏障
memory_fence_cta()     // fence.acq_rel.cta - CTA 级屏障
```

用于显式控制内存操作的可见性顺序,配合 acquire/release 语义实现无锁同步。

### 3. Warp 级并行工具

**`UNROLLED_WARP_COPY` 宏 (lines 5-28):**
```cpp
UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)
```
- 用 warp 的 32 个 lane 并行拷贝数据
- `UNROLL_FACTOR` 控制每个 lane 的循环展开次数
- 主循环处理对齐的部分,尾部处理边界条件
- `LD_FUNC`/`ST_FUNC` 可以是任意读写函数(如 `ld_nc_global`),实现灵活的内存语义

**Warp Reduce 操作 (lines 582-638):**
```cpp
warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T, Op>(value, op)
```
- 通用 warp 规约框架,支持任意二元操作符 `Op`
- `kNumLanesPerGroup`: 规约组大小 (1/2/4/8/16/32)
- `kIntergroupReduce`: 是否跨组规约
- 预定义: `warp_reduce_sum/max/min/and/or`
- 用 `__shfl_xor_sync` 实现高效的蝶形规约

**其他 warp 工具:**
- `get_lane_id()` - 获取 warp 内 lane ID
- `elect_one_sync()` - SM90 特性,选举一个 lane 执行操作
- `broadcast<T>()` - 广播任意类型数据到 warp 所有 lane

### 4. 同步原语

**Barrier 同步 (lines 504-534):**
```cpp
barrier_block<kNumRanks, kSyncOnly>(barrier_signal_ptrs, rank)
```
- 多 GPU 间的块级屏障
- 用原子操作实现 arrival/departure 计数
- 支持超时检测 (`NUM_TIMEOUT_CYCLES`)
- `kSyncOnly=true` 时只做线程同步,否则包含 `memory_fence()`

**锁原语 (lines 548-557):**
```cpp
acquire_lock(mutex)   // CAS-based spinlock
release_lock(mutex)
```
- 基于 `atomic_cas_cta_acquire` / `atomic_exch_cta_release`
- acquire/release 语义保证临界区内存操作的正确性

### 5. TMA (Tensor Memory Accelerator) 指令 (SM90 only, lines 336-416)

需要 `#ifndef DISABLE_SM90_FEATURES`:

**Mbarrier 操作:**
- `mbarrier_init()` / `mbarrier_inval()` - 初始化/失效屏障
- `mbarrier_arrive_and_expect_tx()` - 到达并期待传输
- `mbarrier_wait<kWithMultiStages>()` - 等待屏障完成
- `fence_barrier_init()` - 集群级屏障初始化

**异步拷贝:**
- `tma_load_1d()` - 全局内存 → 共享内存 (带 mbarrier 完成通知)
- `tma_store_1d()` - 共享内存 → 全局内存
- `tma_store_wait<N>()` - 等待 N 个写入完成
- 支持 `evict_first` 缓存提示 (`kEvictFirst`/`kEvictNormal`)

### 6. 数学工具

**对齐/除法:**
```cpp
ceil_div(a, b)        // ⌈a/b⌉
align_up(a, b)        // 向上对齐到 b 的倍数
align_down(a, b)      // 向下对齐
```

**任务分配:**
```cpp
get_channel_task_range(num_tokens, num_sms, sm_id, start, end)
```
将 `num_tokens` 均匀分配给 `num_sms` 个 SM。

**FP8 量化工具 (lines 467-502):**
- `calculate_fp8_scales()` - 计算 FP8 量化 scale/scale_inv
  - 支持 round_scale (2 的幂次) 或连续 scale
  - 基于 E4M3 格式的最大值 448.0
- `fast_pow2()` / `fast_log2_ceil()` - 快速 2 的幂次计算 (位操作实现)
- `extract_required_scale_format<kIsUE8M0>()` - 提取 scale 为 UE8M0 或 FP32 格式

**近似数学:**
```cpp
log2f_approx(x)   // lg2.approx.f32 PTX
exp2f_approx(x)   // ex2.approx.f32 PTX
```

### 7. 类型工具

**VecInt 模板 (lines 32-53):**
将字节数映射到整型向量类型:
- 1 byte → `int8_t`
- 2 bytes → `int16_t`
- 4 bytes → `int`
- 8 bytes → `int64_t`
- 16 bytes → `int4`

用于泛型内存操作的类型擦除。

**Pack/Unpack (lines 440-454):**
```cpp
pack2<dtype_a_t, dtype_b_t>(x, y)      // 两个 dtype_a_t 打包成 dtype_b_t
unpack2<dtype_a_t, dtype_b_t>(packed, x, y)
```

**PatternVisitor (lines 55-62):**
函数对象包装器,用于索引模式访问 (`visitor[i]`)。

### 8. 其他工具

- `trap()` - 触发 GPU trap 异常 (用于错误处理)
- `EP_STATIC_ASSERT` / `EP_DEVICE_ASSERT` - 静态/运行时断言 (定义在 `exception.cuh`)

## 设计思路

1. **内存语义精确控制**: 用 PTX 汇编绕过 C++ 内存模型的保守优化,实现 relaxed/acquire/release 语义的精确组合。

2. **Cache-aware 优化**: `nc` (non-coherent) 和 `na` (no-allocate) 指令避免无效缓存污染,适合流式数据访问。

3. **SM90 条件编译**: TMA 和 `elect_one_sync` 等新特性用宏隔离,兼容 SM80。

4. **零抽象成本**: 全部 `__forceinline__` + 模板特化,运行时零开销。

5. **激进的 PTX 使用**: 
   - `ld.global.nc.L1::no_allocate.L2::256B` 在部分文档中属于未定义行为
   - 实践中在 Hopper 上有效,但需要 `DISABLE_AGGRESSIVE_PTX_INSTRS` 逃生舱

---

**次要内容 (Backward/ROCm):**
- 文件中没有显式的 backward 相关代码
- 没有 ROCm/HIP 特定代码 (DeepEP 纯 CUDA 实现)
