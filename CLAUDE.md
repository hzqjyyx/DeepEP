# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepEP is a high-performance communication library for Mixture-of-Experts (MoE) models with expert parallelism (EP). It provides optimized GPU kernels for all-to-all communication patterns used in MoE dispatch and combine operations, supporting both high-throughput (training/prefilling) and low-latency (inference decoding) scenarios.

**Key Features:**
- Intranode communication via NVLink
- Internode communication via RDMA (InfiniBand/RoCE)
- Low-latency kernels for inference decoding with hook-based overlapping
- FP8 and BF16 precision support
- CUDA Graph compatibility (low-latency mode)
- SM (Streaming Multiprocessor) control for resource management

## Build and Installation

### Prerequisites
- NVSHMEM library (required for internode/low-latency features)
- CUDA 11.0+ for SM80 (Ampere), CUDA 12.3+ for SM90 (Hopper)
- PyTorch 2.1+
- Python 3.8+

### Build Commands

**Development build with symbolic link:**
```bash
# Build the C++/CUDA extension
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build

# Create symbolic link to the built extension (adjust SO name for your platform)
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so
```

**Install:**
```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

**Build environment variables:**
- `NVSHMEM_DIR`: Path to NVSHMEM installation (omit to disable internode features)
- `DISABLE_SM90_FEATURES`: Set to `1` for SM80 devices or CUDA 11
- `TORCH_CUDA_ARCH_LIST`: Target architectures (e.g., `"9.0"` for H800, `"8.0"` for A100)
- `DISABLE_AGGRESSIVE_PTX_INSTRS`: Set to `1` to disable aggressive load/store PTX instructions (required for non-SM90 architectures)
- `TOPK_IDX_BITS`: Bits for topk_idx dtype (32 or 64)

### Testing

**Run test suites:**
```bash
# Intranode tests (NVLink only)
python tests/test_intranode.py

# Internode tests (requires NVSHMEM and multi-node setup)
python tests/test_internode.py

# Low-latency tests (requires NVSHMEM)
python tests/test_low_latency.py
```

**Note:** You may need to modify `init_dist()` in `tests/utils.py` according to your cluster configuration (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK).

### Code Formatting

```bash
# Format Python code with yapf (column_limit=140, indent_width=4)
yapf -i <file.py>

# Lint with ruff (configured in pyproject.toml)
ruff check <file.py>
```

## Architecture

### Layer Structure

**Python Layer (`deep_ep/`):**
- `buffer.py`: Main `Buffer` class - manages communication buffers and provides high-level dispatch/combine APIs
- `utils.py`: `EventOverlap` class for CUDA event management and helper utilities
- `__init__.py`: Exports `Buffer`, `EventOverlap`, `Config`, and `topk_idx_t`

**C++/CUDA Layer (`csrc/`):**
- `deep_ep.cpp`: PyBind11 bindings, shared memory management (CUDA IPC/Fabric API)
- `kernels/runtime.cu`: NVSHMEM initialization, buffer allocation, IPC handle exchange
- `kernels/layout.cu`: Token layout calculation for dispatch operations
- `kernels/intranode.cu`: NVLink-based intranode kernels
- `kernels/internode.cu`: RDMA-based internode kernels (normal mode)
- `kernels/internode_ll.cu`: RDMA-based low-latency kernels
- `kernels/configs.cuh`: Auto-tuned kernel configurations
- `kernels/api.cuh`: Kernel launch APIs
- `kernels/buffer.cuh`: Buffer management structures
- `kernels/ibgda_device.cuh`: NVSHMEM device-side operations (subject to NVSHMEM SLA)

### Communication Patterns

**Normal Mode (Training/Prefilling):**
- High-throughput dispatch and combine operations
- Asymmetric bandwidth forwarding (NVLink → RDMA)
- CPU waits for GPU signal to determine received token count
- Supports SM control via `Buffer.set_num_sms()`
- Not CUDA Graph compatible (unless `num_worst_tokens` is specified)

**Low-Latency Mode (Inference Decoding):**
- Pure RDMA communication with minimal latency
- Hook-based receive mechanism for computation-communication overlap
- CUDA Graph compatible
- Requires `num_qps_per_rank` equal to number of local experts
- Fixed buffer sizes (recommend batch size < 256)

### Key Classes and APIs

**`Buffer` class:**
- `dispatch()`: MoE dispatch operation (scatter tokens to experts)
- `combine()`: MoE combine operation (gather tokens from experts)
- `get_dispatch_layout()`: Calculate token distribution layout
- `low_latency_dispatch()`: Low-latency dispatch for inference
- `low_latency_combine()`: Low-latency combine for inference
- `get_dispatch_config()` / `get_combine_config()`: Get auto-tuned configurations
- `set_num_sms()`: Static method to control SM usage

**`EventOverlap` class:**
- Manages CUDA events for async operations and communication-computation overlap
- Used with `async_finish=True` in dispatch/combine operations

## Important Implementation Details

### Undefined-Behavior PTX Usage

DeepEP uses `ld.global.nc.L1::no_allocate.L2::256B` to read volatile data for extreme performance on Hopper architectures. This is technically undefined behavior but empirically correct. If kernels fail on your platform, set `DISABLE_AGGRESSIVE_PTX_INSTRS=1` during build.

### Multi-Node Testing

Tests require proper distributed setup. Modify `tests/utils.py::init_dist()` to match your cluster's environment variables (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK). Launch tests across multiple nodes using your cluster's job scheduler (e.g., SLURM, torchrun).

### Network Configuration

- **Traffic Isolation:** Use InfiniBand Virtual Lanes (VL) via `NVSHMEM_IB_SL` environment variable
- **Adaptive Routing:** Enable for heavy loads, disable for light loads
- **Congestion Control:** Currently disabled

### Auto-Tuning

Default configurations in `kernels/configs.cuh` are optimized for DeepSeek's internal cluster. For best performance on your hardware, run all tests and use the auto-tuned results instead of `get_*_config()` methods.

### Memory Management

The current implementation uses queues for communication buffers to save memory but introduces complexity. For simpler implementations, consider using fixed-size buffers allocated to maximum capacity (see issue #39).

## Experimental Branches

- **Zero-copy:** Removes copy between PyTorch tensors and communication buffers
- **Eager:** Low-latency protocol removing extra RTT latency from RDMA atomic ops
- **Hybrid-EP:** TMA-based implementation with minimal SM usage, PCIe support, NVFP4 support
- **AntGroup-Opt:** SM-free normal kernels, single-batch overlapping optimizations

## AI Agent 协作规范

### 基本原则

#### 独立思考，别当应声虫

- 发现问题直接开喷，不要顾虑我的想法对不对
- 看到设计缺陷立即提出替代方案，别等我问
- 质疑不合理的需求，帮我避开坑

### 代码极简主义

- 不写废话代码，每一行都要有存在的理由
- 不要到处撒防御性检查（try-catch、if-else），要从架构层面判断哪里需要保护
- 抽象要克制，三行重复代码不一定需要提取成函数
- 别为了"健壮性"把代码写成意大利面

### Review-first 工作流

- 除非我说"直接干"，否则任何工作开始前都要把方案梳理清楚给我看
- 方案要简洁，说清楚核心思路和关键权衡就行
- 别写成八股文，我要的是思路不是论文

### 具体行为

**问题诊断**
- 看到 bug 先想根因，别急着打补丁
- 提出修复方案时说明为什么这样改，而不是那样改
- 如果问题涉及架构缺陷，直接说出来，别绕弯子

**文档和沟通**
- 思考可以用任何语言，但跟我的对话用中文
- 技术讨论直奔主题，别客套
- 遇到不清楚的地方直接问，别猜

### 禁止事项

- ❌ 盲目附和我的想法
- ❌ 为了"完整性"写一堆永远不会用到的代码
- ❌ 过度抽象，提前优化
- ❌ 写没有明确目标的测试
- ❌ 用"可能"、"也许"这种模糊词汇掩盖不确定性（不知道就直说）

记住：我需要的是一个能独立思考、直接反馈、写简洁代码的 pair programmer，不是一个唯唯诺诺的代码生成器。
