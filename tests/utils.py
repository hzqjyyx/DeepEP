import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Union


# 初始化分布式环境并设置默认设备
# 设计决策: 兼容不同版本的 PyTorch (device_id 参数在旧版本中不存在)
# 注意: WORLD_SIZE/RANK 指的是节点数,不是总进程数
def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))  # 节点数,非进程数
    node_rank = int(os.getenv('RANK', 0))  # 节点编号,非进程编号

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,  # 总进程数 = 节点数 × 每节点 GPU 数
        'rank': node_rank * num_local_ranks + local_rank,  # 全局 rank = 节点 rank × GPU 数 + 本地 GPU ID
    }
    if 'device_id' in sig.parameters:  # 兼容 PyTorch 2.0+ 的新参数
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)  # DeepEP 默认使用 BF16 训练
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


# 计算两个张量的相对差异 (基于余弦相似度的变体)
# 设计动机: +1 避免全零张量导致除零,double 提高数值精度
# 返回值: [0, 2] 范围,0 表示完全相同
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1  # +1 避免除零,转 double 提高精度
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator  # 类似 Dice 系数的相似度度量
    return (1 - sim).item()


def align_up(x, y):
    return (x + y - 1) // y * y


# Per-token 量化到 FP8 E4M3 格式
# 关键设计: 以 128 为粒度分组量化,每组独立计算 scale (平衡精度和存储开销)
# 为什么 128: TMA 指令和 NVSHMEM 传输的最小对齐单位,避免跨 cacheline 访问
# 为什么 448.0: FP8 E4M3 的最大表示值,保证量化后不溢出
def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align_up(n, 128)  # 对齐到 128,匹配 TMA 传输粒度
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 128)  # 每 128 个元素一组
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)  # 每组的绝对值最大值作为 scale 基准
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)  # 返回 (量化值, scale)


# 反量化 FP8 E4M3 到 BF16
# 关键技巧: 支持 int 类型的 scale (通过移位直接构造 float32 尾数,避免 int->float 转换)
# 为什么移位 23: IEEE 754 单精度浮点数的尾数占 23 位,直接构造指数部分
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:  # 处理空张量的边界情况
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align_up(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:  # 支持整数编码的 scale (节省传输带宽)
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23  # 移位构造 float32 的指数
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()


# 原地去重并按出现频次排序 (保留高频 ID,用于 rank 列表去重)
# 设计动机: 避免分配新张量,直接在原张量上修改 (节省内存和拷贝开销)
# 应用场景: 将 topk_idx 映射到 rank_idx 后,去除重复的 rank (每个 token 只需通知涉及的 rank 一次)
def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0  # -1 表示无效 ID
    x_padded = x.masked_fill(mask, num_slots)  # 将 -1 临时映射到 num_slots,避免 scatter 越界
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))  # 统计每个 ID 的出现次数
    bin_count = bin_count[:, :num_slots]  # 丢弃临时占位的 num_slots 列
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)  # 按频次降序
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)  # 未出现的 ID 标记为 -1
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values  # 将 -1 排到末尾
    x[:, :].fill_(-1)  # 清空原张量
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]  # 写回去重后的 ID 列表


# 生成分组受限的 expert scores (用于测试 grouped top-k)
# 设计动机: 模拟 DeepSeek-V3 的 grouped top-k 机制 (每个 group 只选一个 expert)
# 实现方式: 将非选中 group 的 scores 清零,确保 top-k 只从指定 group 中选择
def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)  # 按 group 重塑
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)  # 只保留选中 group
    return (scores * mask).view(num_tokens, num_experts)  # 非选中 group 的 scores 清零


# 基准测试工具 (带 L2 cache flush 的精确计时)
# 关键设计: flush L2 避免 cache 热身影响首次测量,CUDA Event 计时避免 CPU-GPU 同步开销
# 为什么 256MB: H100 L2 cache 为 50MB,256MB 足以完全刷新 (超过 5 倍 L2 容量)
# 为什么丢弃首次: 首次测量可能受 CUDA context 初始化影响
def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')  # 256MB buffer

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()  # 写操作清空 L2 cache

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:  # 可选的后处理函数,用于测试 overlapping
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]  # 丢弃首次测量
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# 抑制 stdout/stderr 的 context manager (用于消除 Kineto profiler 的冗余输出)
# 关键设计: 同时重定向 Python 层 (sys.stdout) 和 OS 层 (文件描述符) 的输出
# 为什么两层: C++ 扩展的输出直接写文件描述符,仅重定向 sys.stdout 无法拦截
class suppress_stdout_stderr:

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())  # 备份原始文件描述符
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)  # 重定向文件描述符到 /dev/null
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file  # 重定向 Python 层输出
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout  # 恢复 Python 层输出
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)  # 恢复文件描述符
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)  # 关闭备份的文件描述符
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# 使用 PyTorch Kineto profiler 进行精确的 kernel 级性能分析
# 关键设计: 通过 barrier + 大 kernel 消除 CPU launch 不均衡导致的计时偏差
# 为什么需要 barrier: 多进程环境下,不同 rank 的 CPU launch 时间不同步,导致测量误差
# 为什么用大 kernel (8192x8192 matmul): 强制 GPU 进入稳定状态,避免前序 kernel 的尾延迟干扰
def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)  # 跳过首轮,仅分析第二轮
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for _ in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:  # 消除 CPU launch 不均衡的影响
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs  # 大 kernel 让所有 GPU 进入稳定状态
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))  # 同步所有 rank 的 CPU 时间线
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'  # 确保 kernel 名唯一匹配

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)  # 导出 Chrome tracing 格式,用于可视化分析

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}  # 支持毫秒和微秒两种单位
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]  # 倒数第二列是时间字符串
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, '')) / scale)  # 统一转换为秒
                        break
                break

    # Expand the kernels by periods
    # 关键设计: 分离周期性 kernel 的各阶段耗时 (如 dispatch 分多个 chunk 执行)
    # 应用场景: 测量 pipelined kernel 的各 stage 耗时分布
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]  # 提取 kernel 事件
            events = sorted(events, key=lambda event: event['ts'])  # 按时间戳排序
            durations = [event['dur'] / 1e6 for event in events]  # 转换为秒
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns for j in range(num_kernels_per_period)]  # 分离各阶段平均耗时

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()
