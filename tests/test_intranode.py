import argparse
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


# Intranode 通信正确性验证与性能调优
# 测试流程: 1) 布局计算 2) Dispatch/Combine 正确性验证 3) 性能调优 (遍历 nvl_chunk_size)
# 关键验证点: FP8/BF16 精度,async/sync 模式,with/without topk,缓存 dispatch
# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int, buffer: deep_ep.Buffer,
              group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0  # 专家必须均分到各 rank
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank  # 常量张量,用于验证正确性
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')  # 随机张量,用于测试数值精度
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None  # FP8 仅在 SM90+ 编译时可用
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None  # T.T 技巧保证 scale 内存布局连续
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1  # +1 确保非零
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]  # 选出 top-k experts
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)  # 转换为配置的 topk_idx 类型 (int32 或 int64)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank  # 常量权重,便于验证
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # 将 expert_id 映射到 rank_id，并进行去重
    # 相当于一个映射表
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)  # 去重 rank_id,每个 token 只需与相关 rank 通信一次

    # Expert meta
    # 计算每个 expert 将接收的 token 数 (用于验证 dispatch 正确性)
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()  # 统计本地选择该 expert 的 token 数
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)  # 汇总全局每个 expert 的 token 数

    # Rank layout meta
    # 计算每个 token 在目标 rank 的接收 buffer 中的索引位置
    # 关键设计: token_idx_in_rank[rank][token] 表示 token 在 rank 的接收 buffer 中的位置
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()  # 统计发往该 rank 的 token 数
        token_sel = (rank_idx == i).max(dim=-1)[0]  # token 是否会发往 rank=i
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]  # 提取 token indices
        tokens[:count] = torch.sort(tokens[:count])[0]  # 按 token_id 排序,保证确定性
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')  # 分配接收 buffer 位置
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0  # 标记哪些 token 会发往哪些 rank
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)  # 汇总全局每个 rank 的接收 token 数

    # 验证 DeepEP 的 layout kernel 与参考实现一致
    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()  # 同步所有 rank,避免后续测试的时间线混乱
    time.sleep(1)

    # Config
    nvl_buffer_size = 256  # NVLink buffer 大小 (单位: 元素数),控制 pipelining 粒度
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)  # num_sms, nvl_chunk_size, nvl_buffer_size

    # Test dispatch
    # 验证接收到的数据是否来自正确的发送方
    # 关键验证: 每行数据应该全是同一个 rank 的值 (因为 x 初始化为 rank 常量)
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))  # 每行数据应该是常量
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()  # 从 rank i 接收的数据在 [check_start, check_end) 区间
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0  # 验证数据值等于发送方 rank_id
            check_start = check_end

    # 多维度正确性测试: previous_mode × async_mode × dtype × with_topk
    # previous_mode: 测试与前序操作的依赖管理 (CUDA event)
    # async_mode: 测试异步执行模式 (dispatch 返回后立即继续,通过 event 同步)
    # dtype: BF16/FP8 精度测试
    # with_topk: 是否传递 topk_idx 和 topk_weights (影响 dispatch kernel 的代码路径)
    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end='')
                    dispatch_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'config': config,
                        'async_finish': async_mode
                    }
                    if with_topk:  # topk_idx/weights 是可选参数
                        dispatch_args.update({
                            'topk_idx': topk_idx,
                            'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        })
                    if previous_mode:  # 测试 event 依赖链
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(
                        **dispatch_args)
                    event.current_stream_wait() if async_mode else ()  # 异步模式需要显式等待
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x  # FP8 反量化

                    # Checks
                    rank_prefix_matrix = handle[0]  # handle 包含布局信息,用于验证
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'  # 验证接收 token 数正确
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list  # 验证 expert 分布正确
                    if current_x is not x_pure_rand:  # 常量张量才能验证来源
                        check_data(recv_x, rank_prefix_matrix)
                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        # topk_idx 已从全局 expert_id 重映射为本地 expert_id (范围 [0, num_experts/num_ranks))
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count  # 验证每个本地 expert 的 token 计数

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:  # 常量权重才能验证来源
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                                dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]  # 填充 -1 位置,避免影响验证
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    # Test `num_worst_tokens != 0`
                    # num_worst_tokens: 预分配固定大小的接收 buffer,补齐到指定数量 (CUDA Graph 兼容性要求)
                    # 关键设计: 多余位置用 -1 填充,确保每次 dispatch 的输出 shape 固定
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks  # 最坏情况下的 token 数
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x
                        assert len(empty_list) == 0  # num_worst_tokens 模式不返回 expert 计数
                        assert num_worst_tokens == recv_worst_x.size(0)  # 固定 shape
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])  # 前半部分应与动态 shape 结果相同
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])
                        assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()  # 多余位置用 -1 填充

                    # Test cached dispatch (must without top-k staffs)
                    # 缓存模式: 复用上次 dispatch 的 handle (布局信息),跳过 layout 计算
                    # 限制: 不支持 topk_idx/weights,因为 topk 会改变每次的 layout
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)

                    # Test combine
                    # Combine 是 dispatch 的逆操作: 将数据从各 rank 收集回原始 token 位置
                    # 关键验证: dispatch→combine 应恢复原始数据 (考虑多个 rank 的累加)
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)  # 除以参与的 rank 数,恢复原始值
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6  # BF16 精度损失容忍度
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x
                                                                       is x_pure_rand) else (combined_topk_weights /
                                                                                             is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9  # FP32 权重精度更高

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance
    # 性能调优目标: 找到最优的 nvl_chunk_size (控制 NVLink pipelining 粒度)
    # nvl_chunk_size: 每次 NVLink 传输的 chunk 大小 (单位: 128B),影响 SM 利用率和延迟
    # 为什么遍历: 最优值取决于 token 数、hidden size、rank 数等多个因素,需要实测
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2  # FP8 带宽系数: 数据占 1 字节,scale 占 4/(128*1) 开销
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):  # 遍历 4~32 的偶数 + 默认配置
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)  # 使用自动调优的默认配置
            tune_args = {'x': current_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]  # noqa: B023
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',  # 报告有效带宽
                    flush=True)
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True)
            print('', flush=True)

        # Gather the best config from rank 0 and the first test setting
        # 关键设计: 所有 rank 使用相同的配置 (取 rank 0 的最优配置),避免通信不匹配
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor([best_results[0], best_results[1]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()  # 统一使用 rank 0 的配置
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': dispatch_config if dispatch_config is not None else config
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    # Combine 调优: 与 dispatch 类似,但 chunk 范围不同 (combine 对小 chunk 更敏感)
    # 为什么范围更小: combine 是 gather 操作,每个 token 需要从多个 rank 收集,带宽压力更大
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0, ):  # 遍历 1~16 + 默认配置
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {'x': recv_x, 'handle': handle, 'config': config}
        t = bench(lambda: buffer.combine(**tune_args))[0]  # noqa: B023
        if local_rank == 0:
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                flush=True)
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
            flush=True)
        print('', flush=True)


# 测试入口函数 (每个进程执行一次)
# 关键流程: 1) 初始化分布式环境 2) 创建 Buffer 对象 3) 运行 intranode 测试 4) 可选的 low-latency 兼容性测试
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility, num_rdma_bytes = False, 0  # 是否测试 low-latency 功能兼容性
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

    # 创建通信 buffer (2GB NVLink buffer + 可选的 RDMA buffer)
    # num_nvl_bytes: 2GB
    # 这是在为每个 GPU/rank 预先分配的一段“可被同节点其它 GPU 通过 NVLink P2P 直接访问”的显存共享区，
    # 用来做 intranode（同节点）all-to-all（dispatch/combine）的中转/队列/流水化分块（pipelining）缓冲。
    buffer = deep_ep.Buffer(group,
                            int(2e9),  # NVLink buffer 大小
                            num_rdma_bytes,  # RDMA buffer 大小 (仅 low-latency 模式需要)
                            low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),  # 每个 rank 的 QP 数
                            explicitly_destroy=True,  # 显式销毁,避免 Python GC 延迟
                            allow_mnnvl=args.allow_mnnvl,  # 是否允许 Multi-Node NVLink
                            use_fabric=args.use_fabric)  # 是否使用 CUDA Fabric API
    torch.manual_seed(rank)  # 设置随机种子,确保不同 rank 生成不同的测试数据

    for i in (24, ):  # 使用 24 个 SM (H100 有 132 个 SM,24 是常用的保守配置)
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    # 验证 intranode buffer 与 low-latency 模式的兼容性 (同一个 Buffer 对象支持两种模式)
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)  # 清理 low-latency buffer
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()  # 显式销毁 NVSHMEM 资源
    dist.barrier()  # 确保所有 rank 都完成销毁
    dist.destroy_process_group()


# 主入口: 使用 torch.multiprocessing.spawn 启动多进程测试
# 默认配置模拟 DeepSeek-V3 的典型场景: 4096 tokens, 7168 hidden, 256 experts, top-8
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Enable MNNVL support')  # Multi-Node NVLink
    parser.add_argument('--use-fabric', action="store_true", help='Enable fabric mode')  # CUDA Fabric API
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)  # 启动多进程
