// 核心配置参数（kernel launch 参数、auto-tuned 配置等）
#include "configs.cuh"
// 异常处理宏（EP_DEVICE_ASSERT、EP_STATIC_ASSERT）
#include "exception.cuh"
// kernel launch 相关宏（SETUP_LAUNCH_CONFIG、LAUNCH_KERNEL）
#include "launch.cuh"

namespace deep_ep {

namespace layout {

// MoE dispatch 预处理 kernel：统计 token 分布到各 expert/rank 的数量
// 核心设计：使用 shared memory 做线程级归约避免全局 atomic contention
// 关键参数：
//   - kNumThreads: 每个 block 的线程数（256）
//   - kNumExpertsPerSM: 前段 block 每个负责统计的 expert 数（4）
//   - kNumRanksPerSM: 后段 block 每个负责统计的 rank 数（8）
// 注意：kernel 分两部分，前段 block 统计 expert，后段 block 统计 rank
template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,  // [num_tokens, num_topk] 每个 token 选中的 expert ID
                                    int* num_tokens_per_rank,  // [num_ranks] 每个 rank 收到的 token 总数
                                    int* num_tokens_per_rdma_rank,  // [num_rdma_ranks] 每个 RDMA rank 收到的 token 数（可选）
                                    int* num_tokens_per_expert,  // [num_experts] 每个 expert 收到的 token 数
                                    bool* is_token_in_rank,  // [num_tokens, num_ranks] token i 是否被发送到 rank j
                                    int num_tokens,  // 总 token 数
                                    int num_topk,  // 每个 token 选中的 expert 数（通常是 8）
                                    int num_ranks,  // 总 rank 数，一般为 8～32
                                    int num_experts) {  // 总 expert 数，DeepSeek 为 256
    auto sm_id = static_cast<int>(blockIdx.x);  // block ID（前段统计 expert，后段统计 rank）
    auto thread_id = static_cast<int>(threadIdx.x);  // 线程 ID（0 到 kNumThreads-1）

    // =========================
    // 第一部分：前段 block 统计每个 expert 收到的 token 数
    // 核心优化：每个线程维护独立计数器，最后 reduce，避免 atomic add
    // =========================
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];  // shared memory：[thread_id][expert_local_id] -> count
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);  // 本 block 负责的 expert 范围 [expert_begin_idx, expert_end_idx)
    if (expert_begin_idx < expert_end_idx) {  // 本 block 负责统计 expert（前段 block 才会进入这个分支）
        // 初始化每个线程的 expert 计数器为 0
        // unroll 展开循环以减少分支开销
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;

        // 每个线程 stride 访问所有 token，统计在所有 token 中落在本 block 负责范围内的 expert
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;  // 定位到 token i 的 topk expert 列表起始位置
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {  // 遍历该 token 的 num_topk 个 expert
                expert_idx = static_cast<int>(shifted_topk_idx[j]);  // 读取 expert ID
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)  // 检查是否落在本 block 负责的 expert 范围
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];  // 本线程的对应 expert 计数器 +1（使用 local index）
            }
        }
        __syncthreads();  // 同步确保所有线程完成统计再进行 reduce

        // Reduce：对每个 expert，将所有线程的计数器求和
        // 限制条件：kNumExpertsPerSM <= kNumThreads，保证每个 expert 至少有一个线程负责 reduce
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {  // thread_id < kNumExpertsPerSM 的线程负责 reduce（每个线程对应一个 expert）
            int sum = 0;  // 累加所有线程对该 expert 的统计结果
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];  // thread_id 对应的 expert，累加所有线程的计数
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;  // 写回全局内存（使用 global expert ID）
        }
        return;  // 前段 block 完成任务后直接返回
    }

    // 健全性检查：如果启用 rdma_rank 统计，确保 rank 数是 NUM_MAX_NVL_PEERS 的倍数且大于 NUM_MAX_NVL_PEERS
    // NUM_MAX_NVL_PEERS 是 NVLink 节点内通信的最大 peer 数（通常是 8）
    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    // =========================
    // 第二部分：后段 block 统计每个 rank 收到的 token 数
    // 核心逻辑：
    //   1. 统计 num_tokens_per_rank（intranode 通信所需，每个 rank 单独统计）
    //   2. 统计 num_tokens_per_rdma_rank（internode 通信所需，NUM_MAX_NVL_PEERS 个 rank 合并为一个 rdma_rank）
    //   3. 填充 is_token_in_rank（标记每个 token 是否被发送到某个 rank）
    // =========================
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;  // 每个 block 负责的 rdma_rank 数（8 个 rank = 1 个 rdma_rank）
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];  // shared memory：[thread_id][rank_local_id] -> count
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];  // shared memory：[thread_id][rdma_rank_local_id] -> count
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;  // 计算有多少个 block 用于统计 expert（向上取整）
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);  // 本 block 负责的 rank 范围（后段 block 的 ID 需要减去 sm_begin）
    int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;  // 对应的 rdma_rank 范围（NUM_MAX_NVL_PEERS 个 rank 合并为 1 个 rdma_rank）
    if (rank_begin_idx < rank_end_idx) {  // 本 block 负责统计 rank（后段 block 才会进入这个分支）
        const auto num_expert_per_rank = num_experts / num_ranks;  // 每个 rank 负责的 expert 数（假设均匀分布）
        auto expert_begin = rank_begin_idx * num_expert_per_rank;  // 本 block 负责的 rank 范围对应的 expert 起始索引
        auto expert_end = rank_end_idx * num_expert_per_rank;  // 对应的 expert 结束索引

        // 初始化每个线程的 rank 和 rdma_rank 计数器为 0
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;

        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;  // 定位到 token i 的 topk expert 列表起始位置
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};  // 本线程处理的当前 token 是否在各 rank/rdma_rank 中（用于去重）
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {  // 遍历该 token 的 num_topk 个 expert
                expert_idx = static_cast<int>(shifted_topk_idx[j]);  // 读取 expert ID
                if (expert_begin <= expert_idx and expert_idx < expert_end) {  // 检查是否落在本 block 负责的 expert 范围
                    // 根据 expert ID 计算对应的 rank ID（expert 均匀分布到各 rank）
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;  // 计算 rank 的 local index
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;  // 标记该 token 在这个 rank/rdma_rank 中（可能重复标记，后续会去重）
                }
            }

            // 写入 is_token_in_rank （bool 数组，标记 token i 是否被发送到 rank j）
            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;  // 定位到 token i 对应的 rank 列表起始位置
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);  // 写入 bool 值（去重：只要 is_in_rank[j] > 0 就说明至少有一个 expert 在该 rank）
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);  // 去重后累加到 rank 计数器（每个 token 对每个 rank 最多计数 1 次）
            }

            // 累加 rdma_rank 计数器（去重）
            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);  // 去重后累加（每个 token 对每个 rdma_rank 最多计数 1 次）
        }
        __syncthreads();  // 同步确保所有线程完成统计再进行 reduce

        // Reduce：对每个 rank，将所有线程的计数器求和
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {  // thread_id < kNumRanksPerSM 的线程负责 reduce（每个线程对应一个 rank）
            int sum = 0;  // 累加所有线程对该 rank 的统计结果
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];  // thread_id 对应的 rank，累加所有线程的计数
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;  // 写回全局内存（使用 global rank ID）
        }

        // Reduce rdma_rank 统计结果（仅在需要时）
        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {  // 仅在 num_tokens_per_rdma_rank 不为空且 thread_id < kNumRDMARanksPerSM 时执行
            int sum = 0;  // 累加所有线程对该 rdma_rank 的统计结果
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];  // thread_id 对应的 rdma_rank，累加所有线程的计数
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;  // 写回全局内存（使用 global rdma_rank ID）
        }
    }
}

// Host-side API：启动 layout kernel
// 核心设计：分配前段 block 统计 expert，后段 block 统计 rank
void get_dispatch_layout(const topk_idx_t* topk_idx,  // [num_tokens, num_topk]
                         int* num_tokens_per_rank,  // [num_ranks]
                         int* num_tokens_per_rdma_rank,  // [num_rdma_ranks] 可选
                         int* num_tokens_per_expert,  // [num_experts]
                         bool* is_token_in_rank,  // [num_tokens, num_ranks]
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;  // 调优参数：每个 block 256 线程，统计 4 个 expert 或 8 个 rank
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;  // 计算总 block 数：前段统计 expert 的 block 数 + 后段统计 rank 的 block 数
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");  // 编译期检查：kNumRanksPerSM 必须是 NUM_MAX_NVL_PEERS 的倍数（保证 rdma_rank 对齐）

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);  // 配置 kernel launch 参数（grid size, block size, stream）
    LAUNCH_KERNEL(&cfg,  // 启动 kernel（宏会处理错误检查）
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),  // 模板实例化
                  topk_idx,
                  num_tokens_per_rank,
                  num_tokens_per_rdma_rank,
                  num_tokens_per_expert,
                  is_token_in_rank,
                  num_tokens,
                  num_topk,
                  num_ranks,
                  num_experts);
}

}  // namespace layout

}  // namespace deep_ep
