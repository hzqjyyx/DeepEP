import torch

def gen_random_data(num_tokens, hidden, num_experts, num_topk, rank, num_ranks):
    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cpu') * rank  # 常量张量,用于验证正确性
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cpu')  # 随机张量,用于测试数值精度
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cpu').abs() + 1  # +1 确保非零
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]  # 选出 top-k experts
    topk_idx = topk_idx.to(torch.int64)  # 转换为配置的 topk_idx 类型 (int32 或 int64)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cpu') * rank  # 常量权重,便于验证
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cpu')
    rank_idx = topk_idx // (num_experts // num_ranks)  # 将 expert_id 映射到 rank_id
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)  # 无效 expert 对应的 rank 也标记为 -1
    inplace_unique(rank_idx, num_ranks)  # 去重 rank_id,每个 token 只需与相关 rank 通信一次

    # Expert meta
    # 计算每个 expert 将接收的 token 数 (用于验证 dispatch 正确性)
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cpu')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()  # 统计本地选择该 expert 的 token 数

        # Rank layout meta
    # 计算每个 token 在目标 rank 的接收 buffer 中的索引位置
    # 关键设计: token_idx_in_rank[rank][token] 表示 token 在 rank 的接收 buffer 中的位置
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cpu')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cpu')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()  # 统计发往该 rank 的 token 数
        token_sel = (rank_idx == i).max(dim=-1)[0]  # token 是否会发往 rank=i
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]  # 提取 token indices
        tokens[:count] = torch.sort(tokens[:count])[0]  # 按 token_id 排序,保证确定性
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cpu')  # 分配接收 buffer 位置
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0  # 标记哪些 token 会发往哪些 rank


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


if __name__ == "__main__":
    num_tokens = 4096
    hidden = 7168
    num_experts = 256
    num_topk = 8
    num_ranks = 16
    rank = 0
    gen_random_data(num_tokens, hidden, num_experts, num_topk, rank, num_ranks)
