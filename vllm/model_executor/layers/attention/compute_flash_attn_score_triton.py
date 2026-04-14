import math

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1

@triton.jit
def _compute_key_importance_kernel(
    Q, K, LSE, Out_Score,
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_lse_b, stride_lse_h, stride_lse_n,
    stride_out_b, stride_out_h, stride_out_n,
    sm_scale,
    H, Q_LEN, K_LEN,
    ACTUAL_HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    计算 Key 的重要性分数 (Column Sum of Attention Matrix).
    原理: Sum_over_Queries( Softmax(Q @ K.T) )
    BLOCK_D 是 ACTUAL_HEAD_DIM 向上对齐到 2 的幂的值。
    """
    # 获取当前的 Batch ID, Head ID 和 Key-Block ID
    cur_n_idx = tl.program_id(0) # K 维度的分块索引
    cur_head_idx = tl.program_id(1)
    cur_batch_idx = tl.program_id(2)

    # 计算 Key 的偏移量
    # K 形状: [Batch, Heads, K_Len, Head_Dim]
    k_start_ptr = K + (cur_batch_idx * stride_k_b + cur_head_idx * stride_k_h)
    
    # 定义 K 的块范围 (BLOCK_N)
    offs_n = cur_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 加载 K 的块 (转置以便于做点积) -> [BLOCK_D, BLOCK_N]
    # mask 处理边界情况 (seq 和 head_dim 两个维度)
    k_mask = (offs_n[None, :] < K_LEN) & (offs_d[:, None] < ACTUAL_HEAD_DIM)
    k_ptrs = k_start_ptr + (offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    # 初始化累加器，用于存储这一块 Key 的总重要性分数
    # 形状 [BLOCK_N]
    acc_score = tl.zeros([BLOCK_N], dtype=tl.float32)

    # 遍历所有的 Query (分块进行)
    # Q 形状: [Batch, Heads, Q_Len, Head_Dim]
    q_start_ptr = Q + (cur_batch_idx * stride_q_b + cur_head_idx * stride_q_h)
    lse_start_ptr = LSE + (cur_batch_idx * stride_lse_b + cur_head_idx * stride_lse_h)

    for start_m in range(0, Q_LEN, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        # 1. 加载 Q 块 [BLOCK_M, BLOCK_D]
        q_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < ACTUAL_HEAD_DIM)
        q_ptrs = q_start_ptr + (offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # 2. 加载对应的 LSE [BLOCK_M]
        # LSE 形状通常是 [Batch, Heads, Q_Len]
        lse_ptrs = lse_start_ptr + offs_m * stride_lse_n
        lse_mask = offs_m < Q_LEN
        lse_block = tl.load(lse_ptrs, mask=lse_mask, other=0.0)

        # 3. 计算 Attention Scores: S = Q @ K.T
        # q_block: [BLOCK_M, BLOCK_D], k_block: [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_block, k_block)
        qk *= sm_scale

        # 4. 计算概率 P = exp(S - LSE)
        # 注意: LSE 需要广播到 [BLOCK_M, BLOCK_N]
        p_block = tl.exp(qk - lse_block[:, None])
        
        # 边界掩码处理：超出 Q_LEN 的部分概率置 0
        q_seq_mask = offs_m[:, None] < Q_LEN
        p_block = tl.where(q_seq_mask, p_block, 0.0)

        # 5. 列求和 (Sum over Q dimension) -> 结果加到 accumulator
        # tl.sum(p_block, axis=0) 返回 [BLOCK_N]
        acc_score += tl.sum(p_block, axis=0)

    # 循环结束后，将结果写入 Output
    # Out 形状: [Batch, Heads, K_Len]
    out_start_ptr = Out_Score + (cur_batch_idx * stride_out_b + cur_head_idx * stride_out_h)
    out_ptrs = out_start_ptr + offs_n * stride_out_n
    
    # 写入显存
    tl.store(out_ptrs, acc_score, mask=offs_n < K_LEN)

def compute_token_importance_triton(q, k, softmax_lse, softmax_scale=None):
    """
    Python 包装函数
    Args:
        q: [Batch, Seq_Q, Heads, Dim] 或 [Batch, Heads, Seq_Q, Dim]
        k: [Batch, Seq_K, Heads, Dim] 或 [Batch, Heads, Seq_K, Dim]
        softmax_lse: [Batch, Heads, Seq_Q] (Flash Attention 的输出)
    Returns:
        token_importance: [Batch, Heads, Seq_K]
    """
    # 1. 统一维度格式，确保是 [Batch, Heads, Seq, Dim]
    # 依据 softmax_lse 的 heads 维度来判断是否需要 transpose
    expected_heads = softmax_lse.shape[1]
    if q.shape[1] != expected_heads:
        q = q.transpose(1, 2)
    if k.shape[1] != expected_heads:
        k = k.transpose(1, 2)
    
    # 确保连续，避免 stride 问题
    q = q.contiguous()
    k = k.contiguous()
    
    batch, heads, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    # 2. 准备输出 Tensor [Batch, Heads, K_Len] (只有 O(N) 大小!)
    out_score = torch.empty((batch, heads, k_len), device=q.device, dtype=torch.float32)
    
    # 3. 设定 Kernel 参数
    BLOCK_M = 128
    BLOCK_N = 64 # K 维度的切分块大小
    
    # Grid: [K_Blocks, Heads, Batch]
    grid = (triton.cdiv(k_len, BLOCK_N), heads, batch)
    
    # 4. 启动 Kernel
    BLOCK_D = _next_power_of_2(head_dim)
    _compute_key_importance_kernel[grid](
        q, k, softmax_lse, out_score,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        out_score.stride(0), out_score.stride(1), out_score.stride(2),
        softmax_scale,
        heads, q_len, k_len,
        ACTUAL_HEAD_DIM=head_dim, BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    token_importance = out_score.mean(dim=1) # [batch， K_len]
    
    return token_importance


@triton.jit
def _compute_key_importance_varlen_kernel(
    Q, K, LSE, Out_Score, CuSeqLens, n_heads,
    # Strides (步长)
    stride_q_tok, stride_q_h, stride_q_d,  # Q: [Total_Tokens, Heads, Head_Dim]
    stride_k_tok, stride_k_h, stride_k_d,  # K: [Total_Tokens, Heads, Head_Dim]
    stride_lse_h, stride_lse_tok,          # LSE: [Heads, Total_Tokens] (Packed 格式)
    stride_out_tok,          # Out: [Total_Tokens,]
    # Meta parameters
    sm_scale,
    ACTUAL_HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    """
    针对 Varlen (Packed Sequence) 的 Key 重要性计算。
    LSE 使用 Packed 格式 [Heads, Total_Tokens]，与 flash_attn_varlen_func 返回格式一致。
    BLOCK_D 是 ACTUAL_HEAD_DIM 向上对齐到 2 的幂的值。
    """
    # 1. 获取当前程序的 Grid ID
    # pid_seq: 当前处理的是第几个序列 (Batch 维度)
    pid_seq = tl.program_id(2) 
    # pid_head: 当前处理的是第几个 Head
    pid_head = tl.program_id(1)
    # pid_n: 当前处理的是 Key 的第几个 Block
    pid_n = tl.program_id(0)

    # 2. 读取 cu_seqlens 获取当前序列的边界
    # cu_seqlens 形状 [Batch+1]，存储每个序列的累积长度
    # start_idx: 当前序列在 Total_Tokens 中的起始索引
    start_idx = tl.load(CuSeqLens + pid_seq)
    end_idx = tl.load(CuSeqLens + pid_seq + 1)
    seq_len = end_idx - start_idx

    # 3. 边界检查：如果当前 Key Block 超出了该序列的实际长度，直接退出
    if pid_n * BLOCK_N >= seq_len:
        return

    # 4. 定位 Key 的指针 (Packed Layout)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 掩码：确保不读取超过当前序列长度的 Key，以及不超过实际 head_dim
    k_mask = (offs_n[None, :] < seq_len) & (offs_d[:, None] < ACTUAL_HEAD_DIM)
    
    # 计算 K 的物理地址 [BLOCK_D, BLOCK_N]
    # 注意：start_idx 是当前序列的全局偏移
    k_ptrs = K + ((start_idx + offs_n[None, :]) * stride_k_tok + pid_head * stride_k_h + offs_d[:, None] * stride_k_d)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    # 5. 初始化累加器
    acc_score = tl.zeros([BLOCK_N], dtype=tl.float32)

    # 6. 遍历 Query (FlashAttention 原理：Softmax(Q @ K.T))
    # 我们需要在当前序列内部，遍历所有的 Query Token
    
    # LSE 指针初始化 (Packed 格式 [Heads, Total_Tokens])
    lse_base_ptr = LSE + (pid_head * stride_lse_h + start_idx * stride_lse_tok)
    
    # Q 指针初始化 (Packed Layout)
    q_base_ptr = Q + (start_idx * stride_q_tok + pid_head * stride_q_h)

    # 循环步进 BLOCK_M
    for start_m in range(0, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        # 加载 Q Block [BLOCK_M, BLOCK_D]
        q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < ACTUAL_HEAD_DIM)
        q_ptrs = q_base_ptr + (offs_m[:, None] * stride_q_tok + offs_d[None, :] * stride_q_d)
        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # 加载 LSE Block [BLOCK_M]
        lse_ptrs = lse_base_ptr + offs_m * stride_lse_tok
        lse_block = tl.load(lse_ptrs, mask=offs_m < seq_len, other=0.0)

        # S = Q @ K.T -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_block, k_block)
        qk *= sm_scale

        # P = exp(S - LSE)
        p_block = tl.exp(qk - lse_block[:, None])

        # 掩码处理：超出 Q 长度的部分概率置 0
        q_seq_mask = offs_m[:, None] < seq_len
        p_block = tl.where(q_seq_mask, p_block, 0.0)

        # 列求和 (Sum over Queries)
        acc_score += tl.sum(p_block, axis=0)

    # 7. 写回结果 (Packed Layout)
    # Out 形状: [Total_Tokens, Heads]
    avg_score = acc_score / n_heads
    out_ptrs = Out_Score + (start_idx + offs_n)
    tl.atomic_add(out_ptrs, avg_score, mask=offs_n < seq_len)


def compute_varlen_importance(q, k, cu_seqlens, max_seqlen, softmax_lse, softmax_scale=None):
    """
    计算 Varlen (Packed Sequence) 格式的 token importance。
    
    Args:
        q: [Total_Tokens, Heads, Dim] - Packed Sequence
        k: [Total_Tokens, Heads, Dim] - Packed Sequence
        cu_seqlens: [Batch + 1] - Int32 tensor, e.g., [0, 5, 12, ...]
        max_seqlen: int - 最大序列长度 (用于确定 Grid 维度)
        softmax_lse: [Heads, Total_Tokens] - Packed 格式，来自 flash_attn_varlen_func 的输出
        softmax_scale: float - Softmax 缩放因子，默认为 1/sqrt(head_dim)
    
    Returns:
        token_importance: [Total_Tokens] - 每个 token 的重要性分数 (在 heads 维度上平均)
    """
    # 检查输入维度
    assert q.dim() == 3, "Q should be packed [Total_Tokens, Heads, Dim]"
    total_tokens, n_heads, head_dim = q.shape
    batch_size = cu_seqlens.numel() - 1
    
    # 确保 LSE 的形状匹配 (Packed 格式)
    # flash_attn_varlen_func 输出: [Heads, Total_Tokens]
    assert softmax_lse.dim() == 2, f"LSE should be 2D [Heads, Total_Tokens], got {softmax_lse.dim()}D"
    assert softmax_lse.shape[0] == n_heads, f"LSE heads mismatch: {softmax_lse.shape[0]} vs {n_heads}"
    assert softmax_lse.shape[1] == total_tokens, f"LSE tokens mismatch: {softmax_lse.shape[1]} vs {total_tokens}"

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)

    # 准备输出
    out_score = torch.zeros((total_tokens,), dtype=torch.float32, device=q.device)

    # Kernel 配置
    BLOCK_M = 128
    BLOCK_N = 64 # K 维度的切分
    
    # Grid 策略:
    # X轴: K 的分块数量 (用 max_seqlen 估算最大需要的块数，kernel 内会用 cu_seqlens 截断)
    # Y轴: Heads
    # Z轴: Batch Size (对应 cu_seqlens 的区间)
    grid = (triton.cdiv(max_seqlen, BLOCK_N), n_heads, batch_size)

    BLOCK_D = _next_power_of_2(head_dim)
    _compute_key_importance_varlen_kernel[grid](
        q, k, softmax_lse, out_score, cu_seqlens, n_heads,
        # Strides
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1),  # [Heads, Total_Tokens]
        out_score.stride(0),
        # Meta
        softmax_scale,
        ACTUAL_HEAD_DIM=head_dim, BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out_score
