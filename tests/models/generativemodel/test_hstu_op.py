import torch
try:
    from hstu_attn import hstu_attn_varlen_func
except:
    from hstu_attn_interface import hstu_attn_varlen_func

def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 超参对齐示例调用
    num_heads = 16
    attention_dim = 256
    linear_dim = 256
    target_group_size = 1
    is_causal = True

    # 变长序列配置
    batch = 2
    seq_lens = [16, 24]
    total_tokens = sum(seq_lens)
    max_seqlen = max(seq_lens)

    # 构造offsets(cu_seqlens)
    offsets = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    for i in range(batch):
        offsets[i+1] = offsets[i] + seq_lens[i]

    # 输入tensor
    tq = torch.randn(total_tokens, num_heads * attention_dim, dtype=dtype, device=device)
    tk = torch.randn(total_tokens, num_heads * attention_dim, dtype=dtype, device=device)
    tv = torch.randn(total_tokens, num_heads * linear_dim, dtype=dtype, device=device)

    # HSTU专属参数
    num_contextuals = torch.tensor([8, 12], dtype=torch.int32, device=device)
    num_candidates = torch.tensor([8, 12], dtype=torch.int32, device=device)
    alpha = 1.0 / (attention_dim ** 0.5)
    window_size = (-1, 0) if is_causal else (-1, -1)

    # 调用算子，和业务逻辑保持一致
    out = hstu_attn_varlen_func(
        q=tq.view(-1, num_heads, attention_dim),
        k=tk.view(-1, num_heads, attention_dim),
        v=tv.view(-1, num_heads, linear_dim),
        cu_seqlens_q=offsets,
        cu_seqlens_k=offsets,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        num_contexts=num_contextuals,
        num_targets=num_candidates,
        target_group_size=target_group_size,
        window_size=window_size,
        rab=None,
        alpha=alpha,
        has_drab=False,
    ).view(-1, num_heads * linear_dim)

    print(f"Input q shape: {tq.shape}")
    print(f"Output shape: {out.shape}")
    print("Run success")

if __name__ == "__main__":
    torch.manual_seed(42)
    main()