"""
Stub flash_attn_interface for JGS simulator.
Provides a pure PyTorch implementation of flash_attn_varlen_func.
"""
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q=None,
    k=None, 
    v=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    out=None,
    block_table=None,
    seqused_k=None,
    window_size=None,
    softcap=None,
    alibi_slopes=None,
    s_aux=None,
    **kwargs,
):
    """Pure PyTorch fallback for flash attention on JGS simulator.
    
    Supports both:
    - Prefill mode: cu_seqlens_q + cu_seqlens_k (variable length sequences)
    - Decode mode: block_table + seqused_k (paged KV cache)
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    num_heads_q = q.shape[1] if q.dim() == 3 else q.shape[-2]
    head_dim = q.shape[-1]

    if out is None:
        out = torch.empty_like(q)

    if block_table is not None and seqused_k is not None:
        # Decode mode with paged KV cache
        # q: [total_q_tokens, num_heads, head_dim]
        # k, v: [num_blocks, block_size, num_kv_heads, head_dim] (paged cache)
        # block_table: [batch, max_blocks_per_seq]
        # seqused_k: [batch] - actual k sequence lengths
        
        batch_size = seqused_k.shape[0]
        block_size = k.shape[1]
        num_kv_heads = k.shape[2]
        
        # GQA ratio
        num_groups = num_heads_q // num_kv_heads if num_kv_heads > 0 else 1
        
        q_offset = 0
        for b in range(batch_size):
            kv_len = seqused_k[b].item()
            if kv_len <= 0:
                continue
            
            # Gather K, V from paged cache for this batch element
            num_blocks_needed = (kv_len + block_size - 1) // block_size
            k_gathered = []
            v_gathered = []
            for blk_idx in range(min(num_blocks_needed, block_table.shape[1])):
                blk_num = block_table[b, blk_idx].item()
                if blk_num < 0 or blk_num >= k.shape[0]:
                    continue
                remaining = min(block_size, kv_len - blk_idx * block_size)
                k_gathered.append(k[blk_num, :remaining])
                v_gathered.append(v[blk_num, :remaining])
            
            if not k_gathered:
                q_offset += 1
                continue
                
            k_cat = torch.cat(k_gathered, dim=0)  # [kv_len, num_kv_heads, head_dim]
            v_cat = torch.cat(v_gathered, dim=0)
            
            # q for this batch: assume 1 token per batch in decode
            qi = q[q_offset:q_offset+1]  # [1, num_heads_q, head_dim]
            
            # Transpose for matmul: [num_heads, seq, head_dim]
            qi_t = qi.transpose(0, 1)  # [num_heads_q, 1, head_dim]
            
            # Expand KV heads for GQA
            if num_groups > 1:
                k_t = k_cat.transpose(0, 1)  # [num_kv_heads, kv_len, head_dim]
                v_t = v_cat.transpose(0, 1)
                k_t = k_t.repeat_interleave(num_groups, dim=0)  # [num_heads_q, kv_len, head_dim]
                v_t = v_t.repeat_interleave(num_groups, dim=0)
            else:
                k_t = k_cat.transpose(0, 1)
                v_t = v_cat.transpose(0, 1)
            
            # Attention: [num_heads_q, 1, head_dim] x [num_heads_q, head_dim, kv_len]
            attn = torch.matmul(qi_t, k_t.transpose(-2, -1)) * softmax_scale
            attn = torch.softmax(attn.float(), dim=-1).to(qi.dtype)
            attn_out = torch.matmul(attn, v_t)  # [num_heads_q, 1, head_dim]
            out[q_offset:q_offset+1] = attn_out.transpose(0, 1)
            
            q_offset += 1

    elif cu_seqlens_q is not None and cu_seqlens_k is not None:
        # Prefill mode with variable length sequences
        batch_size = cu_seqlens_q.shape[0] - 1
        num_kv_heads = k.shape[1] if k.dim() == 3 else k.shape[-2]
        num_groups = num_heads_q // num_kv_heads if num_kv_heads > 0 else 1
        
        for b in range(batch_size):
            sq = cu_seqlens_q[b].item()
            eq = cu_seqlens_q[b + 1].item()
            sk = cu_seqlens_k[b].item()
            ek = cu_seqlens_k[b + 1].item()
            
            qi = q[sq:eq]  # [seq_q, num_heads_q, head_dim]
            ki = k[sk:ek]  # [seq_k, num_kv_heads, head_dim]
            vi = v[sk:ek]
            
            qi_t = qi.transpose(0, 1)  # [num_heads_q, seq_q, head_dim]
            
            if num_groups > 1:
                ki_t = ki.transpose(0, 1).repeat_interleave(num_groups, dim=0)
                vi_t = vi.transpose(0, 1).repeat_interleave(num_groups, dim=0)
            else:
                ki_t = ki.transpose(0, 1)
                vi_t = vi.transpose(0, 1)
            
            attn = torch.matmul(qi_t, ki_t.transpose(-2, -1)) * softmax_scale
            
            if causal:
                seq_q, seq_k = qi_t.shape[1], ki_t.shape[1]
                mask = torch.triu(
                    torch.ones(seq_q, seq_k, device=q.device, dtype=torch.bool),
                    diagonal=seq_k - seq_q + 1
                )
                attn = attn.masked_fill(mask.unsqueeze(0), float('-inf'))
            
            attn = torch.softmax(attn.float(), dim=-1).to(qi.dtype)
            attn_out = torch.matmul(attn, vi_t)
            out[sq:eq] = attn_out.transpose(0, 1)
    else:
        # Fallback: just zero the output
        out.zero_()

    return out
