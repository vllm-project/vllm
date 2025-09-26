import random

import torch

from vllm.v1.attention.backends.mla.indexer import kv_spans_from_batches
from vllm.utils.tile_lang_kernels import act_quant, fp8_index
from vllm import _custom_ops as ops
from vllm.model_executor.models.deepseek_v2 import indexer_k_quant_and_cache
from vllm.utils.deep_gemm import (
    fp8_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_paged_mqa_logits,
)
from vllm.utils.tile_lang_kernels import act_quant

def ref_compute_logits_fp8(q, kv, weights, mask, block_size):
    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    k_fp8, k_scale = act_quant(kv, block_size, "ue8m0")
    
    weights = weights.unsqueeze(-1) * q_scale
    weights = weights * (128**(-0.5))
    index_score = fp8_index(
        q_fp8.contiguous(), weights,
        k_fp8.contiguous(),
        k_scale.contiguous())
    if mask is not None:
        index_score += mask
    return index_score

def ref_indexer(seq_len, q, kv, weights, block_size, topk):
    B = seq_len.shape[0]
    total_seqlen = torch.sum(seq_len)
    varlen_logits = torch.full((total_seqlen, total_seqlen), float("-inf"), device="cuda")
    
    current_context_ptr = 0
    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous().unsqueeze(0)
        kv_s = kv[i][:S].contiguous().unsqueeze(0)
        weights_s = weights[i][:S].contiguous().unsqueeze(0)
        mask = torch.full(
            (S, S), float("-inf"),
            device="cuda").triu_(1)
        logits = ref_compute_logits_fp8(q_s, kv_s, weights_s, mask, block_size)
        logits = logits.squeeze(0)
        
        varlen_logits[current_context_ptr:current_context_ptr + S, current_context_ptr: current_context_ptr + S] = logits
        current_context_ptr += S
    return varlen_logits

def deepgemm_mqa_indexer(seq_len, q, kv, weights, block_size, topk):
    B = seq_len.shape[0]
    concat_q = []
    concat_kv = []
    concat_weights = []

    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous()
        kv_s = kv[i][:S].contiguous()
        weight_s = weights[i][:S].contiguous()
        concat_q.append(q_s)
        concat_kv.append(kv_s)
        concat_weights.append(weight_s)
    
    q = torch.cat(concat_q, dim=0)
    kv = torch.cat(concat_kv, dim=0)
    weights = torch.cat(concat_weights, dim=0)
    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    kv_fp8, kv_scale = act_quant(kv, block_size, "ue8m0")
    
    weights = weights.unsqueeze(-1) * (128**(-0.5)) * q_scale
    weights = weights.squeeze(-1)
    query_start_loc = torch.empty((B + 1), device="cuda")
    query_start_loc[0] = 0
    query_start_loc[1:] = seq_len.cumsum(dim=0).to(dtype=torch.int32)

    cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(query_start_loc, seq_len)

    logits = fp8_mqa_logits(
        q_fp8, 
        (kv_fp8, kv_scale), 
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke
    )
    topk_indices = logits.topk(topk, dim=-1)[1]
    mask_lo = topk_indices >= cu_seqlen_ks[:, None]
    mask_hi = topk_indices < cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi
    topk_indices = topk_indices.masked_fill(~mask, -1)
    return logits

def test_prefill_indexer():
    B = 3
    S = 128
    SKV = S
    H = 64
    HKV = 1
    D = 128
    block_size = 128
    topk = 64
    device = "cuda"
    seq_len = torch.randint(low=64, high=S, size=(B,))
    
    q = torch.randn(B, S, H, D, device="cuda",
                    dtype=torch.bfloat16)
    kv = torch.randn(B, SKV, D, device="cuda",
                     dtype=torch.bfloat16)
    weights = torch.randn(B, S, H, device=device, dtype=torch.float32) * H**-0.5

    ref_logits = ref_indexer(seq_len, q, kv, weights, block_size, topk)
    deepgemm_logits = deepgemm_mqa_indexer(seq_len, q, kv, weights, block_size, topk)
    torch.testing.assert_close(ref_logits, deepgemm_logits)


if __name__ == "__main__":
    test_prefill_indexer()