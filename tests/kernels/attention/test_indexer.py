# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import torch

from vllm import _custom_ops as ops
from vllm.utils import cdiv
from vllm.utils.deep_gemm import (calc_diff, fp8_mqa_logits,
                                  fp8_paged_mqa_logits, get_num_sms,
                                  get_paged_mqa_logits_metadata)
from vllm.utils.tile_lang_kernels import act_quant, fp8_index
from vllm.v1.attention.backends.mla.indexer import kv_spans_from_batches


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)),
                        device=x.device,
                        dtype=torch.uint8)
    x_fp8[:, :block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[:,
          block_size * head_dim:] = sf.view(num_blocks,
                                            block_size).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def ref_compute_logits_fp8(q, kv, weights, mask, block_size):
    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    k_fp8, k_scale = act_quant(kv, block_size, "ue8m0")

    weights = weights.unsqueeze(-1) * q_scale
    weights = weights * (128**(-0.5))
    index_score = fp8_index(q_fp8.contiguous(), weights, k_fp8.contiguous(),
                            k_scale.contiguous())
    if mask is not None:
        index_score += mask
    return index_score


def ref_indexer(seq_len, q, kv, weights, block_size, topk):
    B = seq_len.shape[0]
    total_seqlen = torch.sum(seq_len)
    varlen_logits = torch.full((total_seqlen, total_seqlen),
                               float("-inf"),
                               device="cuda")

    current_context_ptr = 0
    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous().unsqueeze(0)
        kv_s = kv[i][:S].contiguous().unsqueeze(0)
        weights_s = weights[i][:S].contiguous().unsqueeze(0)
        mask = torch.full((S, S), float("-inf"), device="cuda").triu_(1)
        logits = ref_compute_logits_fp8(q_s, kv_s, weights_s, mask, block_size)
        logits = logits.squeeze(0)

        varlen_logits[current_context_ptr:current_context_ptr + S,
                      current_context_ptr:current_context_ptr + S] = logits
        current_context_ptr += S
    return varlen_logits


def deepgemm_mqa_indexer(seq_len,
                         query_seq_len,
                         q,
                         kv,
                         weights,
                         block_size,
                         topk,
                         is_kv_batched=True):
    B = seq_len.shape[0]
    concat_q = []
    concat_kv = []
    concat_weights = []

    for i in range(B):
        S = seq_len[i]
        q_s = q[i][:S].contiguous()
        if is_kv_batched:
            kv_s = kv[i][:S].contiguous()
            weight_s = weights[i][:S].contiguous()
        concat_q.append(q_s)
        if is_kv_batched:
            concat_kv.append(kv_s)
            concat_weights.append(weight_s)

    q = torch.cat(concat_q, dim=0)
    if is_kv_batched:
        kv = torch.cat(concat_kv, dim=0)
        weights = torch.cat(concat_weights, dim=0)
    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    kv_fp8, kv_scale = act_quant(kv, block_size, "ue8m0")

    weights = weights.unsqueeze(-1) * (128**(-0.5)) * q_scale
    weights = weights.squeeze(-1)
    query_start_loc = torch.empty((B + 1), device="cuda")
    query_start_loc[0] = 0
    query_start_loc[1:] = query_seq_len.cumsum(dim=0).to(dtype=torch.int32)

    cu_seqlen_ks, cu_seqlen_ke = kv_spans_from_batches(query_start_loc,
                                                       seq_len)

    logits = fp8_mqa_logits(q_fp8, (kv_fp8, kv_scale), weights, cu_seqlen_ks,
                            cu_seqlen_ke)
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
    seq_len = torch.randint(low=64, high=S, size=(B, ))

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, SKV, D, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(B, S, H, device=device,
                          dtype=torch.float32) * H**-0.5

    ref_logits = ref_indexer(seq_len, q, kv, weights, block_size, topk)
    deepgemm_logits = deepgemm_mqa_indexer(seq_len, seq_len, q, kv, weights,
                                           block_size, topk)
    torch.testing.assert_close(ref_logits, deepgemm_logits)


def test_decode_paged_indexer():
    num_blocks, blocksize = 111 * 3000, 64
    B = 3
    S = 128
    SKV = S
    H = 64
    HKV = 1
    D = 128
    block_size = 128
    topk = 64
    device = "cuda"
    seq_len = torch.randint(low=64, high=S, size=(B, ), device="cuda")

    query_seq_len = torch.ones(B, device="cuda")

    q = torch.randn((B, 1, H, D), device='cuda', dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, blocksize, 1, D),
                           device='cuda',
                           dtype=torch.bfloat16)
    weights = torch.randn(
        (B * 1, H), device='cuda', dtype=torch.float32) * H**-0.5
    max_block_len = (seq_len.max().item() + blocksize -
                     1) // blocksize * blocksize

    block_tables = torch.zeros((B, max_block_len),
                               device='cuda',
                               dtype=torch.int32)

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(B):
        ctx_len = seq_len[i].item()
        for j in range(cdiv(ctx_len, blocksize)):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1

    flatten_kv = torch.empty([seq_len.sum(), D],
                             device="cuda",
                             dtype=torch.bfloat16)
    cu_seq_lens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        seq_len.cumsum(dim=0)
    ]).to(torch.int32).cuda()

    ops.cp_gather_cache(
        kv_cache,
        flatten_kv,
        block_tables,
        cu_seq_lens,
        B,
    )

    ref_logits = deepgemm_mqa_indexer(seq_len,
                                      query_seq_len,
                                      q,
                                      flatten_kv,
                                      weights,
                                      block_size,
                                      topk,
                                      is_kv_batched=False)

    q_fp8, q_scale = act_quant(q, block_size, "ue8m0")
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(seq_len.int(), blocksize,
                                                      get_num_sms())

    weights = weights.unsqueeze(-1) * (128**(-0.5)) * q_scale.squeeze(1)
    weights = weights.squeeze(-1)

    logits = fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seq_len.int(),
                                  block_tables, schedule_metadata, 4096)

    concat_logit = []
    context = 0
    for i in range(B):
        per_seq_logits = torch.zeros(4096, device="cuda")
        S = seq_len[i]
        per_seq_logits[:S] = ref_logits[i][context:context + S]
        concat_logit.append(per_seq_logits)
        context += S
    ref_logits = torch.stack(concat_logit, dim=0)
    logits[logits == float("-inf")] = 0
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"


if __name__ == "__main__":
    test_prefill_indexer()
    test_decode_paged_indexer()
