# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Segmented FlashAttention/FlashInfer execution preserving DASH route priors.

Each selected remote block and the local window is an independent segment.
Merging their natural-log LSEs plus the block priors exactly recovers the
softmax over the union, without requiring a backend-specific score modifier.
"""

from functools import cache

import torch


@cache
def _indptr(device, n, topk, bs, local):
    lengths = torch.tensor([bs] * topk + [local], dtype=torch.int32)
    kv = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.repeat(n).cumsum(0)))
    qo = torch.arange(n * (topk + 1) + 1, dtype=torch.int32)
    return qo, kv.to(torch.int32), qo.to(device), kv.to(device, dtype=torch.int32)


@cache
def _workspace(device):
    return torch.empty(128 * 1024**2, dtype=torch.uint8, device=device)


@cache
def _flashinfer_plan(device, n, topk, bs, local, heads, kv_heads, dim, dtype):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DASH FlashInfer plans must be warmed up before capture")
    qo, kv, qo_gpu, kv_gpu = _indptr(device, n, topk, bs, local)
    total = n * (topk * bs + local)
    packed_mask = torch.empty(total // 8, dtype=torch.uint8, device=device)
    mask_indptr = torch.empty_like(qo_gpu)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(device),
        kv_layout="NHD",
        use_cuda_graph=True,
        qo_indptr_buf=qo_gpu,
        kv_indptr_buf=kv_gpu,
        custom_mask_buf=packed_mask,
        mask_indptr_buf=mask_indptr,
        backend="fa2",
    )
    wrapper.plan(
        qo,
        kv,
        heads,
        kv_heads,
        dim,
        custom_mask=torch.ones(total, dtype=torch.bool, device=device),
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
        sm_scale=dim**-0.5,
        disable_split_kv=True,
    )
    return wrapper, packed_mask


def flash_dash_attention(q, k, v, used, bias, backend, bs, local):
    """Attend gathered segments and combine normalized partial outputs.

    k/v are contiguous [N*(topk*bs+local), Hkv, D]; used contains the
    valid prefix length of each segment, including zero for inactive rows.
    """
    n, heads, dim = q.shape
    topk = bias.shape[1]
    groups = topk + 1
    device = q.device
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("DASH FlashAttention/FlashInfer require FP16 or BF16")
    if bs % 8 or local % 8:
        raise ValueError("DASH Flash backends require byte-aligned segments")
    _, _, cuq, cuk = _indptr(device, n, topk, bs, local)
    queries = q[:, None].expand(n, groups, heads, dim).reshape(-1, heads, dim)
    if backend == "FLASH_ATTN":
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        version = get_flash_attn_version(head_size=dim)
        if version is None:
            raise RuntimeError(
                "DASH FLASH_ATTN requires a supported CUDA FlashAttention"
            )
        out, lse = flash_attn_varlen_func(
            queries,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cuq,
            max_seqlen_k=max(bs, local),
            cu_seqlens_k=cuk,
            seqused_k=used,
            softmax_scale=dim**-0.5,
            causal=False,
            return_softmax_lse=True,
            fa_version=version,
        )
        lse = lse.T
    elif backend == "FLASHINFER":
        wrapper, packed = _flashinfer_plan(
            device, n, topk, bs, local, heads, k.shape[1], dim, q.dtype
        )
        remote = (
            torch.arange(bs, device=device)[None, None]
            < used.view(n, groups)[:, :topk, None]
        )
        nearby = (
            torch.arange(local, device=device)[None] < used.view(n, groups)[:, -1, None]
        )
        mask = torch.cat((remote.flatten(1), nearby), -1).reshape(-1, 8)
        bits = torch.arange(8, device=device)
        packed.copy_((mask.to(torch.int32) << bits).sum(-1).to(torch.uint8))
        out, lse = wrapper.run(
            queries, k, v, return_lse=True, lse_base="ln", lse_layout="NH"
        )
    else:
        raise ValueError(f"Unsupported DASH Flash backend: {backend}")
    live = used.view(n, groups, 1) > 0
    prior = torch.cat((bias, torch.zeros_like(bias[:, :1])), -1)
    logits = lse.reshape(n, groups, heads).float() + prior[:, :, None]
    weights = logits.masked_fill(~live, -torch.inf).softmax(1).nan_to_num()
    partial = out.view(n, groups, heads, dim).float().masked_fill(~live[..., None], 0)
    return (partial * weights[..., None]).sum(1).to(q.dtype)
