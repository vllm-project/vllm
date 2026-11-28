# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA prefill attention with FlashInfer's QToken-KvBlock-Sparse kernel.

It attends GROUP_SIZE rows of a request as one tile over the union of their
selected blocks, so each K/V block is read once per group rather than per row.
"""

import torch

from vllm.config import AttentionConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, next_power_of_2

# Query rows per FlashInfer route, the best on Qwen3.8 prefill shapes.
GROUP_SIZE = 8


def get_q_dtype(
    kv_dtype: torch.dtype, attention_config: AttentionConfig
) -> torch.dtype:
    """FlashInfer prefill Q dtype: FP8 for an opted-in FP8 cache, else BF16."""
    if (
        kv_dtype == torch.uint8
        and attention_config.use_prefill_query_quantization
        and not attention_config.disable_flashinfer_q_quantization
    ):
        return torch.float8_e4m3fn
    return torch.bfloat16


def is_flashinfer_qsa_supported(
    kv_dtype: torch.dtype,
    q_dtype: torch.dtype,
    block_size: int,
    block_topk: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> bool:
    """Whether FlashInfer 0.7.4 can plan this geometry on this device."""
    return (
        current_platform.is_device_capability_family(100)
        # An FP8 cache is allocated as uint8 holding e4m3 bytes.
        and (q_dtype, kv_dtype)
        in ((torch.bfloat16, torch.bfloat16), (torch.float8_e4m3fn, torch.uint8))
        and has_flashinfer()
        and block_size in (4, 8, 16, 32, 64, 128)
        and head_dim in (64, 128, 192, 256)
        and block_topk <= 512
        # A tile holds GROUP_SIZE rows x the query heads of one KV head.
        and GROUP_SIZE * (num_heads // num_kv_heads) <= 128
    )


def get_workspace_size(
    max_num_tokens: int,
    max_num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_seq_len_kv: int,
    block_size: int,
    block_topk: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    device: torch.device,
) -> int:
    """Workspace bytes for the largest prefill. Only shapes matter, so
    placeholder tensors stand in."""
    from flashinfer.attention.prims_ts import (
        get_q_token_kv_block_sparse_workspace_size,
    )

    if kv_dtype == torch.uint8:
        kv_dtype = torch.float8_e4m3fn
    q = torch.empty((1, 1, 1), dtype=q_dtype, device=device).expand(
        max_num_tokens, num_heads, head_dim
    )
    k_cache = torch.empty(
        (1, num_kv_heads, page_size, head_dim), dtype=kv_dtype, device=device
    )
    block_table = torch.empty((1, 1), dtype=torch.int32, device=device).expand(
        max_num_reqs, cdiv(max_seq_len_kv, page_size)
    )
    # One route per request plus one per further GROUP_SIZE rows.
    num_reqs = min(max_num_tokens, max_num_reqs)
    qo_indptr = torch.empty(
        num_reqs + (max_num_tokens - num_reqs) // GROUP_SIZE + 1,
        dtype=torch.int32,
        device=device,
    )
    return get_q_token_kv_block_sparse_workspace_size(
        q,
        k_cache,
        block_table,
        block_topk=block_topk,
        max_seq_len_kv=max_seq_len_kv,
        qo_indptr=qo_indptr,
        group_size=GROUP_SIZE,
        kv_block_size=block_size,
        split_kv=False,
        out_dtype=torch.bfloat16,
    )


# Both tables start num_decodes rows in, so their alignment varies per step;
# they are only read one element at a time.
@triton.jit(do_not_specialize_on_alignment=["block_table_ptr", "seq_lens_ptr"])
def _zero_partial_tail_kernel(
    kv_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    stride_block,
    stride_head,
    stride_token,
    stride_table,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    ROW: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int64)
    block_end = (seq_len + KV_BLOCK - 1) // KV_BLOCK * KV_BLOCK
    cols = tl.arange(0, ROW_BLOCK)
    for k in tl.static_range(KV_BLOCK - 1):
        pos = seq_len + k
        if pos < block_end:
            page = tl.load(block_table_ptr + req * stride_table + pos // PAGE_SIZE)
            base = page.to(tl.int64) * stride_block + (pos % PAGE_SIZE) * stride_token
            for h in tl.static_range(NUM_KV_HEADS):
                tl.store(
                    kv_cache_ptr + base + h * stride_head + cols,
                    tl.zeros([ROW_BLOCK], dtype=kv_cache_ptr.dtype.element_ty),
                    mask=cols < ROW,
                )


@triton.jit
def _sigmoid_gate_mul_kernel(
    output_ptr,
    gate_ptr,
    numel,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    output = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr + offsets,
        output.to(tl.float32) * tl.sigmoid(gate.to(tl.float32)),
        mask=mask,
    )


def qsa_sparse_paged_prefill(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    num_decodes: int,
    seq_lens: torch.Tensor,
    block_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    qo_indptr: torch.Tensor,
    block_size: int,
    max_seq_len_kv: int,
    workspace_size: int,
    scale: float,
    output_gate: torch.Tensor,
    out: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Write `sigmoid(output_gate) * attention` of prefill rows into `out`.

    `block_indices` holds the indexer's selected blocks, -1 padded;
    `token_to_req` indexes all of `block_table`, whose prefill requests start
    at `num_decodes`; `seq_lens` holds those requests' lengths. An FP8 cache
    (uint8 storage) takes FP8 `query` (unit scale) and its per-tensor
    `k_scale`/`v_scale`.
    """
    from flashinfer.attention.prims_ts import QTokenKvBlockSparsePagedTSWrapper

    _, num_kv_heads, page_size, row = kv_cache.shape
    head_dim = row // 2
    # FlashInfer loads whole blocks and its causal mask does not clear NaN/Inf
    # in V; unwritten tail slots may hold another cache group's bytes.
    _zero_partial_tail_kernel[(seq_lens.numel(),)](
        kv_cache,
        block_table[num_decodes:],
        seq_lens,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        PAGE_SIZE=page_size,
        NUM_KV_HEADS=num_kv_heads,
        ROW=row,
        ROW_BLOCK=next_power_of_2(row),
        KV_BLOCK=block_size,
    )
    if kv_cache.dtype == torch.uint8:
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    k_cache, v_cache = kv_cache[..., :head_dim], kv_cache[..., head_dim:]
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=query.device)
    # A plan binds this layer's K/V and the batch shapes, so plan every call;
    # its ~0.35 ms of host time hides behind the GPU except on short prefills.
    # TODO: fix in FlashInfer (bind K/V and shapes per run()) and plan once.
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        qo_indptr.numel() - 1,
        query.shape[1],
        num_kv_heads,
        head_dim,
        block_size,
        page_size,
        block_indices.shape[1],
        max_seq_len_kv,
        device=query.device,
        workspace_buffer=workspace,
        group_size=GROUP_SIZE,
        packed_query=True,
        split_kv=False,
        q_dtype=query.dtype,
        kv_dtype=kv_cache.dtype,
        out_dtype=out.dtype,
    )
    wrapper.run(
        query,
        (k_cache, v_cache),
        block_table,
        block_indices,
        token_to_req,
        positions,
        qo_indptr=qo_indptr,
        # Per-tensor K/V dequantization: K folds into the softmax scale, V
        # scales the output. Both are 1 for a BF16 cache.
        sm_scale=scale * k_scale,
        v_scale=v_scale,
        out=out,
    )
    numel = out.numel()
    BLOCK = 2048
    _sigmoid_gate_mul_kernel[(triton.cdiv(numel, BLOCK),)](
        out,
        output_gate,
        numel,
        BLOCK=BLOCK,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=4,
    )
