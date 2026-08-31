# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch Triton checkpoint scatter for GDN ``mamba_cache_mode='all'``.

Replaces the Python-looped ``gdn_scatter_block_checkpoints``
(all_mode_utils.py) — which issues four ``int(tensor)`` GPU syncs plus
arange/index_put chains per prefill sequence, per layer — with one kernel
launch that emits bit-identical stores: the FLA +1-chunk shift
(``h[c]`` = state at the START of chunk c), the per-sequence ``seq_hi``
clamp, and the skip-unaligned-interior rule (never poison APC) are preserved
exactly. Gated by ``VLLM_GDN_INKERNEL_CKPT_WRITE`` (default on; "0" restores
the Python loop).
"""

import os

import torch

from vllm.triton_utils import tl, triton


def gdn_inkernel_ckpt_write_enabled() -> bool:
    return os.environ.get("VLLM_GDN_INKERNEL_CKPT_WRITE", "1") != "0"


@triton.jit
def _gdn_scatter_block_ckpt_kernel(
    ssm_state_ptr,  # (num_blocks, ...) written in place
    inter_ptr,  # (NT, ...) FLA h, state at the START of each chunk
    final_ptr,  # (num_prefills, ...) final recurrent states
    block_table_ptr,  # (num_prefills, max_blocks)
    first_ptr,  # (num_prefills,) block_idx_first_scheduled_token_p
    last_ptr,  # (num_prefills,) block_idx_last_scheduled_token_p
    ncomp_ptr,  # (num_prefills,) num_computed_tokens_p
    first_chunk_ptr,  # (first_chunk_len,) chunk_offsets
    nt,
    first_chunk_len,
    numel,
    stride_ssm,
    stride_inter,
    stride_final,
    stride_bt,
    BLOCK_SIZE: tl.constexpr,  # mamba_block_size
    CHUNK: tl.constexpr,  # FLA chunk size
    BLOCK_N: tl.constexpr,
):
    i_s = tl.program_id(0)
    j_rel = tl.program_id(1)
    i_tile = tl.program_id(2)
    first = tl.load(first_ptr + i_s).to(tl.int64)
    last = tl.load(last_ptr + i_s).to(tl.int64)
    # n interior blocks (may be <= 0: final block only, like the loop).
    n = tl.maximum(last - first, 0)
    if j_rel > n:
        return
    offs = i_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < numel
    # src is upcast to fp32 (lossless from bf16/fp16/fp32 chunk states) so
    # both branches agree on one type; the store rounds once to the pool
    # dtype, matching the eager .to(ssm_state.dtype) cast.
    if j_rel == n:
        # Final (possibly partial) block <- the sequence's final state.
        dst = tl.load(block_table_ptr + i_s * stride_bt + last).to(tl.int64)
        src = tl.load(final_ptr + i_s * stride_final + offs, mask=mask).to(
            tl.float32
        )
    else:
        ncomp = tl.load(ncomp_ptr + i_s).to(tl.int64)
        if ncomp % CHUNK != 0:
            # Not chunk-aligned: interior boundaries don't map to exact FLA
            # chunk states — skip rather than poison APC (the final block
            # above is exact regardless).
            return
        j = first + j_rel
        fc = tl.load(first_chunk_ptr + i_s).to(tl.int64)
        if i_s + 1 < first_chunk_len:
            seq_hi = tl.load(first_chunk_ptr + i_s + 1).to(tl.int64) - 1
        else:
            seq_hi = tl.zeros((), tl.int64) + nt - 1
        # Interior block j ends after (j+1)*B sequence tokens, i.e.
        # ((j+1)*B - ncomp) // CHUNK scheduled chunks in; h[c] is the state
        # at the START of chunk c (+1-chunk shift vs Mamba2), clamped to
        # this sequence's chunk range (defense-in-depth).
        k = ((j + 1) * BLOCK_SIZE - ncomp) // CHUNK
        cidx = tl.minimum(tl.maximum(fc + k, fc), seq_hi)
        dst = tl.load(block_table_ptr + i_s * stride_bt + j).to(tl.int64)
        src = tl.load(inter_ptr + cidx * stride_inter + offs, mask=mask).to(
            tl.float32
        )
    p_dst = ssm_state_ptr + dst * stride_ssm + offs
    tl.store(p_dst, src.to(p_dst.dtype.element_ty), mask=mask)


def gdn_scatter_block_checkpoints_triton(
    ssm_state: torch.Tensor,
    inter_states: torch.Tensor,
    final_states: torch.Tensor,
    block_table_p: torch.Tensor,
    block_idx_first_scheduled_token_p: torch.Tensor,
    block_idx_last_scheduled_token_p: torch.Tensor,
    num_computed_tokens_p: torch.Tensor,
    first_chunk_p: torch.Tensor,
    mamba_block_size: int,
    chunk_size: int,
    num_prefill_tokens: int,
) -> None:
    """Drop-in kernel equivalent of ``gdn_scatter_block_checkpoints``.

    ``num_prefill_tokens`` (host int) bounds the per-sequence interior block
    count so the grid needs no GPU sync.
    """
    num_prefills = block_idx_last_scheduled_token_p.shape[0]
    if num_prefills == 0:
        return
    numel = ssm_state[0].numel()
    assert ssm_state[0].is_contiguous()
    assert inter_states[0].is_contiguous() and inter_states[0].numel() == numel
    assert final_states[0].is_contiguous() and final_states[0].numel() == numel
    assert block_table_p.stride(-1) == 1
    max_j = num_prefill_tokens // mamba_block_size + 2
    block_n = 2048
    grid = (num_prefills, max_j, triton.cdiv(numel, block_n))
    _gdn_scatter_block_ckpt_kernel[grid](
        ssm_state,
        inter_states,
        final_states,
        block_table_p,
        block_idx_first_scheduled_token_p,
        block_idx_last_scheduled_token_p,
        num_computed_tokens_p,
        first_chunk_p,
        inter_states.shape[0],
        first_chunk_p.shape[0],
        numel,
        ssm_state.stride(0),
        inter_states.stride(0),
        final_states.stride(0),
        block_table_p.stride(0),
        BLOCK_SIZE=mamba_block_size,
        CHUNK=chunk_size,
        BLOCK_N=block_n,
    )
