# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing FP16 sparse decode for DeepSeek V4.

Adapts ``vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py`` to plain FP16
KV: the XPU path dequantizes UE8M0 FP8 cache pages to BF16 on the fly; the
Turing path copies already-FP16 KV rows with ``gather_fp16_slots`` and feeds
the FP16 ``triton_mla_sparse_interface``.
"""

import torch

from vllm.models.deepseek_v4.turing.constants import HEAD_DIM
from vllm.models.deepseek_v4.turing.gather import gather_fp16_slots
from vllm.models.deepseek_v4.turing.sparse import triton_mla_sparse_interface

_BLOCK_N = 16


def turing_sparse_decode(
    q: torch.Tensor,  # [num_tokens, num_heads, head_dim] fp16
    kv_cache: torch.Tensor | None,  # [num_blocks, block_size, head_dim] fp16
    swa_kv_cache: torch.Tensor,  # [num_blocks, swa_block_size, head_dim] fp16
    swa_only: bool,
    topk_indices: torch.Tensor | None,  # [num_tokens, 1, topk] global slots
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,  # [num_tokens, 1, swa_k] global slots
    swa_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,  # [num_tokens, num_heads, head_dim] fp16
) -> None:
    """Gather FP16 KV rows and run FP16 sparse MLA attention on the decode set."""
    num_tokens = q.shape[0]
    device = q.device

    if not swa_only and topk_indices is not None:
        topk_idx_2d = (
            topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices
        )
        max_topk = topk_idx_2d.shape[1]
    else:
        topk_idx_2d = None
        max_topk = 0

    swa_idx_2d = swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices
    max_swa = swa_idx_2d.shape[1]

    K_total = max_topk + max_swa

    workspace = torch.empty(
        (num_tokens * K_total, HEAD_DIM), dtype=torch.float16, device=device
    )
    ws_3d = workspace.view(num_tokens, K_total, HEAD_DIM)

    if not swa_only and topk_idx_2d is not None and kv_cache is not None:
        topk_flat = topk_idx_2d.reshape(-1).to(torch.int32)
        topk_buf = torch.empty(
            (num_tokens * max_topk, HEAD_DIM),
            dtype=torch.float16,
            device=device,
        )
        gather_fp16_slots(topk_buf, kv_cache, topk_flat, kv_cache.shape[1])
        ws_3d[:, :max_topk, :] = topk_buf.view(num_tokens, max_topk, HEAD_DIM)

    swa_flat = swa_idx_2d.reshape(-1).to(torch.int32)
    swa_buf = torch.empty(
        (num_tokens * max_swa, HEAD_DIM), dtype=torch.float16, device=device
    )
    gather_fp16_slots(swa_buf, swa_kv_cache, swa_flat, swa_kv_cache.shape[1])
    ws_3d[:, max_topk:, :] = swa_buf.view(num_tokens, max_swa, HEAD_DIM)

    # Workspace layout per token t: [topk_0..topk_{max_topk-1}, swa_0..swa_{max_swa-1}].
    # The attention kernel reads only indices[0..combined_lens-1], so indices
    # must be packed contiguously: [valid_topk..., valid_swa..., -1 padding...].
    if not swa_only and topk_lens is not None:
        combined_lens = (topk_lens + swa_lens).to(torch.int32)
    else:
        combined_lens = swa_lens.to(torch.int32)

    max_combined = int(combined_lens.max().item()) if combined_lens.numel() > 0 else 0
    max_combined_padded = ((max_combined + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N

    combined_indices = torch.full(
        (num_tokens, max_combined_padded),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.int32) * K_total

    if not swa_only and topk_lens is not None:
        max_tlen = int(topk_lens.max().item())
        topk_range = torch.arange(max_tlen, device=device, dtype=torch.int32).unsqueeze(
            0
        )
        topk_valid = topk_range < topk_lens.unsqueeze(1)
        topk_ws_indices = token_offsets.unsqueeze(1) + topk_range
        combined_indices[:, :max_tlen] = torch.where(
            topk_valid,
            topk_ws_indices,
            torch.tensor(-1, dtype=torch.int32, device=device),
        )
        swa_range = torch.arange(max_swa, device=device, dtype=torch.int32).unsqueeze(0)
        swa_valid = swa_range < swa_lens.unsqueeze(1)
        swa_ws_indices = token_offsets.unsqueeze(1) + max_topk + swa_range
        for t_idx in range(num_tokens):
            tlen = int(topk_lens[t_idx].item())
            slen = int(swa_lens[t_idx].item())
            combined_indices[t_idx, tlen : tlen + slen] = swa_ws_indices[t_idx, :slen]
    else:
        effective_swa = min(max_swa, max_combined_padded)
        swa_range = torch.arange(
            effective_swa, device=device, dtype=torch.int32
        ).unsqueeze(0)
        swa_valid = swa_range < swa_lens.unsqueeze(1)
        swa_ws_indices = token_offsets.unsqueeze(1) + swa_range
        combined_indices[:, :effective_swa] = torch.where(
            swa_valid,
            swa_ws_indices,
            torch.tensor(-1, dtype=torch.int32, device=device),
        )

    out_attn, _, _ = triton_mla_sparse_interface(
        q=q,
        kv=workspace.unsqueeze(1),
        indices=combined_indices.unsqueeze(1),
        sm_scale=softmax_scale,
        d_v=q.shape[-1],
        # q and KV are fully RoPE'd on insert, so the full-dim dot product is
        # equivalent and keeps BLOCK_DMODEL (= 512) a power of two.
        block_dpe=0,
    )
    out.copy_(out_attn)
