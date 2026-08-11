# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 MLA decode for the Kimi-K3 DSpark draft group.

The draft block is non-causal: every one of a request's ``query_len`` positions
attends to the same committed KV prefix. The Triton path expresses that by
flattening the block to one decode row per query token, which makes each row
re-read the whole KV span -- ``query_len`` times the traffic. This kernel keeps
the block folded into a single workgroup tile and reads the span once.
"""

import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

D_QK = 576  # 512 latent + 64 rope
D_V = 512
KV_TILE = 128
LOG2E = 1.4426950408889634
_NUM_CU = 256

_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _supported() -> bool:
    # v_mfma_scale_f32_16x16x128_f8f6f4 and ds_read_b64_tr_b8 are gfx950-only.
    return current_platform.is_rocm() and on_gfx950()


def h40_draft_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    sm_scale: float,
    q_scale: float,
    kv_scale: float,
) -> bool:
    """Fill ``out`` in place. Returns False if the shape is not supported."""
    if not _supported():
        return False
    if q.dtype is not torch.float8_e4m3fn or kv_cache.dtype is not torch.float8_e4m3fn:
        return False
    if out.dtype is not torch.bfloat16:
        return False
    if q.dim() != 3 or q.size(2) != D_QK or out.size(-1) != D_V:
        return False

    num_reqs = seq_lens.numel()
    if query_len < 2 or q.size(0) != num_reqs * query_len:
        return False
    if block_table.dim() != 2 or block_table.stride(1) != 1:
        return False

    page_size = kv_cache.size(1)
    if page_size % KV_TILE:
        # A KV tile would straddle a page; the kernel resolves one page per tile.
        return False

    num_heads = q.size(1)
    rows = num_heads * query_len
    blocks = -(-rows // KV_TILE)
    # One workgroup per CU for exactly one round. Derived from shapes only, so
    # it stays correct under CUDA graph capture.
    num_splits = max(1, _NUM_CU // (num_reqs * blocks))

    workgroups = num_reqs * num_splits
    key = (rows, q.device.index)
    ws = _WORKSPACE.get(key)
    if ws is None or ws[0].size(0) < workgroups:
        cap = max(workgroups, _NUM_CU)
        opts = {"dtype": torch.float32, "device": q.device}
        ws = (
            torch.empty(cap, rows, **opts),
            torch.empty(cap, rows, **opts),
            torch.empty(cap, rows, D_V, **opts),
        )
        _WORKSPACE[key] = ws

    torch.ops._C.kimi_k3_h40_draft_mla_decode(
        q,
        kv_cache,
        block_table,
        seq_lens,
        ws[0][:workgroups].view(num_reqs, num_splits, rows),
        ws[1][:workgroups].view(num_reqs, num_splits, rows),
        ws[2][:workgroups].view(num_reqs, num_splits, rows, D_V),
        out,
        query_len,
        num_splits,
        sm_scale * q_scale * kv_scale * LOG2E,
        kv_scale,
    )
    return True
