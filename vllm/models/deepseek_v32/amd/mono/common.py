# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the GLM-5.2 MonoKernel live mode (live.py) and its debug harnesses.

Only torch is imported at module level (vLLM / kernel imports are lazy), so the pure
helpers are CPU-testable."""

from __future__ import annotations

from typing import Any

import torch

# GLM-5.2 decode geometry the kernel is built for (kernel/config.py GLM5_CONFIG).
TOPK = 2048  # index_topk: sparse keys per query row
HIDDEN = 6144
MLA_ROW = 576  # kv_lora 512 + rope 64: one fused MLA cache row
ROPE_HALF = 32  # cos / sin columns per position (rotary dim 64, interleaved pairs)


# ------------------------------------------------------------------ distributed
def tp_uniform(ok: bool, cpu_group) -> bool:
    """Rank-uniform go/no-go + launch alignment: device sync, then MIN all-reduce over
    the CPU group."""
    import torch.distributed as dist

    torch.accelerator.synchronize()
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(flag.item())


# ------------------------------------------------------------------ RoPE
def rope_tables(
    layer, max_len: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Kernel RoPE ABI (ATOM copy): separate contiguous BF16 [max_pos, 32] cos / sin,
    interleaved pairs, sliced from vLLM's ``rotary_emb.cos_sin_cache`` [max_pos, 64] =
    [cos | sin]. Returns (cos, sin, src dtype)."""
    cache = layer.self_attn.rotary_emb.cos_sin_cache
    c = cache if max_len is None else cache[:max_len]
    half = c.shape[1] // 2
    assert half == ROPE_HALF, c.shape
    cos = c[:, :half].to(torch.bfloat16).contiguous()
    sin = c[:, half:].to(torch.bfloat16).contiguous()
    return cos, sin, str(cache.dtype)


# ------------------------------------------------------------------ forward-context
# metadata
def layer_metadata(layer, list_policy: str = "first", fc=None):
    """(attention metadata, slot mapping) of ``layer`` for the current step.

    ``list_policy``: what to do with a list-form ``attn_metadata`` (ubatching / spec):
    "first" takes element 0 (live mode), "none" returns (None, None) (check mode).
    ``fc`` = a forward context (default: vLLM's current one)."""
    if fc is None:
        from vllm.forward_context import get_forward_context

        fc = get_forward_context()
    md = fc.attn_metadata
    if isinstance(md, list):
        if list_policy == "none":
            return None, None
        md = md[0] if md else None
    if isinstance(md, dict):
        md = md.get(layer.self_attn.layer_name)
    sm = fc.slot_mapping
    sm = sm.get(layer.self_attn.layer_name) if isinstance(sm, dict) else None
    return md, sm


# ------------------------------------------------------------------ kernel ops
def build_width_ops(
    order,
    weights,
    S: int,
    *,
    rank: int,
    npes: int,
    group,
    poll_limit,
    attention_weight,
    prepared_for=None,
    launches_per_step: int = 1,
    poll_early_out: bool = False,
    extra_kwargs_for=None,
    stage_opts: dict | None = None,
) -> dict:
    """One ``Glm5MonoKernel`` per mono layer for kernel width ``S``, all sharing the
    first one's runtime (scratch, peer buffer, step counter). ``prepared_for(L)`` ->
    packed weights to reuse or None. ``extra_kwargs_for(L)`` -> extra Glm5MonoKernel
    kwargs (e.g. the fused indexer) or None; layers with a different scratch geometry
    (with_indexer) get their own shared runtime (the first such op owns it)."""
    from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel

    ops: dict[tuple[int, int], Any] = {}
    runtimes: dict[bool, Any] = {}
    for L in order:
        extra = (extra_kwargs_for(L) if extra_kwargs_for is not None else None) or {}
        key = bool(extra.get("with_indexer", False))
        op = Glm5MonoKernel(
            weights[L],
            S,
            rank=rank,
            npes=npes,
            group=group,
            topk=TOPK,
            attention_weight=attention_weight,
            poll_limit=poll_limit,
            prepared_weights=None if prepared_for is None else prepared_for(L),
            runtime=runtimes.get(key),
            launches_per_step=launches_per_step,
            poll_early_out=poll_early_out,
            **extra,
            **stage_kwargs(stage_opts, S),
        )
        runtimes.setdefault(key, op)
        ops[(L, S)] = op
    return ops


def stage_kwargs(stage_opts: dict | None, S: int) -> dict:
    """The enabled build-time stage options valid at width ``S`` (64-key split tasks
    spill at S10 / S12)."""
    return {
        k: v
        for k, v in (stage_opts or {}).items()
        if v and (k != "split_keys64" or S <= 8)
    }


def warm_up_launch(
    op, S: int, curpos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dev
) -> None:
    """JIT compile / code-object load: one launch on a scratch cache with every row
    inactive (slot -1, empty sparse range), then a device sync. Callers align ranks
    around it (``tp_uniform``)."""
    wc = torch.zeros(64, MLA_ROW, dtype=torch.bfloat16, device=dev)
    z = torch.zeros(S, dtype=torch.int64, device=dev)
    op.forward(
        torch.zeros(S, HIDDEN, dtype=torch.bfloat16, device=dev),
        curpos,
        wc,
        wc,
        torch.zeros(1, dtype=torch.int32, device=dev),
        cos,
        sin,
        positions=z,
        slot_mapping=z - 1,
        sparse_kv_indptr=torch.zeros(S + 1, dtype=torch.int32, device=dev),
    )
    torch.accelerator.synchronize()


def convert_topk(
    md, topk_rows: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor
) -> None:
    """Convert vLLM's logical per-row top-k to the kernel's CSR physical slot ids
    (vLLM's own Triton kernel)."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    T = topk_rows.shape[0]
    triton_convert_req_index_to_global_index(
        md.req_id_per_token[:T],
        md.block_table,
        topk_rows,
        indptr,
        indices,
        BLOCK_SIZE=md.block_size,
        NUM_TOPK_TOKENS=TOPK,
    )
