# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CDNA HIP per-token-head attention dispatch.

This module owns the decision of *when* to route a per-token-head INT8/INT4
KV-cache attention call to the native CDNA HIP kernels, and the call itself.
It is kept separate from the attention backend so the backend only needs a
single thin call.

Dispatch is fully automatic — there is no opt-in env var. The native kernels
are used iff:

* we are on an AMD MI3xx (gfx942 / gfx950) device,
* the compiled CDNA ops are present in the build (the ``is_available``
  helpers in ``cdna_int8_prefill`` / ``cdna_int4_prefill``),
* ``kv_cache_dtype`` is ``int8_per_token_head`` or ``int4_per_token_head``,
* the request uses none of alibi / sliding-window / sinks / logits-softcap,
* ``head_size`` is 64 or 128 (the MFMA tile sizes the kernels support), and
* the batch is homogeneous — all-prefill or all-decode. Mixed prefill+decode
  batches fall through to the Triton path (the HIP kernel handles a single
  ``(q_len, ctx_len)`` regime per CTA; the unified-batch split is a later
  phase of the port).

When none of those hold the dispatcher returns ``False`` and the caller falls
through to the Triton ``unified_attention`` path unchanged.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.attention.ops import cdna_int4_prefill, cdna_int8_prefill

logger = init_logger(__name__)

# Remember which (regime, dtype, head_size) tags we've already logged so the
# one-time "dispatching" info line does not spam the log on every forward.
_dispatch_logged: set[str] = set()


def _log_once(tag: str) -> None:
    if tag not in _dispatch_logged:
        _dispatch_logged.add(tag)
        logger.info("[CDNA-PTH] dispatching %s kernel", tag)


def maybe_dispatch_cdna_pth(
    *,
    kv_cache_dtype: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor | None,
    v_scale_cache: torch.Tensor | None,
    scale: float,
    alibi_slopes,
    sliding_window,
    sinks,
    logits_soft_cap,
) -> bool:
    """Run the CDNA HIP per-token-head attention kernel if eligible.

    Returns ``True`` if the native kernel handled the call (``output`` is
    written in place), or ``False`` to signal the caller should fall through
    to the Triton path. See the module docstring for the eligibility rules.
    """
    is_int4 = kv_cache_dtype == "int4_per_token_head"
    is_int8 = kv_cache_dtype == "int8_per_token_head"
    if not (is_int4 or is_int8):
        return False

    # Platform + compiled-op gate. is_available() already checks ROCm and
    # on_mi3xx(), and caches its result.
    if is_int4:
        if not cdna_int4_prefill.is_available():
            return False
    elif not cdna_int8_prefill.is_available():
        return False

    if (
        alibi_slopes is not None
        or sliding_window != (-1, -1)
        or sinks is not None
        or (logits_soft_cap is not None and logits_soft_cap > 0)
    ):
        return False

    head_size = query.shape[-1]
    if head_size not in (64, 128):
        return False
    if k_scale_cache is None or v_scale_cache is None:
        return False

    dtype_tag = "int4" if is_int4 else "int8"

    # Homogeneous-batch gate. If max_query_len <= 1 every sequence is a
    # single-token decode; otherwise treat the batch as all-prefill. Reading
    # max_query_len avoids a host sync on per-sequence query lengths.
    if attn_metadata.max_query_len <= 1:
        _log_once(f"decode-{dtype_tag}-hs{head_size}")
        decode_op = (torch.ops._C.pth_decode_int4_cdna if is_int4
                     else torch.ops._C.pth_decode_int8_cdna)
        decode_op(
            output.view(-1, query.shape[-2], head_size),
            query.view(-1, query.shape[-2], head_size),
            key_cache, value_cache, k_scale_cache, v_scale_cache,
            attn_metadata.block_table, attn_metadata.seq_lens, scale,
        )
        return True

    _log_once(f"prefill-{dtype_tag}-hs{head_size}")
    prefill = (cdna_int4_prefill.cdna_int4_paged_prefill if is_int4
               else cdna_int8_prefill.cdna_int8_paged_prefill)
    prefill(
        output,        # [num_tokens, num_heads, head_size]
        query,         # [num_tokens, num_heads, head_size]
        key,           # [num_tokens, num_kv_heads, head_size] (chunk)
        value,         # [num_tokens, num_kv_heads, head_size] (chunk)
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        attn_metadata.block_table,
        attn_metadata.query_start_loc,
        attn_metadata.seq_lens,
        attn_metadata.max_query_len,
        float(scale),
        True,          # causal
    )
    return True
