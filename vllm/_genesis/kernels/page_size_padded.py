# SPDX-License-Identifier: Apache-2.0
"""Helpers for Patch 5b — TQFullAttentionSpec `page_size_padded` pass-through.

Context (from `wiring/patch_5_page_size.py` HISTORY)
----------------------------------------------------
P5 v1 (active) uses the LCM-pad-up strategy: it enlarges the MAX page size
to the least common multiple of all layer pages. Works but costs ~51.6 %
overhead on Qwen3.6-35B-A3B when GCD(max, small) is small (observed
1,073,152 vs 813,248 → GCD=64 → LCM ~13.6 GB clipped to ceil(max/small) × small = 1,626,496 ≈ +51 % per block).

P5 v2 (attempted, reverted): pad-smaller-to-max. Mathematically optimal
per the monolith comment — frees ~34 % of KV cache VRAM. But it crashed
at runtime because:

  - Storage allocator sizes num_blocks from `layer_spec.page_size_bytes`
    (= padded size).
  - TurboQuant attention kernel reshapes that storage using NATURAL shape
    `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` which
    doesn't include the padding.
  - `num_blocks * block_size * num_kv_heads * slot_size_aligned` mismatch
    with the allocated storage → `RuntimeError: shape [...] invalid for
    input of size <padded>`.

P5b (this module + wiring/patch_5b_*) — the CORRECT way forward:

  1. Add a companion attribute `real_page_size_bytes` on TQFullAttentionSpec
     (pointing to the natural, un-padded size).
  2. Teach the TurboQuant kernel to compute reshape shape from
     `real_page_size_bytes` so padding is ignored by the kernel but
     honoured by the allocator.
  3. Enable pad-smaller-to-max behind an explicit env gate so the rollout
     is reversible: `GENESIS_ENABLE_P5B=1`.

This module holds the helper functions + decision predicate. The wiring
module (`patch_5b_tq_spec_passthrough.py`) applies the text-patch when the
env gate is set.

Status: SHIP AS DISABLED SCAFFOLDING. Enable only after VM 100 integration
measurement proves no regression vs P5 v1 on GSM8K + long-context OOM.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.page_size_padded")

_ENV_ENABLE_P5B = "GENESIS_ENABLE_P5B"


def is_p5b_enabled() -> bool:
    """Explicit env gate. Default OFF because P5 v2 crashed once already.

    Accepts truthy values in any case (1/true/TRUE/yes/Yes/on/ON).
    """
    return os.environ.get(_ENV_ENABLE_P5B, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def compute_real_page_size_bytes(layer_spec: Any) -> int:
    """Resolve the NATURAL (un-padded) page_size_bytes of a layer spec.

    Used by the TurboQuant kernel to reshape the KV cache tensor with the
    shape that actually corresponds to the layer's data layout, even when
    the allocator reserved a padded block.

    Heuristic resolution order:
      1. If spec has `real_page_size_bytes` attr (P5b-aware) → use it.
      2. Else if spec has `page_size_bytes_natural` → use it.
      3. Else compute from (block_size, num_kv_heads, slot_size_aligned)
         and fall back to upstream `page_size_bytes`.
    """
    for attr in ("real_page_size_bytes", "page_size_bytes_natural"):
        val = getattr(layer_spec, attr, None)
        if val is not None:
            return int(val)
    # Fallback — assumes spec exposes page_size_bytes (universally true
    # in current vLLM v1 API).
    return int(layer_spec.page_size_bytes)


def clamp_to_real_shape(
    tensor_shape: tuple[int, ...],
    layer_spec: Any,
) -> tuple[int, ...]:
    """Compute the natural reshape shape for a cache tensor.

    If the allocator reserved a PADDED block but the kernel expects
    NATURAL bytes, we must reshape with (natural_page_bytes // element_bytes)
    last-dim rather than the padded one.

    Returns the shape tuple unchanged if the spec doesn't expose padding.
    """
    real = compute_real_page_size_bytes(layer_spec)
    stated = getattr(layer_spec, "page_size_bytes", real)
    if real == stated:
        return tensor_shape
    # Scale the last dim by (real / stated). Both are bytes so the ratio
    # carries over into element counts.
    if stated == 0:
        return tensor_shape
    scaled = list(tensor_shape)
    scaled[-1] = int(tensor_shape[-1] * real / stated)
    return tuple(scaled)
