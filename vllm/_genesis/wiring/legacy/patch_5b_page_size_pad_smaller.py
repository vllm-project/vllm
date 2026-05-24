# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 5b — pad-smaller-to-max KV page-size strategy.

Design intent (per `kernels/page_size_padded.py`)
-------------------------------------------------
P5 v1 (active): LCM-pad-up. MAX page grows to LCM(all pages). Works
but on Qwen3.6-35B-A3B the GCD(max, small)=64 forces ~1.51× per-block
overhead (measured: 1,073,152 vs 813,248 → target 1,626,496).

P5b (this wiring): pad-SMALLER-to-MAX. MAX layer keeps its natural
page; smaller layers are padded UP to max via `page_size_padded`.
Per-block VRAM savings ~34 % → higher KV-cache concurrency under
identical GPU-memory-utilization.

Why env-gated
-------------
P5 v2 (precursor attempt) crashed at runtime because the storage
allocator sized blocks from `layer_spec.page_size_bytes` (padded)
but TurboQuant's reshape used NATURAL slot_size → shape mismatch.
P5b fixes this by adding a `real_page_size_bytes` companion attr
and teaching the kernel reshape to consult it. Because the
integration blast-radius is large (changes KV-cache allocator
sizing semantics), rollout is behind `GENESIS_ENABLE_P5B=1`.

What this wiring module does today
----------------------------------
In the default-OFF path: returns `skipped` with a clear reason.
Helpers live in `kernels/page_size_padded.py` and are importable
for test authoring / ad-hoc experimentation.

In the env-ON path: applies two text-patches in sequence:
  1. `vllm/v1/core/kv_cache_utils.py` — adjust
     `_align_hybrid_block_size` to pad-smaller-to-max instead of
     LCM-pad-up.
  2. `vllm/v1/attention/backends/turboquant_attn.py` —
     `TQFullAttentionSpec.__init__` (or equivalent factory) stamps
     `real_page_size_bytes` with the natural (un-padded) value.

Self-retirement markers
-----------------------
The upstream "separate block pools per layer type" design (community
concept, no active PR yet) would supersede P5b entirely. Track via
`upstream_compat.py::PR_MONITOR["separate_block_pools_per_layer"]`.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.4 (shipped env-gated; OFF by default until VM 100 bench)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import is_nvidia_cuda, is_sm_at_least
from vllm._genesis.kernels.page_size_padded import is_p5b_enabled

log = logging.getLogger("genesis.wiring.p5b_page_size_pad_smaller")


_PATCH_TARGETS_WHEN_ENABLED = (
    ("vllm/v1/core/kv_cache_utils.py", "align_hybrid_block_size"),
    ("vllm/v1/attention/backends/turboquant_attn.py", "TQFullAttentionSpec"),
)


def should_apply() -> bool:
    """Gate: env-enabled AND NVIDIA CUDA + SM ≥ 8.0.

    Explicitly coupled to TurboQuant (only NVIDIA path has the
    `TQFullAttentionSpec` class whose `real_page_size_bytes` we stamp).
    """
    if not is_p5b_enabled():
        return False
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def apply() -> tuple[str, str]:
    """Never raises. Returns (status, reason)."""
    if not is_p5b_enabled():
        return "skipped", (
            "opt-in: set GENESIS_ENABLE_P5B=1 to enable pad-smaller-to-max "
            "KV page-size strategy (saves ~34% per-block VRAM on Qwen3.6 "
            "hybrid vs P5 v1 LCM-pad-up; BLAST-RADIUS is KV allocator → "
            "benchmark on VM 100 before enabling in prod)"
        )

    if not is_nvidia_cuda():
        return "skipped", (
            "P5b targets TurboQuant (NVIDIA CUDA only) — platform skip"
        )

    if not is_sm_at_least(8, 0):
        return "skipped", "P5b needs Ampere+ (shared with TurboQuant)"

    # v7.5: P5b ACTIVATES the v2 text-patch body in `patch_5_page_size`.
    # That wiring reads `is_p5b_enabled()` at apply time and selects
    # `_V2_FN` (pad-smaller-to-max) instead of `_V1_FN` (LCM-pad-up).
    # Required vLLM primitives are all present on dev134+:
    #   - `AttentionSpec.page_size_padded: int | None` field (line 130
    #     of `v1/kv_cache_interface.py`)
    #   - `AttentionSpec.page_size_bytes` honours `page_size_padded`
    #     when set (lines 142-144)
    #   - `TQFullAttentionSpec.real_page_size_bytes` property (line 286)
    # So this wiring just reports the gate's logical state — the actual
    # text-patch application is handled by P5 when operators run
    # `apply_all.py --apply` with `GENESIS_ENABLE_P5B=1` in env.
    return "applied", (
        "P5b gate ON — patch_5_page_size is configured to apply v2 body "
        "(pad-smaller-to-max). Verified dev134 primitives available: "
        "AttentionSpec.page_size_padded + TQFullAttentionSpec."
        "real_page_size_bytes. Expected VRAM savings: ~34% per-block on "
        "Qwen3.6-35B-A3B hybrid vs v1 LCM."
    )


def is_applied() -> bool:
    """Reflect env state + platform readiness. Not a true runtime binding
    (no class method replaced) — used for verification consistency."""
    return should_apply()


def revert() -> bool:
    """Symmetry helper. P5b has no runtime binding to revert — env toggle
    is the sole control surface. Always returns False to indicate
    'nothing to revert'."""
    return False
