# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 46 — GDN `fused_gdn_gating` output buffer pool.

Replaces two per-call `torch.empty(...)` allocations with persistent
pool acquires (`GdnGatingBufferManager.acquire_g/acquire_beta`) at
the helper `fused_gdn_gating` in
`vllm/model_executor/layers/mamba/gdn_linear_attn.py`.

Strategy
--------
Text-patch: 2 sub-patches (one per `torch.empty`). Both anchors are
unambiguous in the dev134 baseline (only callsite). Drift-markers
watch for upstream adopting their own persistent-buffer variant.

Applied by default on NVIDIA SM ≥ 8.0 (matches existing P28 / P22
gates). No env-opt-in — the patch is byte-exact and has no semantic
change.

Upstream drift markers
----------------------
- `GdnGatingBufferManager` — already-present means we applied
- `acquire_gating_g` — if upstream adopts similar helper
- `fused_gdn_gating_persistent_buffers`

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.7 (default-on NVIDIA SM 8.0+)
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import (
    is_nvidia_cuda, is_sm_at_least, resolve_vllm_file, vllm_install_root,
)
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p46_gdn_gating_buffers")

GENESIS_P46_MARKER = "Genesis P46 GDN gating buffer pool v7.7"

UPSTREAM_DRIFT_MARKERS = [
    "GdnGatingBufferManager",
    "acquire_gating_g",
    "fused_gdn_gating_persistent_buffers",
]


# Anchors pinned from `reference/dev134_gdn_linear_attn.py:1195-1196`.
# Both lines use explicit dtype — we keep the dtype extraction.
_OLD_G = (
    "    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)"
)
_NEW_G = (
    "    # [Genesis P46] persistent `g` buffer (one-per-shape-key pool)\n"
    "    from vllm._genesis.kernels.gdn_gating_buffer import (\n"
    "        GdnGatingBufferManager as _GenesisGdnGatingBuf,\n"
    "    )\n"
    "    g = _GenesisGdnGatingBuf.acquire_g(\n"
    "        batch=batch, num_heads=num_heads,\n"
    "        device=a.device, dtype=torch.float32,\n"
    "    )"
)

_OLD_BETA = (
    "    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)"
)
_NEW_BETA = (
    "    # [Genesis P46] persistent `beta_output` buffer\n"
    "    beta_output = _GenesisGdnGatingBuf.acquire_beta(\n"
    "        batch=batch, num_heads=num_heads,\n"
    "        device=b.device, dtype=b.dtype,\n"
    "    )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="P46 GDN gating buffer pool",
        target_file=target,
        marker=GENESIS_P46_MARKER,
        sub_patches=[
            TextPatch(
                name="p46_g_buffer",
                anchor=_OLD_G,
                replacement=_NEW_G,
                required=True,
            ),
            TextPatch(
                name="p46_beta_buffer",
                anchor=_OLD_BETA,
                replacement=_NEW_BETA,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def apply() -> tuple[str, str]:
    """Never raises. Returns (status, reason)."""
    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA: GDN pool is CUDA-only"
    if not is_sm_at_least(8, 0):
        return "skipped", "SM < 8.0"

    # P53 (v7.9): Hybrid-active dispatch gate. fused_gdn_gating only fires
    # on hybrid linear-attention layers. On pure-attention models the
    # text-patch target file won't even be imported.
    try:
        from vllm._genesis.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P46 GDN gating buffer pool",
                "pure-attention model (no GDN layers)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P46] model_detect probe failed (proceeding): %s", e)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn_linear_attn.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", (
            "text-patch applied — fused_gdn_gating now uses "
            "GdnGatingBufferManager pool (eliminates ~24k allocs/sec on "
            "Qwen3.6-35B-A3B decode)"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already patched this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"


def is_applied() -> bool:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn_linear_attn.py"
    )
    if target is None or not target.exists():
        return False
    try:
        return GENESIS_P46_MARKER in target.read_text()
    except Exception:
        return False


def revert() -> bool:
    """Text-patch — no in-process revert. Restart container with
    `compose down && up -d`."""
    return False
