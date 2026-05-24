# SPDX-License-Identifier: Apache-2.0
"""Centralized buffer-mode selection for Genesis prealloc patches.

Genesis prealloc patches (P22/P26/P28/P37/P38/P44/P46) historically attached
buffers per-attention-layer (`getattr(layer, "_tq_k_buf", None)`). On models
with many layers (Qwen3.6-MoE has 36 attention layers), this multiplies the
GPU-memory footprint by N_layers even though execution is sequential and only
ONE layer is active at any moment.

This module exposes a single `buffer_mode_for(patch_id)` API that reads the
operator's choice from environment variables and returns either:

  - ``"per_layer"``: legacy behaviour — each layer keeps its own buffer
    attribute. Maximum compatibility with hypothetical pipeline-parallel /
    multi-stream futures, but wastes ~30-40x memory on Qwen3.6.
  - ``"shared"``: singleton buffer pool — all layers share one buffer keyed
    by (namespace, shape, dtype, device) via `GenesisPreallocBuffer`.
    Safe on our execution model (TP only, single CUDA stream, sequential
    layer execution within forward pass) and saves multi-GB on long-context.

Env precedence (most specific wins):

  ``GENESIS_BUFFER_MODE_<PATCH_ID>``  per-patch override (e.g. ``GENESIS_BUFFER_MODE_P38=per_layer``)
  ``GENESIS_BUFFER_MODE``             global default for all patches
  built-in default                    ``"shared"`` (memory-efficient)

Usage in a patch:

    from vllm._genesis.buffer_mode import buffer_mode_for

    if buffer_mode_for("P38") == "shared":
        from vllm._genesis.prealloc import GenesisPreallocBuffer as GPB
        k_buf = GPB.get_or_create("p38_k_dequant", shape, dtype, device)
    else:
        k_buf = getattr(layer, "_tq_k_dequant_buf", None)
        if k_buf is None or k_buf.shape[2] < alloc_len:
            k_buf = torch.empty(shape, dtype=dtype, device=device)
            layer._tq_k_dequant_buf = k_buf

This keeps the legacy path alive for safe rollback while new path defaults to
the memory-efficient mode on fresh deployments.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.buffer_mode")

VALID_MODES = ("shared", "per_layer")
DEFAULT_MODE = "shared"


def _normalize(value: str) -> str:
    v = value.strip().lower()
    if v in ("shared", "singleton", "pool", "global"):
        return "shared"
    if v in ("per_layer", "per-layer", "perlayer", "legacy", "attached"):
        return "per_layer"
    return ""


def buffer_mode_for(patch_id: str) -> str:
    """Return ``"shared"`` or ``"per_layer"`` for the given patch.

    Resolution order (first match wins):
      1. ``GENESIS_BUFFER_MODE_<PATCH_ID>`` (case-normalised, e.g. P38 / p38 / P67)
      2. ``GENESIS_BUFFER_MODE``
      3. ``DEFAULT_MODE`` (= ``"shared"``)

    Invalid values fall back to the next level with a one-time warning.
    """
    pid = patch_id.upper().strip()

    # 1. Per-patch override
    specific_var = f"GENESIS_BUFFER_MODE_{pid}"
    raw = os.environ.get(specific_var)
    if raw:
        norm = _normalize(raw)
        if norm in VALID_MODES:
            return norm
        log.warning(
            "[Genesis buffer_mode] %s=%r is not in %s — ignoring, falling back",
            specific_var, raw, VALID_MODES,
        )

    # 2. Global default
    raw = os.environ.get("GENESIS_BUFFER_MODE")
    if raw:
        norm = _normalize(raw)
        if norm in VALID_MODES:
            return norm
        log.warning(
            "[Genesis buffer_mode] GENESIS_BUFFER_MODE=%r is not in %s — "
            "ignoring, using default %r",
            raw, VALID_MODES, DEFAULT_MODE,
        )

    # 3. Built-in default
    return DEFAULT_MODE


def log_mode_decision(patch_id: str, mode: str, reason_extra: str = "") -> None:
    """Emit a one-time INFO log for observability of which mode is active.

    Patches should call this once during apply() / first forward to make the
    choice visible in `docker logs` for diagnostics.
    """
    extra = f" — {reason_extra}" if reason_extra else ""
    log.info("[Genesis buffer_mode] %s using %r mode%s", patch_id, mode, extra)
