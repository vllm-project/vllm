# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN55 — vllm#41602 backport.

Fixes `init_fp8_kv_scales()` AttributeError on Mamba/DeltaNet hybrid
models после `/sleep` → `/wake_up`. `MambaSpec` хранит per-layer state
как `list[Tensor]`, а не tensor; original loop наивно зовёт `.zero_()`
прямо на list → AttributeError ломает entire wake-up.

Affects: 27B Lorbus Qwen3.6 hybrid (GDN = MambaSpec). Crash trigger —
любой `/sleep` + `/wake_up` через mgmt API. У нас в active scripts
sleep не вызывается, но crash possible через external trigger.

Default OFF — defensive backport; включить при необходимости sleep/wake.

Backport: Joachim Studnia / Mistral, vllm#41602 (OPEN as of 2026-05-04).
Author: Sandermage backport.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn55_wake_up_hybrid_kv")

GENESIS_PN55_MARKER = "Genesis PN55 wake_up hybrid KV (vllm#41602)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV", ""
    ).strip().lower() in ("1", "true", "yes", "on")


ANCHOR_OLD = (
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        for cache_tensor in kv_caches:\n"
    "            if cache_tensor is not None:\n"
    "                cache_tensor.zero_()"
)

ANCHOR_NEW = (
    "        kv_caches = getattr(self, \"kv_caches\", [])\n"
    "        # [Genesis PN55 vllm#41602] Hybrid models (Mamba, DeltaNet)\n"
    "        # store per-layer state as `list[Tensor]` not single tensor.\n"
    "        # Original loop AttributeError'd on .zero_() over list.\n"
    "        # [Audit A-12 fix 2026-05-05] Guard each list element: hybrid cache\n"
    "        # may contain None or non-tensor sentinels — skip without raising.\n"
    "        for cache_entry in kv_caches:\n"
    "            if cache_entry is None:\n"
    "                continue\n"
    "            if isinstance(cache_entry, list):\n"
    "                for _pn55_t in cache_entry:\n"
    "                    if _pn55_t is None:\n"
    "                        continue\n"
    "                    if not hasattr(_pn55_t, \"zero_\"):\n"
    "                        continue\n"
    "                    _pn55_t.zero_()\n"
    "            elif hasattr(cache_entry, \"zero_\"):\n"
    "                cache_entry.zero_()"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN55 wake_up hybrid KV (vllm#41602)",
        target_file=str(target),
        marker=GENESIS_PN55_MARKER,
        sub_patches=[TextPatch(
            name="pn55_isinstance_list",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "isinstance(cache_entry, list)",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN55")
    log_decision("PN55", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gpu_model_runner.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "PN55 applied: wake_up no longer crashes on hybrid"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged"
    return "failed", failure.reason if failure else "unknown failure"
