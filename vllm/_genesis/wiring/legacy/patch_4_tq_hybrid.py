# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 4 — TurboQuant KV cache for hybrid (Mamba+attention) models.

Problem (baseline image @ vllm-main 2026-04-23, commit fe9c3d6c5)
----------------------------------------------------------------
`vllm/engine/arg_utils.py:1648-1668` contains this code path:

    if resolved_cache_dtype.startswith("turboquant_"):
        if model_config.is_hybrid:
            raise NotImplementedError(
                "TurboQuant KV cache is not supported for hybrid "
                "(attention + Mamba) models. Boundary layer protection "
                "requires uniform attention layers."
            )
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        num_layers = model_config.hf_text_config.num_hidden_layers
        boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)
        existing = set(cache_config.kv_cache_dtype_skip_layers)
        merged = sorted(existing | set(boundary), key=lambda x: int(x))
        cache_config.kv_cache_dtype_skip_layers = merged

This rejects Qwen3.6-35B-A3B + `turboquant_k8v4` because the model is
hybrid (mamba + attention blocks). But TurboQuant CAN work on hybrid
models — we just need to identify which layers are attention vs mamba
and apply TQ only to the full-attention ones.

Fix (Genesis Patch 4 v2 — logically equivalent to monolith v5.14.1 P4)
-----------------------------------------------------------------------
Replace the monolithic block with branching logic:

  - Non-hybrid models: keep upstream behavior (standard boundary skip).
  - Hybrid models: identify full-attention layer indices via the model's
    own config conventions (``layer_types``, ``layers_block_type``,
    ``attn_type_list``). Apply TQ to those layers; skip none by default
    (the model's own mamba layers are naturally excluded because they
    don't use KV cache).

Implementation strategy
-----------------------
This MUST be a text-patch because:
  - The raise is inside a method body (EngineArgs.create_engine_config).
  - Bypassing requires injecting new control flow + a helper function.
  - Monkey-patching the whole method would duplicate ~100 lines of the
    surrounding method that's unrelated to TQ — fragile across vLLM
    versions.

The patcher is designed to FAIL SOFT — on anchor drift, it skips with a
warning rather than crashing, letting prod stay on v5.14.1 monolith path.

Platform compatibility
----------------------
  NVIDIA CUDA (SM 8.0+): primary target (TurboQuant is CUDA-only upstream)
  Other platforms: no-op (resolved_cache_dtype would not start with
                   "turboquant_" on non-NVIDIA anyway)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
# Module-level imports so tests can monkeypatch them. Top-level also means
# we fail fast on a broken install rather than halfway through apply().
from vllm._genesis.guards import (
    is_nvidia_cuda,
    resolve_vllm_file,
    vllm_install_root,
)
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p4_tq_hybrid")


# Unique marker — appears in the patched file header so re-runs are idempotent.
GENESIS_P4_MARKER = "Genesis P4 TQ hybrid v7.0"


# Upstream forward-compat markers. If ANY of these appear in arg_utils.py, the
# upstream has merged a fix that makes our patch obsolete — we skip gracefully.
#
# Retirement pathways tracked as of 2026-04-24:
#   - PR #39931 (@JartX, OPEN with `ready` label): "hybrid TurboQuant" —
#     full retire of P4. Signature tokens listed below.
UPSTREAM_DRIFT_MARKERS = [
    # Hypothetical: if upstream ever refactors get_boundary_skip_layers to take
    # model_config, our old anchor will miss and this marker will flag it.
    "TurboQuantConfig.get_boundary_skip_layers(model_config",
    # Hypothetical: if upstream removes the NotImplementedError altogether.
    "# TurboQuant: hybrid-aware boundary protection",
    # PR #39931 landing signatures: JartX adds explicit full-attention-layer
    # discovery via `_is_full_attention_layer` helper + removes the raise on
    # hybrid models. Catching any of these flavours self-retires us.
    "_is_full_attention_layer",
    "def is_full_attention_layer_index",
    "full_attention_layer_types",
    "# TurboQuant hybrid support: skip boundary check on mamba layers",
    # PR #41123 (cderinbogaz, OPEN 2026-04-29) signature tokens — when this PR
    # lands upstream the `_get_turboquant_boundary_skip_layers` helper appears
    # at module level in arg_utils.py and the message text changes from "is
    # not supported" to "skip layers are not supported". Either marker
    # confirms #41123 has merged and P4 should self-retire.
    "_get_turboquant_boundary_skip_layers",
    "TurboQuant KV cache skip layers are not supported for ",
]


# ──────────────────────────────────────────────────────────────────────────
# Helper function injected at module level in arg_utils.py.
# Uses common vLLM conventions to identify full-attention layer indices.
# ──────────────────────────────────────────────────────────────────────────
_HELPER_FN_SOURCE = '''

def _genesis_p4_full_attention_indices(model_config) -> list[int]:
    """Return global layer indices of full-attention blocks in a hybrid model.

    Supports the three conventions currently used across vLLM:
      - ``layer_types`` list (Qwen3.5-Next, Qwen3.6-MoE)
      - ``layers_block_type`` list (Jamba, Zamba2)
      - ``attn_type_list`` list (Minimax M1/Text)

    Returns empty list if none recognized — caller should treat as error.

    [Genesis P4] Helper injected by genesis_vllm_plugin.
    """
    text_cfg = model_config.hf_text_config
    hf_cfg = model_config.hf_config

    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        return [
            i for i, t in enumerate(layer_types)
            if t in ("full_attention", "attention")
        ]

    layers_block_type = getattr(text_cfg, "layers_block_type", None)
    if layers_block_type is not None:
        return [
            i for i, t in enumerate(layers_block_type)
            if t in ("attention", "hybrid")
        ]

    attn_type_list = getattr(hf_cfg, "attn_type_list", None)
    if attn_type_list is not None:
        return [i for i, t in enumerate(attn_type_list) if t == 1]

    return []
'''


def _make_patcher() -> TextPatcher | None:
    """Build the TextPatcher for P4. Returns None if target file missing."""
    target = resolve_vllm_file("engine/arg_utils.py")
    if target is None:
        return None

    # Sub-patch A: insert helper function at module scope. We place it right
    # before the `@dataclass` that defines EngineArgs — a stable anchor.
    anchor_before_engineargs = "\n\n@dataclass\nclass EngineArgs:"
    helper_inject = _HELPER_FN_SOURCE + "\n\n@dataclass\nclass EngineArgs:"

    # Sub-patch B: replace the TQ+hybrid block with branched logic.
    # The new block:
    #   - For hybrid: call _genesis_p4_full_attention_indices, log, boundary=[]
    #   - For non-hybrid: keep upstream standard-boundary behavior
    old_tq_block = (
        "        if resolved_cache_dtype.startswith(\"turboquant_\"):\n"
        "            if model_config.is_hybrid:\n"
        "                raise NotImplementedError(\n"
        "                    \"TurboQuant KV cache is not supported for hybrid \"\n"
        "                    \"(attention + Mamba) models. Boundary layer protection \"\n"
        "                    \"requires uniform attention layers.\"\n"
        "                )\n"
        "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
        "                TurboQuantConfig,\n"
        "            )\n"
        "\n"
        "            num_layers = model_config.hf_text_config.num_hidden_layers\n"
        "            boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)\n"
        "            existing = set(cache_config.kv_cache_dtype_skip_layers)\n"
        "            merged = sorted(existing | set(boundary), key=lambda x: int(x))\n"
        "            cache_config.kv_cache_dtype_skip_layers = merged\n"
        "            logger.info(\n"
        "                \"TQ: skipping layers %s for boundary protection (num_layers=%d)\",\n"
        "                merged,\n"
        "                num_layers,\n"
        "            )\n"
    )

    new_tq_block = (
        "        if resolved_cache_dtype.startswith(\"turboquant_\"):\n"
        "            # [Genesis P4] Hybrid-aware boundary: identify full-attention\n"
        "            # layers instead of rejecting. Mamba layers naturally skip\n"
        "            # KV cache, so we only need to protect attention-layer\n"
        "            # boundaries in dense models.\n"
        "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
        "                TurboQuantConfig,\n"
        "            )\n"
        "            num_layers = model_config.hf_text_config.num_hidden_layers\n"
        "            if model_config.is_hybrid:\n"
        "                attn_indices = _genesis_p4_full_attention_indices(model_config)\n"
        "                if not attn_indices:\n"
        "                    raise NotImplementedError(\n"
        "                        \"TurboQuant KV cache on hybrid model requires \"\n"
        "                        \"identifiable full-attention layers (layer_types / \"\n"
        "                        \"layers_block_type / attn_type_list); none found.\"\n"
        "                    )\n"
        "                logger.info(\n"
        "                    \"[Genesis P4] TQ hybrid: full-attention layers %s \"\n"
        "                    \"(total layers=%d)\",\n"
        "                    attn_indices, num_layers,\n"
        "                )\n"
        "                boundary: list[int] = []  # hybrid: rely on model topology\n"
        "            else:\n"
        "                boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)\n"
        "            existing = set(cache_config.kv_cache_dtype_skip_layers)\n"
        "            merged = sorted(existing | set(boundary), key=lambda x: int(x))\n"
        "            cache_config.kv_cache_dtype_skip_layers = merged\n"
        "            logger.info(\n"
        "                \"TQ: skipping layers %s for boundary protection \"\n"
        "                \"(num_layers=%d, hybrid=%s)\",\n"
        "                merged, num_layers, model_config.is_hybrid,\n"
        "            )\n"
    )

    return TextPatcher(
        patch_name="P4 TurboQuant hybrid model support",
        target_file=target,
        marker=GENESIS_P4_MARKER,
        sub_patches=[
            TextPatch(
                name="p4_helper_fn",
                anchor=anchor_before_engineargs,
                replacement=helper_inject,
                required=True,
            ),
            TextPatch(
                name="p4_tq_block",
                anchor=old_tq_block,
                replacement=new_tq_block,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P4 wiring. Returns (status, reason).

    status ∈ {"applied", "idempotent", "skipped", "failed"}.
    Never raises — the plugin layer above treats a non-success as graceful
    skip, not engine crash.
    """
    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA: TurboQuant only on CUDA"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "engine/arg_utils.py not found in vllm install"

    result, failure = patcher.apply()

    if result == TextPatchResult.APPLIED:
        return "applied", "text-patch succeeded"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip reason"
    return "failed", failure.reason if failure else "unknown failure"
