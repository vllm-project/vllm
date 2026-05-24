# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 5 v2 — KV cache page size unification (pad-smaller-to-max).

History
-------
v1 (2026-04-24): LCM-pad-up algorithm. Mirror of monolith v5.14.1 P5 + JartX
    PR #10. Worked but incurred 51.6% overhead on our Qwen3.6+TQ topology
    when max_page_size and smaller pages had small GCD (LCM ~ 0.76 × max
    → ceil-up to 2 × LCM = 1.51 × max).

v2 (2026-04-24, this file): pad-smaller-to-max. The MAX page size layer
    keeps its natural page; smaller layers get padded UP to max via their
    own `page_size_padded` field (which `AttentionSpec` and `MambaSpec`
    both expose). This is mathematically optimal under the constraint
    "all layers must report the same page_size_bytes".

Math comparison on our integration log numbers
----------------------------------------------
Observed:
  max_page_size  = 1,073,152 B   (10 attention layers @ TQ K8V4)
  smaller_page   =   813,248 B   (30 mamba/GDN layers)
  GCD = 64;  LCM = max × small / 64 = ~13.6 GB (huge)

v1 algorithm:
  smaller_lcm   = 813,248
  target        = ceil(1,073,152 / 813,248) × 813,248 = 1,626,496
  Per-block memory = 40 layers × 1,626,496 = 65,059,840 B
  Overhead vs natural = 51.6 % on every layer.

v2 algorithm:
  target = max_page_size = 1,073,152 (no LCM trick)
  10 attn layers: 1,073,152 each (no padding)
  30 mamba layers: padded to 1,073,152 via page_size_padded
  Per-block memory = 40 × 1,073,152 = 42,926,080 B
  Savings = 22,133,760 B per block (~34 %).
  Per-layer overhead now 0 % on max-layers, (max-small)/small = 32 %
  on mamba ONLY.

In real terms: with our `GPU KV cache size: 98,512 tokens` measured under
v1, v2 frees ~34 % of KV-cache VRAM. At identical VRAM budget that
translates to roughly 132,000-token KV cache (or higher concurrency).

Why not "scale block_size down on max-layers to match smaller pages"?
We can't reach equality without changing slot_size (TQ-specific), and
shrinking attn block_size below the upstream-mandated mamba alignment
breaks vLLM's `_align_hybrid_block_size` invariant. Pad-smaller-to-max
respects all upstream invariants.

Why not the architectural fix (separate block pools per layer type, swtb3
PR #37429)? That changes the KV cache manager + tensor allocator + block
pool — too invasive for a runtime text-patch. P5 v2 is a tactical
improvement; #37429 (when it lands) supersedes us with zero overhead.

Migration
---------
This module supports applying:
  - On baseline (unmodified upstream) → v2 directly.
  - On v1-patched file → replace v1 logic with v2 logic.
  - Idempotent if v2 marker is already present.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Credits: math/idea inspired by JartX/vllm#10 (LCM v1) + insight from
deep-study of vLLM v1/kv_cache_interface.py (page_size_padded already
supported on AttentionSpec + MambaSpec).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import (
    is_nvidia_cuda,
    resolve_vllm_file,
    vllm_install_root,
)
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p5_page_size")


# v7.0 marker (v1 algorithm = LCM-pad-max).
# HISTORY:
#   v7.0   (initial)   — LCM-pad-up-max; active. 51.6% overhead but stable.
#   v7.0.1 (attempted) — pad-smaller-to-max. Integration round 4 (2026-04-24)
#     discovered that vLLM's storage allocator uses `page_size_padded` for
#     num_blocks sizing, but the TurboQuant attention kernel reshapes the
#     cache tensor using NATURAL shape (num_blocks × block_size × num_kv_heads
#     × slot_size_aligned). When P5 v2 padded smaller layers, the allocated
#     tensor became larger than the expected natural shape →
#     `RuntimeError: shape '[524, 16, 1, 388]' is invalid for input of
#     size 4292608`. Extra 32% bytes = exactly our padding overhead.
#
#     v2 algorithm stays in this file as _V2_FN for future exploration
#     (would need a companion vLLM change to use `real_page_size_bytes`
#     for kernel reshape instead of `page_size_bytes`). Active = v1.
GENESIS_P5_MARKER = "Genesis P5 page_size unification v7.0"

UPSTREAM_DRIFT_MARKERS = [
    # Hypothetical upstream resolution (PR #37429 or successor) using
    # per-group block pools — would skip page unification entirely.
    "_has_mixed_mamba_attention",
    "mamba_num_blocks",
]


# ──────────────────────────────────────────────────────────────────────────
# v2 function body — pad-smaller-to-max strategy.
# ──────────────────────────────────────────────────────────────────────────
_V2_FN = (
    "def unify_kv_cache_spec_page_size(\n"
    "    kv_cache_spec: dict[str, KVCacheSpec],\n"
    ") -> dict[str, KVCacheSpec]:\n"
    "    \"\"\"\n"
    "    Unify the page size of the given KVCacheSpec.\n"
    "\n"
    "    [Genesis P5 v2] Pad-smaller-to-max strategy:\n"
    "      - Layers already at max page size: kept as-is (no overhead).\n"
    "      - Smaller layers whose page divides max evenly: scale block_size\n"
    "        up (matches upstream fast path; preserves token-per-block).\n"
    "      - Smaller layers with non-divisible page: padded UP to max via\n"
    "        the layer spec's `page_size_padded` field. Per-layer overhead\n"
    "        is (max - layer_page) bytes, localized to the smaller layer\n"
    "        type (typically Mamba in TurboQuant + hybrid models).\n"
    "\n"
    "    This is a tactical fix for hybrid TQ models. PR #37429 (per-type\n"
    "    block pools) is the architectural alternative; when it lands,\n"
    "    this patch becomes obsolete.\n"
    "\n"
    "    Args:\n"
    "        kv_cache_spec: The KVCacheSpec of each attention layer in the model\n"
    "\n"
    "    Returns:\n"
    "        The updated KVCacheSpec with the same page_size_bytes.\n"
    "    \"\"\"\n"
    "    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}\n"
    "    if len(page_sizes) <= 1:\n"
    "        # All layers have the same page size, no need to unify.\n"
    "        return kv_cache_spec\n"
    "\n"
    "    target_page_size = max(page_sizes)\n"
    "\n"
    "    new_kv_cache_spec = {}\n"
    "    for layer_name, layer_spec in kv_cache_spec.items():\n"
    "        layer_page = layer_spec.page_size_bytes\n"
    "        if layer_page == target_page_size:\n"
    "            # Layer already at target — no change. Zero overhead.\n"
    "            new_kv_cache_spec[layer_name] = layer_spec\n"
    "        elif target_page_size % layer_page == 0:\n"
    "            # Divisible: scale block_size up (upstream fast path).\n"
    "            ratio = target_page_size // layer_page\n"
    "            new_block_size = layer_spec.block_size * ratio\n"
    "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
    "            assert new_spec.page_size_bytes == target_page_size, (\n"
    "                f\"Page size mismatch after block_size adjust: \"\n"
    "                f\"{new_spec.page_size_bytes} != {target_page_size}\"\n"
    "            )\n"
    "            new_kv_cache_spec[layer_name] = new_spec\n"
    "        else:\n"
    "            # Non-divisible: pad smaller layer UP to max via\n"
    "            # page_size_padded. Overhead localized to this layer.\n"
    "            try:\n"
    "                new_spec = replace(layer_spec, page_size_padded=target_page_size)\n"
    "            except TypeError:\n"
    "                raise NotImplementedError(\n"
    "                    f\"[Genesis P5 v2] Cannot pad page size for \"\n"
    "                    f\"{type(layer_spec).__name__}: page_size_padded \"\n"
    "                    f\"not supported. Layer page={layer_page}, \"\n"
    "                    f\"target={target_page_size}\"\n"
    "                )\n"
    "            assert new_spec.page_size_bytes == target_page_size, (\n"
    "                f\"Page size mismatch after padding: \"\n"
    "                f\"{new_spec.page_size_bytes} != {target_page_size}\"\n"
    "            )\n"
    "            logger.info(\n"
    "                \"[Genesis P5 v2] padded layer %s: %d -> %d (+%.2f%%)\",\n"
    "                layer_name, layer_page, target_page_size,\n"
    "                (target_page_size - layer_page) / layer_page * 100,\n"
    "            )\n"
    "            new_kv_cache_spec[layer_name] = new_spec\n"
    "    return new_kv_cache_spec"
)


# ──────────────────────────────────────────────────────────────────────────
# Old anchors. Two distinct anchors so we can apply v2 either to a fresh
# baseline file OR migrate from v1 in place.
# ──────────────────────────────────────────────────────────────────────────

# Anchor A: pristine upstream baseline (engine/arg_utils.py:1648-1668).
_BASELINE_FN = (
    "def unify_kv_cache_spec_page_size(\n"
    "    kv_cache_spec: dict[str, KVCacheSpec],\n"
    ") -> dict[str, KVCacheSpec]:\n"
    "    \"\"\"\n"
    "    Unify the page size of the given KVCacheSpec. If the page size of all layers\n"
    "    are the same, return the original KVCacheSpec. If not same, unify the page\n"
    "    size by increasing the block size of layers with smaller page size. Raise\n"
    "    NotImplementedError if failed to unify the page size.\n"
    "\n"
    "    Args:\n"
    "        kv_cache_spec: The KVCacheSpec of each attention layer in the model\n"
    "\n"
    "    Returns:\n"
    "        The updated KVCacheSpec with the same page_size_bytes.\n"
    "    \"\"\"\n"
    "    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}\n"
    "    if len(page_sizes) <= 1:\n"
    "        # All layers have the same page size, no need to unify.\n"
    "        return kv_cache_spec\n"
    "\n"
    "    max_page_size = max(page_sizes)\n"
    "    new_kv_cache_spec = {}\n"
    "    for layer_name, layer_spec in kv_cache_spec.items():\n"
    "        if layer_spec.page_size_bytes == max_page_size:\n"
    "            new_kv_cache_spec[layer_name] = layer_spec\n"
    "        else:\n"
    "            layer_page_size = layer_spec.page_size_bytes\n"
    "            if max_page_size % layer_page_size != 0:\n"
    "                raise NotImplementedError(\n"
    "                    \"The page size of the layer is not divisible by the \"\n"
    "                    \"maximum page size. Cannot unify by adjusting block_size.\"\n"
    "                )\n"
    "            ratio = max_page_size // layer_page_size\n"
    "            new_block_size = layer_spec.block_size * ratio\n"
    "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
    "            assert new_spec.page_size_bytes == max_page_size\n"
    "            new_kv_cache_spec[layer_name] = new_spec\n"
    "    return new_kv_cache_spec"
)


# Anchor B: our prior P5 v1 (LCM-pad-max). For in-place upgrade on
# already-patched containers. Match the EXACT body of v1's _NEW_FN.
_V1_FN = (
    "def unify_kv_cache_spec_page_size(\n"
    "    kv_cache_spec: dict[str, KVCacheSpec],\n"
    ") -> dict[str, KVCacheSpec]:\n"
    "    \"\"\"\n"
    "    Unify the page size of the given KVCacheSpec. If the page size of all layers\n"
    "    are the same, return the original KVCacheSpec. If not same, unify the page\n"
    "    size by increasing the block size of layers with smaller page size.\n"
    "\n"
    "    [Genesis P5] For hybrid models (e.g. TurboQuant + DeltaNet/Mamba), page\n"
    "    sizes may not be naturally divisible. In that case, the largest page size\n"
    "    is padded UP to the nearest multiple of all smaller page sizes using the\n"
    "    page_size_padded field on the layer spec. Memory overhead is typically\n"
    "    <0.1%.\n"
    "\n"
    "    Args:\n"
    "        kv_cache_spec: The KVCacheSpec of each attention layer in the model\n"
    "\n"
    "    Returns:\n"
    "        The updated KVCacheSpec with the same page_size_bytes.\n"
    "    \"\"\"\n"
    "    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}\n"
    "    if len(page_sizes) <= 1:\n"
    "        # All layers have the same page size, no need to unify.\n"
    "        return kv_cache_spec\n"
    "\n"
    "    max_page_size = max(page_sizes)\n"
    "\n"
    "    # [Genesis P5] Fast path: all smaller page sizes already divide max evenly.\n"
    "    smaller_sizes = sorted(ps for ps in page_sizes if ps < max_page_size)\n"
    "    all_divide = all(max_page_size % ps == 0 for ps in smaller_sizes)\n"
    "\n"
    "    if all_divide:\n"
    "        target_page_size = max_page_size\n"
    "    else:\n"
    "        # Hybrid model: pad max UP to nearest multiple of LCM(smaller_sizes).\n"
    "        smaller_lcm = math.lcm(*smaller_sizes)\n"
    "        target_page_size = ((max_page_size + smaller_lcm - 1) // smaller_lcm) * smaller_lcm\n"
    "        logger.info(\n"
    "            \"[Genesis P5] page size unification: max %d -> %d (LCM=%d, \"\n"
    "            \"overhead %.3f%%)\",\n"
    "            max_page_size, target_page_size, smaller_lcm,\n"
    "            (target_page_size - max_page_size) / max_page_size * 100,\n"
    "        )\n"
    "\n"
    "    new_kv_cache_spec = {}\n"
    "    for layer_name, layer_spec in kv_cache_spec.items():\n"
    "        layer_page = layer_spec.page_size_bytes\n"
    "        if layer_page == target_page_size:\n"
    "            new_kv_cache_spec[layer_name] = layer_spec\n"
    "        elif layer_page < target_page_size and target_page_size % layer_page == 0:\n"
    "            # Scale up block_size so page matches target\n"
    "            ratio = target_page_size // layer_page\n"
    "            new_block_size = layer_spec.block_size * ratio\n"
    "            new_spec = replace(layer_spec, block_size=new_block_size)\n"
    "            assert new_spec.page_size_bytes == target_page_size, (\n"
    "                f\"Page size mismatch after block_size adjust: \"\n"
    "                f\"{new_spec.page_size_bytes} != {target_page_size}\"\n"
    "            )\n"
    "            new_kv_cache_spec[layer_name] = new_spec\n"
    "        else:\n"
    "            # Layer had original max page size but target was padded up.\n"
    "            # Pad this layer to target via page_size_padded.\n"
    "            try:\n"
    "                new_spec = replace(layer_spec, page_size_padded=target_page_size)\n"
    "            except TypeError:\n"
    "                raise NotImplementedError(\n"
    "                    f\"[Genesis P5] Cannot pad page size for \"\n"
    "                    f\"{type(layer_spec).__name__}: page_size_padded not \"\n"
    "                    f\"supported. Layer page={layer_page}, target={target_page_size}\"\n"
    "                )\n"
    "            assert new_spec.page_size_bytes == target_page_size, (\n"
    "                f\"Page size mismatch after padding: \"\n"
    "                f\"{new_spec.page_size_bytes} != {target_page_size}\"\n"
    "            )\n"
    "            new_kv_cache_spec[layer_name] = new_spec\n"
    "    return new_kv_cache_spec"
)


# v1 active algorithm needs math.lcm — add `import math` to module scope.
_IMPORT_OLD = (
    "import copy\n"
    "import hashlib\n"
    "import os\n"
)
_IMPORT_NEW = (
    "import copy\n"
    "import hashlib\n"
    "import math\n"
    "import os\n"
)


def _make_patcher() -> TextPatcher | None:
    """Build the text-patcher. Active body is v1 (LCM-pad-up-max) by
    default; v2 (pad-smaller-to-max) activates when `GENESIS_ENABLE_P5B=1`.

    v2 requires:
      - `AttentionSpec.page_size_padded` field (dev134+ ✅)
      - `TQFullAttentionSpec.real_page_size_bytes` property (dev134+ ✅)

    Both are present on our dev134 baseline. v2 saves ~34% per-block
    VRAM on Qwen3.6-35B-A3B hybrid vs v1 LCM-pad-up (math in
    `kernels/page_size_padded.py` module docstring).

    The migration path is staged:
      - First apply on pristine baseline: _BASELINE_FN → active body
      - Re-apply on v1-patched container: _V1_FN → v2 body (when P5B=1)
      - Re-apply on v2-patched container: idempotent (marker match)
    """
    from vllm._genesis.kernels.page_size_padded import is_p5b_enabled

    target = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if target is None:
        return None

    active_body = _V2_FN if is_p5b_enabled() else _V1_FN
    body_name = "v2_pad_smaller" if is_p5b_enabled() else "v1_lcm_pad_max"

    # Two migration anchors so we can switch algorithm in place:
    #   baseline → active      (fresh install)
    #   v1 → v2                (when P5B env is flipped on an already-
    #                           patched container; becomes no-op if
    #                           active_body==v1)
    sub_patches = [
        TextPatch(
            name="p5_import_math",
            anchor=_IMPORT_OLD,
            replacement=_IMPORT_NEW,
            required=False,
        ),
        TextPatch(
            name=f"p5_{body_name}_from_baseline",
            anchor=_BASELINE_FN,
            replacement=active_body,
            required=False,
        ),
    ]

    if is_p5b_enabled():
        # Also allow migration from v1 → v2 on already-patched containers.
        # The v1 body was stamped with its own comment block so the
        # anchor is unique and won't trigger on v2 itself (idempotent).
        sub_patches.append(
            TextPatch(
                name="p5_v2_migrate_from_v1",
                anchor=_V1_FN,
                replacement=_V2_FN,
                required=False,
            )
        )

    return TextPatcher(
        patch_name=f"P5 KV cache page size unification ({body_name})",
        target_file=target,
        marker=GENESIS_P5_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P5 v2. Never raises. Returns (status, reason)."""
    import os as _g_p5_os
    # ════════════════════════════════════════════════════════════════════════
    # [Genesis P5 conditional defer to upstream]
    # When operator runs with --mamba-cache-mode={align|all}, vLLM's own
    # `_align_hybrid_block_size` (PR #25752 / #30877) enforces block_size to
    # be a multiple of mamba_chunk_size (256). Our P5 LCM-pad may produce
    # non-256-aligned block_size that breaks
    # MambaManager.find_longest_cache_hit's alignment check (returns 0 hits).
    #
    # Set GENESIS_DISABLE_P5=1 alongside --mamba-cache-mode={align|all} to
    # defer page-size unification to upstream and unlock cache hits.
    # WARNING: without our P5 v2, hybrid TQ models that need page padding
    # may crash at KV cache init with NotImplementedError. Only set
    # GENESIS_DISABLE_P5=1 when also setting --mamba-cache-mode != none.
    # ════════════════════════════════════════════════════════════════════════
    if _g_p5_os.environ.get("GENESIS_DISABLE_P5", "").strip().lower() in (
            "1", "true", "yes", "on"):
        return "skipped", (
            "GENESIS_DISABLE_P5=1 set — deferring KV page-size unification "
            "to upstream's `_align_hybrid_block_size` (PR #25752/#30877). "
            "Required for compatibility with --mamba-cache-mode={align|all}. "
            "Without P5, hybrid TQ models may need explicit alignment via "
            "--block-size or upstream alignment to function."
        )

    # ════════════════════════════════════════════════════════════════════════
    # [Genesis P5 auto-retire on PR #39931 merge — added 2026-04-30]
    #
    # JartX's PR #39931 (TurboQuant hybrid + uniform quant) is the upstream
    # superset for P5's dominant use case: hybrid models with TQ k8v4 KV.
    # Once #39931 merges, vLLM's `_align_hybrid_block_size` builds the
    # TQ branch with `lcm(tq_page, skip_page)` natively, so P5's
    # planner-level rewrite of `unify_kv_cache_spec_page_size` becomes
    # redundant for the hybrid+TQ path.
    #
    # Detect by probing for the canonical PR #39931 symbols. If present,
    # auto-skip rather than running redundantly. Residual edge cases
    # (non-hybrid TQ with mixed-preset layers; hybrid models without
    # `layer_types`/`layers_block_type` hints) still go through the
    # normal P5 path because the probe will return False there.
    # ════════════════════════════════════════════════════════════════════════
    try:
        import importlib
        # Probe 1: TQFullAttentionSpec (added by #39931 in
        # vllm/model_executor/layers/quantization/turboquant/config.py)
        try:
            tq_cfg = importlib.import_module(
                "vllm.model_executor.layers.quantization.turboquant.config"
            )
            has_tq_full_spec = hasattr(tq_cfg, "TQFullAttentionSpec")
            has_filter_helper = hasattr(
                tq_cfg, "_get_full_attention_layer_indices"
            )
        except Exception:
            has_tq_full_spec = False
            has_filter_helper = False

        if has_tq_full_spec and has_filter_helper:
            return "skipped", (
                "PR #39931 detected (TQFullAttentionSpec + "
                "_get_full_attention_layer_indices present in upstream "
                "vllm.model_executor.layers.quantization.turboquant.config). "
                "Genesis P5 deferring to upstream's TQ-aware "
                "_align_hybrid_block_size — no planner-level rewrite "
                "needed for hybrid+TQ. Set GENESIS_DISABLE_P5_AUTORETIRE=1 "
                "to override and force P5 anyway (e.g. for non-hybrid "
                "edge cases where #39931 is incomplete)."
            )
    except Exception as e:
        # Probe failure is non-fatal — fall through to normal P5 apply.
        # Operator can still set GENESIS_DISABLE_P5=1 if needed.
        log.debug("P5 auto-retire probe raised %s: %s — proceeding with "
                  "normal apply.", type(e).__name__, e)

    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA: TQ+hybrid only matters with CUDA TurboQuant"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "v1/core/kv_cache_utils.py not found in vllm install"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "text-patch v2 succeeded (pad-smaller-to-max)"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "v2 already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
