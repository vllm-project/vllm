# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 6 — TurboQuant-aware attention page size in interface.py.

Problem
-------
`vllm/platforms/interface.py:546` computes `attn_page_size_1_token` for hybrid
block alignment using `FullAttentionSpec`. For `turboquant_*` cache dtypes this
formula returns the standard layout (e.g. 1024 bytes/token for k8v4 head_dim=128)
instead of the TurboQuant packed layout (≈776 bytes/token using `slot_size_aligned`).
The over-estimate then propagates into the mamba alignment logic, causing
sub-optimal block_size choices.

Reference: vLLM PR [#39931](https://github.com/vllm-project/vllm/pull/39931).

Fix
---
When `cache_config.cache_dtype.startswith("turboquant_")`, use `TQFullAttentionSpec`
instead. Also account for `kv_cache_dtype_skip_layers` (boundary-protected layers
that fall back to the standard layout) by taking `max(tq_page, skip_page)` so the
mamba padding covers the largest actual page in the model.

Two sub-patches:
1. Add `TQFullAttentionSpec` to the kv_cache_interface import block.
2. Insert the TQ branch before the existing FullAttentionSpec else.

Compatibility windows:
- if `get_kv_quant_mode` is NOT in the file → upstream scaffolding for #39931
  hasn't landed → SKIP cleanly.
- if `TQFullAttentionSpec` is ALREADY imported → upstream merged the full PR
  → SKIP cleanly (our patch is then redundant).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
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

log = logging.getLogger("genesis.wiring.p6_tq_block_size")

GENESIS_P6_MARKER = "Genesis P6 TQ-aware block size alignment v7.0"

# Drift markers — full PR #39931 merge would import TQFullAttentionSpec here.
#
# Retirement pathways tracked as of 2026-04-24:
#   - PR #39931 (JartX hybrid TurboQuant, `ready` label): adds
#     TQFullAttentionSpec import — triggers the import-line markers below.
#   - PR #36701 (tdoublep, hybrid Mamba FA block-size restriction retire,
#     OPEN): removes the `block_sizes in {16, 32, 64}` check entirely
#     because #35219 (KVBlockZeroer) already resolved the NaN propagation
#     root cause. When that PR merges, our P6 becomes a no-op. Markers
#     below tracking a code comment that upstream uses in the PR body.
UPSTREAM_DRIFT_MARKERS = [
    "TQFullAttentionSpec,",
    "from vllm.v1.kv_cache_interface import (\n            FullAttentionSpec,\n            MambaSpec,\n            MLAAttentionSpec,\n            TQFullAttentionSpec,",
    # PR #36701 signatures — removes the FA block-size restriction.
    "# FA block-size restriction removed (NaN fix via KVBlockZeroer, #35219)",
    "# block_size restriction removed for hybrid Mamba models",
]


_IMPORT_OLD = (
    "        from vllm.v1.kv_cache_interface import (\n"
    "            FullAttentionSpec,\n"
    "            MambaSpec,\n"
    "            MLAAttentionSpec,\n"
    "            get_kv_quant_mode,\n"
    "        )"
)
_IMPORT_NEW = (
    "        from vllm.v1.kv_cache_interface import (\n"
    "            FullAttentionSpec,\n"
    "            MambaSpec,\n"
    "            MLAAttentionSpec,\n"
    "            TQFullAttentionSpec,  # [Genesis P6]\n"
    "            get_kv_quant_mode,\n"
    "        )"
)


_BRANCH_OLD = (
    "        else:\n"
    "            attn_page_size_1_token = FullAttentionSpec(\n"
    "                block_size=1,\n"
    "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
    "                head_size=model_config.get_head_size(),\n"
    "                dtype=kv_cache_dtype,\n"
    "                kv_quant_mode=kv_quant_mode,\n"
    "            ).page_size_bytes"
)


_BRANCH_NEW = (
    "        elif cache_config.cache_dtype.startswith(\"turboquant_\"):\n"
    "            # [Genesis P6] TQ has packed K|V layout; FullAttentionSpec\n"
    "            # over-sizes it. Use TQFullAttentionSpec; if there are skip\n"
    "            # layers (boundary-protected, standard layout), take max so\n"
    "            # mamba padding covers the largest actual page. (PR #39931)\n"
    "            from vllm.model_executor.layers.quantization.turboquant.config import (\n"
    "                TurboQuantConfig,\n"
    "            )\n"
    "            _tq_cfg = TurboQuantConfig.from_cache_dtype(\n"
    "                cache_config.cache_dtype,\n"
    "                model_config.get_head_size(),\n"
    "            )\n"
    "            tq_page = TQFullAttentionSpec(\n"
    "                block_size=1,\n"
    "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
    "                head_size=model_config.get_head_size(),\n"
    "                head_size_v=model_config.get_head_size(),\n"
    "                dtype=kv_cache_dtype,\n"
    "                kv_quant_mode=kv_quant_mode,\n"
    "                tq_slot_size=_tq_cfg.slot_size_aligned,\n"
    "            ).page_size_bytes\n"
    "            if cache_config.kv_cache_dtype_skip_layers:\n"
    "                skip_page = FullAttentionSpec(\n"
    "                    block_size=1,\n"
    "                    num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
    "                    head_size=model_config.get_head_size(),\n"
    "                    dtype=model_config.dtype,\n"
    "                ).page_size_bytes\n"
    "                attn_page_size_1_token = max(tq_page, skip_page)\n"
    "            else:\n"
    "                attn_page_size_1_token = tq_page\n"
    "        else:\n"
    "            attn_page_size_1_token = FullAttentionSpec(\n"
    "                block_size=1,\n"
    "                num_kv_heads=model_config.get_num_kv_heads(parallel_config),\n"
    "                head_size=model_config.get_head_size(),\n"
    "                dtype=kv_cache_dtype,\n"
    "                kv_quant_mode=kv_quant_mode,\n"
    "            ).page_size_bytes"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("platforms/interface.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P6 TQ-aware block size alignment",
        target_file=target,
        marker=GENESIS_P6_MARKER,
        sub_patches=[
            TextPatch(
                name="p6_import_tqspec",
                anchor=_IMPORT_OLD,
                replacement=_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="p6_tq_branch",
                anchor=_BRANCH_OLD,
                replacement=_BRANCH_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Apply P6 wiring. Never raises."""
    if not is_nvidia_cuda():
        return "skipped", "non-NVIDIA — TurboQuant only relevant for CUDA"

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "platforms/interface.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "TQ-aware page-size branch inserted"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied this image layer (idempotent)"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown skip"
    return "failed", failure.reason if failure else "unknown failure"
