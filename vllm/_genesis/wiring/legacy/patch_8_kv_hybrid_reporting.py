# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 8 — KV cache reporting for hybrid (attention + Mamba) models.

Critical finding (2026-04-24 after round 6): without this patch, the
Genesis v7.0 stack reports ~3.76× LESS KV-cache capacity than the
production monolith stack — on the SAME hardware, SAME VRAM budget, and
otherwise identical settings.

Root cause
----------
`vllm/v1/core/kv_cache_utils.py:_report_kv_cache_config()` and
`vllm/v1/core/sched/scheduler.py:Scheduler.__init__()` both divide
`kv_cache_config.num_blocks` by `len(kv_cache_config.kv_cache_groups)`
— counting EVERY group (attention + mamba) in the per-token divisor.

Mamba groups hold O(1) state per request, NOT O(n) per token. Including
them in the divisor under-reports KV capacity by the mamba-to-attention
group ratio. For Qwen3.6-35B-A3B (1 attn group + 1 mamba group), that's
2× under-reporting at log time AND a 2× wrong `max_num_kv_tokens` fed
into the scheduler's routed-experts buffer sizing.

Measured on 2× A5000 + Qwen3.6 + TQ K8V4 + max_model_len=262144:
  prod v5.14.1 (with monolith P8): GPU KV cache = **1,104,432 tokens**
  Genesis v7.0 round 6 (no P8):    GPU KV cache =   **293,408 tokens**
  Ratio: 3.76×

Upstream PR tracker: [#40384](https://github.com/vllm-project/vllm/pull/40384)
by jhsmith409, with @Sandermage co-author credit. Still OPEN — Genesis
P8 mirrors its exact architecture so adoption of the eventual merge is
a no-op.

Fix architecture (matches upstream PR shape)
---------------------------------------------
1. Extract a shared helper `token_capacity_kv_cache_groups()` into
   `kv_cache_utils.py` that returns the subset of kv_cache_groups that
   should be counted in per-token capacity:
     - always include AttentionSpec groups
     - include MambaSpec groups only when `mamba_cache_mode == 'all'`
     - fall back to the full set if the filter produces an empty list
2. Use the helper in `_report_kv_cache_config` (log + concurrency).
3. Import the helper in `scheduler.py` and use it in
   `Scheduler.__init__` for `self.max_num_kv_tokens`.

Four sub-patches, two files.

Scope note (from jhsmith409 on PR #40384, 2026-04-21)
-----------------------------------------------------
This patch applies the formula to BOTH kv_cache_utils.py (log) AND
scheduler.py (max_num_kv_tokens). The scheduler change is correct for
single-attention-group hybrids (our Qwen3.6-A3B: 1 attn + 1 mamba group).
On multi-attention-group hybrids like Nemotron-H, the scheduler's
max_num_kv_tokens feeds a routed-experts side buffer that needs the FULL
address space of the chosen attention group (num_blocks * attn_group
block_size) — see PR [#37118](https://github.com/vllm-project/vllm/pull/37118).
Our filter-formula would under-size it there. Since Qwen3.6-A3B has a
single attention group, filter-result == single attn group == full
address space, and both formulas produce identical values, so this is
production-correct for our stack. If ever porting to Nemotron-H-style,
revisit.

Platform compatibility
----------------------
  All platforms ✅ — pure Python spec-routing logic, no kernel code.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch, TextPatcher, TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p8_kv_hybrid_reporting")

GENESIS_P8_MARKER_KV = "Genesis P8 KV hybrid reporting kv_cache_utils v7.62.13_v2anchor"
GENESIS_P8_MARKER_SCHED = "Genesis P8 KV hybrid reporting scheduler v7.62.13_v2anchor"

# Retirement pathways tracked as of 2026-04-24:
#   - PR #40384 (@jhsmith409, OPEN): exports `token_capacity_kv_cache_groups`
#     as a module-level helper — direct retirement of our text-patch.
#   - PR #37429 (@swtb3, OPEN): per-group block pools for hybrid Mamba/attn;
#     7× memory waste fix. Introduces new code paths (`_has_mixed_mamba_attention`,
#     `mamba_num_blocks`) that supersede both P5 and P8 simultaneously.
UPSTREAM_DRIFT_MARKERS_KV = [
    # If upstream PR #40384 merges, helper function will exist at module scope.
    "def token_capacity_kv_cache_groups(",
    # PR #37429 signatures — per-group block pools.
    "_has_mixed_mamba_attention",
    "def compute_mamba_num_blocks",
    "mamba_num_blocks:",
]
UPSTREAM_DRIFT_MARKERS_SCHED = [
    "from vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups",
    # PR #37429 scheduler-side signatures.
    "mamba_block_pool",
    "self.mamba_num_blocks",
]


# ── kv_cache_utils.py sub-patches ──────────────────────────────────────

# [Genesis P8 v7.62.13] B-fix: anchor pair to handle BOTH pre- and post-MambaSpec
# vllm versions. The pre-MambaSpec form (older nightly) needs both AttentionSpec
# AND MambaSpec injected. The post-MambaSpec form (newer nightly, PR #40384 part
# merged) only needs AttentionSpec injected. Apply() tries _OLD first, falls back
# to _OLD_V2.
_KV_IMPORT_OLD = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheConfig,\n"
    "    KVCacheGroupSpec,\n"
    "    KVCacheSpec,\n"
    "    KVCacheTensor,\n"
    "    SlidingWindowSpec,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")"
)
_KV_IMPORT_NEW = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,  # [Genesis P8]\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheConfig,\n"
    "    KVCacheGroupSpec,\n"
    "    KVCacheSpec,\n"
    "    KVCacheTensor,\n"
    "    MambaSpec,  # [Genesis P8]\n"
    "    SlidingWindowSpec,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")"
)
# Newer vllm form (post-MambaSpec auto-imported, post-MLA additions).
# We anchor on the closing `)` preceded by `UniformTypeKVCacheSpecs,\n` and
# inject `AttentionSpec,` before `ChunkedLocalAttentionSpec`. The actual
# matched anchor is the full block; any new MLA specs in between are kept.
_KV_IMPORT_OLD_V2 = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheConfig,\n"
    "    KVCacheGroupSpec,\n"
    "    KVCacheSpec,\n"
    "    KVCacheTensor,\n"
    "    MambaSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    SlidingWindowSpec,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")"
)
_KV_IMPORT_NEW_V2 = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,  # [Genesis P8 v2]\n"
    "    ChunkedLocalAttentionSpec,\n"
    "    FullAttentionSpec,\n"
    "    KVCacheConfig,\n"
    "    KVCacheGroupSpec,\n"
    "    KVCacheSpec,\n"
    "    KVCacheTensor,\n"
    "    MambaSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    SlidingWindowSpec,\n"
    "    UniformTypeKVCacheSpecs,\n"
    ")"
)


_KV_REPORT_FN_OLD = (
    "def _report_kv_cache_config(\n"
    "    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig\n"
    ") -> None:"
)

_KV_HELPER_FN = (
    "def token_capacity_kv_cache_groups(\n"
    "    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig\n"
    ") -> list[KVCacheGroupSpec]:\n"
    "    \"\"\"[Genesis P8] KV cache groups that contribute to per-token capacity.\n"
    "\n"
    "    Mirrors upstream PR #40384 (jhsmith409 + Sandermage co-author).\n"
    "    Attention groups always scale with sequence length. Mamba groups only\n"
    "    scale when `mamba_cache_mode == 'all'`; in 'none' and 'align' they\n"
    "    hold O(1) state per request and pre-reserve a fixed number of blocks,\n"
    "    so counting them in the per-token divisor under-reports capacity on\n"
    "    hybrid models.\n"
    "\n"
    "    Falls back to all groups if the filter produces an empty list.\n"
    "    \"\"\"\n"
    "    mamba_scales = (\n"
    "        getattr(vllm_config.cache_config, \"mamba_cache_mode\", \"none\") == \"all\"\n"
    "    )\n"
    "    groups = [\n"
    "        g\n"
    "        for g in kv_cache_config.kv_cache_groups\n"
    "        if isinstance(g.kv_cache_spec, AttentionSpec)\n"
    "        or (isinstance(g.kv_cache_spec, MambaSpec) and mamba_scales)\n"
    "    ]\n"
    "    return groups or list(kv_cache_config.kv_cache_groups)\n"
    "\n"
    "\n"
)
_KV_REPORT_FN_NEW = _KV_HELPER_FN + _KV_REPORT_FN_OLD


_KV_CALLSITE_OLD = (
    "    min_block_size = min(\n"
    "        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]\n"
    "    )\n"
    "\n"
    "    # Log the KV cache size and maximum concurrency.\n"
    "    num_tokens = (\n"
    "        kv_cache_config.num_blocks\n"
    "        // len(kv_cache_config.kv_cache_groups)\n"
    "        * min_block_size\n"
    "    )"
)
_KV_CALLSITE_NEW = (
    "    # [Genesis P8] Use filter-helper that excludes O(1) mamba groups.\n"
    "    capacity_groups = token_capacity_kv_cache_groups(vllm_config, kv_cache_config)\n"
    "    min_block_size = min(g.kv_cache_spec.block_size for g in capacity_groups)\n"
    "\n"
    "    # Log the KV cache size and maximum concurrency.\n"
    "    num_tokens = (\n"
    "        kv_cache_config.num_blocks // len(capacity_groups) * min_block_size\n"
    "    )"
)


# ── scheduler.py sub-patches ──────────────────────────────────────────

_SCHED_IMPORT_OLD = (
    "from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager\n"
    "from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n"
    "from vllm.v1.core.sched.interface import PauseState, SchedulerInterface"
)
_SCHED_IMPORT_NEW = (
    "from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager\n"
    "from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector\n"
    "from vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups  # [Genesis P8]\n"
    "from vllm.v1.core.sched.interface import PauseState, SchedulerInterface"
)


_SCHED_CALLSITE_OLD = (
    "            min_block_size = min(\n"
    "                [\n"
    "                    group.kv_cache_spec.block_size\n"
    "                    for group in kv_cache_config.kv_cache_groups\n"
    "                ]\n"
    "            )\n"
    "            num_groups = len(kv_cache_config.kv_cache_groups)\n"
    "            self.max_num_kv_tokens = (\n"
    "                kv_cache_config.num_blocks // num_groups\n"
    "            ) * min_block_size"
)
_SCHED_CALLSITE_NEW = (
    "            # [Genesis P8] Exclude O(1) mamba groups from per-token divisor.\n"
    "            capacity_groups = token_capacity_kv_cache_groups(\n"
    "                self.vllm_config, kv_cache_config\n"
    "            )\n"
    "            min_block_size = min(g.kv_cache_spec.block_size for g in capacity_groups)\n"
    "            self.max_num_kv_tokens = (\n"
    "                kv_cache_config.num_blocks // len(capacity_groups)\n"
    "            ) * min_block_size"
)


# ── Dispatch ──────────────────────────────────────────────────────────

def _patcher_kv() -> TextPatcher | None:
    """Build the kv_cache_utils.py patcher with V1/V2 import-anchor auto-pick.

    [Genesis P8 v7.62.13 fix] Detect upstream import-block layout at
    construction time. Old vllm: V1 anchor (no MambaSpec). New vllm
    (post upstream MambaSpec auto-export + MLA additions): V2 anchor.
    Either way, _KV_REPORT_FN_OLD and _KV_CALLSITE_OLD anchors stay
    the same — only the import block layout differs.
    """
    target = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if target is None:
        return None
    # Auto-pick imports anchor by reading the actual file once.
    import_anchor = _KV_IMPORT_OLD
    import_replacement = _KV_IMPORT_NEW
    try:
        import os as _os
        if _os.path.isfile(str(target)):
            with open(str(target)) as _f:
                _src = _f.read()
            # [Genesis P8 v3 2026-05-02] Distinguish "already patched"
            # (idempotent skip — INFO) from "real anchor drift" (WARNING).
            # Previously emitted alarming "may have drifted" log on every
            # restart even when the patch was correctly already-applied
            # (idempotent path triggers when marker comment + replacement
            # block are in source instead of the original anchor).
            already_applied = (
                "[Genesis P8" in _src
                or _KV_IMPORT_NEW in _src
                or _KV_IMPORT_NEW_V2 in _src
            )
            if _KV_IMPORT_OLD in _src:
                pass  # V1 layout — already set
            elif _KV_IMPORT_OLD_V2 in _src:
                import_anchor = _KV_IMPORT_OLD_V2
                import_replacement = _KV_IMPORT_NEW_V2
                log.info("[P8] using V2 import anchor (post-MambaSpec layout)")
            elif already_applied:
                log.info(
                    "[P8] kv_cache_utils.py already has Genesis P8 marker — "
                    "idempotent skip (this is the expected restart path; "
                    "patch persists across container restart since file "
                    "edits are on the bind-mount layer)"
                )
            else:
                log.warning(
                    "[P8] neither V1 nor V2 import anchor matches AND no "
                    "Genesis marker present — kv_cache_utils.py drifted "
                    "from upstream layout (real upstream change); patcher "
                    "will skip safely. If this persists across pin-bumps, "
                    "P8 may need a V3 anchor for new layout"
                )
    except Exception as e:
        log.warning("[P8] anchor auto-pick failed: %s — defaulting to V1", e)
    return TextPatcher(
        patch_name="P8 KV hybrid reporting (kv_cache_utils)",
        target_file=target,
        marker=GENESIS_P8_MARKER_KV,
        sub_patches=[
            TextPatch(
                name="p8_kv_imports",
                anchor=import_anchor,
                replacement=import_replacement,
                required=True,
            ),
            TextPatch(
                name="p8_kv_helper_injection",
                anchor=_KV_REPORT_FN_OLD,
                replacement=_KV_REPORT_FN_NEW,
                required=True,
            ),
            TextPatch(
                name="p8_kv_callsite",
                anchor=_KV_CALLSITE_OLD,
                replacement=_KV_CALLSITE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS_KV,
    )


def _patcher_sched() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P8 KV hybrid reporting (scheduler)",
        target_file=target,
        marker=GENESIS_P8_MARKER_SCHED,
        sub_patches=[
            TextPatch(
                name="p8_sched_import",
                anchor=_SCHED_IMPORT_OLD,
                replacement=_SCHED_IMPORT_NEW,
                required=True,
            ),
            TextPatch(
                name="p8_sched_callsite",
                anchor=_SCHED_CALLSITE_OLD,
                replacement=_SCHED_CALLSITE_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=UPSTREAM_DRIFT_MARKERS_SCHED,
    )


def apply() -> tuple[str, str]:
    """Apply BOTH P8 sub-patches (kv_cache_utils + scheduler). Never raises.

    [Genesis P8 v7.62.13 atomic-apply guard]: if kv_cache_utils.py FAILS or is
    SKIPPED (anchor drift / partial-applied marker), scheduler.py MUST NOT
    apply. Otherwise the scheduler.py P8 import line `from
    vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups` would
    reference a missing function → `ImportError` at engine boot → container
    crash loop. This was the v7.62.12 PROD-restore fail incident.
    """
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    messages: list[str] = []

    # Apply to kv_cache_utils.py FIRST
    pk = _patcher_kv()
    if pk is None:
        return "skipped", "kv_cache_utils.py not found"
    result_kv, failure_kv = pk.apply()

    # ─── ATOMIC GUARD ───────────────────────────────────────────────
    # Only proceed to scheduler.py if kv_cache_utils.py was successfully
    # APPLIED or already-IDEMPOTENT (helper exists). If it FAILED or
    # SKIPPED, the scheduler.py import would crash the engine.
    #
    # [Genesis Issue #5 hardening, 2026-04-30]: Even when result_kv
    # reports APPLIED, the helper symbol may not actually be importable
    # — vLLM v0.20.0 reorganized `kv_cache_utils.py` such that our
    # anchor still matches but the surrounding context drifted.
    # Verify by attempting an actual import; if it fails, refuse to
    # advance to scheduler.py (which would inject an import line that
    # crashes engine boot with the reported `ImportError: cannot import
    # name 'token_capacity_kv_cache_groups'`).
    if result_kv in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
        try:
            import importlib
            mod = importlib.import_module("vllm.v1.core.kv_cache_utils")
            if not hasattr(mod, "token_capacity_kv_cache_groups"):
                log.warning(
                    "[P8 Issue #5 guard] kv_cache_utils.py reports "
                    "applied/idempotent BUT the helper symbol "
                    "'token_capacity_kv_cache_groups' is NOT importable. "
                    "This is the vLLM v0.20.0 case: upstream "
                    "reorganized kv_cache_utils.py and our anchor still "
                    "matches but the helper injection landed in a place "
                    "that doesn't expose it at module scope. REFUSING "
                    "to apply scheduler.py P8 — engine would crash with "
                    "`ImportError: cannot import name "
                    "'token_capacity_kv_cache_groups'`. To unblock: pin "
                    "to a pre-v0.20.0 vLLM commit, or set "
                    "GENESIS_ENABLE_P8=0, or wait for a P8 v0.20.0 "
                    "anchor refresh."
                )
                return "skipped", (
                    "P8 Issue #5 guard: kv_cache_utils.py applied but "
                    "helper not importable (vLLM v0.20.0 layout); "
                    "scheduler.py P8 NOT applied to avoid ImportError. "
                    "Engine boots with vanilla scheduler."
                )
        except Exception as e:
            # If we cannot even import kv_cache_utils, scheduler.py
            # certainly won't load with our injection — bail safely.
            log.warning(
                "[P8 Issue #5 guard] kv_cache_utils.py post-apply "
                "import probe raised %s: %s — refusing to advance to "
                "scheduler.py.",
                type(e).__name__, e,
            )
            return "skipped", (
                f"P8 Issue #5 guard: post-apply import probe failed "
                f"({type(e).__name__}); scheduler.py P8 NOT applied."
            )

    # Special case: SKIPPED with reason=upstream_merged means the helper
    # already exists in the file — scheduler.py SHOULD still apply because
    # its import will succeed. The Issue #5 guard above already verified
    # importability when result_kv was APPLIED/IDEMPOTENT; for upstream-
    # merged we trust upstream and proceed.
    reason_kv_str = failure_kv.reason if failure_kv else ""
    upstream_already = (
        result_kv == TextPatchResult.SKIPPED
        and reason_kv_str == "upstream_merged"
    )

    if result_kv not in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT) and not upstream_already:
        reason_kv = failure_kv.reason if failure_kv else "unknown"
        log.warning(
            "[P8] kv_cache_utils.py apply did NOT succeed (%s: %s) — "
            "REFUSING to apply scheduler.py P8 to avoid ImportError crash. "
            "Engine will boot with vanilla scheduler (no Mamba groups exclusion). "
            "Fix the kv_cache_utils.py anchor and re-deploy.",
            result_kv.name if hasattr(result_kv, 'name') else result_kv,
            reason_kv,
        )
        return "skipped", (
            f"P8 atomic guard: kv_cache_utils.py {result_kv} "
            f"({reason_kv}); scheduler.py P8 NOT applied to avoid ImportError. "
            "Engine boots with vanilla scheduler."
        )

    # Apply to scheduler.py (only reached if kv_cache_utils.py succeeded)
    ps = _patcher_sched()
    if ps is None:
        return "skipped", "scheduler.py not found"
    result_sched, failure_sched = ps.apply()

    # Status policy: if BOTH applied or idempotent → "applied".
    # If EITHER skipped → log partial. If EITHER failed → overall failed.
    def _state(r, f) -> tuple[str, str]:
        if r == TextPatchResult.APPLIED:
            return "applied", "ok"
        if r == TextPatchResult.IDEMPOTENT:
            return "applied", "idempotent"
        if r == TextPatchResult.SKIPPED:
            return "skipped", f.reason if f else "unknown"
        return "failed", f.reason if f else "unknown"

    s_kv, r_kv = _state(result_kv, failure_kv)
    s_sched, r_sched = _state(result_sched, failure_sched)

    messages.append(f"kv_cache_utils={s_kv}({r_kv})")
    messages.append(f"scheduler={s_sched}({r_sched})")
    combined_reason = ", ".join(messages)

    if s_kv == "failed" or s_sched == "failed":
        return "failed", combined_reason
    if s_kv == "skipped" and s_sched == "skipped":
        return "skipped", combined_reason
    # At least one applied successfully; treat as applied overall
    return "applied", combined_reason
