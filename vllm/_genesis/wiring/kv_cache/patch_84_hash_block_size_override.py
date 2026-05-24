# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 84 — hash_block_size override (vllm#38182 actual root cause).

================================================================
ROOT CAUSE
================================================================

vllm/v1/core/sched/scheduler.py:234 hard-codes:

    self.kv_cache_manager = KVCacheManager(
        ...
        hash_block_size=self.block_size,  # ← BUG for hybrid models
        ...
    )

For hybrid models (Qwen3.6-MoE GDN+attention, Mamba models, etc.) the
effective `self.block_size` is the LCM of all group block sizes,
typically padded UP to the largest group requirement (Mamba state
size, often >= 2048 tokens).

vLLM's `request_block_hasher` only computes hashes for FULL blocks:

    while True:
        end_token_idx = start_token_idx + block_size
        if end_token_idx > num_tokens:
            break  # We only hash full blocks

So a 1424-token request with block_size=2048 produces ZERO block
hashes → coordinator returns 0 hits trivially → prefix-cache is
USELESS for short prompts even though the cache machinery still
runs (per-step overhead).

Empirical confirmation (this rig, debug-instrumented):
- 1424-token identical request, prefix-cache ON: num_hashes=0
- Cache hit rate: 0% (literally cannot hit because no hashes computed)
- Throughput cost of cache machinery overhead: -30% vs cache OFF

================================================================
FIX
================================================================

vLLM's KVCacheCoordinator architecture ALREADY supports
`hash_block_size != block_size` — see:

    vllm/v1/core/kv_cache_coordinator.py:397-405
    # hash_block_size: the block size used to compute block hashes.
    # The actual block size usually equals hash_block_size, but in cases
    # where different KV cache groups have different block sizes, the
    # actual block size can be a multiple of hash_block_size.
    self.hash_block_size = hash_block_size
    assert all(
        g.kv_cache_spec.block_size % hash_block_size == 0
        for g in kv_cache_config.kv_cache_groups
    ), "block_size must be divisible by hash_block_size"

Constraint: hash_block_size must divide ALL groups' block_size. For our
hybrid Qwen3.6, smallest group block_size = 16 (full attention default),
largest = Mamba state alignment (e.g., 2048 or 4096). Valid choices:
{1, 2, 4, 8, 16}.

P84 text-patches scheduler.py:234 to read `hash_block_size` from env
`GENESIS_P84_HASH_BLOCK_SIZE` when set, falling back to `self.block_size`
when unset (= upstream behavior).

================================================================
EXPECTED OUTCOME
================================================================

With `GENESIS_P84_HASH_BLOCK_SIZE=16`:
- 1424-token request → 89 hashes computed (1424 // 16)
- Multi-turn shared prefix can actually cache-hit
- Predicted: TTFT regression on multi-turn 2.5K shared prefix
  drops from ~280ms (currently) to <50ms

================================================================
SAFETY MODEL
================================================================

- Default OFF (env unset → unchanged upstream behavior).
- Drift detection on scheduler.py L234 anchor.
- If env value doesn't divide all group block_sizes → vLLM's own
  assertion fires at startup, container fails fast (NOT silent corruption).
- MTP/Eagle/no-spec all benefit (the bug is in cache layer, not
  spec-decode layer).

Status: opt-in via `GENESIS_P84_HASH_BLOCK_SIZE=<int>`. Default OFF.

Tunable knobs
-------------
- `GENESIS_P84_HASH_BLOCK_SIZE=<int>` — override hash_block_size.
  Recommended: 16 (matches default full-attention block_size on most
  configs). Must divide every group's block_size or container fails.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Discovery: Genesis P83 DEBUG instrumentation 2026-04-27 — empirically
confirmed num_hashes=0 for 1424-token requests on Qwen3.6-MoE prod.
Related: vllm#38182 (which identified the WRONG root cause —
the L457 pop is a downstream symptom; the upstream cause is hash_block_size).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p84_hash_block_size_override")


GENESIS_P84_MARKER = "Genesis P84 hash_block_size override (vllm#38182 actual root cause) v7.53.3_dual_site"


# ─── Anchor: scheduler.py:234 hash_block_size=self.block_size ──────────────

P84_OLD = (
    "            hash_block_size=self.block_size,\n"
)

P84_NEW = (
    "            # ════════════════════════════════════════════════════════════════\n"
    "            # [Genesis P84 vllm#38182 actual root cause]\n"
    "            # When GENESIS_P84_HASH_BLOCK_SIZE=<int> is set, override the\n"
    "            # default hash_block_size = self.block_size (which on hybrid\n"
    "            # models is LCM-padded up to largest group's block_size, often\n"
    "            # 2048+ → no hashes computed for short prompts → cache useless).\n"
    "            # The override must divide every group's block_size or vLLM's\n"
    "            # own assertion fires at startup. Recommended: 16 (full-attn default).\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            hash_block_size=int(__import__('os').environ.get(\n"
    "                'GENESIS_P84_HASH_BLOCK_SIZE', str(self.block_size))),\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P84 v1/core/sched/scheduler.py — hash_block_size override "
            "(vllm#38182 actual root cause)"
        ),
        target_file=str(target),
        marker=GENESIS_P84_MARKER,
        sub_patches=[
            TextPatch(
                name="p84_hash_block_size_env_override",
                anchor=P84_OLD,
                replacement=P84_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P84",
            "GENESIS_P84_HASH_BLOCK_SIZE",
            # Upstream-side markers if vLLM adds its own knob:
            "--hash-block-size",
            "hash_block_size_arg",
            "args.hash_block_size",
            # [v7.62.13 audit] Upstream PR landed in nightly 7923b48047be:
            # cache_config.hash_block_size, Scheduler.__init__(hash_block_size),
            # engine/core.py:210 uses hash_block_size+caching_hash_fn directly.
            "self.hash_block_size",
            "cache_config.hash_block_size",
        ],
    )


# ─── Site 2: engine/core.py request_block_hasher init ──────────────────────
# This is THE critical site — request_block_hasher is what computes the
# block hashes per request. P84 v1 only patched scheduler.py (which controls
# the coordinator-side hash_block_size). Without patching this site, the
# request hasher still uses scheduler_block_size = cache_config.block_size
# (LCM-padded), so num_hashes=0 for short prompts even with hash_block_size
# override on the lookup side.

P84_SITE2_OLD = (
    "            self.request_block_hasher = get_request_block_hasher(\n"
    "                scheduler_block_size, caching_hash_fn\n"
    "            )\n"
)

P84_SITE2_NEW = (
    "            # ════════════════════════════════════════════════════════════════\n"
    "            # [Genesis P84 vllm#38182 actual root cause — Site 2]\n"
    "            # request_block_hasher MUST use the same hash_block_size as the\n"
    "            # coordinator, otherwise hashes don't get computed for short prompts.\n"
    "            # ════════════════════════════════════════════════════════════════\n"
    "            import os as _genesis_p84_os\n"
    "            _genesis_p84_hbs = _genesis_p84_os.environ.get('GENESIS_P84_HASH_BLOCK_SIZE', '')\n"
    "            _genesis_p84_block_size = int(_genesis_p84_hbs) if _genesis_p84_hbs else scheduler_block_size\n"
    "            self.request_block_hasher = get_request_block_hasher(\n"
    "                _genesis_p84_block_size, caching_hash_fn\n"
    "            )\n"
)


def _make_patcher_engine_core() -> TextPatcher | None:
    target = resolve_vllm_file("v1/engine/core.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P84 Site 2 v1/engine/core.py — request_block_hasher uses env hash_block_size"
        ),
        target_file=str(target),
        marker=GENESIS_P84_MARKER + "_engine_core",
        sub_patches=[
            TextPatch(
                name="p84_engine_core_request_block_hasher",
                anchor=P84_SITE2_OLD,
                replacement=P84_SITE2_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P84",
            "_genesis_p84_block_size",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P84 — hash_block_size override."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P84")
    log_decision("P84", decision, reason)
    if not decision:
        return "skipped", reason

    # Special: P84 has TWO opt-in conditions:
    # 1. The standard dispatcher gate (GENESIS_ENABLE_P84=1)
    # 2. AND the hash_block_size value itself must be set
    # If neither is set, skip (no-op patch).
    if not os.environ.get("GENESIS_P84_HASH_BLOCK_SIZE", ""):
        return "skipped", (
            "GENESIS_P84_HASH_BLOCK_SIZE is unset — patch would be a no-op "
            "(falls back to self.block_size). Set the env to engage."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/sched/scheduler.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P84] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P84" and m in content:
            continue  # our marker; handled above
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have added its own --hash-block-size knob",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: surface SKIPPED as skipped (was masked as applied)
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    # Site 2: engine/core.py — required for hashes to actually be computed.
    # Without this, scheduler.py override is no-op (lookup side only).
    site2 = _make_patcher_engine_core()
    if site2 is None:
        log.warning(
            "[P84] Site 2 engine/core.py not found — request_block_hasher will "
            "still use scheduler_block_size; hashes won't be computed for short prompts"
        )
    elif os.path.isfile(site2.target_file):
        with open(site2.target_file) as f:
            site2_content = f.read()
        if site2.marker not in site2_content:
            site2_result, site2_failure = site2.apply()
            if site2_result == TextPatchResult.FAILED:
                return "failed", (
                    f"P84 Site 2: {site2_failure.reason if site2_failure else 'unknown'} "
                    f"({site2_failure.detail if site2_failure else ''})"
                )
            log.info("[P84] Site 2 engine/core.py applied: %s", site2_result)

    hbs = os.environ.get("GENESIS_P84_HASH_BLOCK_SIZE", "")
    return "applied", (
        f"P84 applied (DUAL SITE): hash_block_size will be overridden to {hbs} "
        "BOTH at coordinator (scheduler.py) AND at request_block_hasher (engine/core.py). "
        "For hybrid Qwen3.6-MoE this enables block hashing at finer granularity → "
        "prefix-cache can actually hit on short prompts. If hash_block_size doesn't "
        "divide all group block_sizes, vLLM's own assertion will fire at startup."
    )
