# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 85 — hybrid fine-shadow prefix cache (vllm#38182 followup).

================================================================
ROOT CAUSE (proven empirically + via deep code analysis)
================================================================

vLLM v1's prefix-cache for hybrid models (Qwen3.6-MoE GDN+Mamba+attention)
has TWO distinct mismatches that combine to make caching non-functional
for short-prompt single-user workloads:

**Mismatch A (short prompts < largest spec.block_size, e.g., < 2048):**

`single_type_kv_cache_manager.py:251` `SingleTypeKVCacheManager.cache_blocks`:

    num_full_blocks = num_tokens // self.block_size

For 1424-token requests with `MambaManager.self.block_size = 2048`:
- `num_full_blocks = 1424 // 2048 = 0` → early return → nothing stored.

The HybridKVCacheCoordinator then gates the final hit_length on the MIN
across all groups (kv_cache_coordinator.py:497-540 iterative loop).
Even though FullAttentionManager correctly stores 89 fine-grained hashes
for the 1424-token request, the Mamba group returns 0 hits → final
hit_tokens = 0.

**Mismatch B (long prompts ≥ 2048 tokens, e.g., 5018):**

`MambaManager.allocate_new_blocks` in align mode pads the prefix with
`null_block`s — only the LAST `1 + num_speculative_blocks` real blocks
are populated. So for `num_full_blocks = 5018 // 2048 = 2`:
- `new_full_blocks = blocks[0:2]` = `[null_block, null_block]`
- `block_pool.cache_full_blocks` skip-loops at `if blk.is_null: continue`
- **Zero entries actually inserted in `cached_block_hash_to_block`.**

Both mismatches manifest empirically as `hit_tokens = 0` even on three
identical requests in a row.

================================================================
PRIOR PATCHES (insufficient alone)
================================================================

- **P83** (skip Eagle pop): correct fix for one downstream symptom but
  pop site is never reached because hashes / coordinator paths cut earlier.
- **P84** (dual-site `hash_block_size` override): enables fine-grained
  hash COMPUTATION (89 hashes for 1424-token request) but doesn't fix
  the STORE/LOOKUP mismatch on Mamba group.

P83 + P84 together produce: `num_hashes=89` ✓ but still `hit_tokens=0`.

================================================================
P85 FIX — fine-shadow entries on Mamba store + lookup
================================================================

Approach (a) from architectural analysis: when MambaManager stores a
coarse Mamba block, ALSO register `scale_factor = mamba_block_size /
hash_block_size` shadow fine-hash entries in `cached_block_hash_to_block`,
all pointing to the SAME `KVCacheBlock`.

On Mamba lookup (find_longest_cache_hit), when env P85 is set, walk
fine hashes (request.block_hashes directly) instead of coarse-adapted
hashes. This finds matches that the coarse scan would miss.

**Memory model invariants preserved:**
- Same `KVCacheBlock` objects, no new allocations.
- No ref-count changes (shadow entries are pure lookup keys).
- Eviction safety: lookup branch verifies the cached block's
  `block_hash` field still matches our expected coarse hash before
  returning. On mismatch (block was recycled), treat as miss.

**For Mismatch A (short prompts):** Mamba never stores → no shadows
created → architectural limit honored (Mamba state genuinely cannot be
recovered from cache for prompts < block_size). P85 doesn't claim to
fix this — it's a fundamental limitation of incremental Mamba state.

**For Mismatch B (long prompts):** Mamba stores 1+ blocks → shadows
register → Mamba lookup finds matches → coordinator gate passes →
real cache hits → multi-turn TTFT improves.

================================================================
SAFETY MODEL
================================================================

- Default OFF. Opt-in via `GENESIS_ENABLE_P85=1`.
- Both store and lookup hunks gated on the same env. Only ONE side
  enabled would be a no-op (lookup finds nothing, or store creates
  unused shadows — both safe).
- Stale-shadow eviction safety: lookup branch verifies
  `cached_block.block_hash == expected_coarse_hash` before returning.
- Drift detection on both anchor sites.

Status: opt-in via `GENESIS_ENABLE_P85=1`. Default OFF.

Tunable knobs
-------------
- `GENESIS_ENABLE_P85` (default unset/0): master switch
- Requires `GENESIS_ENABLE_P84=1` + `GENESIS_P84_HASH_BLOCK_SIZE=<N>`
  with N dividing every group block_size (typically 16).

Compatibility
-------------
- Mamba / GDN / hybrid models (where this patch is needed).
- Pure attention models: P85 hunks would activate but shadow registration
  is a no-op (block_size == hash_block_size → scale_factor = 1, shadows
  are duplicates of coarse).
- MTP / Eagle / ngram: cache layer agnostic to spec method.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Discovery: 6-round empirical investigation 2026-04-27 + deep code analysis
synthesis. See sprint report SPRINT_REPORT_20260427_phase4_*.md.
Related: P83 (Eagle pop skip), P84 (dual-site hash_block_size override).
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

log = logging.getLogger("genesis.wiring.p85_hybrid_fine_shadow_prefix_cache")


GENESIS_P85_MARKER = "Genesis P85 hybrid fine-shadow prefix cache (vllm#38182 followup) v7.53.8_debug"


# ─── Site 1: MambaManager.cache_blocks adds shadow entries ────────────────

P85_SITE1_OLD = (
    "    def cache_blocks(self, request: Request, num_tokens: int) -> None:\n"
    "        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)\n"
    "        super().cache_blocks(request, num_tokens)\n"
    "        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)\n"
    "        if num_cached_blocks_after > num_cached_blocks_before:\n"
    "            for block in self.req_to_blocks[request.request_id][\n"
    "                num_cached_blocks_before:num_cached_blocks_after\n"
    "            ]:\n"
    "                if block.is_null:\n"
    "                    continue\n"
    "                assert block.block_hash is not None\n"
    "                self.cached_blocks_this_step.add(block.block_hash)\n"
    "\n"
    "    def new_step_starts(self) -> None:\n"
)

P85_SITE1_NEW = (
    "    def cache_blocks(self, request: Request, num_tokens: int) -> None:\n"
    "        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)\n"
    "        super().cache_blocks(request, num_tokens)\n"
    "        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)\n"
    "        if num_cached_blocks_after > num_cached_blocks_before:\n"
    "            for block in self.req_to_blocks[request.request_id][\n"
    "                num_cached_blocks_before:num_cached_blocks_after\n"
    "            ]:\n"
    "                if block.is_null:\n"
    "                    continue\n"
    "                assert block.block_hash is not None\n"
    "                self.cached_blocks_this_step.add(block.block_hash)\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P85] Shadow fine-grained hash entries for hybrid lookup.\n"
    "        # When GENESIS_ENABLE_P85=1 and self.block_size != hash_block_size,\n"
    "        # also register fine sub-block hash entries (one per hash_block_size\n"
    "        # tokens) pointing at the coarse Mamba block. This lets lookup find\n"
    "        # multi-turn cache hits that the coarse-only store would miss.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        import os as _g_p85_os\n"
    "        if _g_p85_os.environ.get('GENESIS_ENABLE_P85', '').strip().lower() in (\n"
    "                '1', 'true', 'yes', 'on'):\n"
    "            from vllm.v1.core.kv_cache_utils import (\n"
    "                make_block_hash_with_group_id as _g_p85_mk)\n"
    "            _g_p85_hbs = self.block_pool.hash_block_size\n"
    "            if self.block_size != _g_p85_hbs and self.block_size % _g_p85_hbs == 0:\n"
    "                _g_p85_scale = self.block_size // _g_p85_hbs\n"
    "                _g_p85_blocks = self.req_to_blocks[request.request_id]\n"
    "                _g_p85_fine = request.block_hashes\n"
    "                _g_p85_committed = self.num_cached_block.get(request.request_id, 0)\n"
    "                for _g_p85_i in range(_g_p85_committed):\n"
    "                    _g_p85_blk = _g_p85_blocks[_g_p85_i]\n"
    "                    if _g_p85_blk.is_null:\n"
    "                        continue\n"
    "                    _g_p85_base = _g_p85_i * _g_p85_scale\n"
    "                    _g_p85_end = _g_p85_base + _g_p85_scale\n"
    "                    if _g_p85_end > len(_g_p85_fine):\n"
    "                        break\n"
    "                _g_p85_inserted = 0\n"
    "                _g_p85_skipped_null = 0\n"
    "                for _g_p85_i in range(_g_p85_committed):\n"
    "                    _g_p85_blk = _g_p85_blocks[_g_p85_i]\n"
    "                    if _g_p85_blk.is_null:\n"
    "                        _g_p85_skipped_null += 1\n"
    "                        continue\n"
    "                    _g_p85_base = _g_p85_i * _g_p85_scale\n"
    "                    _g_p85_end = _g_p85_base + _g_p85_scale\n"
    "                    if _g_p85_end > len(_g_p85_fine):\n"
    "                        break\n"
    "                    for _g_p85_j in range(_g_p85_base, _g_p85_end):\n"
    "                        _g_p85_key = _g_p85_mk(\n"
    "                            _g_p85_fine[_g_p85_j], self.kv_cache_group_id)\n"
    "                        self.block_pool.cached_block_hash_to_block.insert(\n"
    "                            _g_p85_key, _g_p85_blk)\n"
    "                        _g_p85_inserted += 1\n"
    "                if _g_p85_os.environ.get('GENESIS_P85_DEBUG', '') == '1':\n"
    "                    import sys as _g_p85_sys\n"
    "                    _g_p85_sys.stderr.write(\n"
    "                        '[GENESIS_P85_STORE] req=' + request.request_id[:8]\n"
    "                        + ' bs=' + str(self.block_size)\n"
    "                        + ' hbs=' + str(_g_p85_hbs)\n"
    "                        + ' scale=' + str(_g_p85_scale)\n"
    "                        + ' committed=' + str(_g_p85_committed)\n"
    "                        + ' skipped_null=' + str(_g_p85_skipped_null)\n"
    "                        + ' shadows_inserted=' + str(_g_p85_inserted)\n"
    "                        + ' fine_hashes=' + str(len(_g_p85_fine))\n"
    "                        + '\\n')\n"
    "                    _g_p85_sys.stderr.flush()\n"
    "\n"
    "    def new_step_starts(self) -> None:\n"
)


# ─── Site 2: MambaManager.find_longest_cache_hit walks fine hashes ────────

P85_SITE2_OLD = (
    "        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(\n"
    "            [] for _ in range(len(kv_cache_group_ids))\n"
    "        )\n"
    "\n"
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
    "                # across full attention layers and Mamba layers to ensure the\n"
    "                # prefix hit length aligned at block\n"
    "                if (\n"
    "                    block_size != alignment_tokens  # Faster for common case.\n"
    "                    and (i + 1) * block_size % alignment_tokens != 0\n"
    "                ):\n"
    "                    continue\n"
    "                for computed, cached in zip(computed_blocks, cached_block):\n"
    "                    # the hit length logic later assumes:\n"
    "                    #  hit_length = len(hit_blocks_other_attn[0])\n"
    "                    #               * self.other_block_size\n"
    "                    # so we insert dummy blocks at the beginning:\n"
    "                    computed.extend([block_pool.null_block] * i)\n"
    "                    computed.append(cached)\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        return computed_blocks\n"
)

P85_SITE2_NEW = (
    "        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(\n"
    "            [] for _ in range(len(kv_cache_group_ids))\n"
    "        )\n"
    "\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        # [Genesis P85] Fine-shadow lookup branch.\n"
    "        # When env is set AND fine hashes are available (BlockHashListWithBlockSize\n"
    "        # adapter NOT applied — i.e. coordinator passed raw fine hashes),\n"
    "        # scan at fine granularity to find shadows registered by\n"
    "        # MambaManager.cache_blocks. Eviction-safe: re-derive the coarse\n"
    "        # hash and verify cached_block.block_hash matches before returning.\n"
    "        # ════════════════════════════════════════════════════════════════\n"
    "        import os as _g_p85_os2\n"
    "        if _g_p85_os2.environ.get('GENESIS_ENABLE_P85', '').strip().lower() in (\n"
    "                '1', 'true', 'yes', 'on'):\n"
    "            from vllm.v1.core.kv_cache_utils import (\n"
    "                BlockHashListWithBlockSize as _g_p85_BHLBS,\n"
    "                make_block_hash_with_group_id as _g_p85_mk2,\n"
    "            )\n"
    "            _g_p85_hbs2 = block_pool.hash_block_size\n"
    "            if (kv_cache_spec.block_size != _g_p85_hbs2\n"
    "                    and kv_cache_spec.block_size % _g_p85_hbs2 == 0\n"
    "                    and not isinstance(block_hashes, _g_p85_BHLBS)):\n"
    "                _g_p85_scale2 = kv_cache_spec.block_size // _g_p85_hbs2\n"
    "                _g_p85_max_fine = max_length // _g_p85_hbs2\n"
    "                _g_p85_max_fine = min(_g_p85_max_fine, len(block_hashes))\n"
    "                _g_p85_align_fine = alignment_tokens // _g_p85_hbs2\n"
    "                # Search from right to left at fine granularity, but only\n"
    "                # consider indices that align with both scale and alignment.\n"
    "                for _g_p85_i in range(_g_p85_max_fine - 1, -1, -1):\n"
    "                    if (_g_p85_i + 1) % _g_p85_scale2 != 0:\n"
    "                        continue\n"
    "                    if _g_p85_align_fine > 0 and (_g_p85_i + 1) % _g_p85_align_fine != 0:\n"
    "                        continue\n"
    "                    _g_p85_fine_key = _g_p85_mk2(\n"
    "                        block_hashes[_g_p85_i], kv_cache_group_ids[0])\n"
    "                    _g_p85_cached = block_pool.get_cached_block(\n"
    "                        block_hashes[_g_p85_i], kv_cache_group_ids)\n"
    "                    if _g_p85_cached is None:\n"
    "                        continue\n"
    "                    # Eviction safety: the cached block's block_hash field\n"
    "                    # is the COARSE hash (set by block_pool.cache_full_blocks).\n"
    "                    # Re-derive it and verify match. If block was evicted +\n"
    "                    # recycled, the field will mismatch → treat as miss.\n"
    "                    _g_p85_coarse_base = (_g_p85_i + 1 - _g_p85_scale2)\n"
    "                    _g_p85_coarse_end = _g_p85_i + 1\n"
    "                    if _g_p85_coarse_end > len(block_hashes):\n"
    "                        continue\n"
    "                    _g_p85_merged = bytes(block_hashes[_g_p85_coarse_base])\n"
    "                    for _g_p85_j2 in range(_g_p85_coarse_base + 1, _g_p85_coarse_end):\n"
    "                        _g_p85_merged += bytes(block_hashes[_g_p85_j2])\n"
    "                    _g_p85_expected_coarse = _g_p85_mk2(\n"
    "                        _g_p85_merged, kv_cache_group_ids[0])\n"
    "                    _g_p85_first = _g_p85_cached[0]\n"
    "                    if _g_p85_first.block_hash != _g_p85_expected_coarse:\n"
    "                        # Stale shadow — block was evicted/recycled. Skip.\n"
    "                        continue\n"
    "                    # Coarse-level index from fine: (i+1)/scale - 1\n"
    "                    _g_p85_coarse_idx = (_g_p85_i + 1) // _g_p85_scale2 - 1\n"
    "                    for computed, cached in zip(computed_blocks, _g_p85_cached):\n"
    "                        computed.extend(\n"
    "                            [block_pool.null_block] * _g_p85_coarse_idx)\n"
    "                        computed.append(cached)\n"
    "                    return computed_blocks\n"
    "                # No fine match — fall through to coarse logic below\n"
    "                # (which will likely return empty for short prompts).\n"
    "        block_size = kv_cache_spec.block_size\n"
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
    "                # across full attention layers and Mamba layers to ensure the\n"
    "                # prefix hit length aligned at block\n"
    "                if (\n"
    "                    block_size != alignment_tokens  # Faster for common case.\n"
    "                    and (i + 1) * block_size % alignment_tokens != 0\n"
    "                ):\n"
    "                    continue\n"
    "                for computed, cached in zip(computed_blocks, cached_block):\n"
    "                    # the hit length logic later assumes:\n"
    "                    #  hit_length = len(hit_blocks_other_attn[0])\n"
    "                    #               * self.other_block_size\n"
    "                    # so we insert dummy blocks at the beginning:\n"
    "                    computed.extend([block_pool.null_block] * i)\n"
    "                    computed.append(cached)\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        return computed_blocks\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "P85 v1/core/single_type_kv_cache_manager.py — hybrid fine-shadow "
            "prefix cache (vllm#38182 followup)"
        ),
        target_file=str(target),
        marker=GENESIS_P85_MARKER,
        sub_patches=[
            TextPatch(
                name="p85_mamba_cache_blocks_shadow",
                anchor=P85_SITE1_OLD,
                replacement=P85_SITE1_NEW,
                required=True,
            ),
            TextPatch(
                name="p85_mamba_find_longest_cache_hit_fine",
                anchor=P85_SITE2_OLD,
                replacement=P85_SITE2_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P85",
            "_g_p85_",
            # Upstream-side markers if vLLM ships its own hybrid fine cache:
            "MambaManager.find_longest_cache_hit_fine",
            "fine_shadow_prefix_cache",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P85 — hybrid fine-shadow prefix cache."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P85")
    log_decision("P85", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/single_type_kv_cache_manager.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P85] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P85" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed hybrid fine-cache fix",
            )

    result, failure = patcher.apply()
    # Audit P1 fix 2026-05-05: route SKIPPED/IDEMPOTENT honestly via shared helper
    from vllm._genesis.wiring.text_patch import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P85 applied: hybrid fine-shadow prefix cache installed at MambaManager. "
            "cache_blocks now registers fine-grained shadow entries; "
            "find_longest_cache_hit prefers fine-scan with eviction-safety verify. "
            "Requires GENESIS_ENABLE_P84=1 + GENESIS_P84_HASH_BLOCK_SIZE=<N> for "
            "fine hashes to be computed in the first place."
        ),
        patch_name=patcher.patch_name,
    )
