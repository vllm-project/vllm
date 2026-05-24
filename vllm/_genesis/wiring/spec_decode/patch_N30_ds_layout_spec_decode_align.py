# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N30 — DS conv state layout + spec-decode AL>1 fix.

================================================================
Issue
================================================================
https://github.com/Sandermage/genesis-vllm-patches/issues/17 (noonghunna)

`get_conv_copy_spec` in `vllm/model_executor/layers/mamba/mamba_utils.py`
raises NotImplementedError when:
- VLLM_SSM_CONV_STATE_LAYOUT=DS (dim-strided layout, +6% TPS on 27B)
- num_accepted_tokens > 1 (every prefill with MTP K=3 + AL>1)
- mamba_cache_mode='align' (default)

50/50 LiveCodeBench v6 problems failed instantly on noonghunna's
27B Lorbus + TQ3 + MTP K=3 + TP=1 + structured-CoT config. Container
exited with status 0 after first batch.

================================================================
ROOT CAUSE
================================================================

DS layout: tensor shape (num_blocks, dim, state_len), strides
(dim*state_len, state_len, 1). Slicing `state[block, :, offset:]`
yields a NON-contiguous view because rows of `dim` are interleaved
with `state_len` chunks in memory; the slice picks `state_len-offset`
elements from each row but rows are strided by `state_len`.

Downstream `do_mamba_copy_block` consumes `MambaCopySpec.start_addr`
as a raw pointer for `batch_memcpy`. Non-contiguous source = invalid
for memcpy. Upstream conservatively raises NotImplementedError rather
than silently corrupt state.

================================================================
FIX
================================================================

Two-file text-patch with module-level temp-tensor list + delayed
cleanup pattern:

1. **`mamba_utils.py:get_conv_copy_spec`** — replace the
   NotImplementedError with `.contiguous()` copy + module-level list
   append. Tensor stays alive until next batch.

2. **`v1/worker/mamba_utils.py:do_mamba_copy_block`** — wrap to clear
   temp-tensor list AFTER batch_memcpy, with stream sync ONLY when
   the DS+offset>0 path was actually exercised (cheap predicate).

================================================================
LIFECYCLE CORRECTNESS
================================================================

`batch_memcpy` enqueues async copy on default CUDA stream. To safely
free contiguous-copy temp tensors, we either need to:
(a) Stream-sync after batch_memcpy and free immediately
(b) Delay free until next batch (FIFO ordering on stream guarantees
    previous async ops completed before new ops execute)

We use approach (a) — explicit `current_stream().synchronize()` is
~10-50us per batch and only fires when DS+offset>0 path was used.
Negligible cost for the workload that triggers this (spec-decode
prefill with structured CoT, where TPS is already dominated by
prefill compute).

================================================================
SAFETY MODEL
================================================================
- Default OFF (opt-in via GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE=1)
- Pure text-patch, idempotent via marker
- Drift-aware: anchor includes the exact NotImplementedError raise
  block — if upstream fixes this differently, our anchor won't match
- Anchor missing → SKIPPED, source stays vanilla
- Worst case: extra contiguous() copy + stream sync per batch when
  DS layout active and AL>1; cost amortized across long-form generation

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (issue #17, 2026-05-01).
"""
from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN30_ds_layout_spec_decode")

GENESIS_PN30_MARKER = (
    "Genesis PN30 DS conv state + spec-decode AL>1 (issue #17) v7.68"
)


# ─── Sub-patch 1: model_executor/layers/mamba/mamba_utils.py ────────
#
# v7.68: replaced the v7.65 compact `.contiguous()` path with a
# fail-closed RuntimeError. The v7.65 approach materialized
# `state[src_block_id, :, offset:].contiguous()` (compact dim×(state_len-offset)
# buffer) and raw-memcpy'd that into `state[dest_block_id]` — but the
# destination block is strided by the FULL state_len. So row 1+ of the
# compact source landed at wrong destination offsets, corrupting DS
# conv state row strides on every offset>0 copy.
#
# The corrected fix lives in part3 (`collect_mamba_copy_meta`), where
# both src AND dst block ids are known at the same callsite and a
# dst-shaped temp can be built without losing destination stride. The
# old part1 path becomes unreachable on the AL>1 + DS layout path; if
# anything ever reaches it, fail closed with a clear error rather than
# silently corrupt.
#
# Diagnosis credit: noonghunna + ChatGPT/Codex CLI cross-check
# (club-3090 commit 9af1a52, 2026-05-02). The compact-temp approach
# can't preserve DS layout because destination's stride information is
# lost when you flatten to a compact buffer.

PN30_PART1_ANCHOR = (
    "    if is_conv_state_dim_first():\n"
    "        # DS layout: (num_blocks, dim, state_len) — state_len is last.\n"
    "        if offset > 0:\n"
    "            # Slicing along the last dim yields a non-contiguous view\n"
    "            # because features (dim) are strided by state_len.\n"
    "            raise NotImplementedError(\n"
    "                \"DS conv state layout does not yet support speculative \"\n"
    "                \"decoding with mamba_cache_mode='align' \"\n"
    "                \"(num_accepted_tokens > 1).\"\n"
    "            )\n"
    "        src_state = state[src_block_id]\n"
)

PN30_PART1_REPLACEMENT = (
    "    if is_conv_state_dim_first():\n"
    "        # DS layout: (num_blocks, dim, state_len) — state_len is last.\n"
    "        # [Genesis PN30 v7.68] DS offset>0 is handled in\n"
    "        # collect_mamba_copy_meta (part3), where the destination block\n"
    "        # id is known and a dst-shaped temp can preserve row strides.\n"
    "        # If this path is reached, fail closed rather than corrupt.\n"
    "        # See part3 patch + noonghunna club-3090 commit 9af1a52 for\n"
    "        # the diagnosis of why the v7.65 compact .contiguous() path\n"
    "        # silently corrupted DS conv state row strides.\n"
    "        if offset > 0:\n"
    "            raise RuntimeError(\n"
    "                \"[Genesis PN30 v7.68] DS conv state offset>0 must be \"\n"
    "                \"handled by collect_mamba_copy_meta's dst-shaped temp \"\n"
    "                \"path; refusing compact copy that would corrupt row \"\n"
    "                \"strides. PN30 part3 should be applied — check \"\n"
    "                \"v1/worker/mamba_utils.py for marker.\"\n"
    "            )\n"
    "        src_state = state[src_block_id]\n"
)

# Sub-patch 1b: add module-level state for the temp-tensor list + flag.
# Inserted after `class MambaCopySpec` definition.

PN30_PART1B_ANCHOR = (
    "MambaStateCopyFunc: TypeAlias = Callable[\n"
    "    [torch.Tensor, list[int], int, int], MambaCopySpec\n"
    "]\n"
)

PN30_PART1B_REPLACEMENT = (
    "MambaStateCopyFunc: TypeAlias = Callable[\n"
    "    [torch.Tensor, list[int], int, int], MambaCopySpec\n"
    "]\n"
    "\n"
    "# [Genesis PN30 issue #17] Module-level state for DS layout + spec-decode\n"
    "# AL>1 fix. Temp tensors hold contiguous copies of non-contiguous slices;\n"
    "# cleared by patched do_mamba_copy_block in v1/worker/mamba_utils.py\n"
    "# after batch_memcpy + stream sync. Flag is single-element list (mutable\n"
    "# by reference) to avoid ambiguity with module-level rebinds.\n"
    "_GENESIS_PN30_TEMP_TENSORS: list = []\n"
    "_GENESIS_PN30_FLAG: list = [False]\n"
)


# ─── Sub-patch 2: v1/worker/mamba_utils.py:do_mamba_copy_block ──────
# Wrap to clear PN30 temp-tensor list with stream sync.

PN30_PART2_ANCHOR = (
    "def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):\n"
    "    n = copy_bufs.offset\n"
    "    if n == 0:\n"
    "        return\n"
    "    batch_memcpy(\n"
    "        copy_bufs.src_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.dst_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.sizes.copy_to_gpu(n),\n"
    "    )\n"
)

PN30_PART2_REPLACEMENT = (
    "def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):\n"
    "    n = copy_bufs.offset\n"
    "    if n == 0:\n"
    "        # [Genesis PN30 issue #17] Even on n==0, opportunistic clear of\n"
    "        # leftover DS temp tensors (defensive — should be empty).\n"
    "        try:\n"
    "            from vllm.model_executor.layers.mamba.mamba_utils import (\n"
    "                _GENESIS_PN30_TEMP_TENSORS, _GENESIS_PN30_FLAG,\n"
    "            )\n"
    "            _GENESIS_PN30_TEMP_TENSORS.clear()\n"
    "            _GENESIS_PN30_FLAG[0] = False\n"
    "        except (ImportError, AttributeError):\n"
    "            pass\n"
    "        return\n"
    "    batch_memcpy(\n"
    "        copy_bufs.src_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.dst_ptrs.copy_to_gpu(n),\n"
    "        copy_bufs.sizes.copy_to_gpu(n),\n"
    "    )\n"
    "    # [Genesis PN30 issue #17] If DS layout + offset>0 path was used\n"
    "    # this batch, the contiguous-copy temp tensors are still alive in\n"
    "    # _GENESIS_PN30_TEMP_TENSORS. Sync the stream to ensure batch_memcpy\n"
    "    # consumed them, then clear. Cost: ~10-50us, only fires when DS+\n"
    "    # offset>0 actually triggered (typical AL=1 fast path is no-op).\n"
    "    try:\n"
    "        from vllm.model_executor.layers.mamba.mamba_utils import (\n"
    "            _GENESIS_PN30_TEMP_TENSORS, _GENESIS_PN30_FLAG,\n"
    "        )\n"
    "        if _GENESIS_PN30_FLAG[0]:\n"
    "            import torch as _torch_pn30\n"
    "            _torch_pn30.cuda.current_stream().synchronize()\n"
    "            _GENESIS_PN30_TEMP_TENSORS.clear()\n"
    "            _GENESIS_PN30_FLAG[0] = False\n"
    "    except (ImportError, AttributeError):\n"
    "        pass\n"
)


# ─── Sub-patch 3 (v7.68): collect_mamba_copy_meta dst-shaped temp ────
#
# This is the CORRECT fix for the DS layout + spec-decode AL>1 problem.
# Both src AND dst block ids are available here, so we can build a temp
# shaped like the destination block (preserving DS row stride) and patch
# in only the source tail. Memcpy then sees a normal contiguous full-block
# copy that lands at the right address.
#
# Algorithm:
#   tmp = state[dest_block_id].clone()           # full dst-shaped temp
#   tmp[..., :tail].copy_(state[src_block_id, ..., offset:offset+tail])
#   memcpy: src=tmp.data_ptr(), dst=state[dest].data_ptr(), size=tmp.numel()*esz
#
# This preserves DS row stride end-to-end. Reuses PN30 part1's existing
# `_GENESIS_PN30_TEMP_TENSORS` lifecycle (just changes the temp shape).
#
# Credit: noonghunna + ChatGPT/Codex CLI (club-3090 commit 9af1a52,
# 2026-05-02). Validated on 1×3090 + 2×3090 across all 4 TQ3 composes,
# probes 4 (multi-turn agent) + 5 (LCB-coding) pass cleanly with this
# fix; both crashed with the v7.65 compact-temp path.

PN30_PART3_ANCHOR = (
    "def collect_mamba_copy_meta(\n"
    "    copy_bufs: MambaCopyBuffers,\n"
    "    kv_cache_config: KVCacheConfig,\n"
    "    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],\n"
    "    mamba_group_ids: list[int],\n"
    "    src_block_idx: int,\n"
    "    dest_block_idx: int,\n"
    "    accept_token_bias: int,\n"
    "    req_state: CachedRequestState,\n"
    "    forward_context: dict[str, Any],\n"
    ") -> None:\n"
    "    if src_block_idx == dest_block_idx and accept_token_bias == 0:\n"
    "        return\n"
    "\n"
    "    src_ptrs_np = copy_bufs.src_ptrs.np\n"
    "    dst_ptrs_np = copy_bufs.dst_ptrs.np\n"
    "    sizes_np = copy_bufs.sizes.np\n"
    "    offset = copy_bufs.offset\n"
    "\n"
    "    for mamba_group_id in mamba_group_ids:\n"
    "        block_ids = req_state.block_ids[mamba_group_id]\n"
    "        dest_block_id = block_ids[dest_block_idx]\n"
    "        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names\n"
    "        for layer_name in layer_names:\n"
    "            attention = forward_context[layer_name]\n"
    "            kv_caches: list[torch.Tensor] = attention.kv_cache\n"
    "            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):\n"
    "                copy_spec = state_copy_func(\n"
    "                    state, block_ids, src_block_idx, accept_token_bias + 1\n"
    "                )\n"
    "\n"
    "                src_ptrs_np[offset] = copy_spec.start_addr\n"
    "                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()\n"
    "                sizes_np[offset] = copy_spec.num_elements * state.element_size()\n"
    "                offset += 1\n"
    "\n"
    "    copy_bufs.offset = offset\n"
)

PN30_PART3_REPLACEMENT = (
    "def collect_mamba_copy_meta(\n"
    "    copy_bufs: MambaCopyBuffers,\n"
    "    kv_cache_config: KVCacheConfig,\n"
    "    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],\n"
    "    mamba_group_ids: list[int],\n"
    "    src_block_idx: int,\n"
    "    dest_block_idx: int,\n"
    "    accept_token_bias: int,\n"
    "    req_state: CachedRequestState,\n"
    "    forward_context: dict[str, Any],\n"
    ") -> None:\n"
    "    if src_block_idx == dest_block_idx and accept_token_bias == 0:\n"
    "        return\n"
    "\n"
    "    src_ptrs_np = copy_bufs.src_ptrs.np\n"
    "    dst_ptrs_np = copy_bufs.dst_ptrs.np\n"
    "    sizes_np = copy_bufs.sizes.np\n"
    "    offset = copy_bufs.offset\n"
    "    num_accepted_tokens = accept_token_bias + 1\n"
    "\n"
    "    # [Genesis PN30 v7.68 dst-shaped] Pull PN30 module-level state and\n"
    "    # the conv-copy-spec function reference so we can detect the path\n"
    "    # that needs the dst-shaped temp + bypass the compact-copy entry.\n"
    "    try:\n"
    "        from vllm.model_executor.layers.mamba.mamba_utils import (\n"
    "            _GENESIS_PN30_FLAG,\n"
    "            _GENESIS_PN30_TEMP_TENSORS,\n"
    "            get_conv_copy_spec as _GENESIS_PN30_GET_CONV_COPY_SPEC,\n"
    "            is_conv_state_dim_first as _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST,\n"
    "        )\n"
    "    except (ImportError, AttributeError):\n"
    "        _GENESIS_PN30_FLAG = None\n"
    "        _GENESIS_PN30_TEMP_TENSORS = None\n"
    "        _GENESIS_PN30_GET_CONV_COPY_SPEC = None\n"
    "        _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST = None\n"
    "\n"
    "    for mamba_group_id in mamba_group_ids:\n"
    "        block_ids = req_state.block_ids[mamba_group_id]\n"
    "        dest_block_id = block_ids[dest_block_idx]\n"
    "        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names\n"
    "        for layer_name in layer_names:\n"
    "            attention = forward_context[layer_name]\n"
    "            kv_caches: list[torch.Tensor] = attention.kv_cache\n"
    "            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):\n"
    "                # [Genesis PN30 v7.68] DS layout + offset>0 needs\n"
    "                # dst-shaped temp to preserve row stride. Detect via\n"
    "                # function identity (with __name__ fallback for import\n"
    "                # cycles).\n"
    "                _genesis_pn30_is_conv_copy = (\n"
    "                    state_copy_func is _GENESIS_PN30_GET_CONV_COPY_SPEC\n"
    "                    or getattr(state_copy_func, '__name__', '')\n"
    "                    == 'get_conv_copy_spec'\n"
    "                )\n"
    "                if (\n"
    "                    num_accepted_tokens > 1\n"
    "                    and _genesis_pn30_is_conv_copy\n"
    "                    and _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST is not None\n"
    "                    and _GENESIS_PN30_IS_CONV_STATE_DIM_FIRST()\n"
    "                    and state.dim() >= 3\n"
    "                ):\n"
    "                    # Build dst-shaped temp; copy source tail into prefix.\n"
    "                    # Memcpy lands the full block, preserving DS layout.\n"
    "                    src_block_id = block_ids[src_block_idx]\n"
    "                    token_offset = num_accepted_tokens - 1\n"
    "                    state_len = int(state.shape[-1])\n"
    "                    tail = max(state_len - int(token_offset), 0)\n"
    "                    tmp_state = state[dest_block_id].clone()\n"
    "                    if tail > 0:\n"
    "                        tmp_state[..., :tail].copy_(\n"
    "                            state[\n"
    "                                src_block_id,\n"
    "                                ...,\n"
    "                                token_offset:token_offset + tail,\n"
    "                            ]\n"
    "                        )\n"
    "                    if _GENESIS_PN30_TEMP_TENSORS is not None:\n"
    "                        _GENESIS_PN30_TEMP_TENSORS.append(tmp_state)\n"
    "                    if _GENESIS_PN30_FLAG is not None:\n"
    "                        _GENESIS_PN30_FLAG[0] = True\n"
    "\n"
    "                    src_ptrs_np[offset] = tmp_state.data_ptr()\n"
    "                    dst_ptrs_np[offset] = state[dest_block_id].data_ptr()\n"
    "                    sizes_np[offset] = (\n"
    "                        tmp_state.numel() * state.element_size()\n"
    "                    )\n"
    "                    offset += 1\n"
    "                    continue\n"
    "\n"
    "                copy_spec = state_copy_func(\n"
    "                    state, block_ids, src_block_idx, num_accepted_tokens\n"
    "                )\n"
    "\n"
    "                src_ptrs_np[offset] = copy_spec.start_addr\n"
    "                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()\n"
    "                sizes_np[offset] = copy_spec.num_elements * state.element_size()\n"
    "                offset += 1\n"
    "\n"
    "    copy_bufs.offset = offset\n"
)


def _make_patcher_part1() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/mamba_utils.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 model_executor/layers/mamba/mamba_utils.py — DS layout "
            "spec-decode AL>1 fix (issue #17)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part1",
        sub_patches=[
            TextPatch(
                name="pN30_get_conv_copy_spec_contiguous",
                anchor=PN30_PART1_ANCHOR,
                replacement=PN30_PART1_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pN30_module_level_state",
                anchor=PN30_PART1B_ANCHOR,
                replacement=PN30_PART1B_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN30",
            # If upstream removes NotImplementedError or rewrites the path,
            # our anchor won't match → no-op apply.
        ],
    )


def _make_patcher_part2() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 v1/worker/mamba_utils.py — do_mamba_copy_block stream "
            "sync + temp tensor cleanup (issue #17)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part2",
        sub_patches=[
            TextPatch(
                name="pN30_do_mamba_copy_block_cleanup",
                anchor=PN30_PART2_ANCHOR,
                replacement=PN30_PART2_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN30",
        ],
    )


def _make_patcher_part3() -> TextPatcher | None:
    """v7.68 dst-shaped temp patch on collect_mamba_copy_meta.

    This is the layout-correct fix. part1's old compact path is now
    fail-closed; this part3 path is the live route for DS+offset>0.

    Drift-marker note (v7.68 hotfix): part3 patches the SAME file as
    part2 (`v1/worker/mamba_utils.py`). part2's replacement inserts
    the substring `[Genesis PN30 issue #17]` into that file, so a
    generic `[Genesis PN30` drift marker would false-positive on
    part3's first apply (after part2 runs in the same `apply()` call)
    and fail with `upstream_merged`. Drift markers here must be
    SPECIFIC to part3's own insertion or to known external sidecars.
    Reported by noonghunna 2026-05-02 on club-3090 cross-rig test of
    v7.68 commit `18e65e3`.
    """
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN30 v1/worker/mamba_utils.py — collect_mamba_copy_meta "
            "dst-shaped DS temp (issue #17, v7.68)"
        ),
        target_file=str(target),
        marker=GENESIS_PN30_MARKER + " part3 dst-shaped-temp",
        sub_patches=[
            TextPatch(
                name="pN30_collect_mamba_copy_meta_dst_shaped_temp",
                anchor=PN30_PART3_ANCHOR,
                replacement=PN30_PART3_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # part3-specific marker (matches part3's own replacement
            # comment) — re-runs hit Layer 2 (idempotency) before
            # Layer 3 (drift), so this is correct.
            "[Genesis PN30 v7.68 dst-shaped]",
            # noonghunna's setup-time sidecar (different file, but if
            # someone ports it inline this signal still works)
            "club-3090: PN30 dst-shaped DS temp",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN30 v7.68 — DS layout spec-decode AL>1 fix (3-file text-patch).

    v7.68 supersedes v7.65: part3 (dst-shaped temp on
    collect_mamba_copy_meta) replaces the v7.65 compact `.contiguous()`
    layout-correctness bug. part1 path is now fail-closed.

    All three coordinated patches must apply. In particular, part3
    bypasses PN30's original compact-temp path; without it, DS offset>0
    would corrupt row strides. Treat any required-anchor skip as failed
    when PN30 is explicitly enabled (operator picked an inconsistent
    half-patched state).

    Diagnosis credit (v7.68): noonghunna + ChatGPT/Codex CLI cross-check
    on club-3090, 2026-05-02 (commit 9af1a52).
    """
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN30")
    log_decision("PN30", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # All three coordinated patches must be present.
    p1 = _make_patcher_part1()
    p2 = _make_patcher_part2()
    p3 = _make_patcher_part3()
    if p1 is None or p2 is None or p3 is None:
        return "skipped", (
            "target file(s) not resolvable — vllm tree may differ "
            "from expected layout"
        )

    patch_results = [
        ("part1 mamba_utils.py:get_conv_copy_spec", *p1.apply()),
        (
            "part2 v1/worker/mamba_utils.py:do_mamba_copy_block",
            *p2.apply(),
        ),
        (
            "part3 v1/worker/mamba_utils.py:collect_mamba_copy_meta",
            *p3.apply(),
        ),
    ]
    for label, result, failure in patch_results:
        if result not in (
            TextPatchResult.APPLIED,
            TextPatchResult.IDEMPOTENT,
        ):
            reason_text = failure.reason if failure else "unknown"
            detail = (
                failure.detail
                if failure and failure.detail
                else "unknown"
            )
            return "failed", (
                f"PN30 {label} did not apply safely: "
                f"{reason_text} — {detail}"
            )

    status_result = (
        TextPatchResult.APPLIED
        if any(
            r == TextPatchResult.APPLIED for _, r, _ in patch_results
        )
        else TextPatchResult.IDEMPOTENT
    )
    return result_to_wiring_status(
        status_result,
        None,
        applied_message=(
            "PN30 v7.68 applied: DS conv state layout + spec-decode AL>1 "
            "now uses collect_mamba_copy_meta dst-shaped temp blocks for "
            "DS conv offset>0, preserving destination row stride. "
            "get_conv_copy_spec fails closed if the collect-time bypass "
            "is missed; do_mamba_copy_block keeps PN30's stream sync + "
            "temp clear lifecycle. Supersedes v7.65 compact .contiguous() "
            "approach which silently corrupted DS row strides "
            "(diagnosed by noonghunna/ChatGPT-Codex 2026-05-02)."
        ),
        patch_name="PN30 DS layout + spec-decode AL>1",
    )
