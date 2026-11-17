# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution seam for scheduler-admitted KV repairs.

Ported from the reference 0003-vllm-worker-repair-seam patch (originally
based on the db9527a4 stack) onto the current PR-#68140 scheduler
lifecycle. The scheduler admits RepairWork descriptors on exclusive
repair-only steps; this module executes them: a native layer replay over
each work item's span for layers [0, layer_end) only, guarded by a pre-write
ownership fence against the scheduler-authorized physical write set, with
abort containment (re-zero of partial writes) and a fail-closed TP consensus
on every verdict.

The module lives beside the GPU model runner because the old
kv_connector_model_runner_mixin module no longer exists upstream: its
connector plumbing was absorbed into vllm/v1/worker/gpu/kv_connector.py.
"""

import os
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import RepairResultReport

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def execute_repair_only(
    model: Any,
    scheduler_output: "SchedulerOutput",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    batch_req_ids: list[str],
) -> list[RepairResultReport]:
    """Execute the repair works of one exclusive repair-only step.

    Runs native layer.forward over each work item's repair span for
    layers [0, layer_end) only, discards the output (no logits, no
    sampling), and returns one verdict per work for the caller to attach
    to the step's connector output. Must be called inside the live
    forward context (the caller's execute_model) with the runner's own
    preprocessed inputs; every scheduled request in the step is a repair
    work (the step is exclusive by construction).

    Each work item's rows are its own window of the step's batch
    (batch_req_ids is the row order), each item runs its own layer
    range, and the validation battery (positions, slot ownership, write
    coverage, layer coverage) applies per work item.
    """
    works = scheduler_output.repair_works
    assert works is not None
    # Repair steps are exclusive: no ordinary request rides the batch.
    assert set(works) == set(scheduler_output.num_scheduled_tokens), (
        "hole repair: a repair step must not carry ordinary work "
        f"({len(works)} repair works vs "
        f"{len(scheduler_output.num_scheduled_tokens)} scheduled requests)"
    )
    assert set(batch_req_ids) == set(scheduler_output.num_scheduled_tokens), (
        "hole repair: batch_req_ids must name exactly the repair step's "
        f"requests ({batch_req_ids!r} vs "
        f"{sorted(scheduler_output.num_scheduled_tokens)})"
    )
    global _REPAIR_SYNC_PROFILE
    _REPAIR_SYNC_PROFILE = {} if _repair_sync_profile_enabled() else None

    row_starts: dict[str, int] = {}
    row = 0
    for req_id in batch_req_ids:
        row_starts[req_id] = row
        row += scheduler_output.num_scheduled_tokens[req_id]

    results: list[RepairResultReport] = []
    for work_row, req_id in enumerate(batch_req_ids):
        work = works[req_id]
        lo = row_starts[req_id]
        hi = lo + scheduler_output.num_scheduled_tokens[req_id]
        span_start, span_end = work.repair_span
        pos_first = _int_timed(positions[lo], f"positions[{lo}].int")
        pos_last = _int_timed(positions[hi - 1], f"positions[{hi - 1}].int")
        assert pos_first == span_start and pos_last == span_end - 1, (
            f"hole repair: physical positions [{pos_first},"
            f"{pos_last}] do not match the repair span "
            f"[{span_start}, {span_end}) of {req_id}"
        )

        layers = model.model.layers
        assert work.layer_end <= len(layers), (
            f"hole repair: layer_end {work.layer_end} exceeds the stack "
            f"({len(layers)} layers)"
        )

        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict) and attn_metadata
        meta: Any = next(iter(attn_metadata.values()))
        block_size = _repair_block_size(layers[0])

        # Pre-write safety gate: the slot window and the block-table row
        # are fetched once to the host (a single D2H per work item -- the
        # same lists feed the post-write attestation) and the geometry is
        # validated BEFORE the first cache-writing kernel, so a malformed
        # slot mapping can never write into another request's KV.
        slots_host = _tolist_timed(
            meta.slot_mapping[lo:hi], "pre_write.slot_mapping.tolist"
        )
        block_table_host = _tolist_timed(
            _repair_block_table_row(meta, work_row),
            "pre_write.block_tables.tolist",
        )
        # Ownership fence: the expected physical blocks come from the WORK
        # (the scheduler snapshotted them from KVCacheManager at the
        # admission), NOT from this worker's metadata. The expected
        # physical slots are reconstructed from the authorization and
        # compared against the ACTUAL slot mapping: two
        # consistently-corrupted worker structures (the block table and
        # the slot mapping both pointing at a foreign request's blocks)
        # cannot reproduce the authorized slot multiset, so the foreign
        # write is refused.
        auth_blocks = tuple(work.expected_write_block_ids)
        auth_present = bool(auth_blocks)
        span_blocks = -(-(span_end - span_start) // block_size)
        # The authorization must cover exactly the span's blocks: a
        # partial or over-long snapshot is malformed -- refused, never a
        # crash.
        auth_len_ok = len(auth_blocks) == span_blocks
        expected_slots: Counter[int] = Counter()
        if auth_present and auth_len_ok:
            first_blk = span_start // block_size
            for tok in range(span_start, span_end):
                block_id = auth_blocks[(tok // block_size) - first_blk]
                expected_slots[block_id * block_size + tok % block_size] += 1
        auth_slots_ok = (
            auth_present and auth_len_ok and Counter(slots_host) == expected_slots
        )
        # The worker block table's span slice must AGREE with the
        # authorization (an independent check of the row itself).
        table_span = block_table_host[
            span_start // block_size : -(-span_end // block_size)
        ]
        auth_table_ok = auth_present and list(table_span) == list(auth_blocks)
        pre_write_ownership = _slots_owned_by_request_list(
            slots_host, block_table_host, block_size
        )
        pre_write_coverage = _slots_cover_span_list(
            slots_host, block_table_host, span_start, span_end, block_size
        )
        pre_write_ok = (
            pre_write_ownership
            and pre_write_coverage
            and auth_present
            and auth_slots_ok
            and auth_table_ok
        )
        if not pre_write_ok:
            logger.error(
                "Repair pre-write fence refused req=%s plan_id=%d "
                "span=[%d,%d) ownership=%s coverage=%s auth_present=%s "
                "auth_slots=%s auth_table=%s -- no cache write executed",
                work.request_id,
                work.plan_id,
                span_start,
                span_end,
                pre_write_ownership,
                pre_write_coverage,
                auth_present,
                auth_slots_ok,
                auth_table_ok,
            )

        hidden_states = model.model.embed_input_ids(input_ids[lo:hi])
        residual = None
        executed_layers = 0
        backend_ok = True
        if pre_write_ok:
            for idx, layer in enumerate(layers[: work.layer_end]):
                try:
                    hidden_states, residual = layer(
                        positions[lo:hi], hidden_states, residual
                    )
                except Exception:
                    # Exception containment: a failure inside the repair
                    # stream is request-local. Stop the stream here, re-zero
                    # the executed layers' partial writes (the abort handler
                    # below), report the failure, and let the scheduler's
                    # recompute fallback own the request. A poisoned device
                    # context fails loudly at the next step; a successful
                    # transaction is never fabricated.
                    logger.exception(
                        "Repair layer exception req=%s idx=%d span=[%d,%d) "
                        "-> contained as abort",
                        work.request_id,
                        idx,
                        span_start,
                        span_end,
                    )
                    backend_ok = False
                    break
                executed_layers += 1
        del hidden_states, residual

        aborted = executed_layers < work.layer_end
        if aborted:
            _rezero_repair_span(layers[:executed_layers], meta, lo, hi)

        commit_check = {
            "transfer_complete": not aborted,
            # The post-write attestation reuses the pre-write host lists
            # (the slot mapping is immutable across the forward): the
            # executed writes targeted exactly this request's blocks and
            # exactly the span; the pre-write gate above is what PREVENTED
            # a malformed geometry from ever executing.
            "pre_write_geometry": pre_write_ok,
            "authorized_ownership": auth_present and auth_slots_ok,
            "authorized_table": auth_table_ok,
            "slot_ownership": pre_write_ownership,
            "write_coverage": pre_write_coverage,
            "layer_coverage": executed_layers == work.layer_end,
            "backend_ok": backend_ok,
            # TP consensus, fail-closed: every rank executes the SAME works
            # (the SchedulerOutput is broadcast) and all-gathers its packed
            # verdict summary. The transaction commits only when every rank
            # agrees and succeeded -- a single failed or divergent rank
            # aborts the transaction on EVERY rank, never the driver's
            # verdict alone.
            "ranks_agree": _ranks_repair_agree(
                aborted,
                executed_layers,
                pre_write_ok,
                auth_present and auth_slots_ok,
                auth_table_ok,
                pre_write_ownership,
                pre_write_coverage,
                work.layer_end,
                backend_ok,
            ),
            "identity_ok": work.plan_id >= 0 and work.epoch >= 0,
        }
        structural_check = all(commit_check.values())
        result = RepairResultReport(
            request_id=work.request_id,
            plan_id=work.plan_id,
            epoch=work.epoch,
            structural_check=structural_check,
            executed_layers=executed_layers,
            span=(span_start, span_end),
            aborted=aborted,
            commit_check=commit_check,
        )
        logger.info(
            "Repair executed req=%s plan_id=%d epoch=%d span=[%d,%d) "
            "layer_end=%d executed_layers=%d aborted=%s "
            "structural_check=%s attn_state=%s",
            work.request_id,
            work.plan_id,
            work.epoch,
            span_start,
            span_end,
            work.layer_end,
            executed_layers,
            aborted,
            structural_check,
            str(getattr(meta, "attn_state", None)),
        )
        results.append(result)
    if _REPAIR_SYNC_PROFILE is not None:
        profile = {k: round(v * 1e3, 3) for k, v in _REPAIR_SYNC_PROFILE.items()}
        logger.info(
            "Repair sync profile total_ms=%.3f ops=%s",
            sum(profile.values()),
            profile,
        )
        _REPAIR_SYNC_PROFILE = None
    return results


# Per-repair-step sync profile: {label: accumulated seconds}. None (the
# production default) makes the _*_timed helpers pass-through calls.
_REPAIR_SYNC_PROFILE: dict[str, float] | None = None


def _repair_sync_profile_enabled() -> bool:
    return os.environ.get("KV_REPAIR_SYNC_PROFILE") == "1"


def _tolist_timed(t: torch.Tensor, label: str) -> list:
    if _REPAIR_SYNC_PROFILE is None:
        return t.tolist()
    t0 = time.perf_counter()
    out = t.tolist()
    _record_sync(label, time.perf_counter() - t0)
    return out


def _item_timed(t: torch.Tensor, label: str) -> int:
    if _REPAIR_SYNC_PROFILE is None:
        return int(t)
    t0 = time.perf_counter()
    out = int(t)
    _record_sync(label, time.perf_counter() - t0)
    return out


def _int_timed(t: torch.Tensor, label: str) -> int:
    return _item_timed(t, label)


def _record_sync(label: str, seconds: float) -> None:
    if _REPAIR_SYNC_PROFILE is not None:
        _REPAIR_SYNC_PROFILE[label] = _REPAIR_SYNC_PROFILE.get(label, 0.0) + seconds


def _tp_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _ranks_repair_agree(
    aborted: bool,
    executed_layers: int,
    pre_write_ok: bool,
    auth_ownership_ok: bool,
    auth_table_ok: bool,
    slot_ownership: bool,
    write_coverage: bool,
    layer_end: int,
    backend_ok: bool,
) -> bool:
    """The TP cross-rank repair agreement (fail-closed).

    Every rank executes the same repair works and reaches its own local
    verdict. Each rank packs its verdict summary into one int64 tensor
    and all-gathers it: the transaction succeeds only when EVERY rank's
    summary is identical to the local one AND every rank's summary is a
    success (not aborted, full layer coverage, all local checks green).
    A single failed or divergent rank aborts the transaction on every
    rank -- the driver's verdict alone never commits.
    """
    ws = _tp_world_size()
    if ws <= 1:
        return True
    summary = torch.tensor(
        [
            int(aborted),
            int(executed_layers),
            int(pre_write_ok),
            int(auth_ownership_ok),
            int(auth_table_ok),
            int(slot_ownership),
            int(write_coverage),
            int(layer_end),
            int(backend_ok),
        ],
        dtype=torch.int64,
    )
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        summary = summary.to("npu")
    world = [torch.empty_like(summary) for _ in range(ws)]
    torch.distributed.all_gather(world, summary)
    local_success = (
        not aborted
        and executed_layers == layer_end
        and pre_write_ok
        and auth_ownership_ok
        and auth_table_ok
        and slot_ownership
        and write_coverage
        and backend_ok
    )
    for remote in world:
        if not torch.equal(remote, summary):
            return False
        remote_success = (
            remote[0].item() == 0
            and int(remote[1].item()) == int(remote[7].item())
            and all(int(remote[i].item()) == 1 for i in (2, 3, 4, 5, 6, 8))
        )
        if not remote_success:
            return False
    return local_success


def _slots_owned_by_request_list(
    slots_host: list[int], block_table_host: list[int], block_size: int
) -> bool:
    """Ownership check over the host-side slot and block-table lists:
    every span slot lives inside a block the work's own block-table row
    names."""
    owned = set(block_table_host)
    return all((s // block_size) in owned for s in slots_host)


def _slots_cover_span_list(
    slots_host: list[int],
    block_table_host: list[int],
    span_start: int,
    span_end: int,
    block_size: int,
) -> bool:
    """Multiset coverage check over the host-side lists: the expected
    slots are derived from the block table per span token and the actual
    slot window must match exactly (a duplicated slot aliases two tokens
    onto one cache row). The block table row lists the request's blocks
    from its block 0, so the absolute block index (tok // block_size)
    indexes it directly."""
    expected: Counter[int] = Counter()
    for tok in range(span_start, span_end):
        block_id = block_table_host[tok // block_size]
        expected[block_id * block_size + tok % block_size] += 1
    return Counter(slots_host) == expected


def _slots_owned_by_request(
    meta, block_size: int, row: int = 0, lo: int = 0, hi: int | None = None
) -> bool:
    """Commit-check element: every span slot lives inside a block the
    request's own block table names. ``row`` is the work item's row in
    the step's block tables; the slot window is the item's [lo, hi) rows
    of the step's slot mapping."""
    owned = {
        int(b)
        for b in _tolist_timed(
            _repair_block_table_row(meta, row), "ownership.block_tables.tolist"
        )
    }
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    for s in _tolist_timed(slots, "ownership.slot_mapping.tolist"):
        if int(s) // block_size not in owned:
            return False
    return True


def _slots_cover_span(
    meta,
    block_size: int,
    span_start: int,
    span_end: int,
    row: int = 0,
    lo: int = 0,
    hi: int | None = None,
) -> bool:
    """Commit-check element: the work item's slot window covers exactly
    the span's slots, each expected row exactly once. A multiset
    comparison: a duplicated slot (two tokens aliased to one cache row,
    silently losing one row's KV) must fail the check, which a set
    comparison passes. ``row`` is the item's block-table row; the actual
    slots are the item's [lo, hi) rows of the step's slot mapping."""
    expected: Counter[int] = Counter()
    block_table = _repair_block_table_row(meta, row)
    for tok in range(span_start, span_end):
        block_idx = tok // block_size
        block_id = _item_timed(block_table[block_idx], "coverage_expected.item")
        expected[block_id * block_size + tok % block_size] += 1
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    actual = Counter(
        int(s) for s in _tolist_timed(slots, "coverage.slot_mapping.tolist")
    )
    return actual == expected


def _repair_cache_views(layer) -> list[torch.Tensor]:
    """The per-layer KV cache tensor(s) that the repair re-zero writes.

    A layer's cache is backend-dependent: vLLM core binds one 4D
    `[B, H, N, C]` tensor per layer (K and V share it, split along the
    channel dim); the Ascend backend binds a `[2, ...]`-stacked (or
    two-element) cache whose leading dim separates K and V. Both forms
    are indexed by the manager block id on dim 0, so re-zeroing selects
    `cache[bidx, oidx]` rows through the token dim that follows.
    """
    cache = layer.self_attn.attn.kv_cache
    if isinstance(cache, (list, tuple)):
        return list(cache)
    if isinstance(cache, torch.Tensor) and cache.dim() >= 1 and cache.shape[0] == 2:
        return [cache[0], cache[1]]
    return [cache]


def _repair_block_size(layer) -> int:
    """The manager block size of a repair-written KV cache layer.

    Derives the token capacity of one block from the cache geometry:
    the 4D core layout `[B, H, N, C]` carries the block size on dim 2
    (`N`, tokens per block); the Ascend layouts carry it on dim 2 of the
    per-K/V views (`[num_blocks, block_size, ...]` after the leading
    K/V dim is peeled).
    """
    caches = _repair_cache_views(layer)
    cache = caches[0]
    dim = cache.dim()
    # [B, H, N, C] core layout: tokens-per-block is dim 2.
    if dim == 4:
        return int(cache.shape[2])
    # Ascend [num_blocks, block_size, heads, dim] and older
    # [num_blocks, block_size, x] forms: tokens-per-block is dim 1.
    if dim in (2, 3, 4):
        return int(cache.shape[1])
    raise ValueError(f"unrecognized repair KV cache geometry: {tuple(cache.shape)}")


def _repair_block_table_row(meta, row: int) -> torch.Tensor:
    """The `row`-th request's block-table row from attention metadata.

    Backends name the batched block table differently: the Ascend
    metadata exposes `block_tables` ([num_reqs, max_blocks]); the CUDA
    backends expose `block_table` (same shape). Both carry the manager
    block ids for one request per row.
    """
    table = getattr(meta, "block_tables", None)
    if table is None:
        table = getattr(meta, "block_table", None)
    if table is None:
        raise AttributeError(
            "repair metadata exposes neither block_tables nor block_table"
        )
    return table[row]


def _rezero_repair_span(layers, meta, lo: int = 0, hi: int | None = None) -> None:
    """Abort-path partial-write invalidation: re-zero the span's slot rows
    in every executed layer's K and V cache. A partial repair is not a
    valid prefix; no partial content is left readable.

    The write indexes the cache tensor's own block and token dims -- the
    per-layer cache can be a non-contiguous view (a reshape would copy,
    and the zero would land in the copy). The core 4D `[B, H, N, C]`
    layout carries the token dim at index 2; the Ascend per-K/V views
    carry it at index 1. The slot window is the work item's [lo, hi)
    rows of the step's slot mapping.
    """
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    slots = slots.long()
    max_abs_after = 0.0
    for layer in layers:
        caches = _repair_cache_views(layer)
        block_size = _repair_block_size(layer)
        bidx = slots // block_size
        oidx = slots % block_size
        for cache in caches:
            if cache.dim() == 4 and cache.shape[2] == block_size:
                # Core [B, H, N, C]: zero every head row of the block.
                cache[bidx, :, oidx, :] = 0
            else:
                cache[bidx, oidx] = 0
        probe = caches[0]
        if probe.dim() == 4 and probe.shape[2] == block_size:
            k_rows = probe[bidx, :, oidx, :]
        else:
            k_rows = probe[bidx, oidx]
        max_abs_after = max(max_abs_after, float(k_rows.abs().max()))
    logger.info(
        "Repair abort re-zero layers=%d rows=%d max_abs_after=%.6f",
        len(layers),
        int(slots.numel()),
        max_abs_after,
    )
