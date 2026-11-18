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

Known limitation: kernel-block-split cache groups (AttentionGroup
kernel_block_size below the manager block size) bind layer views whose
derived block unit is the kernel block, not the manager unit the
authorization uses; the fence detects the unit mismatch and refuses every
such repair loudly (the request falls back to recompute). Supporting those
layouts needs the manager unit plumbed into the work descriptor.

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
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import RepairWork
from vllm.v1.outputs import RepairResultReport

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


def _format_repair_exception(exc: BaseException) -> str:
    """The envelope's diagnostic record of a contained exception: the
    exception's type and message (the full traceback goes to the log via
    logger.exception). Kept on the RepairResultReport so the scheduler
    side sees the original failure even when worker logs are lost."""
    text = type(exc).__name__ + ": " + str(exc)
    return text if len(text) <= 300 else text[:297] + "..."


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
    global _REPAIR_SYNC_PROFILE
    _REPAIR_SYNC_PROFILE = {} if _repair_sync_profile_enabled() else None
    try:
        # Repair steps are exclusive: no ordinary request rides the batch.
        # The works and the scheduling counts are broadcast data (identical
        # on every rank); only batch_req_ids is worker-local. A step-level
        # violation on ONE rank converges to failed verdicts on every rank,
        # but NOT promptly: the failing rank skips ahead to its verdict
        # collective while peers already inside a layer collective block
        # there until the backend timeout (the process-group timeout, the
        # NCCL watchdog, or the executor RPC bound), and only then do the
        # failed verdicts and their collectives proceed.
        assert set(works) == set(scheduler_output.num_scheduled_tokens), (
            "hole repair: a repair step must not carry ordinary work "
            f"({len(works)} repair works vs "
            f"{len(scheduler_output.num_scheduled_tokens)} scheduled requests)"
        )
        assert len(batch_req_ids) == len(scheduler_output.num_scheduled_tokens) and set(
            batch_req_ids
        ) == set(scheduler_output.num_scheduled_tokens), (
            "hole repair: batch_req_ids must name exactly the repair step's "
            f"requests, each once ({batch_req_ids!r} vs "
            f"{sorted(scheduler_output.num_scheduled_tokens)})"
        )
    except Exception as exc:
        step_error = _format_repair_exception(exc)
        logger.exception(
            "Repair step validation failed (%d works) -> failed verdicts "
            "for every work",
            len(works),
        )
        step_results: list[RepairResultReport] = []
        for step_work in works.values():
            try:
                _ranks_repair_agree(
                    True,
                    0,
                    False,
                    False,
                    False,
                    False,
                    False,
                    step_work.layer_end,
                    False,
                    plan_id=step_work.plan_id,
                    epoch=step_work.epoch,
                    span=step_work.repair_span,
                )
            except Exception as collect_exc:
                logger.exception(
                    "Repair TP consensus failed req=%s -> abort",
                    step_work.request_id,
                )
                step_error = step_error or _format_repair_exception(collect_exc)
            step_results.append(
                RepairResultReport(
                    request_id=step_work.request_id,
                    plan_id=step_work.plan_id,
                    epoch=step_work.epoch,
                    structural_check=False,
                    executed_layers=0,
                    span=step_work.repair_span,
                    aborted=True,
                    commit_check=None,
                    error_detail=step_error,
                )
            )
        if _REPAIR_SYNC_PROFILE is not None:
            _REPAIR_SYNC_PROFILE = None
        return step_results

    row_starts: dict[str, int] = {}
    row = 0
    for req_id in batch_req_ids:
        row_starts[req_id] = row
        # A worker-local batch corruption (a missing scheduling entry) must
        # not strand the TP peers in the FIRST collective: the missing entry
        # degrades to a zero-width window, which the per-work envelope below
        # contains as that work's failed verdict while the other works run
        # normally.
        row += scheduler_output.num_scheduled_tokens.get(req_id, 0)

    results: list[RepairResultReport] = []
    for work_row, req_id in enumerate(batch_req_ids):
        # Failure envelope: a rank-local recoverable exception anywhere
        # between the work lookup and the verdict construction (positions
        # fetch, metadata access, the host-side fence, embedding, the layer
        # stream, the re-zero, or the verdict collective itself) must not
        # ESCAPE this function: it is captured here, preserved as
        # error_detail, and converged into a consistent failed verdict that
        # still participates in consensus. The VERDICT collective is
        # never skipped; peers already blocked inside an inner LAYER
        # collective at the moment of the exception still wait for that
        # collective's backend timeout -- containment is fail-closed, not
        # latency-free.
        work: RepairWork | None = None
        span_start, span_end = -1, -1
        lo = row_starts.get(req_id, 0)
        hi = lo + scheduler_output.num_scheduled_tokens.get(req_id, 0)
        executed_layers = 0
        layer_failed_at: int | None = None
        backend_ok = True
        pre_write_ok = False
        auth_present = False
        auth_slots_ok = False
        auth_table_ok = False
        auth_distinct_ok = False
        pre_write_ownership = False
        pre_write_coverage = False
        error_detail: str | None = None
        meta: Any = None
        layers: Any = None
        try:
            work = works[req_id]
            span_start, span_end = work.repair_span
            # A zero-width window is malformed (and positions[hi - 1]
            # with hi == 0 would read the whole batch's LAST row via
            # negative indexing).
            assert hi > lo, (
                f"hole repair: zero-width repair window for {req_id} "
                f"(num_scheduled_tokens="
                f"{scheduler_output.num_scheduled_tokens.get(req_id, 0)})"
            )
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
            meta = next(iter(attn_metadata.values()))
            block_size = _repair_block_size(layers[0])

            # Pre-write safety gate: the slot window and the block-table row
            # are fetched once to the host (two D2H copies per work item --
            # the same lists feed the post-write attestation) and the geometry is
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
            # Grid range, matching the scheduler's
            # _repair_write_set_block_ids slice
            # (req_block_ids[rollback // bs : ceil(corrupt_end / bs)]): a
            # span starting mid-block covers one block more than its
            # rounded-up length.
            first_blk = span_start // block_size
            span_blocks = -(-span_end // block_size) - first_blk
            # The authorization must cover exactly the span's blocks: a
            # partial or over-long snapshot is malformed -- refused, never a
            # crash.
            auth_len_ok = len(auth_blocks) == span_blocks
            # An aliased authorization (one block named twice) would alias
            # two span tokens onto one cache row while every multiset
            # derived from it carries the same alias: the blocks must be
            # distinct.
            auth_distinct_ok = len(set(auth_blocks)) == len(auth_blocks)
            expected_slots: Counter[int] = Counter()
            if auth_present and auth_len_ok:
                for tok in range(span_start, span_end):
                    block_id = auth_blocks[(tok // block_size) - first_blk]
                    expected_slots[block_id * block_size + tok % block_size] += 1
            auth_slots_ok = (
                auth_present
                and auth_len_ok
                and auth_distinct_ok
                and Counter(slots_host) == expected_slots
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
                    "auth_slots=%s auth_table=%s auth_distinct=%s -- no "
                    "cache write executed",
                    work.request_id,
                    work.plan_id,
                    span_start,
                    span_end,
                    pre_write_ownership,
                    pre_write_coverage,
                    auth_present,
                    auth_slots_ok,
                    auth_table_ok,
                    auth_distinct_ok,
                )

            hidden_states = model.model.embed_input_ids(input_ids[lo:hi])
            residual = None
            if pre_write_ok:
                for idx, layer in enumerate(layers[: work.layer_end]):
                    try:
                        hidden_states, residual = layer(
                            positions[lo:hi], hidden_states, residual
                        )
                    except Exception as exc:
                        # Exception containment: a failure inside the repair
                        # stream is request-local. Stop the stream here,
                        # re-zero the partial writes INCLUDING this raising
                        # layer's rows (it may have written some KV rows
                        # before raising), report the failure, and let the
                        # scheduler's recompute fallback own the request. A
                        # poisoned device context fails loudly at the next
                        # step; a successful transaction is never fabricated.
                        layer_failed_at = idx
                        backend_ok = False
                        error_detail = _format_repair_exception(exc)
                        logger.exception(
                            "Repair layer exception req=%s idx=%d span=[%d,%d) "
                            "-> contained as abort",
                            work.request_id,
                            idx,
                            span_start,
                            span_end,
                        )
                        break
                    executed_layers += 1
            del hidden_states, residual
        except Exception as exc:
            # The envelope: any rank-local recoverable failure OUTSIDE the
            # per-layer containment (the work lookup itself, position
            # fetch, metadata access, host fence, embedding, re-zero)
            # converges to the failed verdict below instead of escaping and
            # stranding the TP ranks in consensus.
            backend_ok = False
            if error_detail is None:
                error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair pre-verdict exception req=%s plan_id=%s span=[%d,%d) "
                "-> contained as abort",
                req_id,
                work.plan_id if work is not None else "unknown",
                span_start,
                span_end,
            )

        if work is None:
            # The work lookup itself failed on this rank: no geometry, no
            # identity. Still enter the collective with a placeholder
            # identity -- the peers carry the real one, the summaries
            # diverge, and consensus fails everywhere (fail-closed). The
            # scheduler's identity check rejects the placeholder echo as
            # stale, and its missing-verdict synthesis aborts the plan.
            # The collective here is REQUIRED: this rank's per-work loop
            # iteration count matches the peers', so skipping it would
            # leave every peer's next collective unpaired.
            if error_detail is None:
                error_detail = "RepairWork lookup failed on this rank"
            logger.error(
                "Repair work lookup failed req=%s -> placeholder failed "
                "verdict (error_detail=%s)",
                req_id,
                error_detail,
            )
            try:
                _ranks_repair_agree(
                    True,
                    0,
                    False,
                    False,
                    False,
                    False,
                    False,
                    0,
                    False,
                    plan_id=-1,
                    epoch=-1,
                    span=(-1, -1),
                )
            except Exception:
                logger.exception("Repair TP consensus failed req=%s -> abort", req_id)
            results.append(
                RepairResultReport(
                    request_id=req_id,
                    plan_id=-1,
                    epoch=-1,
                    structural_check=False,
                    executed_layers=0,
                    span=(-1, -1),
                    aborted=True,
                    commit_check=None,
                    error_detail=error_detail,
                )
            )
            continue

        aborted = executed_layers < work.layer_end or error_detail is not None
        if aborted and meta is not None and layers is not None:
            # Best-effort partial-write invalidation. Depth: the raising
            # layer is included (it may have written rows before raising);
            # a failure before the layer loop wrote nothing (depth 0, no
            # kernel launched). A clean full-depth run never reaches this
            # block. The scheduler-side eviction is the authoritative
            # invalidation -- this zeroing must never mask its failure, so
            # its own exception is contained too.
            depth = (
                min(layer_failed_at + 1, work.layer_end)
                if layer_failed_at is not None
                else executed_layers
            )
            if depth > 0:
                try:
                    _rezero_repair_span(layers[:depth], meta, lo, hi)
                except Exception as exc:
                    if error_detail is None:
                        error_detail = _format_repair_exception(exc)
                    logger.exception(
                        "Repair abort re-zero failed req=%s plan_id=%d "
                        "span=[%d,%d) depth=%d (scheduler-side eviction "
                        "remains the authoritative invalidation)",
                        work.request_id,
                        work.plan_id,
                        span_start,
                        span_end,
                        depth,
                    )

        ranks_agree = False
        try:
            # TP consensus, fail-closed: every rank executes the SAME works
            # (the SchedulerOutput is broadcast) and all-gathers its packed
            # verdict summary -- outcome flags AND the work's identity. The
            # transaction commits only when every rank agrees on identity
            # AND outcome. This call is UNCONDITIONAL on every path the
            # envelope controls: skipping it on one rank would deadlock the
            # others inside the collective.
            ranks_agree = _ranks_repair_agree(
                aborted,
                executed_layers,
                pre_write_ok,
                auth_present and auth_slots_ok,
                auth_table_ok,
                pre_write_ownership,
                pre_write_coverage,
                work.layer_end,
                backend_ok,
                plan_id=work.plan_id,
                epoch=work.epoch,
                span=work.repair_span,
            )
        except Exception as exc:
            # The collective itself failed (process-group error, or a
            # backend timeout): consensus is unachievable, which by
            # fail-closure is a failed verdict.
            if error_detail is None:
                error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair TP consensus failed req=%s plan_id=%d -> abort",
                work.request_id,
                work.plan_id,
            )

        # The final abort flag: the run stopped short of its layer range,
        # OR the envelope contained any exception (including a verdict-
        # phase or collective failure after a clean run -- the transaction
        # did not complete, so its writes must be treated as partial
        # forever after).
        aborted = aborted or error_detail is not None
        commit_check = {
            "transfer_complete": not aborted,
            # The post-write attestation reuses the pre-write host lists
            # (the slot mapping is immutable across the forward): the
            # executed writes targeted exactly this request's blocks and
            # exactly the span; the pre-write gate above is what PREVENTED
            # a malformed geometry from ever executing.
            "pre_write_geometry": pre_write_ok,
            "authorized_ownership": auth_present and auth_slots_ok,
            "authorized_distinct": auth_distinct_ok,
            "authorized_table": auth_table_ok,
            "slot_ownership": pre_write_ownership,
            "write_coverage": pre_write_coverage,
            "layer_coverage": executed_layers == work.layer_end,
            "backend_ok": backend_ok,
            "ranks_agree": ranks_agree,
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
            error_detail=error_detail,
        )
        logger.info(
            "Repair executed req=%s plan_id=%d epoch=%d span=[%d,%d) "
            "layer_end=%d executed_layers=%d aborted=%s "
            "structural_check=%s attn_state=%s error_detail=%s",
            work.request_id,
            work.plan_id,
            work.epoch,
            span_start,
            span_end,
            work.layer_end,
            executed_layers,
            aborted,
            structural_check,
            str(getattr(meta, "attn_state", None)) if meta is not None else None,
            error_detail,
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
    plan_id: int = -1,
    epoch: int = -1,
    span: tuple[int, int] = (-1, -1),
) -> bool:
    """The TP cross-rank repair agreement (fail-closed).

    Every rank executes the same repair works and reaches its own local
    verdict. Each rank packs its verdict summary -- the outcome flags AND
    the work's identity (plan_id, epoch, span) -- into one int64 tensor
    and all-gathers it: the transaction succeeds only when EVERY rank's
    summary is identical to the local one AND every rank's summary is a
    success (not aborted, full layer coverage, all local checks green).
    A single failed or divergent rank aborts the transaction on every
    rank -- the driver's verdict alone never commits. The identity fields
    make a rank executing stale or foreign work (a divergent broadcast,
    a replayed schedule) abort on every rank instead of committing a
    verdict whose geometry happened to agree.
    """
    ws = _tp_world_size()
    if ws <= 1:
        # Uniform semantics: with one rank, "every rank agrees and
        # succeeded" reduces to the local summary being a success --
        # the same predicate the multi-rank loop derives from the
        # gathered summaries.
        return (
            not aborted
            and executed_layers == layer_end
            and pre_write_ok
            and auth_ownership_ok
            and auth_table_ok
            and slot_ownership
            and write_coverage
            and backend_ok
        )
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
            int(plan_id),
            int(epoch),
            int(span[0]),
            int(span[1]),
        ],
        dtype=torch.int64,
    )
    # The summary must ride the process group's accelerator when there
    # is one: the default world group is NCCL/HCCN on GPU and NPU hosts,
    # which only accept device tensors. Derive the device from the
    # platform (portable across cuda/npu/xpu) rather than referencing a
    # vendor backend; on a CPU-only host the group is gloo and the CPU
    # tensor is already correct.
    device_type = current_platform.device_type
    if device_type:
        summary = summary.to(device_type)
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
    request's own block table names. `row` is the work item's row in
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
    comparison passes. `row` is the item's block-table row; the actual
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
    channel dim); the Ascend backend binds a two-element (or
    `[2, ...]`-stacked) cache whose leading dim separates K and V. A
    single tensor is NEVER split on its leading shape -- a core-layout
    cache with exactly two blocks must not be mistaken for the Ascend
    stacked form -- only explicit list/tuple bindings and a genuine
    `[2, B, ...]` 3-D stack are treated as (K, V) pairs.
    """
    cache = layer.self_attn.attn.kv_cache
    if isinstance(cache, (list, tuple)):
        return list(cache)
    if isinstance(cache, torch.Tensor) and cache.dim() == 3 and cache.shape[0] == 2:
        if cache.shape[1] == cache.shape[2]:
            # [2, x, x] is ambiguous between the core
            # [num_blocks=2, block_size, C] layout and the Ascend
            # [2, num_blocks, block_size] stack: both readings derive the
            # same block size, so the fence cannot separate them. Refuse
            # rather than guess (a wrong guess re-zeros the wrong cells).
            raise ValueError(
                "ambiguous 3-D repair KV cache geometry "
                f"{tuple(cache.shape)}: cannot distinguish the core "
                "[2, block_size, C] layout from the Ascend "
                "[2, num_blocks, block_size] K/V stack"
            )
        # The Ascend stacked [2, num_blocks, tokens] form: dim 0 is the K/V
        # selector, not a block index (a core-layout view is 4-D).
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

    Backends name the batched block table differently: the CUDA backends
    expose `block_table` (FlashAttention), the Ascend metadata
    `block_tables`, and the HPC backend `block_table_tensor` -- all the
    same `[num_reqs, max_blocks]` manager-block-id table. FlashInfer
    keeps its tables inside prefill/decode sub-objects and exposes no
    top-level field; such backends fail here with a named error, which
    the failure envelope converts into that work's failed verdict.
    """
    table = getattr(meta, "block_tables", None)
    if table is None:
        table = getattr(meta, "block_table", None)
    if table is None:
        table = getattr(meta, "block_table_tensor", None)
    if table is None:
        raise AttributeError(
            "repair metadata exposes none of block_tables/block_table/"
            "block_table_tensor (backend "
            + type(meta).__name__
            + "); its repair plan cannot be fenced"
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
