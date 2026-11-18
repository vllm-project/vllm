# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution seam for scheduler-admitted KV repairs.

Ported from the reference 0003-vllm-worker-repair-seam patch (originally
based on the db9527a4 stack) onto the current PR-#68140 scheduler
lifecycle. The scheduler admits RepairWork descriptors on exclusive
repair-only steps; this module executes them: a native layer replay over
each work item's span for layers [0, layer_end) only, guarded by a pre-write
ownership fence against the scheduler-authorized physical write set, a
distributed PRE-EXECUTION preflight consensus (every TP rank all-gathers the
repair operation identity -- capability reason, plan_id, epoch, span, layer
range, and authorized write geometry -- BEFORE any rank enters native layer
execution, so a rank-local refusal can never strand a peer inside a native
TP collective), abort containment (re-zero of partial writes), and a
fail-closed TP consensus on every verdict.

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

import dataclasses
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


# Repair preflight reason codes (the pre-execution consensus' capability
# field). Stable across processes: the code rides the all-gathered preflight
# summary tensor, so every rank sees not just THAT a peer refused but WHY.
# 0 is the only "locally authorized" value; the seam accepts external
# capability probes (a model-capability verdict computed off-process) as
# distinct nonzero codes.
_REPAIR_PF_OK = 0
_REPAIR_PF_STEP_INVALID = 1
_REPAIR_PF_WORK_LOOKUP = 2
_REPAIR_PF_GEOMETRY = 3
_REPAIR_PF_FENCE = 4
_REPAIR_PF_METADATA = 5
_REPAIR_PF_CAPABILITY = 6


class _ReplayProbeAbort(BaseException):
    """Stops the compatibility probe's forward at layer 0, before any
    attention or KV write runs (BaseException so a model-internal
    except-Exception cannot swallow it)."""


def verify_replay_compatibility(model: Any) -> None:
    """Refuse the repair policy unless the model's own forward reduces to
    exactly what the replay executes: embed_input_ids feeding
    layer(positions, hidden_states, residual).

    Drives one tiny forward of the inner model with a pre-hook on layer
    0 that captures the layer's actual arguments and aborts the call
    before any attention or cache write runs, then checks that the
    layer is called with exactly the replay's three positional arguments
    (models whose loop supplies per-layer state -- e.g. per-layer-input
    embeddings -- would silently diverge in the replay) and that the
    hidden states fed to layer 0 equal embed_input_ids of the same
    tokens (models that transform embeddings after the embed call --
    e.g. muP input scaling -- would silently diverge too).

    The probe sees only the layer-call shape and the layer-0 input:
    state the model forward sets on itself and layers read internally
    (per-forward model state) is invisible to it and remains outside the
    verified set.

    Raises:
        RuntimeError: with the model-agnostic reason when either check
            fails; the engine then refuses to start under
            kv_load_failure_policy='repair'.

    """
    inner = model.model
    layers = getattr(inner, "layers", None)
    if not layers:
        raise RuntimeError(
            "kv_load_failure_policy='repair' requires a native model with "
            "a layer stack (none found on the inner model)"
        )
    embed_fn = getattr(inner, "embed_input_ids", None)
    if embed_fn is None:
        raise RuntimeError(
            "kv_load_failure_policy='repair' requires a model exposing "
            "embed_input_ids (the replay's embedding seam); switch the "
            "policy to 'recompute'"
        )
    try:
        device = next(inner.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    ids = torch.zeros(4, dtype=torch.long, device=device)
    positions = torch.arange(4, dtype=torch.int64, device=device)
    with torch.no_grad():
        embed = embed_fn(ids)
    captured: dict[str, Any] = {}

    def _probe(_module, args, kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        raise _ReplayProbeAbort()

    handle = layers[0].register_forward_pre_hook(_probe, with_kwargs=True)
    try:
        with torch.no_grad():
            inner(input_ids=ids, positions=positions, intermediate_tensors=None)
    except _ReplayProbeAbort:
        pass
    except Exception as exc:
        raise RuntimeError(
            "kv_load_failure_policy='repair' cannot verify this model's "
            "forward (the probe forward failed before layer 0: "
            + _format_repair_exception(exc)
            + "); switch the policy to 'recompute'"
        ) from exc
    finally:
        handle.remove()
    args = captured.get("args")
    kwargs = captured.get("kwargs") or {}
    if args is None or len(args) != 3 or kwargs:
        n_args = len(args) if args is not None else 0
        raise RuntimeError(
            "kv_load_failure_policy='repair' requires native layers called "
            "as layer(positions, hidden_states, residual) with no extra "
            "per-layer inputs; this model's layer loop supplies "
            f"{n_args} positional and {sorted(kwargs)} keyword arguments "
            "the replay cannot reproduce -- switch the policy to "
            "'recompute'"
        )
    hidden_states = args[1]
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.shape != embed.shape
        or hidden_states.dtype != embed.dtype
        or not torch.equal(hidden_states, embed)
    ):
        raise RuntimeError(
            "kv_load_failure_policy='repair' requires the hidden states "
            "fed to layer 0 to equal embed_input_ids of the same tokens; "
            "this model transforms the embedding after embed_input_ids "
            "(e.g. muP input scaling) and the replay would write "
            "silently-wrong KV -- switch the policy to 'recompute'"
        )


# The authorized-write-geometry fingerprint carried by the preflight
# summary: a stable hash of the sorted authorized block ids (the exact
# physical region the scheduler authorized). Distinct authorizations
# produce distinct fingerprints with overwhelming probability; a
# fingerprint COLLISION would need two different block-id tuples hashing
# equal, which the (sorted ids, count, span bounds) packing already
# separates.
_AUTH_FINGERPRINT_REFUSED = -1


def _auth_fingerprint(auth_blocks: tuple[int, ...]) -> int:
    """Stable cross-process fingerprint of the authorized write set."""
    if not auth_blocks:
        return _AUTH_FINGERPRINT_REFUSED
    h = 2166136261
    for block_id in auth_blocks:
        h = ((h ^ (block_id & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


@dataclasses.dataclass
class _RepairPreflight:
    """One rank's pre-execution repair preflight summary.

    Attributes:
        authorized: this rank completed every local preflight check and
            is willing to enter native layer execution for this work.
        reason: machine-readable refusal code (_REPAIR_PF_*). 0 (OK) is
            the only value that may carry authorized=True.
        plan_id/epoch/span/layer_end: the RepairWork identity this rank
            derived -- divergent identities abort on every rank.
        auth_fingerprint: fingerprint of the authorized write geometry.
        capability: the model-capability code (0 = the replay seam is
            present; divergent capabilities abort on every rank).
        error_detail: the contained local failure's diagnostic record.
        pre_write_ownership/pre_write_coverage/auth_present/auth_slots_ok/
            auth_table_ok/auth_distinct_ok/pre_write_ok: the pre-write
            fence flags (the commit check reuses them verbatim).
        peer_reasons: all-gathered refusal codes (filled by the
            consensus helper; None before it).

    """

    authorized: bool
    reason: int
    plan_id: int
    epoch: int
    span: tuple[int, int]
    layer_end: int
    auth_fingerprint: int
    capability: int = 0
    error_detail: str | None = None
    pre_write_ownership: bool = False
    pre_write_coverage: bool = False
    auth_present: bool = False
    auth_slots_ok: bool = False
    auth_table_ok: bool = False
    auth_distinct_ok: bool = False
    pre_write_ok: bool = False
    peer_reasons: list[int] | None = None

    def summary(self) -> torch.Tensor:
        """The all-gatherable int64 identity summary."""
        return torch.tensor(
            [
                int(self.authorized),
                int(self.reason),
                int(self.plan_id),
                int(self.epoch),
                int(self.span[0]),
                int(self.span[1]),
                int(self.layer_end),
                int(self.auth_fingerprint),
                int(self.capability),
            ],
            dtype=torch.int64,
        )


def _repair_preflight(
    reason: int,
    plan_id: int,
    epoch: int,
    span: tuple[int, int],
    layer_end: int,
    auth_fingerprint: int,
    capability: int = 0,
    error_detail: str | None = None,
) -> _RepairPreflight:
    """A refused preflight summary (never authorized)."""
    return _RepairPreflight(
        authorized=False,
        reason=reason,
        plan_id=plan_id,
        epoch=epoch,
        span=span,
        layer_end=layer_end,
        auth_fingerprint=auth_fingerprint,
        capability=capability,
        error_detail=error_detail,
    )


def _preflight_divergence_detail(pf: _RepairPreflight) -> str:
    """A human-readable record of a refused preflight consensus."""
    peers = pf.peer_reasons if pf.peer_reasons is not None else [pf.reason]
    return (
        "Repair preflight consensus refused ("
        f"local_reason={pf.reason}, peer_reasons={peers}, "
        f"plan_id={pf.plan_id}, epoch={pf.epoch}, span={pf.span}, "
        f"layer_end={pf.layer_end}, capability={pf.capability}) -- the "
        "repair aborted before any native layer execution"
    )


def _ranks_repair_preflight_agree(pf: _RepairPreflight) -> bool:
    """The PRE-EXECUTION consensus over the repair operation identity.

    Every rank packs its preflight summary (authorization, refusal
    reason, plan_id, epoch, span, layer range, authorized-write
    fingerprint) and all-gathers it BEFORE any rank enters native layer
    execution. The step is authorized only when every rank's summary is
    IDENTICAL to the local one and every rank is authorized with reason
    0. Any divergence -- a rank-local refusal, a stale or foreign work
    identity, a different authorized geometry -- aborts on EVERY rank
    before the first layer call, so no rank can be left blocked inside
    a native TP collective a peer never enters.

    Single-rank semantics reduce to the local summary being authorized,
    uniform with the multi-rank predicate.
    """
    ws = _tp_world_size()
    if ws <= 1:
        return pf.authorized and pf.reason == _REPAIR_PF_OK
    summary = pf.summary()
    device_type = current_platform.device_type
    if device_type:
        summary = summary.to(device_type)
    world = [torch.empty_like(summary) for _ in range(ws)]
    torch.distributed.all_gather(world, summary)
    reasons = [int(t[1].item()) for t in world]
    pf.peer_reasons = reasons
    for remote in world:
        if not torch.equal(remote, summary):
            return False
    return all(r == _REPAIR_PF_OK for r in reasons)


def _repair_capability(model: Any) -> int:
    """The cheap per-step model-capability code for the preflight.

    0 (OK) when the model exposes the replay seam the executor drives
    (a non-empty layer stack and `embed_input_ids`); the nonzero codes
    mirror the init-time canary's structural refusals. The full
    replay-faithfulness probe (`verify_replay_compatibility`) stays at
    init: a canary failure there aborts the engine on every rank, so
    mid-flight capability DIVERGENCE through the canary is unreachable
    by construction -- this structural check is what the per-step
    preflight can afford, and a divergent code still aborts the step
    on every rank through the consensus below.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        return 1
    if getattr(inner, "embed_input_ids", None) is None:
        return 2
    return 0


def _execute_preflight(
    model: Any,
    scheduler_output: "SchedulerOutput",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    req_id: str,
    work: RepairWork,
    lo: int,
    hi: int,
    work_row: int,
) -> _RepairPreflight:
    """The rank-local pre-execution checks, WITHOUT side effects.

    Completes every check that decides whether this rank may enter
    native layer execution -- the zero-width window, the ordered
    positions and input-shape checks, the layer-stack bound, the
    attention metadata access, the cache-geometry derivation, and the
    FULL pre-write ownership fence (the same ordered slot and
    authorization battery the executor's commit check reports) -- and
    returns the summary for the pre-execution consensus. Any failure is
    CONTAINED here as a refused summary (never an exception), so every
    rank reaches the consensus collective on every path. This function
    must not write anything (no re-zero, no embed): an aborted
    preflight leaves the caches exactly as they were.
    """
    plan_id = work.plan_id
    epoch = work.epoch
    span = work.repair_span
    layer_end = work.layer_end
    span_start, span_end = span
    auth_blocks = tuple(work.expected_write_block_ids)
    auth_fingerprint = _auth_fingerprint(auth_blocks)
    capability = _repair_capability(model)

    try:
        assert hi > lo, (
            f"hole repair: zero-width repair window for {req_id} "
            f"(num_scheduled_tokens="
            f"{scheduler_output.num_scheduled_tokens.get(req_id, 0)})"
        )
        expected_positions = torch.arange(
            span_start, span_end, dtype=positions.dtype, device=positions.device
        )
        # ORDER matters, not just the endpoints: a metadata bug that
        # permutes the span's rows inside the authorized block would
        # write each token's KV into a neighboring row while the
        # boundary pair still matches.
        assert torch.equal(positions[lo:hi], expected_positions), (
            f"hole repair: the work's physical positions do not match "
            f"the repair span [{span_start}, {span_end}) of {req_id} "
            "in order"
        )
        assert input_ids.shape[0] >= hi, (
            f"hole repair: input_ids carries {input_ids.shape[0]} rows, "
            f"fewer than the work's window [{lo}, {hi})"
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

        # Per-layer metadata identity: every replayed layer resolves its
        # OWN metadata from the forward context (attn_metadata[layer_name]),
        # and the KV write path resolves its own per-layer slot mapping
        # (slot_mapping[layer_name]). A repair step is single-KV-group by
        # the scheduler's init guard, so every layer of a group shares ONE
        # metadata object by construction (build_attn_metadata assigns the
        # same builder product to every layer_name of the group) -- but the
        # fence must not TRUST that: a backend (or a corrupted context)
        # serving one replayed layer a DIFFERENT metadata object than the
        # one the fence validated would write that layer through an
        # unfenced geometry. Every replayed layer's metadata object must be
        # the same object the fence validated, and every replayed layer's
        # name must be present (a missing name would raise inside the
        # layer's lookup, mid-execution).
        # Per-layer metadata identity: every replayed layer resolves its
        # OWN metadata from the forward context (attn_metadata[layer_name]),
        # and the KV write path resolves its own per-layer slot mapping
        # (slot_mapping[layer_name]). A repair step is single-KV-group by
        # the scheduler's init guard, so every layer of a group shares ONE
        # metadata object by construction (build_attn_metadata assigns the
        # same builder product to every layer_name of the group) -- but the
        # fence must not TRUST that: a backend (or a corrupted context)
        # serving one replayed layer a DIFFERENT metadata object than the
        # one the fence validated would write that layer through an
        # unfenced geometry. Every replayed layer's metadata object must be
        # the same object the fence validated, and every replayed layer's
        # name must be present (a missing name would raise inside the
        # layer's lookup, mid-execution).
        slot_mappings = get_forward_context().slot_mapping
        layer_slot_mappings = (
            slot_mappings if isinstance(slot_mappings, dict) and slot_mappings else None
        )
        for layer in layers[: work.layer_end]:
            attn = getattr(layer, "self_attn", None)
            layer_name = getattr(getattr(attn, "attn", None), "layer_name", None)
            if layer_name is None:
                # Fallback: a layer that does not expose the attention
                # wrapper's layer_name seam cannot be identity-checked
                # per layer; the fenced first-layer metadata remains the
                # write fence (the single-group construction invariant
                # makes every group layer share it).
                break
            assert layer_name in attn_metadata, (
                "hole repair: replayed layer "
                f"{layer_name} has no attention metadata entry "
                "(the replay would raise inside its metadata lookup)"
            )
            assert attn_metadata[layer_name] is meta, (
                "hole repair: replayed layer "
                f"{layer_name} resolves a DIFFERENT attention metadata "
                "object than the fenced one (per-layer metadata "
                "divergence); the repair write set cannot be fenced"
            )
            if layer_slot_mappings is not None:
                assert layer_slot_mappings.get(layer_name) is not None, (
                    "hole repair: replayed layer "
                    f"{layer_name} has no slot_mapping entry (its KV "
                    "cache update would be silently skipped)"
                )

        # Per-layer cache-layout agreement: the fence derives the block
        # size from layer 0 and validates every slot against it, but each
        # layer writes through its OWN cache view. A layer whose cache
        # geometry implies a different block size (an unexpected attention
        # layout: the single-group scheduler guard makes this unreachable
        # in production, the fence must not trust it) would interpret the
        # SAME slot numbers with a different block/offset split and write
        # outside the authorized rows.
        for layer in layers[: work.layer_end]:
            layer_bs = _repair_block_size(layer)
            assert layer_bs == block_size, (
                "hole repair: a replayed layer derives block size "
                f"{layer_bs}, but the fence validated the span with block "
                f"size {block_size} (per-layer cache layout divergence); "
                "the repair write set cannot be fenced"
            )

        slots_host = _tolist_timed(
            meta.slot_mapping[lo:hi], "pre_write.slot_mapping.tolist"
        )
        block_table_host = _tolist_timed(
            _repair_block_table_row(meta, work_row),
            "pre_write.block_tables.tolist",
        )
        auth_present = bool(auth_blocks)
        first_blk = span_start // block_size
        span_blocks = -(-span_end // block_size) - first_blk
        auth_len_ok = len(auth_blocks) == span_blocks
        auth_distinct_ok = len(set(auth_blocks)) == len(auth_blocks)
        expected_slots: list[int] = []
        if auth_present and auth_len_ok:
            for tok in range(span_start, span_end):
                block_id = auth_blocks[(tok // block_size) - first_blk]
                expected_slots.append(block_id * block_size + tok % block_size)
        auth_slots_ok = (
            auth_present
            and auth_len_ok
            and auth_distinct_ok
            and slots_host == expected_slots
        )
        table_span = block_table_host[
            span_start // block_size : -(-span_end // block_size)
        ]
        auth_table_ok = auth_present and list(table_span) == list(auth_blocks)
        ownership = _slots_owned_by_request_list(
            slots_host, block_table_host, block_size
        )
        coverage = _slots_cover_span_list(
            slots_host, block_table_host, span_start, span_end, block_size
        )
        pre_write_ok = (
            ownership and coverage and auth_present and auth_slots_ok and auth_table_ok
        )
        pf = _RepairPreflight(
            authorized=pre_write_ok and capability == 0,
            reason=_REPAIR_PF_OK if pre_write_ok else _REPAIR_PF_FENCE,
            plan_id=plan_id,
            epoch=epoch,
            span=span,
            layer_end=layer_end,
            auth_fingerprint=auth_fingerprint,
            capability=capability,
            pre_write_ownership=ownership,
            pre_write_coverage=coverage,
            auth_present=auth_present,
            auth_slots_ok=auth_slots_ok,
            auth_table_ok=auth_table_ok,
            auth_distinct_ok=auth_distinct_ok,
            pre_write_ok=pre_write_ok,
        )
        if capability != 0:
            pf.reason = _REPAIR_PF_CAPABILITY
            pf.authorized = False
        if not pre_write_ok:
            logger.error(
                "Repair preflight fence refused plan_id=%d span=[%d,%d) "
                "ownership=%s coverage=%s auth_present=%s auth_slots=%s "
                "auth_table=%s auth_distinct=%s",
                plan_id,
                span_start,
                span_end,
                ownership,
                coverage,
                auth_present,
                auth_slots_ok,
                auth_table_ok,
                auth_distinct_ok,
            )
        return pf
    except Exception as exc:
        error_detail = _format_repair_exception(exc)
        logger.exception(
            "Repair preflight failed plan_id=%d span=%s -> local refusal",
            plan_id,
            span,
        )
        text = error_detail
        if "block_tables" in text or "block_table" in text:
            reason = _REPAIR_PF_METADATA
        else:
            reason = _REPAIR_PF_GEOMETRY
        return _repair_preflight(
            reason,
            plan_id,
            epoch,
            span,
            layer_end,
            auth_fingerprint,
            capability=capability,
            error_detail=error_detail,
        )


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

    A v1 repair-only step carries exactly ONE work (the scheduler
    admits one repair per step; the step-level validation refuses
    more -- per-work attention-metadata slicing is unsupported). The
    work's rows are its own window of the step's batch (batch_req_ids is
    the row order), and the validation battery (positions, slot
    ownership, write coverage, layer coverage) applies per work item.
    """
    works = scheduler_output.repair_works
    assert works is not None
    global _REPAIR_SYNC_PROFILE
    _REPAIR_SYNC_PROFILE = {} if _repair_sync_profile_enabled() else None
    try:
        # Repair steps are exclusive: no ordinary request rides the batch.
        # The works and the scheduling counts are broadcast data (identical
        # on every rank); only batch_req_ids is worker-local. A step-level
        # violation fails identically on every rank for the broadcast
        # conditions; a worker-local batch violation fails on one rank,
        # whose STEP_INVALID preflight summary below diverges the
        # pre-execution consensus on every rank BEFORE any native layer
        # collective is entered -- no rank is stranded inside a layer's
        # MLP/o-proj all-reduce waiting for a peer that skipped it.
        assert len(works) == 1, (
            "hole repair: a v1 repair-only step carries exactly one work "
            "(the scheduler admits one repair per step); per-work "
            "attention-metadata slicing is unsupported, so a multi-work "
            "step cannot be fenced"
        )
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
        # The step-shape failure rides the SAME pre-execution consensus
        # the per-work path uses, so a rank failing only the worker-local
        # batch assert still pairs its collective sequence with the
        # peers' per-work preflight (a single-work step issues exactly
        # one preflight collective on every path).
        first = next(iter(works.values()), None)
        pf = _repair_preflight(
            _REPAIR_PF_STEP_INVALID,
            plan_id=first.plan_id if first is not None else -1,
            epoch=first.epoch if first is not None else -1,
            span=first.repair_span if first is not None else (-1, -1),
            layer_end=first.layer_end if first is not None else -1,
            auth_fingerprint=_AUTH_FINGERPRINT_REFUSED,
            error_detail=step_error,
        )
        try:
            _ranks_repair_preflight_agree(pf)
        except Exception:
            logger.exception("Repair preflight consensus failed (step shape) -> abort")
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
        # PRE-EXECUTION PREFLIGHT CONSENSUS: the work lookup and every
        # pre-layer check (positions and input shapes, layer bound,
        # attention metadata, cache geometry, the full ownership fence,
        # and the model capability) run inside _execute_preflight with
        # their exceptions CONTAINED, and the all-gathered operation
        # identity decides -- BEFORE the embedding or any layer call --
        # whether this rank enters native execution. Either every TP
        # rank is authorized to enter native repair execution for the
        # same work, or every rank aborts before entering it: a
        # rank-local refusal can never strand a peer inside a layer's
        # native TP collectives (the pre-fix behavior: the fence was
        # rank-local and the consensus post-execution, so a refusing
        # rank skipped the layer collectives its passing peer entered,
        # and the passing peer blocked inside the MLP/o-proj all-reduce
        # until the process-group timeout).
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
        except Exception as exc:
            error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair work lookup failed req=%s -> placeholder preflight",
                req_id,
            )

        if work is None:
            # No geometry, no identity: a placeholder-identity preflight
            # still enters the collective -- the peers carry the real
            # identity, the summaries diverge, and the step aborts on
            # every rank BEFORE any native execution (the pre-fix code
            # kept the peers inside their layer collectives here).
            pf = _repair_preflight(
                _REPAIR_PF_WORK_LOOKUP,
                -1,
                -1,
                (-1, -1),
                0,
                _AUTH_FINGERPRINT_REFUSED,
                error_detail=error_detail,
            )
        else:
            pf = _execute_preflight(
                model,
                scheduler_output,
                input_ids,
                positions,
                req_id,
                work,
                lo,
                hi,
                work_row,
            )
        # The fence flags the commit check reports come from the
        # preflight verbatim: the same ordered battery the loop used to
        # derive inline (identical strength), now also converged across
        # ranks before any write can execute.
        pre_write_ok = pf.pre_write_ok
        auth_present = pf.auth_present
        auth_slots_ok = pf.auth_slots_ok
        auth_table_ok = pf.auth_table_ok
        auth_distinct_ok = pf.auth_distinct_ok
        pre_write_ownership = pf.pre_write_ownership
        pre_write_coverage = pf.pre_write_coverage
        if error_detail is None:
            error_detail = pf.error_detail

        try:
            preflight_agree = _ranks_repair_preflight_agree(pf)
        except Exception as exc:
            # The collective itself failed (process-group error or a
            # backend timeout): authorization is unachievable -- fail
            # closed, no native execution on this rank.
            preflight_agree = False
            if error_detail is None:
                error_detail = _format_repair_exception(exc)
            logger.exception(
                "Repair preflight consensus failed req=%s -> abort", req_id
            )
        if not preflight_agree:
            if error_detail is None:
                error_detail = _preflight_divergence_detail(pf)
            logger.error(
                "Repair preflight refused req=%s plan_id=%d reason=%s "
                "peer_reasons=%s -- no native layer execution",
                req_id,
                pf.plan_id,
                pf.reason,
                pf.peer_reasons,
            )

        if work is not None and pf.authorized and preflight_agree:
            # Execution envelope: native replay over the authorized work.
            # A rank-local recoverable failure anywhere between the
            # metadata access and the verdict construction (metadata
            # access, embedding, the layer stream, the re-zero, or the
            # verdict collective itself) must not ESCAPE this function:
            # it is captured here, preserved as error_detail, and
            # converged into a consistent failed verdict that still
            # participates in consensus. The VERDICT collective is never
            # skipped; peers already blocked inside an inner LAYER
            # collective at the moment of the exception still wait for
            # that collective's backend timeout -- containment is
            # fail-closed, not latency-free.
            try:
                layers = model.model.layers
                attn_metadata = get_forward_context().attn_metadata
                assert isinstance(attn_metadata, dict) and attn_metadata
                meta = next(iter(attn_metadata.values()))
                hidden_states = model.model.embed_input_ids(input_ids[lo:hi])
                residual = None
                for idx, layer in enumerate(layers[: work.layer_end]):
                    try:
                        hidden_states, residual = layer(
                            positions[lo:hi], hidden_states, residual
                        )
                    except Exception as exc:
                        # Exception containment: a failure inside the
                        # repair stream is request-local. Stop the stream
                        # here, re-zero the partial writes INCLUDING this
                        # raising layer's rows (it may have written some
                        # KV rows before raising), report the failure,
                        # and let the scheduler's recompute fallback own
                        # the request. A poisoned device context fails
                        # loudly at the next step; a successful
                        # transaction is never fabricated.
                        layer_failed_at = idx
                        backend_ok = False
                        error_detail = _format_repair_exception(exc)
                        logger.exception(
                            "Repair layer exception req=%s idx=%d "
                            "span=[%d,%d) -> contained as abort",
                            work.request_id,
                            idx,
                            span_start,
                            span_end,
                        )
                        break
                    executed_layers += 1
                del hidden_states, residual
            except Exception as exc:
                # The envelope: any rank-local recoverable failure
                # OUTSIDE the per-layer containment (metadata access,
                # embedding) converges to the failed verdict below
                # instead of escaping and stranding the TP ranks in
                # consensus.
                backend_ok = False
                if error_detail is None:
                    error_detail = _format_repair_exception(exc)
                logger.exception(
                    "Repair pre-verdict exception req=%s plan_id=%s "
                    "span=[%d,%d) -> contained as abort",
                    req_id,
                    work.plan_id,
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
            # The pre-execution consensus: every TP rank all-gathered the
            # same authorized operation identity (capability, plan_id,
            # epoch, span, layer range, write-geometry fingerprint)
            # BEFORE any native layer execution.
            "preflight_agree": preflight_agree,
            "capability_ok": pf.capability == 0,
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
    # [B, H, N, C] core layout: tokens-per-block is dim 2. A 4-D
    # per-K/V Ascend view ([num_blocks, block_size, heads, dim]) is NOT
    # supported: it is shape-ambiguous with the core layout, and this
    # reading would derive the head count as the block size, so every
    # such repair fails the fence (fail-closed).
    if dim == 4:
        return int(cache.shape[2])
    # Ascend 2-D/3-D [num_blocks, block_size, ...] views:
    # tokens-per-block is dim 1.
    if dim in (2, 3):
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
