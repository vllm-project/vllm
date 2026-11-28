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

The decoder layer stack is resolved through find_repair_fence_layer across
the model-zoo container names instead of assuming `model.model.layers`, so
the capability verdict and the repair-time replay always target the same
stack; hardware-plugin layout gates import the same resolver.

The module lives beside the GPU model runner because the old
kv_connector_model_runner_mixin module no longer exists upstream: its
connector plumbing was absorbed into vllm/v1/worker/gpu/kv_connector.py.
"""

import dataclasses
import inspect
import os
import time
import typing
from collections import Counter
from collections.abc import Callable
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


def _replay_inert_kwargs(layer: Any, kwargs: dict[str, Any]) -> bool:
    """Whether explicitly-passed keyword extras are inert under the replay.

    The replay calls layer(positions, hidden_states, residual) with the
    extras omitted. That binds identically to the model's explicit call
    only when every passed value is None AND the layer signature defaults
    each name to None; any other default (or an unresolvable signature)
    means the replay would bind a different value than the model's own
    loop, so it must refuse.

    Args:
        layer: The probe-captured layer 0 (signature source).
        kwargs: The keyword arguments the model's loop passed to it.

    Returns:
        True when omitting the extras in the replay is binding-equivalent.

    """
    if any(value is not None for value in kwargs.values()):
        return False
    try:
        signature = inspect.signature(layer.forward)
    except (TypeError, ValueError):
        return False
    for name in kwargs:
        param = signature.parameters.get(name)
        if param is None or param.default is not None:
            return False
    return True


# Decoder-stack container names in precedence order: the standard vLLM
# wrapper (model.layers), the HF-heritage names (ChatGLM
# transformer.encoder.layers, Falcon/GPT-2 transformer.h, MPT
# transformer.blocks), multimodal wrappers nesting the backbone below
# language_model, and a bare stack on the model itself. Hardware-plugin
# layout gates import find_repair_fence_layer, so their init-time checks
# and this executor's repair-time lookup resolve the same layer.
_REPAIR_STACK_CONTAINERS: tuple[tuple[str, ...], ...] = (
    ("model", "layers"),
    ("transformer", "encoder", "layers"),
    ("transformer", "h"),
    ("transformer", "blocks"),
    ("language_model", "model", "layers"),
    ("language_model", "transformer", "encoder", "layers"),
    ("layers",),
)


def _tried_stack_paths() -> str:
    """The container paths a stack lookup walked, for refusal texts."""
    return ", ".join(".".join(path) for path in _REPAIR_STACK_CONTAINERS)


def _walk_repair_stack(model: Any, path: tuple[str, ...]) -> Any:
    """The stack node at one container path, or None when the path does
    not resolve to a non-empty stack."""
    node = model
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            break
    if not node:
        return None
    return node


def _resolve_model_layer_stack(model: Any) -> tuple[Any, Any]:
    """The stack the model's own forward drives, and the module owning it.

    The replay canary's lookup: the first non-empty container, whatever
    its layers expose -- the canary verifies the FORWARD, and the cache
    seam is the capability verdict's own check, so a canary-compatible
    model must not lose its probe over a seam it does not carry yet.
    The owner is the module the path descends through: the replay's
    embedding seam and the probe-forward module (model.model for the
    standard layout, model.transformer for Falcon-style transformer.h
    stacks).

    Returns:
        (stack, owner), or (None, None) when no known container resolves.

    """
    for path in _REPAIR_STACK_CONTAINERS:
        node = _walk_repair_stack(model, path)
        if node is None:
            continue
        try:
            iter(node)
        except TypeError:
            continue
        owner = model
        for attr in path[:-1]:
            owner = getattr(owner, attr, None)
        return node, owner
    return None, None


def _resolve_repair_stack(
    model: Any,
) -> tuple[Any, Any, Any, tuple[str, ...] | None]:
    """The executor's replay stack: the first container holding a layer
    with the repair fence's cache seam (self_attn.attn).

    A container whose layers keep no seam can be neither fenced nor
    re-zeroed, and a leading seam-less container is never mistaken for
    the decoder (a vision encoder's attention keeps no KV-cache seam the
    fence reads). Whenever the init capability verdict authorized
    repairs this is also the canary's stack: the verdict requires the
    first non-empty container's layer 0 to carry the seam, which makes
    that container the first seam-bearing one.

    Returns:
        (stack, owner, fence_layer, path), or
        (None, None, None, None) when no known container resolves.

    """
    for path in _REPAIR_STACK_CONTAINERS:
        node = _walk_repair_stack(model, path)
        if node is None:
            continue
        fence = None
        try:
            for layer in node:
                seam = getattr(getattr(layer, "self_attn", None), "attn", None)
                if seam is not None:
                    fence = layer
                    break
        except TypeError:
            continue
        if fence is None:
            continue
        owner = model
        for attr in path[:-1]:
            owner = getattr(owner, attr, None)
        return node, owner, fence, path
    return None, None, None, None


def find_repair_fence_layer(model: Any) -> tuple[Any, Any]:
    """Locate the first decoder layer the repair fence reads the cache of.

    The repair executor reaches every layer's cache through
    layer.self_attn.attn.kv_cache and derives the block size from that
    cache's geometry (_repair_geometry_from_binding), but model zoos name the
    decoder stack differently (ChatGLM transformer.encoder.layers,
    Falcon transformer.h, MPT transformer.blocks) and multimodal
    wrappers nest the backbone below language_model. Discover the first
    layer that actually carries the fence's cache seam (self_attn.attn),
    which cannot pick up an unrelated attention module (e.g. a vision
    encoder, whose attention keeps no KV-cache seam the fence reads).
    Hardware-plugin layout gates import this resolver so an init-time
    capability check never accepts a stack the executor's repair-time
    lookup would refuse.

    Args:
        model: The model the runner hands to the repair executor.

    Returns:
        (layer, path) of the first candidate with the seam, or
        (None, tried_paths); the caller refuses with an actionable error
        instead of deriving a block size from a wrong module.

    """
    stack, _owner, fence, path = _resolve_repair_stack(model)
    if stack is None:
        return None, _REPAIR_STACK_CONTAINERS
    return fence, ".".join(path)


def verify_replay_compatibility(model: Any) -> None:
    """Refuse the repair policy unless the model's own forward reduces to
    exactly what the replay executes: embed_input_ids feeding
    layer(positions, hidden_states, residual).

    Keyword extras are accepted only when they are replay-inert: every
    explicitly-passed value is None and the layer signature defaults each
    to None (an explicit None binds identically to the default, so the
    replay's 3-positional call is binding-equivalent -- e.g. MistralModel
    threads t_cond=None through LlamaModel.forward's **extra_layer_kwargs
    even for plain text models). Any valued extra (Ministral-3 conditioning
    tensors, Gemma4 per-layer inputs) still refuses.

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
    layers, inner = _resolve_model_layer_stack(model)
    if layers is None:
        raise RuntimeError(
            "kv_load_failure_policy='repair' requires a native model with "
            "a layer stack (none found on this model; tried "
            + _tried_stack_paths()
            + ")"
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
    if (
        args is None
        or len(args) != 3
        or (kwargs and not _replay_inert_kwargs(layers[0], kwargs))
    ):
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
        geometry: the validated write geometry of the authorized work
            (ONE geometry serves the fence, the abort cleanup, and the
            readback); None on every refused path.

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
    geometry: "_RepairGeometry | None" = None

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
    layers, inner = _resolve_model_layer_stack(model)
    if layers is None:
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
    *,
    view_binding: "RepairViewBinding | None" = None,
    full_forward: bool = False,
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

        layers, _inner, _fence, _stack_path = _resolve_repair_stack(model)
        if layers is None:
            raise RuntimeError(
                "hole repair: no decoder layer stack with the repair "
                "fence's cache seam (self_attn.attn) was found on this "
                "model (tried " + _tried_stack_paths() + "); the repair "
                "write set cannot be fenced"
            )
        assert work.layer_end <= len(layers), (
            f"hole repair: layer_end {work.layer_end} exceeds the stack "
            f"({len(layers)} layers)"
        )

        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict) and attn_metadata
        meta = next(iter(attn_metadata.values()))
        if full_forward:
            assert work.layer_end == len(layers), "Mixed repair requires full depth"
            assert meta.query_start_loc[work_row : work_row + 2].tolist() == [lo, hi]
            assert int(meta.seq_lens[work_row]) == span_end, (
                "Repair attention must stop at the hole's end"
            )
        geometry = _repair_geometry_from_binding(
            view_binding, _repair_cache_views(layers[0])
        )
        block_size = geometry.block_size

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
            _verify_repair_layer_geometry(layer, geometry, view_binding)

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
            geometry=geometry,
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


def repair_capability(model: Any) -> dict[str, Any]:
    """Machine-readable replay-capability verdict for the repair policy.

    Wraps the replay canary plus the executor's structural cache-seam
    check into stable reason codes the distributed preflight can
    all-gather: the verdict reads only model structure, so every TP rank
    computes the same dict by construction, and the consensus fails
    closed on any divergence or any refusal. The caller (the model
    runner init) keeps the fail-fast behavior: it raises the refusal
    text when compatible is False.

    Returns:
        A dict with compatible (bool), reason_code (one of ok,
        no_layer_stack, no_embed_seam, layer_signature,
        post_embed_transform, probe_failed, no_self_attn_cache_path,
        kv_sharing_unsupported) and detail (the refusal text).

    """
    verdict: dict[str, Any] = {
        "compatible": False,
        "reason_code": "",
        "detail": "",
    }
    try:
        verify_replay_compatibility(model)
    except RuntimeError as exc:
        message = str(exc)
        verdict["detail"] = message
        # Order matters: the post-embed-transform message also contains
        # the embed_input_ids token, so it must be classified first.
        if "layer stack" in message:
            verdict["reason_code"] = "no_layer_stack"
        elif "transforms the embedding" in message:
            verdict["reason_code"] = "post_embed_transform"
        elif "embed_input_ids" in message:
            verdict["reason_code"] = "no_embed_seam"
        elif "cannot verify" in message:
            verdict["reason_code"] = "probe_failed"
        else:
            verdict["reason_code"] = "layer_signature"
        return verdict
    layers, _owner = _resolve_model_layer_stack(model)
    if layers is None:
        # Unreachable when verify_replay_compatibility passed above (the
        # canary refuses the same condition); kept fail-closed so the
        # verdict stays a verdict, never an AttributeError.
        verdict["reason_code"] = "no_layer_stack"
        verdict["detail"] = (
            "kv_load_failure_policy='repair' requires a native model with "
            "a layer stack (none found on this model; tried "
            + _tried_stack_paths()
            + ")"
        )
        return verdict
    layer0 = layers[0]
    self_attn = getattr(layer0, "self_attn", None)
    for layer in layers:
        attn = getattr(getattr(layer, "self_attn", None), "attn", None)
        if getattr(attn, "kv_sharing_target_layer_name", None) is not None:
            # KV-shared layers (Gemma4 YOCO cross-decoder, eagle sharing)
            # bind several layers to one physical cache: the repair
            # write-set and report geometry assume one cache per layer.
            verdict["reason_code"] = "kv_sharing_unsupported"
            verdict["detail"] = (
                "kv_load_failure_policy='repair': layer "
                + type(layer).__name__
                + " shares its KV cache with another layer "
                "(kv_sharing_target_layer_name); the repair write-set and "
                "report geometry is one-cache-per-layer -- switch the "
                "policy to 'recompute'"
            )
            return verdict
    if not hasattr(self_attn, "attn"):
        # The executor's fence and abort re-zero reach every layer's
        # cache through layer.self_attn.attn (_repair_cache_views); a
        # canary-compatible model without that seam fails every repair
        # loudly at repair time -- surface it as an init-time fact.
        verdict["reason_code"] = "no_self_attn_cache_path"
        verdict["detail"] = (
            "kv_load_failure_policy='repair': layer 0 ("
            + type(layer0).__name__
            + ") lacks the self_attn.attn cache seam the repair executor's "
            "fence and re-zero reach through; switch the policy to "
            "'recompute'"
        )
        return verdict
    verdict["compatible"] = True
    verdict["reason_code"] = "ok"
    return verdict


def execute_repair_only(
    model: Any,
    scheduler_output: "SchedulerOutput",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    batch_req_ids: list[str],
    *,
    view_binding: "RepairViewBinding | None" = None,
    full_forward: Callable[[], Any] | None = None,
) -> list[RepairResultReport]:
    """Validate and execute one scheduler-admitted repair.

    With full_forward, validate the repair row in the complete batch and
    invoke the caller's normal model forward once. Other batch rows retain
    their normal attention metadata and outputs. Without it, execute the
    original exclusive partial-layer replay described below.

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

    The write geometry comes from the caller's RepairViewBinding -- the
    binding contract of the KV views the model runner bound -- and is
    validated against every replayed layer's actual views (shape and
    strides) before the first write; a missing or contradictory binding
    refuses the step.
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
        assert (
            set(works) <= set(scheduler_output.num_scheduled_tokens)
            if full_forward is not None
            else set(works) == set(scheduler_output.num_scheduled_tokens)
        ), (
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
        if view_binding is None:
            raise ValueError(
                "hole repair: the model runner did not supply the "
                "repair view binding (the binding contract of the KV "
                "views the repair writes through); the write geometry "
                "is never derived from shape coincidences -- see the "
                "model runner's _repair_view_binding"
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
        if full_forward is not None and req_id not in works:
            continue
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
        work_geometry: _RepairGeometry | None = None
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
                view_binding=view_binding,
                full_forward=full_forward is not None,
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
        work_geometry = pf.geometry
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
                layers, inner, _fence, _stack_path = _resolve_repair_stack(model)
                if layers is None:
                    raise RuntimeError(
                        "hole repair: no decoder layer stack with the "
                        "repair fence's cache seam (self_attn.attn) was "
                        "found on this model (tried "
                        + _tried_stack_paths()
                        + "); the repair write set cannot be fenced"
                    )
                attn_metadata = get_forward_context().attn_metadata
                assert isinstance(attn_metadata, dict) and attn_metadata
                meta = next(iter(attn_metadata.values()))
                if full_forward is not None:
                    if work.layer_end != len(layers):
                        raise ValueError("Mixed repair requires all model layers")
                    full_forward()
                    executed_layers = work.layer_end
                    replay_layers = ()
                    hidden_states = residual = None
                else:
                    hidden_states = inner.embed_input_ids(input_ids[lo:hi])
                    residual = None
                    replay_layers = layers[: work.layer_end]
                for idx, layer in enumerate(replay_layers):
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
                    if work_geometry is None:
                        raise RuntimeError(
                            "hole repair: the aborted work carries no "
                            "validated geometry; the scheduler-side "
                            "eviction remains the authoritative "
                            "invalidation"
                        )
                    _rezero_repair_span(layers[:depth], meta, work_geometry, lo, hi)
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


class _RepairGeometry(typing.NamedTuple):
    """The explicit indexing contract for repair writes to bound KV views.

    The geometry is VALIDATED FROM THE BINDING, never derived from shape:
    the block size comes from the KV cache spec (authoritative, not
    guessed from a dimension), and the token axis from the binding
    convention the view's owner declared when it bound the view (Core
    logical [B, H, N, C] carries the token axis at dim 2 for every
    physical layout; Ascend per-K/V views carry it at dim 2 for the
    head-major LBHNC form and dim 1 for the token-major LBNHC form).
    Shape and strides VERIFY this contract against the actual bound
    view, never determine it: a matching shape is a confirmation, not a
    definition.
    """

    block_size: int
    """Manager block size (tokens per block), from the KV cache spec."""
    token_axis: int
    """The dim of a per-K/V (or shared Core) view that indexes
    within-block tokens."""
    is_core_shared: bool
    """True: one 4D [B,H,N,C] tensor shared by K and V (Core binding).
    False: separate per-K/V views (Ascend binding)."""


class RepairViewBinding(typing.NamedTuple):
    """The binding contract of the KV views a worker binds.

    Supplied by the code that creates and binds the actual views -- the
    model runner, which allocates the caches and hands every layer its
    view (allocate_kv_cache / create_kv_cache_views, or a backend's own
    unpacking). Repair needs exactly three facts about those views: how
    K and V are represented (one shared Core tensor, or separate
    per-K/V views), which axis indexes within-block tokens, and how an
    authorized slot maps onto the view (slot = block_id * block_size +
    offset indexes the leading block dim and the token axis).

    The binding is AUTHORITATIVE for the token axis: shape alone cannot
    identify it when num_kv_heads == block_size (the two readings are
    then the same tensor), so the executor never guesses -- it validates
    every replayed layer's views against the binding (shape AND strides)
    before the first write and refuses on any missing or contradictory
    fact. A binding that disagrees with the views it describes is the
    one failure mode inspection cannot catch; the runner that supplies
    both derives them from the same resolved CacheConfig, which is the
    construction guarantee the contract rests on.
    """

    kind: str
    """CORE_LOGICAL_VIEW_BINDING or PER_KV_VIEW_BINDING."""
    block_size: int
    """Manager block size (tokens per block) of the KV cache spec."""
    token_axis: int
    """View dim that indexes within-block tokens."""
    blocks_axis: int
    """View dim that indexes blocks (the leading dim, 0)."""
    layout_name: str | None
    """The resolved physical layout the binding was created for."""


CORE_LOGICAL_VIEW_BINDING = "core_logical"
PER_KV_VIEW_BINDING = "per_kv"


def core_logical_repair_binding(
    block_size: int, layout_name: str | None = None
) -> RepairViewBinding:
    """The binding of vLLM Core's KV views: one logical [B, H, N, C]
    tensor per layer, shared by K and V (split along the channel dim C),
    under EVERY physical layout (LBNHC and LBHNC differ in strides
    only)."""
    return RepairViewBinding(
        kind=CORE_LOGICAL_VIEW_BINDING,
        block_size=block_size,
        token_axis=2,
        blocks_axis=0,
        layout_name=layout_name,
    )


def per_kv_repair_binding(
    block_size: int, token_axis: int, layout_name: str | None = None
) -> RepairViewBinding:
    """The binding of a per-K/V backend: separate K and V views whose
    axes follow the backend's own convention (the token axis by the
    resolved layout: head-major LBHNC/HND views carry it at dim 2,
    token-major LBNHC views and flat [num_blocks, block_size, ...] views
    at dim 1)."""
    if token_axis not in (1, 2):
        raise ValueError(
            f"per-K/V repair view binding: token_axis {token_axis} is not "
            "a known per-K/V convention (expected dim 1 or dim 2)"
        )
    return RepairViewBinding(
        kind=PER_KV_VIEW_BINDING,
        block_size=block_size,
        token_axis=token_axis,
        blocks_axis=0,
        layout_name=layout_name,
    )


def _row_major_strides(view: torch.Tensor) -> bool:
    """Strides equal a contiguous row-major tensor of this shape."""
    expected = 1
    for dim in range(view.dim() - 1, -1, -1):
        if view.stride(dim) != expected:
            return False
        expected *= max(int(view.shape[dim]), 1)
    return True


def _core_logical_strides_ok(view: torch.Tensor, layout_name: str | None) -> bool:
    """Strides consistent with the binding contract of a logical
    [B, H, N, C] view under the declared physical layout.

    A head-major (LBHNC) physical allocation viewed as-is is row-major;
    a token-major (LBNHC) allocation is viewed through the permuted
    logical form (dims 1 and 2 swapped). When the physical layout is
    declared, the strides must match THAT form exactly -- a raw
    un-permuted LBNHC physical tensor bound where a logical view was
    promised is refused even when its shapes coincide. An undeclared
    binding accepts either form.
    """
    if layout_name in ("LBHNC", "HND"):
        return _row_major_strides(view)
    permuted = (
        view.stride(3) == 1
        and view.stride(1) == int(view.shape[3])
        and view.stride(2) == int(view.shape[1]) * view.stride(1)
        and view.stride(0) == int(view.shape[2]) * view.stride(2)
    )
    if layout_name in ("LBNHC", "NHD"):
        return permuted
    return _row_major_strides(view) or permuted


def _per_kv_strides_ok(view: torch.Tensor) -> bool:
    """Strides consistent with a per-K/V plane.

    Dims 1..3 must be row-major from the channel dim (how K/V-quantized
    or channel-split views are bound is the CORE convention, refused
    here), and the block stride must be an exact multiple of one block's
    span: a fully contiguous per-K/V view, or K/V planes sliced from a
    [num_kv, num_blocks, ...] stack -- the Ascend binding, where each
    view's block stride carries the sibling planes' spans (verified on
    hardware: [num_blocks, heads, bs, dim] with stride(0) = 2 * span).
    """
    if view.dim() == 4:
        if view.stride(3) != 1:
            return False
        if view.stride(2) != int(view.shape[3]):
            return False
        if view.stride(1) != int(view.shape[2]) * view.stride(2):
            return False
        span = int(view.shape[1]) * view.stride(1)
        return view.stride(0) >= span and view.stride(0) % span == 0
    return _row_major_strides(view)


def _repair_geometry_from_binding(
    binding: "RepairViewBinding | None", views: list[torch.Tensor]
) -> _RepairGeometry:
    """Validate the binding against the actual bound views; the geometry.

    ONE geometry serves the fence, the abort cleanup, and the readback.
    Every check is fail-closed BEFORE the first write: a missing or
    contradictory binding fact refuses the repair with a named error
    instead of guessing an axis from a shape coincidence.
    """
    if binding is None:
        raise ValueError(
            "hole repair: no repair view binding was supplied for the "
            "work's bound KV views; the write geometry is never derived "
            "from shape coincidences (see the model runner's "
            "_repair_view_binding)"
        )
    if binding.block_size is None or binding.block_size <= 0:
        raise ValueError(
            f"hole repair: the binding's block size is not available or "
            f"invalid ({binding.block_size}); the repair geometry cannot "
            "be built without the authoritative block size"
        )
    if binding.blocks_axis != 0:
        raise ValueError(
            f"hole repair: the binding's blocks axis "
            f"({binding.blocks_axis}) is not a supported binding "
            "contract (blocks must be the leading dimension)"
        )
    if not views:
        raise ValueError(
            "hole repair: no KV cache view is bound to the repair fence "
            "layer; the repair write set cannot be fenced"
        )
    view = views[0]
    dim = view.dim()
    if binding.kind == CORE_LOGICAL_VIEW_BINDING:
        # One logical [B, H, N, C] tensor shared by K and V: a per-K/V
        # pair (or any other form) contradicts the binding contract.
        if len(views) != 1 or dim != 4:
            raise ValueError(
                f"hole repair: the bound views contradict the core "
                f"logical binding contract: expected one shared 4D view, "
                f"got {len(views)} view(s) of dim {dim}"
            )
        if int(view.shape[binding.token_axis]) != binding.block_size:
            raise ValueError(
                f"hole repair: the bound 4D KV view shape "
                f"{tuple(view.shape)} does not carry the binding's block "
                f"size {binding.block_size} on the token axis (dim "
                f"{binding.token_axis}): the view geometry contradicts "
                "the KV cache spec (manager block size vs kernel block "
                "size divergence, or a wrong view was bound to the layer)"
            )
        if not _core_logical_strides_ok(view, binding.layout_name):
            raise ValueError(
                f"hole repair: the bound 4D KV view shape "
                f"{tuple(view.shape)} strides {tuple(view.stride())} are "
                f"not consistent with the core logical [B, H, N, C] "
                f"binding under layout '{binding.layout_name}': the view "
                "contradicts the binding contract"
            )
        return _RepairGeometry(
            block_size=binding.block_size,
            token_axis=binding.token_axis,
            is_core_shared=True,
        )
    if binding.kind == PER_KV_VIEW_BINDING:
        # Separate K and V views (possibly more, e.g. quantization scale
        # caches). A lone 4D view is the core-shared binding's shape and
        # contradicts this contract; the flat 2D/3D convention may bind
        # a single view (its rows are zeroed per token regardless of how
        # K and V are packed into it).
        if dim == 4 and len(views) < 2:
            raise ValueError(
                f"hole repair: the bound views contradict the per-K/V "
                f"binding contract: expected separate K and V views, got "
                f"{len(views)} view(s) of dim {dim}"
            )
        for view in views:
            if view.dim() != dim:
                raise ValueError(
                    f"hole repair: the bound per-K/V views mix dims "
                    f"({dim} and {view.dim()}); the views contradict the "
                    "binding contract"
                )
            if dim == 4:
                if int(view.shape[binding.token_axis]) != binding.block_size:
                    raise ValueError(
                        f"hole repair: the bound per-K/V 4D view shape "
                        f"{tuple(view.shape)} does not carry the binding's "
                        f"block size {binding.block_size} on the bound "
                        f"token axis (dim {binding.token_axis}): the view "
                        "geometry contradicts the KV cache spec (manager "
                        "block size vs kernel block size divergence, or a "
                        "wrong view was bound to the layer)"
                    )
            elif dim in (2, 3):
                # Flat [num_blocks, block_size, ...] per-K/V view: the
                # token axis is dim 1 by the binding convention, under
                # every layout.
                if binding.token_axis != 1:
                    raise ValueError(
                        f"hole repair: a flat {dim}D per-K/V view (shape "
                        f"{tuple(view.shape)}) carries its tokens on dim "
                        f"1, but the binding declares dim "
                        f"{binding.token_axis}; the views contradict the "
                        "binding contract"
                    )
                if int(view.shape[1]) != binding.block_size:
                    raise ValueError(
                        f"hole repair: the bound flat per-K/V view shape "
                        f"{tuple(view.shape)} does not carry the binding's "
                        f"block size {binding.block_size} on the token "
                        f"axis (dim 1): the view geometry contradicts the "
                        "KV cache spec"
                    )
            else:
                raise ValueError(
                    f"hole repair: unrecognized per-K/V cache view dim "
                    f"{dim} (shape {tuple(view.shape)}); the binding "
                    "contract covers only flat 2D/3D and 4D views"
                )
            if not _per_kv_strides_ok(view):
                raise ValueError(
                    f"hole repair: the bound per-K/V view shape "
                    f"{tuple(view.shape)} strides {tuple(view.stride())} "
                    "are not consistent with a per-K/V plane (contiguous, "
                    "or sliced from a K/V stack); the view contradicts "
                    "the binding contract"
                )
        return _RepairGeometry(
            block_size=binding.block_size,
            token_axis=binding.token_axis,
            is_core_shared=False,
        )
    raise ValueError(
        f"hole repair: unknown repair view binding kind '{binding.kind}'; "
        f"the binding contract covers only '{CORE_LOGICAL_VIEW_BINDING}' "
        f"and '{PER_KV_VIEW_BINDING}'"
    )


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


def _verify_repair_layer_geometry(
    layer,
    geometry: _RepairGeometry,
    binding: "RepairViewBinding | None" = None,
) -> None:
    """Verify a replayed layer's bound views match the repair geometry.

    The geometry was validated once from the binding (see
    _repair_geometry_from_binding); this check confirms that EVERY view
    involved in the repair work item -- K and V alike, plus any further
    per-K/V views (e.g. quantization scales) -- conforms to the SAME
    binding, by shape AND by strides, before the first write. A mismatch
    is a fail-closed refusal: the repair cannot write through a view
    whose indexing contract differs from the one the fence authorized.
    """
    caches = _repair_cache_views(layer)
    derived = _repair_geometry_from_binding(binding, caches)
    if derived != geometry:
        raise ValueError(
            f"hole repair: layer cache geometry {derived} does not match "
            f"the fence-authorized geometry {geometry}; the bound views "
            "contradict the binding contract"
        )


def _repair_block_size(layer) -> int:
    """Compatibility shim for the published Ascend repair init gate.

    vllm-ascend #22650 (as published) imports this helper and its
    ImportError fallback would SILENTLY DISABLE the init-time
    LBNHC+repair refusal on vLLM builds that carry the binding-contract
    repair executor -- a missing helper must never turn into successful
    admission. The base semantics are preserved EXACTLY (the shape-based
    derivation this module replaced), so the old gate behaves identically
    to how it did on the base build: it refuses LBNHC+repair through the
    block-size mismatch and admits the head-major layouts whose dim 2
    carries the block size. Do not build new consumers on this helper:
    the binding contract (_repair_geometry_from_binding) is the supported
    derivation. Remove together with the vllm-ascend #22650 gate port.
    """
    caches = _repair_cache_views(layer)
    cache = caches[0]
    dim = cache.dim()
    if dim == 4:
        return int(cache.shape[2])
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


def _rezero_repair_span(
    layers, meta, geometry: _RepairGeometry, lo: int = 0, hi: int | None = None
) -> None:
    """Abort-path partial-write invalidation: re-zero the span's slot rows
    in every executed layer's K and V cache. A partial repair is not a
    valid prefix; no partial content is left readable.

    The write indexes the cache tensor's own block and token dims -- the
    per-layer cache can be a non-contiguous view (a reshape would copy,
    and the zero would land in the copy). The geometry (derived from the
    resolved layout + spec, NOT from shape) identifies the view's token
    axis and the K/V binding convention; the slot window is the work
    item's [lo, hi) rows of the step's slot mapping. Shape is used only
    to confirm the already-known contract.
    """
    slots = meta.slot_mapping if hi is None else meta.slot_mapping[lo:hi]
    slots = slots.long()
    bidx = slots // geometry.block_size
    oidx = slots % geometry.block_size
    max_abs_after = 0.0
    for layer in layers:
        caches = _repair_cache_views(layer)
        for cache in caches:
            if geometry.is_core_shared:
                # Core logical [B,H,N,C]: token axis dim 2, zero all heads.
                cache[bidx, :, oidx, :] = 0
            elif cache.dim() == 4 and geometry.token_axis == 2:
                # Ascend head-major [n, h, bs, d]: token axis dim 2.
                cache[bidx, :, oidx, :] = 0
            elif cache.dim() == 4 and geometry.token_axis == 1:
                # Ascend token-major [n, bs, h, d]: token axis dim 1.
                cache[bidx, oidx, :, :] = 0
            else:
                # Flat 2D/3D per-K/V [n, bs, ...]: token axis dim 1.
                cache[bidx, oidx] = 0
        probe = caches[0]
        if geometry.is_core_shared or probe.dim() == 4 and geometry.token_axis == 2:
            k_rows = probe[bidx, :, oidx, :]
        elif probe.dim() == 4 and geometry.token_axis == 1:
            k_rows = probe[bidx, oidx, :, :]
        else:
            k_rows = probe[bidx, oidx]
        max_abs_after = max(max_abs_after, float(k_rows.abs().max()))
    logger.info(
        "Repair abort re-zero layers=%d rows=%d max_abs_after=%.6f",
        len(layers),
        int(slots.numel()),
        max_abs_after,
    )
