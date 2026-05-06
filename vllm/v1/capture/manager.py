# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-consumer capture manager.

``CaptureManager`` is the per-runner object that coordinates activation
capture across an ordered tuple of ``CaptureSink`` instances.  Each sink
corresponds to one registered capture consumer (e.g., filesystem writer,
reward trainer, dashboard).

Key design properties:

- **Union gather:** When multiple consumers want rows from the same
  ``(layer, hook)`` pair, the gather happens once.  Each entry's
  ``consumer_mask`` bitset records which consumers want it so the
  dispatch path can fan-out without redundant GPU reads.

- **Consumer isolation:** A failing ``submit_chunk`` or
  ``submit_finalize`` on one sink never prevents delivery to the others.
  Errors are captured per ``(consumer, request)`` and surfaced through
  ``CaptureResult``.

- **Position expansion:** The manager resolves the five selector
  modes (``last_prompt``, ``all_prompt``, ``all_generated``, ``all``,
  explicit ``list[int]``) and intersects the result with the step's
  ``[num_computed, num_computed + num_scheduled)`` window.

See ``docs/design/capture_consumers.md`` for the full spec.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.v1.capture.plan import (
    CaptureBatchView,
    CapturePositionEntry,
    StepCapturePlan,
)
from vllm.v1.capture.sink import CaptureSink
from vllm.v1.capture.types import (
    CaptureKey,
    CaptureChunk,
    CaptureFinalize,
    CaptureResult,
    CaptureStatus,
    CaptureSpec,
    VllmInternalRequestId,
)

logger = logging.getLogger(__name__)

_CAPTURE_RESULT_SEVERITY: dict[CaptureStatus, int] = {
    "pending": 0,
    "ok": 1,
    "not_requested": 2,
    "partial_error": 3,
    "error": 4,
}


# ---------------------------------------------------------------------------
# Per-request internal state
# ---------------------------------------------------------------------------


@dataclass
class _RequestCaptureState:
    """Bookkeeping for one registered capture request.

    ``consumer_specs`` maps consumer index to the merged spec for this
    request.  ``position_kind`` and ``static_positions`` are per-consumer
    because each consumer may use a different position selector.
    """

    req_id: str
    consumer_specs: dict[int, CaptureSpec]
    position_kind: dict[int, str]
    static_positions: dict[int, list[int] | None]
    num_prompt_tokens: int
    steps_seen: int = 0
    error: str | None = None
    sidecar_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Position expansion helpers
# ---------------------------------------------------------------------------


def _resolve_positions(
    positions: list[int] | str,
    num_prompt_tokens: int,
    num_computed: int,
    num_scheduled: int,
) -> list[int]:
    """Expand a position selector against the current step bounds.

    Returns absolute logical indices.  For symbolic selectors the upper
    bound is ``num_computed + num_scheduled`` (the highest token the
    forward pass will touch).
    """
    upper = num_computed + num_scheduled

    if isinstance(positions, list):
        return list(positions)

    if positions == "last_prompt":
        return [num_prompt_tokens - 1]

    if positions == "all_prompt":
        return list(range(num_prompt_tokens))

    if positions == "all_generated":
        start = num_prompt_tokens
        return list(range(start, upper))

    if positions == "all":
        return list(range(upper))

    msg = f"Unknown position selector: {positions!r}"
    raise ValueError(msg)


def _classify_positions(
    spec: CaptureSpec,
    num_prompt_tokens: int,
) -> tuple[str, list[int] | None]:
    """Return ``(kind, static_positions | None)`` for a spec.

    Static kinds (``last_prompt``, ``all_prompt``, explicit list) can be
    fully resolved once at registration time.  Dynamic kinds
    (``all_generated``, ``all``) must be re-expanded each step.
    """
    positions = spec.positions

    if isinstance(positions, list):
        return "explicit", list(positions)

    if positions == "last_prompt":
        return "last_prompt", [num_prompt_tokens - 1]

    if positions == "all_prompt":
        return "all_prompt", list(range(num_prompt_tokens))

    # Dynamic: will be expanded per-step.
    return positions, None


# ---------------------------------------------------------------------------
# Capture manager
# ---------------------------------------------------------------------------


class CaptureManager:
    """Per-runner multi-consumer capture coordinator.

    Instantiated once per engine worker with an ordered tuple of sinks
    and their (possibly ``None``) global specs.  The manager's lifetime
    matches the runner's.
    """

    def __init__(
        self,
        consumers: tuple[CaptureSink, ...],
        consumer_specs: tuple[CaptureSpec | None, ...],
        num_hidden_layers: int,
        hidden_size: int,
        model_dtype: torch.dtype,
        device: torch.device | str = "cpu",
        finalize_timeout_s: float = 5.0,
    ) -> None:
        if len(consumers) != len(consumer_specs):
            msg = (
                f"consumers length ({len(consumers)}) must match "
                f"consumer_specs length ({len(consumer_specs)})"
            )
            raise ValueError(msg)
        self._consumers = consumers
        self._consumer_specs = consumer_specs
        self._num_hidden_layers = num_hidden_layers
        self._hidden_size = hidden_size
        self._model_dtype = model_dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._finalize_timeout = finalize_timeout_s
        self._requests: dict[str, _RequestCaptureState] = {}
        # Active plan buffered between ``build_step_plan`` (called by the
        # runner pre-forward) and ``on_hook`` fires from inside the
        # compiled forward graph.  Cleared by ``consume_step_plan`` once
        # the runner's finalize path has copied the scratch tensors out.
        self._step_plan: StepCapturePlan | None = None

    # ------------------------------------------------------------------ props

    @property
    def num_consumers(self) -> int:
        return len(self._consumers)

    # ---------------------------------------------------------- registration

    def register_request(
        self,
        req_id: str,
        client_specs: dict[int, CaptureSpec] | None,
        num_prompt_tokens: int,
        sidecar_fields: dict[str, Any] | None = None,
    ) -> None:
        """Register a request for capture.

        ``client_specs`` maps consumer index to a per-request spec.
        These are merged with the global specs: a client spec overrides
        the global spec for that consumer.  A consumer is active for this
        request if it has either a global spec or a client spec.
        """
        if req_id in self._requests:
            msg = f"capture request {req_id!r} is already registered"
            raise ValueError(msg)
        if num_prompt_tokens <= 0:
            msg = (
                f"capture request {req_id!r} has non-positive "
                f"num_prompt_tokens={num_prompt_tokens}"
            )
            raise ValueError(msg)

        merged: dict[int, CaptureSpec] = {}
        for i, global_spec in enumerate(self._consumer_specs):
            if global_spec is not None:
                merged[i] = global_spec

        if client_specs:
            for i, spec in client_specs.items():
                if i < 0 or i >= len(self._consumers):
                    msg = (
                        f"client_specs key {i} out of range [0, {len(self._consumers)})"
                    )
                    raise ValueError(msg)
                merged[i] = spec

        if not merged:
            # No consumer has a spec for this request — nothing to do.
            return

        # Validate hook layers.
        for consumer_idx, spec in merged.items():
            for hook_name, layers in spec.hooks.items():
                for layer_idx in layers:
                    if layer_idx < 0 or layer_idx >= self._num_hidden_layers:
                        msg = (
                            f"capture request {req_id!r} consumer "
                            f"{consumer_idx} hook {hook_name!r} layer "
                            f"{layer_idx} is out of range "
                            f"[0, {self._num_hidden_layers})"
                        )
                        raise ValueError(msg)

        position_kind: dict[int, str] = {}
        static_positions: dict[int, list[int] | None] = {}
        for consumer_idx, spec in merged.items():
            kind, static = _classify_positions(spec, num_prompt_tokens)
            position_kind[consumer_idx] = kind
            static_positions[consumer_idx] = static

        state = _RequestCaptureState(
            req_id=req_id,
            consumer_specs=merged,
            position_kind=position_kind,
            static_positions=static_positions,
            num_prompt_tokens=num_prompt_tokens,
            sidecar_fields=dict(sidecar_fields) if sidecar_fields else {},
        )
        self._requests[req_id] = state

    def unregister_request(self, req_id: str) -> None:
        """Remove all state for ``req_id``.  Silent no-op if unknown."""
        self._requests.pop(req_id, None)

    # ------------------------------------------------------- plan building

    def build_step_plan(
        self,
        batch_view: CaptureBatchView,
    ) -> StepCapturePlan:
        """Build a :class:`StepCapturePlan` for the current batch.

        For each request in the batch that has registered consumers, each
        consumer's position selector is expanded and intersected with the
        step window.  Gather indices reflect the **union** across all
        consumers; each entry's ``consumer_mask`` records which consumers
        want it.
        """
        num_requests = len(batch_view.req_ids)
        if (
            len(batch_view.num_prompt_tokens) != num_requests
            or len(batch_view.num_computed_tokens) != num_requests
            or len(batch_view.num_scheduled_tokens) != num_requests
            or len(batch_view.token_offsets) != num_requests
        ):
            msg = (
                "CaptureBatchView list lengths must match req_ids length "
                f"(got {num_requests})"
            )
            raise ValueError(msg)

        # (layer, hook) -> list of (abs_row, entry_partial)
        # We'll build gather rows and entries together.
        gather_rows: dict[tuple[int, str], list[int]] = {}
        entries: list[CapturePositionEntry] = []
        request_errors: dict[str, str] = {}

        for i in range(num_requests):
            req_id = batch_view.req_ids[i]
            state = self._requests.get(req_id)
            if state is None:
                continue

            if state.error is not None:
                request_errors[req_id] = state.error
                continue

            num_scheduled = batch_view.num_scheduled_tokens[i]
            if num_scheduled <= 0:
                continue

            num_computed = batch_view.num_computed_tokens[i]
            token_offset = batch_view.token_offsets[i]
            step_start = num_computed
            step_end = num_computed + num_scheduled

            # Collect the union of (hook, layers, positions) across all
            # consumers for this request, tracking which consumer wants
            # each (layer, hook, logical_pos).
            #
            # consumer_positions: (layer, hook) -> {logical_pos -> mask}
            consumer_positions: dict[tuple[int, str], dict[int, int]] = defaultdict(
                dict
            )

            has_any = False
            for consumer_idx, spec in state.consumer_specs.items():
                # Resolve positions for this consumer.
                static = state.static_positions[consumer_idx]
                if static is not None:
                    all_positions = static
                else:
                    try:
                        all_positions = _resolve_positions(
                            state.position_kind[consumer_idx],
                            state.num_prompt_tokens,
                            num_computed,
                            num_scheduled,
                        )
                    except ValueError as exc:
                        err_msg = str(exc)
                        state.error = err_msg
                        request_errors[req_id] = err_msg
                        break

                # Intersect with step window.
                in_step = [p for p in all_positions if step_start <= p < step_end]
                if not in_step:
                    continue

                has_any = True
                bit = 1 << consumer_idx

                for hook_name, layers in spec.hooks.items():
                    for layer_idx in layers:
                        key = (layer_idx, hook_name)
                        pos_map = consumer_positions[key]
                        for pos in in_step:
                            pos_map[pos] = pos_map.get(pos, 0) | bit

            # If error was set in the inner loop, skip this request.
            if state.error is not None:
                continue

            if not has_any:
                continue

            # Bump step counter.
            step_index = state.steps_seen
            state.steps_seen += 1

            # Now build gather rows and entries from the union.
            for key in sorted(consumer_positions.keys()):
                pos_map = consumer_positions[key]
                layer_idx, hook_name = key
                rows_list = gather_rows.setdefault(key, [])
                for logical_pos in sorted(pos_map.keys()):
                    mask = pos_map[logical_pos]
                    abs_row = token_offset + (logical_pos - step_start)
                    scratch_row = len(rows_list)
                    rows_list.append(abs_row)
                    entries.append(
                        CapturePositionEntry(
                            request_id=req_id,
                            layer=layer_idx,
                            hook=hook_name,
                            logical_pos=logical_pos,
                            scratch_row=scratch_row,
                            step_index=step_index,
                            consumer_mask=mask,
                        )
                    )

        # Materialize index tensors.  ``gather_indices`` lives on the
        # model's device so ``hidden_states.index_select`` during
        # :meth:`on_hook` is a device-local op.  ``scratch_gpu`` starts
        # empty — :meth:`on_hook` populates it by storing the gathered
        # tensor directly, so there is no point pre-allocating here.
        gather_indices: dict[tuple[int, str], torch.Tensor] = {}
        scratch_gpu: dict[tuple[int, str], torch.Tensor] = {}
        scratch_dtype: dict[tuple[int, str], torch.dtype] = {}
        for key, rows in gather_rows.items():
            gather_indices[key] = torch.tensor(
                rows, dtype=torch.int64, device=self._device
            )
            scratch_dtype[key] = self._model_dtype

        plan = StepCapturePlan(
            gather_indices=gather_indices,
            scratch_gpu=scratch_gpu,
            scratch_dtype=scratch_dtype,
            entries=entries,
            request_errors=request_errors,
        )
        self._step_plan = plan
        return plan

    # ---------------------------------------- runner/custom-op glue helpers

    def set_step_plan(self, plan: StepCapturePlan | None) -> None:
        """Install *plan* as the active plan without re-running the builder.

        Used by unit tests that exercise :meth:`on_hook` in isolation.
        """
        self._step_plan = plan

    def consume_step_plan(self) -> StepCapturePlan | None:
        """Return and clear the active plan.

        The runner's finalize path calls this once per forward step to
        take ownership of scratch tensors before copying them out.
        Returning ``None`` guards the next forward pass against stale
        plans.
        """
        plan = self._step_plan
        self._step_plan = None
        return plan

    def on_hook(
        self,
        layer_idx: int,
        hook_name: str,
        hidden_states: torch.Tensor,
    ) -> None:
        """Custom-op callback fired from inside the compiled forward graph.

        For any ``(layer, hook)`` key the active plan wants, gather the
        rows out of ``hidden_states`` into ``plan.scratch_gpu``.  Keys
        absent from the plan are silently skipped — the op is a no-op
        on any forward step that isn't capturing.

        The tensor passed in is the pristine residual (spec invariant
        1); we must not mutate it.
        """
        plan = self._step_plan
        if plan is None:
            return
        key = (layer_idx, hook_name)
        idx = plan.gather_indices.get(key)
        if idx is None:
            return
        gathered = hidden_states.index_select(0, idx)
        target_dtype = plan.scratch_dtype[key]
        if gathered.dtype != target_dtype:
            gathered = gathered.to(target_dtype)
        plan.scratch_gpu[key] = gathered

    # ----------------------------------------------------- dispatch

    def dispatch_step_captures(self, plan: StepCapturePlan) -> None:
        """Fan out captured rows to each consumer's sink.

        For each consumer, walk the entries where that consumer's bit is
        set in ``consumer_mask``, slice rows out of scratch tensors, and
        call ``submit_chunk``.

        GPU→CPU transfers are coalesced: every scratch tensor is moved
        to host memory non-blocking (one transfer per ``(layer, hook)``
        key), followed by a single ``cuda.synchronize()``.  Consumers
        then slice their rows on the CPU, replacing the previous
        O(consumers × layers) device-sync overhead with O(layers) async
        transfers and O(consumers) cheap CPU index ops.

        Each consumer's dispatch is wrapped in try/except so a failure in
        one sink never blocks delivery to the others.
        """
        if not plan.entries:
            return

        # Transfer all scratch tensors to host in one shot, non-blocking.
        # On-device index_select is not needed here — the union gather
        # already ran in on_hook; we just need the host-side bytes.
        # This replaces the previous O(consumers × layers) per-consumer
        # GPU→CPU round-trips with O(layers) transfers + one sync.
        scratch_cpu: dict[tuple[int, str], torch.Tensor] = {}
        needs_sync = False
        for key, scratch in plan.scratch_gpu.items():
            if scratch.is_cuda:
                scratch_cpu[key] = scratch.to("cpu", non_blocking=True)
                needs_sync = True
            else:
                scratch_cpu[key] = scratch
        if needs_sync:
            torch.cuda.synchronize()

        for consumer_idx, sink in enumerate(self._consumers):
            bit = 1 << consumer_idx

            # Group entries for this consumer by (request_id, layer, hook).
            grouped: dict[tuple[str, int, str], list[CapturePositionEntry]] = (
                defaultdict(list)
            )
            for entry in plan.entries:
                if entry.consumer_mask & bit:
                    grouped_key = (
                        entry.request_id,
                        entry.layer,
                        entry.hook,
                    )
                    grouped[grouped_key].append(entry)

            if not grouped:
                continue

            try:
                for (req_id, layer, hook), chunk_entries in grouped.items():
                    scratch_key = (layer, hook)
                    cpu_scratch = scratch_cpu.get(scratch_key)
                    if cpu_scratch is None:
                        continue

                    # Slice rows for this consumer on the CPU.  All
                    # consumers share the same already-transferred tensor.
                    row_indices = [e.scratch_row for e in chunk_entries]
                    idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=scratch.device)
                    chunk_tensor = scratch.index_select(0, idx_tensor).cpu()

                    step_index = chunk_entries[0].step_index

                    capture_key = (
                        VllmInternalRequestId(req_id),
                        layer,
                        hook,
                    )
                    chunk = CaptureChunk(
                        key=capture_key,
                        tensor=chunk_tensor,
                        dtype=chunk_tensor.dtype,
                        row_offset=0,
                        step_index=step_index,
                        metadata={
                            "consumer_index": consumer_idx,
                            "positions": [e.logical_pos for e in chunk_entries],
                        },
                    )
                    sink.submit_chunk(chunk)
            except Exception:
                logger.exception(
                    "Consumer %d raised during dispatch; "
                    "other consumers are unaffected.",
                    consumer_idx,
                )
                # Record error for each request this consumer was handling.
                for req_id_key in {k[0] for k in grouped}:
                    s = self._requests.get(req_id_key)
                    if s is not None and s.error is None:
                        s.error = f"consumer {consumer_idx} dispatch failed"

    # ----------------------------------------------------- finalization

    def finalize_request(self, req_id: str) -> dict[int, CaptureResult]:
        """Finalize capture for a request across all consumers.

        For each consumer that had a spec for this request, call
        ``submit_finalize`` on the sink and aggregate the terminal
        per-key results.

        Returns a dict mapping consumer index to ``CaptureResult``.
        """
        state = self._requests.pop(req_id, None)
        results: dict[int, CaptureResult] = {}

        if state is None:
            return results

        for consumer_idx, spec in state.consumer_specs.items():
            sink = self._consumers[consumer_idx]

            # Build per-consumer sidecar from request-level sidecar
            # fields plus consumer index.
            sidecar = dict(state.sidecar_fields)
            sidecar["consumer_index"] = consumer_idx

            capture_keys: list[CaptureKey] = []
            for hook_name, layers in spec.hooks.items():
                for layer_idx in layers:
                    capture_key = (
                        VllmInternalRequestId(req_id),
                        layer_idx,
                        hook_name,
                    )
                    capture_keys.append(capture_key)
                    finalize = CaptureFinalize(
                        key=capture_key,
                        sidecar=sidecar,
                    )
                    try:
                        sink.submit_finalize(finalize)
                    except Exception:
                        logger.exception(
                            "Consumer %d submit_finalize failed for %s",
                            consumer_idx,
                            capture_key,
                        )

            if capture_keys:
                per_key_results: list[CaptureResult] = []
                for capture_key in capture_keys:
                    try:
                        result = sink.wait_for_result(
                            capture_key,
                            timeout=self._finalize_timeout,
                        )
                    except Exception:
                        logger.exception(
                            "Consumer %d wait_for_result failed for %s",
                            consumer_idx,
                            capture_key,
                        )
                        result = None

                    if result is None:
                        result = CaptureResult(
                            key=capture_key,
                            status="error",
                            error=f"finalize timed out for {capture_key}",
                        )
                    per_key_results.append(result)

                results[consumer_idx] = _aggregate_capture_results(per_key_results)
            else:
                # No hooks in spec — unusual but not impossible.
                dummy_key = (
                    VllmInternalRequestId(req_id),
                    0,
                    "post_mlp",
                )
                results[consumer_idx] = CaptureResult(
                    key=dummy_key,
                    status="not_requested",
                )

        return results

    # ----------------------------------------------------- error recording

    def record_request_error(self, req_id: str, message: str) -> None:
        """Record a terminal error for ``req_id``."""
        state = self._requests.get(req_id)
        if state is not None:
            state.error = message

    # ----------------------------------------------------- queries

    def is_active(self) -> bool:
        """True if any requests are registered."""
        return bool(self._requests)

    def has_request(self, req_id: str) -> bool:
        """True if ``req_id`` is registered."""
        return req_id in self._requests


def _aggregate_capture_results(results: list[CaptureResult]) -> CaptureResult:
    """Reduce per-key capture results into one per-consumer result."""
    if not results:
        raise ValueError("results must not be empty")

    worst_severity = max(_CAPTURE_RESULT_SEVERITY[r.status] for r in results)
    representative = next(
        result
        for result in results
        if _CAPTURE_RESULT_SEVERITY[result.status] == worst_severity
    )
    errors = [result.error for result in results if result.error]

    if len(results) == 1:
        payload: Any = results[0].payload
    else:
        payload = {result.key: result.payload for result in results}

    return CaptureResult(
        key=representative.key,
        status=representative.status,
        error="; ".join(errors) if errors else None,
        payload=payload,
    )


__all__ = [
    "CaptureManager",
    "_aggregate_capture_results",
]
