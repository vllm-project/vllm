# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from enum import Enum
from typing import Any

import torch


def _record_result_stream(result: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(result, torch.Tensor):
        result.record_stream(stream)
    elif isinstance(result, (tuple, list)):
        for item in result:
            _record_result_stream(item, stream)
    elif isinstance(result, dict):
        for item in result.values():
            _record_result_stream(item, stream)


class AuxStreamType(Enum):
    Attention = 1


class EventType(Enum):
    Main = 0
    Attention = 1


def maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: torch.cuda.Stream | None = None,
) -> tuple[Any, Any]:
    """Run two functions potentially in parallel on separate CUDA streams.

    When aux_stream is provided, fn0 runs on the current (default) stream and
    fn1 runs on aux_stream, synchronized via CUDA events.  When aux_stream is
    None, both functions execute sequentially on the current stream.

    This design follows TensorRT-LLM's maybe_execute_in_parallel pattern
    (tensorrt_llm/_torch/modules/multi_stream_utils.py).

    Args:
        fn0: Callable for the default stream.
        fn1: Callable for the auxiliary stream.
        event0: CUDA event recorded before fn0 so aux_stream can wait.
        event1: CUDA event recorded after fn1 so default stream can wait.
        aux_stream: The second CUDA stream for fn1.
            Multi-stream is disabled when aux_stream is None.

    Returns:
        Tuple of (fn0_result, fn1_result).
    """
    if aux_stream is not None:
        current_stream = torch.cuda.current_stream()
        event0.record(current_stream)
        result0 = fn0()
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(event0)
            result1 = fn1()
            event1.record(aux_stream)
        current_stream.wait_event(event1)
        _record_result_stream(result1, current_stream)
    else:
        result0 = fn0()
        result1 = fn1()
    return (result0, result1)


def execute_in_parallel(
    default_fn: Callable[[], Any],
    aux_fns: list[Callable[[], Any] | None],
    start_event: torch.cuda.Event,
    done_events: list[torch.cuda.Event],
    aux_streams: list[torch.cuda.Stream] | None = None,
    enable: bool = False,
) -> tuple[Any, list[Any]]:
    """Run default_fn on the current stream and aux_fns concurrently on
    aux_streams.

    Generalizes maybe_execute_in_parallel to N aux callables. Slots where
    aux_fns[i] is None are skipped (no stream switch, no event record); their
    corresponding entry in the returned aux_results list is None.

    start_event fans out from the current stream to every launched aux stream;
    done_events[i] is recorded after aux_fns[i] so the current stream joins
    before returning. The default-stream function is enqueued before the aux
    functions so the critical path is not delayed by CPU launch overhead from
    side-stream branches. Falls back to sequential execution on the current
    stream when aux_streams is None or enable is False; in that case default_fn
    runs first, then aux_fns in order.

    Args:
        default_fn: Callable for the default (current) stream.
        aux_fns: Per-aux callables; entries may be None to skip.
        start_event: CUDA event recorded on the current stream before
            default_fn so each launched aux stream can wait on it.
        done_events: One CUDA event per aux slot, recorded after the
            corresponding aux_fn. Length must match aux_fns.
        aux_streams: Per-aux CUDA streams. Length must match aux_fns.
            Multi-stream is disabled when None.
        enable: Opt-in switch for the multi-stream path. Defaults to False,
            so callers that pass aux_streams must also pass enable=True
            (typically gated by an env var) to actually overlap. When False,
            execution falls back to sequential on the current stream.

    Returns:
        Tuple of (default_result, aux_results) where aux_results[i] is the
        result of aux_fns[i] (or None when skipped).
    """
    aux_results: list[Any]
    if aux_streams is None or not enable:
        default_result = default_fn()
        aux_results = [fn() if fn is not None else None for fn in aux_fns]
        return default_result, aux_results

    assert len(aux_fns) == len(aux_streams) == len(done_events), (
        "aux_fns, aux_streams, and done_events must be the same length"
    )

    aux_results = [None] * len(aux_fns)
    current_stream = torch.cuda.current_stream()
    pending: list[tuple[torch.cuda.Event, Any]] = []

    start_event.record(current_stream)
    default_result = default_fn()

    for i, fn in enumerate(aux_fns):
        if fn is None:
            continue
        aux_stream = aux_streams[i]
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(start_event)
            aux_results[i] = fn()
            done_events[i].record(aux_stream)
        pending.append((done_events[i], aux_results[i]))

    for ev, result in pending:
        current_stream.wait_event(ev)
        _record_result_stream(result, current_stream)

    return default_result, aux_results
