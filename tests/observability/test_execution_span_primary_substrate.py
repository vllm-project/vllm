from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import multiprocessing
import pickle
from types import SimpleNamespace

import pytest

import vllm.execution_span as span


def _config(engine=(11, 12)):
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            primary_engine_instance_id_hi=engine[0],
            primary_engine_instance_id_lo=engine[1],
        )
    )


def _output(step=7, output=9, tokens=4):
    return SimpleNamespace(
        total_num_scheduled_tokens=tokens,
        primary_scheduler_step_id=step,
        primary_scheduler_output_id=output,
        num_scheduled_tokens={"r": tokens} if tokens else {},
        execution_span_context=None,
    )


def _multiprocess_roundtrip(context, queue):
    queue.put(pickle.loads(pickle.dumps(context)))


@pytest.fixture(autouse=True)
def _span_environment(monkeypatch):
    monkeypatch.setenv("VLLM_PRIMARY_PROFILE", "primary_current_plus_execution_span")
    monkeypatch.setenv(
        "VLLM_PRIMARY_TRACE_SESSION_ID", "01020304050607081112131415161718"
    )


def test_allocation_first_monotonic_zero_token_and_missing_parent(monkeypatch):
    emitted = []
    monkeypatch.setattr(span, "emit_execution_span_created", emitted.append)
    manager = span.ExecutionSpanManager(_config())
    first = manager.create_if_executable(_output(step=1, output=1))
    assert first is not None and first.execution_span_id == 1
    assert manager.create_if_executable(_output(tokens=0)) is None
    assert manager.create_if_executable(
        _output(step=(1 << 64) - 1, output=2)
    ) is None
    second = manager.create_if_executable(_output(step=2, output=2))
    assert second is not None and second.execution_span_id == 2
    assert [item.execution_span_id for item in emitted] == [1, 2]


def test_retry_gets_new_span_and_incremented_attempt(monkeypatch):
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    manager = span.ExecutionSpanManager(_config())
    first = manager.create_if_executable(_output())
    second = manager.create_if_executable(_output())
    assert first is not None and second is not None
    assert (first.execution_span_id, first.execution_attempt_no) == (1, 0)
    assert (second.execution_span_id, second.execution_attempt_no) == (2, 1)


def test_context_is_frozen_pickle_stable_and_uint64_safe(monkeypatch):
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    context = span.ExecutionSpanManager(_config()).create_if_executable(_output())
    assert context is not None
    assert pickle.loads(pickle.dumps(context)) == context
    with pytest.raises(FrozenInstanceError):
        context.execution_span_id = 3  # type: ignore[misc]


def test_context_survives_spawn_multiprocess_transport(monkeypatch):
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    context = span.ExecutionSpanManager(_config()).create_if_executable(_output())
    assert context is not None
    process_context = multiprocessing.get_context("spawn")
    queue = process_context.Queue()
    process = process_context.Process(
        target=_multiprocess_roundtrip, args=(context, queue)
    )
    process.start()
    transported = queue.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    assert transported == context


def test_engine_namespace_and_uint64_boundary(monkeypatch):
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    left = span.ExecutionSpanManager(_config((11, 12)))
    right = span.ExecutionSpanManager(_config((21, 22)))
    left._next_span_id = span.MAX_EXECUTION_SPAN_ID
    boundary = left.create_if_executable(_output())
    other = right.create_if_executable(_output())
    assert boundary and other
    assert boundary.execution_span_id == span.MAX_EXECUTION_SPAN_ID
    assert boundary.engine_instance_id != other.engine_instance_id
    with pytest.raises(OverflowError):
        left.create_if_executable(_output(step=8, output=10))


def test_async_reordering_retains_original_context(monkeypatch):
    ended = []
    monkeypatch.setattr(
        span,
        "emit_execution_span_rank_end",
        lambda context, **kwargs: ended.append(context.execution_span_id),
    )
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    manager = span.ExecutionSpanManager(_config())
    context_a = manager.create_if_executable(_output(step=1, output=1))
    context_b = manager.create_if_executable(_output(step=2, output=2))
    assert context_a and context_b
    output_a = SimpleNamespace()
    output_b = SimpleNamespace()
    span.bind_async_execution_span(
        output_a,
        context=context_a,
        worker_instance_hash=1,
        worker_rank=0,
        gpu_uuid_hash=2,
    )
    span.bind_async_execution_span(
        output_b,
        context=context_b,
        worker_instance_hash=1,
        worker_rank=0,
        gpu_uuid_hash=2,
    )
    assert span.complete_async_execution_span_once(output_b)
    assert span.complete_async_execution_span_once(output_a)
    assert ended == [context_b.execution_span_id, context_a.execution_span_id]


def test_async_rank_end_and_terminal_are_exactly_once(monkeypatch):
    rank_ends = []
    terminals = []
    monkeypatch.setattr(
        span,
        "emit_execution_span_rank_end",
        lambda context, **kwargs: rank_ends.append(context.execution_span_id),
    )
    monkeypatch.setattr(
        span,
        "emit_execution_span_terminal",
        lambda context, status: terminals.append((context.execution_span_id, status)),
    )
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    manager = span.ExecutionSpanManager(_config())
    context = manager.create_if_executable(_output())
    assert context
    output = SimpleNamespace()
    span.bind_async_execution_span(
        output,
        context=context,
        worker_instance_hash=1,
        worker_rank=0,
        gpu_uuid_hash=2,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: span.complete_async_execution_span_once(output), range(32))) == 1
        assert sum(
            pool.map(
                lambda _: manager.terminal_once(
                    context, span.TerminalStatus.COMPLETED
                ),
                range(32),
            )
        ) == 1
    assert rank_ends == [context.execution_span_id]
    assert terminals == [(context.execution_span_id, span.TerminalStatus.COMPLETED)]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (span.TerminalStatus.COMPLETED, span.TerminalStatus.NOT_EVALUABLE),
        (span.TerminalStatus.FAILED_IN_WORKER, span.TerminalStatus.COMPLETED),
        (
            span.TerminalStatus.ABORTED_BEFORE_DISPATCH,
            span.TerminalStatus.FAILED_IN_WORKER,
        ),
        (
            span.TerminalStatus.PARTIAL_WORKER_COMPLETION,
            span.TerminalStatus.NOT_EVALUABLE,
        ),
        (
            span.TerminalStatus.FAILED_BEFORE_WORKER,
            span.TerminalStatus.ABORTED_BEFORE_DISPATCH,
        ),
    ],
)
def test_terminal_race_matrix_is_exactly_once(monkeypatch, left, right):
    emitted = []
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    monkeypatch.setattr(
        span,
        "emit_execution_span_terminal",
        lambda context, status: emitted.append((context.execution_span_id, status)),
    )
    manager = span.ExecutionSpanManager(_config())
    context = manager.create_if_executable(_output())
    assert context
    statuses = [left, right] * 16
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(
            pool.map(lambda status: manager.terminal_once(context, status), statuses)
        )
    assert sum(accepted) == 1
    assert len(emitted) == 1


def test_sync_and_async_rank_end_paths_are_mutually_exclusive(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        span,
        "emit_execution_span_rank_end",
        lambda context, **kwargs: emitted.append(
            (context.execution_span_id, kwargs["completion_semantics"], kwargs["failed"])
        ),
    )
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    context = span.ExecutionSpanManager(_config()).create_if_executable(_output())
    assert context
    output = SimpleNamespace()
    span.bind_async_execution_span(
        output,
        context=context,
        worker_instance_hash=1,
        worker_rank=0,
        gpu_uuid_hash=2,
    )
    assert emitted == []
    assert span.complete_async_execution_span_once(output, failed=True)
    assert not span.complete_async_execution_span_once(output)
    assert emitted == [
        (
            context.execution_span_id,
            span.RankCompletionSemantics.ASYNC_OUTPUT_READY,
            True,
        )
    ]


def test_worker_validation_rejects_parent_rebinding(monkeypatch):
    monkeypatch.setattr(span, "emit_execution_span_created", lambda context: None)
    output = _output()
    context = span.ExecutionSpanManager(_config()).create_if_executable(output)
    assert context
    output.execution_span_context = context
    assert span.validate_worker_context(output) == context
    output.primary_scheduler_output_id += 1
    with pytest.raises(span.PrimaryUSDTError, match="parent mismatch"):
        span.validate_worker_context(output)


def test_primary_current_profile_has_no_execution_context(monkeypatch):
    monkeypatch.setenv("VLLM_PRIMARY_PROFILE", "primary_current")
    manager = span.ExecutionSpanManager(_config())
    output = _output()
    assert manager.create_if_executable(output) is None
    assert span.validate_worker_context(output) is None
