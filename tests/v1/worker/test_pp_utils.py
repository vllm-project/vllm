# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pipeline-parallel sampled-token handling."""

from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import (
    PendingRecv,
    PPHandler,
    compute_need_sampled_mask,
)


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


@dataclass
class _FakeSlot:
    sequence: int
    event: object | None = None


def _make_queue_handler(
    pp_size: int, delay: int, post_model: bool = False
) -> tuple[PPHandler, list[tuple[int, int]], Callable[[int], None]]:
    handler = PPHandler.__new__(PPHandler)
    handler.queue = deque([None] * pp_size)
    handler.recv_launch_delay = delay
    handler.post_model_recv_launch = post_model
    handler.pending_post_model_receive = None
    handler.is_last_rank = False
    handler.num_deferred_recv_launches = 0
    handler.num_idle_boundaries = 0
    handler.num_idle_flushes = 0
    handler.num_idle_flushed_receives = 0
    handler.num_consume_fallbacks = 0

    launches: list[tuple[int, int]] = []
    current_step = -1

    def launch(slot: PendingRecv) -> None:
        fake_slot = cast(_FakeSlot, slot)
        launches.append((fake_slot.sequence, current_step))
        fake_slot.event = object()

    handler._launch_receive = launch  # type: ignore[method-assign]

    def set_current_step(step: int) -> None:
        nonlocal current_step
        current_step = step

    return handler, launches, set_current_step


@pytest.mark.parametrize(("pp_size", "delay"), [(2, 1), (4, 1), (4, 2), (4, 3)])
def test_deferred_receive_launch_and_consume_cadence(pp_size: int, delay: int) -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size, delay)
    consumed: list[tuple[int, int]] = []
    num_origin_steps = 8

    for step in range(num_origin_steps + pp_size):
        set_current_step(step)
        due_slot = handler._advance_receive_queue()
        if due_slot is not None:
            fake_slot = cast(_FakeSlot, due_slot)
            consumed.append((fake_slot.sequence, step))
        if step < num_origin_steps:
            handler._queue_receive(cast(PendingRecv, _FakeSlot(step)))

    assert launches == [(origin, origin + delay) for origin in range(num_origin_steps)]
    assert consumed == [
        (origin, origin + pp_size) for origin in range(num_origin_steps)
    ]


def test_deferred_receive_empty_steps_preserve_collective_order() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    sampled_steps = {0, 2, 5}
    consumed: list[int] = []

    for step in range(10):
        set_current_step(step)
        due_slot = handler._advance_receive_queue()
        if due_slot is not None:
            consumed.append(cast(_FakeSlot, due_slot).sequence)
        if step in sampled_steps:
            handler._queue_receive(cast(PendingRecv, _FakeSlot(step)))

    assert [sequence for sequence, _ in launches] == sorted(sampled_steps)
    assert consumed == sorted(sampled_steps)


def test_flush_pending_collectives_is_idempotent() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    handler.is_last_rank = False
    slots = [_FakeSlot(0), _FakeSlot(1)]
    handler.queue = deque(
        [cast(PendingRecv, slots[0]), None, cast(PendingRecv, slots[1]), None]
    )
    set_current_step(7)

    assert handler.flush_pending_collectives() == 2
    assert handler.flush_pending_collectives() == 0
    assert launches == [(0, 7), (1, 7)]


def test_immediate_receive_launches_at_origin_step() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=0)
    set_current_step(3)

    handler._queue_receive(cast(PendingRecv, _FakeSlot(3)))

    assert launches == [(3, 3)]


def test_launch_receive_includes_speculative_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = PPHandler.__new__(PPHandler)
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_stream.record_event.return_value = Mock()
    handler.last_rank = 3
    handler.broadcast_group = Mock()
    handler.recv_launch_delay = 3
    handler.num_deferred_recv_launches = 0

    sampled_tokens = Mock()
    combined = Mock()
    draft_tokens = Mock()
    slot = PendingRecv(
        event=None,
        sampled_tokens=sampled_tokens,
        combined=combined,
        num_sampled=Mock(),
        num_rejected=Mock(),
        idx_mapping=Mock(),
        idx_mapping_np=np.array([0]),
        need_sampled_mask=np.array([True]),
        gen_at_receive_np=np.array([0]),
        draft_tokens=draft_tokens,
    )
    broadcast = Mock()
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    handler._launch_receive(slot)

    assert broadcast.call_args_list == [
        call(sampled_tokens, src=3, group=handler.broadcast_group),
        call(combined, src=3, group=handler.broadcast_group),
        call(draft_tokens, src=3, group=handler.broadcast_group),
    ]
    assert slot.event is handler.broadcast_stream.record_event.return_value
    sampled_tokens.record_stream.assert_called_once_with(handler.main_stream)
    combined.record_stream.assert_called_once_with(handler.main_stream)
    draft_tokens.record_stream.assert_called_once_with(handler.main_stream)
    assert handler.num_deferred_recv_launches == 1


def test_post_model_receive_waits_for_explicit_launch() -> None:
    handler, launches, set_current_step = _make_queue_handler(
        pp_size=4, delay=3, post_model=True
    )
    handler.queue[-1] = cast(PendingRecv, _FakeSlot(0))

    for step in range(1, 4):
        set_current_step(step)
        handler._advance_receive_queue()

    assert launches == []
    assert handler.launch_post_model_receive()
    assert launches == [(0, 3)]
    assert not handler.launch_post_model_receive()


def test_post_model_receive_must_launch_before_next_selection() -> None:
    handler, _, set_current_step = _make_queue_handler(
        pp_size=4, delay=1, post_model=True
    )
    handler.queue[-1] = cast(PendingRecv, _FakeSlot(0))
    set_current_step(1)
    handler._advance_receive_queue()
    handler.queue[-1] = cast(PendingRecv, _FakeSlot(1))

    set_current_step(2)
    with pytest.raises(RuntimeError, match="Previous post-model"):
        handler._advance_receive_queue()


def test_flush_launches_pending_post_model_receive_once() -> None:
    handler, launches, set_current_step = _make_queue_handler(
        pp_size=4, delay=3, post_model=True
    )
    slot = _FakeSlot(0)
    handler.pending_post_model_receive = cast(PendingRecv, slot)
    handler.queue[0] = cast(PendingRecv, slot)
    set_current_step(3)

    assert handler.flush_pending_collectives() == 1
    assert handler.flush_pending_collectives() == 0
    assert launches == [(0, 3)]


def test_idle_flush_is_counted() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    handler.is_last_rank = False
    handler.queue[-1] = cast(PendingRecv, _FakeSlot(0))
    set_current_step(1)

    assert handler.flush_pending_collectives(reason="idle") == 1
    assert handler.num_idle_boundaries == 1
    assert handler.num_idle_flushes == 1
    assert handler.num_idle_flushed_receives == 1
    assert launches == [(0, 1)]


def test_consume_fallback_is_counted() -> None:
    handler, launches, set_current_step = _make_queue_handler(pp_size=4, delay=3)
    slot = cast(PendingRecv, _FakeSlot(0))
    set_current_step(4)

    handler._ensure_receive_launched(slot)

    assert handler.num_consume_fallbacks == 1
    assert launches == [(0, 4)]


@pytest.mark.parametrize(
    ("old_computed", "scheduled", "prefill_len", "max_seq_len", "expected"),
    [
        (0, 64, 128, 256, None),  # Non-final chunked prefill.
        (0, 128, 128, 129, [True]),  # Finishing is decided by the scheduler.
        (0, 128, 128, 130, [True]),  # Another decode step needs feedback.
    ],
)
def test_compute_need_sampled_mask_is_shared_skip_predicate(
    old_computed: int,
    scheduled: int,
    prefill_len: int,
    max_seq_len: int,
    expected: list[bool] | None,
) -> None:
    input_batch = type(
        "InputBatchStub",
        (),
        {
            "num_computed_tokens_np": np.array([old_computed]),
            "prefill_len_np": np.array([prefill_len]),
            "max_seq_len_np": np.array([max_seq_len]),
            "num_scheduled_tokens": np.array([scheduled]),
        },
    )()

    result = compute_need_sampled_mask(cast(Any, input_batch))

    assert (None if result is None else result.tolist()) == expected


def test_last_rank_has_no_receives_to_flush() -> None:
    handler, launches, _ = _make_queue_handler(pp_size=4, delay=3)
    handler.is_last_rank = True

    assert handler.flush_pending_collectives() == 0
    assert launches == []
