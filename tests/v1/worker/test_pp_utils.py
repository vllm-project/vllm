# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler


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


def _make_handler(pp_size: int, delay: int, post_model: bool = False):
    handler = PPHandler.__new__(PPHandler)
    handler.queue = deque([None] * pp_size)
    handler.recv_launch_delay = delay
    handler.post_model_recv_launch = post_model
    handler.pending_post_model_receive = None
    handler.is_last_rank = False
    handler.step = -1
    launches = []

    def launch(slot: PendingRecv) -> None:
        fake_slot = cast(_FakeSlot, slot)
        if fake_slot.event is None:
            launches.append((fake_slot.sequence, handler.step))
            fake_slot.event = object()

    handler._launch_receive = launch
    return handler, launches


@pytest.mark.parametrize(
    ("pp_size", "delay", "post_model"),
    [(4, 0, False), (2, 1, False), (4, 2, False), (4, 3, True)],
)
def test_receive_launch_and_consume_cadence(pp_size, delay, post_model):
    handler, launches = _make_handler(pp_size, delay, post_model)
    consumed = []

    for step in range(8 + pp_size):
        handler.step = step
        if (slot := handler._advance_receive_queue()) is not None:
            consumed.append((cast(_FakeSlot, slot).sequence, step))
        if post_model:
            handler.launch_post_model_receive()
        if step < 8:
            handler._queue_receive(cast(PendingRecv, _FakeSlot(step)))

    assert launches == [(origin, origin + delay) for origin in range(8)]
    assert consumed == [(origin, origin + pp_size) for origin in range(8)]


def test_flush_posts_each_pending_receive_once():
    handler, launches = _make_handler(pp_size=4, delay=3, post_model=True)
    slots = [_FakeSlot(0), _FakeSlot(1)]
    handler.pending_post_model_receive = cast(PendingRecv, slots[0])
    handler.queue[0] = cast(PendingRecv, slots[0])
    handler.queue[2] = cast(PendingRecv, slots[1])

    handler.flush_pending_collectives()
    handler.flush_pending_collectives()

    assert [sequence for sequence, _ in launches] == [0, 1]


def test_deferred_receive_includes_speculative_drafts(monkeypatch):
    handler = PPHandler.__new__(PPHandler)
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_stream.record_event.return_value = Mock()
    handler.last_rank = 3
    handler.broadcast_group = Mock()
    sampled_tokens, combined, draft_tokens = Mock(), Mock(), Mock()
    slot = PendingRecv(
        None,
        sampled_tokens,
        combined,
        Mock(),
        Mock(),
        Mock(),
        np.array([0]),
        np.array([True]),
        np.array([0]),
        draft_tokens,
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
