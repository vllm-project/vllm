# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import nullcontext
from unittest.mock import Mock, call

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PPHandler


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


def test_deferred_receive_cadence_fifo_and_flush():
    handler = PPHandler.__new__(PPHandler)
    slots = [Mock(launched=False) for _ in range(3)]
    handler.queue = deque([None, slots[0], slots[1], slots[2]])
    handler.recv_launch_delay = 3
    handler.pending_post_model_receive = None
    handler.is_last_rank = False
    handler._launch_receive = Mock(
        side_effect=lambda slot: setattr(slot, "launched", True)
    )

    handler._advance_receive_queue()
    handler._advance_receive_queue()  # Launches the older pending slot first.
    handler.launch_post_model_receive()
    handler.flush_pending_collectives()
    handler.flush_pending_collectives()

    assert handler._launch_receive.call_args_list == [
        call(slots[0]),
        call(slots[1]),
        call(slots[2]),
    ]


def test_receive_launch_is_idempotent_when_cpu_event_is_none(monkeypatch):
    handler = PPHandler.__new__(PPHandler)
    handler.main_stream, handler.broadcast_stream = Mock(), Mock()
    handler.broadcast_stream.record_event.return_value = None
    handler.last_rank, handler.broadcast_group = 3, Mock()
    tensors = [Mock(), Mock(), Mock()]
    slot = Mock(
        launched=False,
        event=None,
        sampled_tokens=tensors[0],
        combined=tensors[1],
        draft_tokens=tensors[2],
    )
    broadcast = Mock()
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    handler._launch_receive(slot)
    handler._launch_receive(slot)

    assert slot.launched
    handler.broadcast_stream.wait_stream.assert_called_once_with(handler.main_stream)
    assert [item.args[0] for item in broadcast.call_args_list] == tensors
