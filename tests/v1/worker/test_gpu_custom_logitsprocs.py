# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the V2 model runner custom logits processor bridge.

These tests exercise the CPU-side batch-diff and output-token-tracking
logic in CustomLogitsprocs and do not require a GPU.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    LogitsProcessors,
)
from vllm.v1.worker.gpu.sample.logitsprocs import CustomLogitsprocs

MAX_NUM_REQS = 8
VOCAB_SIZE = 32


class RecordingLogitsProcessor(LogitsProcessor):
    """Records update_state calls and masks all but a target token."""

    def __init__(self, argmax_invariant: bool = False):
        self.argmax_invariant = argmax_invariant
        self.batch_updates: list[BatchUpdate | None] = []
        # Row index -> (params, prompt_tok_ids, output_tok_ids)
        self.rows: dict[int, tuple[SamplingParams, list[int] | None, list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return self.argmax_invariant

    def update_state(self, batch_update: BatchUpdate | None):
        self.batch_updates.append(batch_update)
        if not batch_update:
            return
        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            self.rows[index] = (params, prompt_tok_ids, output_tok_ids)
        for index in batch_update.removed:
            self.rows.pop(index, None)
        assert not batch_update.moved

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row in self.rows:
            logits[row] += 1.0
        return logits


class FakeEvent:
    def synchronize(self) -> None:
        pass


def make_input_batch(slots: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        idx_mapping_np=np.array(slots, dtype=np.int32),
        num_draft_tokens=0,
    )


def make_bridge() -> tuple[CustomLogitsprocs, RecordingLogitsProcessor]:
    logitproc = RecordingLogitsProcessor()
    bridge = CustomLogitsprocs(MAX_NUM_REQS, LogitsProcessors([logitproc]))
    return bridge, logitproc


def add_request(
    bridge: CustomLogitsprocs,
    req_idx: int,
    req_id: str,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
) -> None:
    prompt_token_ids = prompt_token_ids or [1, 2, 3]
    prefill_token_ids = prompt_token_ids + (output_token_ids or [])
    bridge.add_request(
        req_idx,
        req_id,
        len(prompt_token_ids),
        SamplingParams(),
        prompt_token_ids,
        prefill_token_ids,
    )


def register_sampled(
    bridge: CustomLogitsprocs,
    input_batch: SimpleNamespace,
    sampled: list[int],
    num_sampled: list[int] | None = None,
) -> None:
    num_rows = len(sampled)
    bridge.register_step_output(
        input_batch,
        np.array(sampled, dtype=np.int64).reshape(num_rows, 1),
        np.array(num_sampled or [1] * num_rows, dtype=np.int32),
        FakeEvent(),
    )


def test_added_and_stable_batch():
    bridge, logitproc = make_bridge()
    add_request(bridge, req_idx=3, req_id="a", prompt_token_ids=[7, 8])
    add_request(bridge, req_idx=5, req_id="b", prompt_token_ids=[9])

    input_batch = make_input_batch([5, 3])
    bridge.update_state(input_batch)

    (batch_update,) = logitproc.batch_updates
    assert batch_update is not None
    assert batch_update.batch_size == 2
    assert not batch_update.removed
    # Row 0 -> slot 5 ("b"), row 1 -> slot 3 ("a").
    assert [(added[0], added[2]) for added in batch_update.added] == [
        (0, [9]),
        (1, [7, 8]),
    ]

    # Unchanged batch: logitsprocs must still be notified with None.
    bridge.update_state(input_batch)
    assert logitproc.batch_updates[-1] is None


def test_row_reorder_and_removal():
    bridge, logitproc = make_bridge()
    add_request(bridge, req_idx=0, req_id="a")
    add_request(bridge, req_idx=1, req_id="b")
    add_request(bridge, req_idx=2, req_id="c")
    bridge.update_state(make_input_batch([0, 1, 2]))
    assert set(logitproc.rows) == {0, 1, 2}

    # Request at slot 1 finishes; remaining rows reorder.
    bridge.remove_request(1)
    bridge.update_state(make_input_batch([2, 0]))

    batch_update = logitproc.batch_updates[-1]
    assert batch_update is not None
    assert batch_update.batch_size == 2
    # Trailing row vacated; reordered rows are re-added in place.
    assert batch_update.removed == [2]
    assert [added[0] for added in batch_update.added] == [0, 1]
    assert set(logitproc.rows) == {0, 1}

    # The re-added rows carry the same live output token lists.
    assert logitproc.rows[0][2] is bridge.records[2].output_token_ids
    assert logitproc.rows[1][2] is bridge.records[0].output_token_ids


def test_sampled_tokens_extend_output_lists():
    bridge, logitproc = make_bridge()
    add_request(bridge, req_idx=0, req_id="a")
    add_request(bridge, req_idx=1, req_id="b", output_token_ids=[42])

    input_batch = make_input_batch([0, 1])
    bridge.update_state(input_batch)
    # Resumed request "b" starts with its prior output tokens.
    assert logitproc.rows[1][2] == [42]

    # Request 0 samples token 11; request 1 is a partial prefill (0 tokens).
    register_sampled(bridge, input_batch, sampled=[11, 99], num_sampled=[1, 0])
    bridge.update_state(input_batch)

    assert logitproc.rows[0][2] == [11]
    assert logitproc.rows[1][2] == [42]

    register_sampled(bridge, input_batch, sampled=[12, 13])
    bridge.update_state(input_batch)
    assert logitproc.rows[0][2] == [11, 12]
    assert logitproc.rows[1][2] == [42, 13]


def test_pending_tokens_skipped_after_slot_reuse():
    bridge, logitproc = make_bridge()
    add_request(bridge, req_idx=0, req_id="a")
    input_batch = make_input_batch([0])
    bridge.update_state(input_batch)
    register_sampled(bridge, input_batch, sampled=[11])

    # Request "a" finishes and slot 0 is reused before the drain.
    bridge.remove_request(0)
    add_request(bridge, req_idx=0, req_id="b")
    bridge.update_state(make_input_batch([0]))

    # The pending token belongs to "a" and must not leak into "b".
    assert logitproc.rows[0][2] == []


def test_apply_skipped_without_active_requests():
    bridge, logitproc = make_bridge()
    # Dummy sampler runs use slots with no tracked requests.
    bridge.update_state(make_input_batch([0, 1]))
    assert logitproc.batch_updates == [None]

    logits = torch.zeros(2, VOCAB_SIZE)
    assert bridge.apply_non_argmax_invariant(logits) is logits
    assert torch.all(logits == 0)


def test_apply_routes_by_argmax_invariance():
    invariant = RecordingLogitsProcessor(argmax_invariant=True)
    non_invariant = RecordingLogitsProcessor(argmax_invariant=False)
    bridge = CustomLogitsprocs(
        MAX_NUM_REQS, LogitsProcessors([invariant, non_invariant])
    )
    add_request(bridge, req_idx=0, req_id="a")
    bridge.update_state(make_input_batch([0]))

    logits = torch.zeros(1, VOCAB_SIZE)
    bridge.apply_non_argmax_invariant(logits)
    assert torch.all(logits == 1.0)
    bridge.apply_argmax_invariant(logits)
    assert torch.all(logits == 2.0)
