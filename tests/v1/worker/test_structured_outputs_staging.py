# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 worker's structured-output staging buffers.

`StructuredOutputsWorker.apply_grammar_bitmask` runs on every decode step of
every batch containing structured-output requests, so the host->device staging
must not allocate per step. These tests pin down both the buffer reuse and the
resulting bitmask/mapping values.
"""

import numpy as np
import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="StructuredOutputsWorker requires CUDA"
)

VOCAB_SIZE = 64
BITMASK_WIDTH = cdiv(VOCAB_SIZE, 32)


class _FakeInputBatch:
    """Minimal stand-in exposing only what apply_grammar_bitmask reads."""

    def __init__(self, req_ids: list[str], cu_num_logits: list[int]):
        self.req_ids = req_ids
        self.cu_num_logits_np = np.array(cu_num_logits, dtype=np.int32)


def _make_worker(max_num_logits: int = 4) -> StructuredOutputsWorker:
    return StructuredOutputsWorker(
        max_num_logits=max_num_logits,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cuda"),
    )


def test_staging_buffers_are_preallocated():
    """Staging memory is allocated once, sized to the batch upper bound."""
    worker = _make_worker(max_num_logits=8)

    assert worker.grammar_bitmask.cpu.shape == (8, BITMASK_WIDTH)
    assert worker.logits_indices.cpu.shape == (8,)
    # The numpy views alias the staging tensors, so writes land in pinned memory
    # rather than in a fresh per-step allocation.
    assert worker.grammar_bitmask.np.base is not None
    assert worker.logits_indices.np.base is not None


def test_no_reallocation_across_steps():
    """Repeated calls must reuse the same host and device buffers."""
    worker = _make_worker(max_num_logits=4)
    input_batch = _FakeInputBatch(req_ids=["a", "b"], cu_num_logits=[0, 1, 2])

    ptrs = (
        worker.grammar_bitmask.cpu.data_ptr(),
        worker.grammar_bitmask.gpu.data_ptr(),
        worker.logits_indices.cpu.data_ptr(),
        worker.logits_indices.gpu.data_ptr(),
    )

    for step in range(4):
        num_masks = 1 + step % 2
        logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device="cuda")
        bitmask = np.full((num_masks, BITMASK_WIDTH), -1, dtype=np.int32)
        worker.apply_grammar_bitmask(
            logits, input_batch, ["a", "b"][:num_masks], bitmask
        )
        torch.accelerator.synchronize()

        assert ptrs == (
            worker.grammar_bitmask.cpu.data_ptr(),
            worker.grammar_bitmask.gpu.data_ptr(),
            worker.logits_indices.cpu.data_ptr(),
            worker.logits_indices.gpu.data_ptr(),
        )


def test_apply_grammar_bitmask_masks_expected_tokens():
    """End-to-end: the staged bitmask masks exactly the disallowed token ids."""
    worker = _make_worker()
    # Two requests, one logits row each.
    input_batch = _FakeInputBatch(req_ids=["a", "b"], cu_num_logits=[0, 1, 2])
    logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device="cuda")

    # Allow only token 0 for req "a" and only token 1 for req "b".
    bitmask = np.zeros((2, BITMASK_WIDTH), dtype=np.int32)
    bitmask[0, 0] = 1 << 0
    bitmask[1, 0] = 1 << 1

    worker.apply_grammar_bitmask(logits, input_batch, ["a", "b"], bitmask)
    torch.accelerator.synchronize()

    assert torch.isfinite(logits[0, 0])
    assert torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[1, 0])
    assert torch.isfinite(logits[1, 1])
    # Every token past the single allowed bit is masked for both rows.
    assert torch.isneginf(logits[:, 2:]).all()


def test_stale_staging_rows_do_not_leak_between_steps():
    """A wide step followed by a narrow one must not reuse the stale rows."""
    worker = _make_worker()
    input_batch = _FakeInputBatch(req_ids=["a", "b"], cu_num_logits=[0, 1, 2])

    # Step 1: two requests, everything masked.
    logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device="cuda")
    worker.apply_grammar_bitmask(
        logits, input_batch, ["a", "b"], np.zeros((2, BITMASK_WIDTH), dtype=np.int32)
    )
    torch.accelerator.synchronize()
    assert torch.isneginf(logits).all()

    # Step 2: only req "b", allowing every token in the low word.
    logits = torch.zeros((2, VOCAB_SIZE), dtype=torch.float32, device="cuda")
    bitmask = np.full((1, BITMASK_WIDTH), -1, dtype=np.int32)
    worker.apply_grammar_bitmask(logits, input_batch, ["b"], bitmask)
    torch.accelerator.synchronize()

    # Row 1 (req "b") is unmasked; row 0 was not scheduled and is untouched.
    assert torch.isfinite(logits[1]).all()
    assert torch.isfinite(logits[0]).all()


def test_apply_grammar_bitmask_is_noop_without_requests():
    worker = _make_worker()
    input_batch = _FakeInputBatch(req_ids=["a"], cu_num_logits=[0, 1])
    logits = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device="cuda")

    worker.apply_grammar_bitmask(
        logits, input_batch, [], np.zeros((0, BITMASK_WIDTH), dtype=np.int32)
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(logits).all()
