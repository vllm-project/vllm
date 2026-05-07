# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm.v1.capture.plan``."""

from __future__ import annotations

import torch

from vllm.v1.capture.plan import (
    CaptureBatchView,
    CapturePositionEntry,
    StepCapturePlan,
)

# ---------------------------------------------------------------------------
# CaptureBatchView construction
# ---------------------------------------------------------------------------


class TestCaptureBatchView:
    def test_construction_basic(self):
        view = CaptureBatchView(
            req_ids=["r1", "r2"],
            num_prompt_tokens=[10, 20],
            num_computed_tokens=[0, 0],
            num_scheduled_tokens=[10, 20],
            token_offsets=[0, 10],
        )
        assert view.req_ids == ["r1", "r2"]
        assert view.num_prompt_tokens == [10, 20]
        assert view.num_computed_tokens == [0, 0]
        assert view.num_scheduled_tokens == [10, 20]
        assert view.token_offsets == [0, 10]

    def test_empty_batch(self):
        view = CaptureBatchView(
            req_ids=[],
            num_prompt_tokens=[],
            num_computed_tokens=[],
            num_scheduled_tokens=[],
            token_offsets=[],
        )
        assert len(view.req_ids) == 0

    def test_single_request(self):
        view = CaptureBatchView(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[5],
            num_scheduled_tokens=[1],
            token_offsets=[0],
        )
        assert view.req_ids[0] == "r1"
        assert view.num_prompt_tokens[0] == 5
        assert view.num_computed_tokens[0] == 5
        assert view.num_scheduled_tokens[0] == 1


# ---------------------------------------------------------------------------
# CapturePositionEntry with consumer_mask
# ---------------------------------------------------------------------------


class TestCapturePositionEntry:
    def test_single_consumer_mask(self):
        entry = CapturePositionEntry(
            request_id="r1",
            layer=0,
            hook="post_mlp",
            logical_pos=9,
            scratch_row=0,
            step_index=0,
            consumer_mask=0b0001,
        )
        assert entry.consumer_mask & (1 << 0)
        assert not (entry.consumer_mask & (1 << 1))

    def test_multi_consumer_mask(self):
        entry = CapturePositionEntry(
            request_id="r1",
            layer=0,
            hook="post_mlp",
            logical_pos=9,
            scratch_row=0,
            step_index=0,
            consumer_mask=0b0101,
        )
        assert entry.consumer_mask & (1 << 0)
        assert not (entry.consumer_mask & (1 << 1))
        assert entry.consumer_mask & (1 << 2)

    def test_all_fields_round_trip(self):
        entry = CapturePositionEntry(
            request_id="req-42",
            layer=7,
            hook="pre_attn",
            logical_pos=100,
            scratch_row=3,
            step_index=5,
            consumer_mask=0b1111,
        )
        assert entry.request_id == "req-42"
        assert entry.layer == 7
        assert entry.hook == "pre_attn"
        assert entry.logical_pos == 100
        assert entry.scratch_row == 3
        assert entry.step_index == 5
        assert entry.consumer_mask == 0b1111

    def test_consumer_mask_zero_means_no_consumer(self):
        entry = CapturePositionEntry(
            request_id="r1",
            layer=0,
            hook="post_mlp",
            logical_pos=0,
            scratch_row=0,
            step_index=0,
            consumer_mask=0,
        )
        for i in range(8):
            assert not (entry.consumer_mask & (1 << i))


# ---------------------------------------------------------------------------
# StepCapturePlan gather_indices shape and dtype
# ---------------------------------------------------------------------------


class TestStepCapturePlan:
    def test_gather_indices_dtype_and_shape(self):
        indices = torch.tensor([0, 3, 7], dtype=torch.int64)
        scratch = torch.empty((3, 16), dtype=torch.float32)
        plan = StepCapturePlan(
            gather_indices={(0, "post_mlp"): indices},
            scratch_gpu={(0, "post_mlp"): scratch},
            scratch_dtype={(0, "post_mlp"): torch.float32},
            entries=[],
        )
        assert plan.gather_indices[(0, "post_mlp")].dtype == torch.int64
        assert plan.gather_indices[(0, "post_mlp")].shape == (3,)
        assert plan.scratch_gpu[(0, "post_mlp")].shape == (3, 16)

    def test_empty_plan(self):
        plan = StepCapturePlan(
            gather_indices={},
            scratch_gpu={},
            scratch_dtype={},
            entries=[],
        )
        assert len(plan.gather_indices) == 0
        assert len(plan.entries) == 0
        assert plan.request_errors == {}

    def test_multiple_layer_hook_pairs(self):
        plan = StepCapturePlan(
            gather_indices={
                (0, "pre_attn"): torch.tensor([0], dtype=torch.int64),
                (0, "post_mlp"): torch.tensor([0, 1], dtype=torch.int64),
                (1, "post_mlp"): torch.tensor([2], dtype=torch.int64),
            },
            scratch_gpu={
                (0, "pre_attn"): torch.empty((1, 8)),
                (0, "post_mlp"): torch.empty((2, 8)),
                (1, "post_mlp"): torch.empty((1, 8)),
            },
            scratch_dtype={
                (0, "pre_attn"): torch.float32,
                (0, "post_mlp"): torch.float32,
                (1, "post_mlp"): torch.float32,
            },
            entries=[],
        )
        assert len(plan.gather_indices) == 3
        assert plan.scratch_gpu[(0, "post_mlp")].shape[0] == 2

    def test_request_errors_default_empty(self):
        plan = StepCapturePlan(
            gather_indices={},
            scratch_gpu={},
            scratch_dtype={},
            entries=[],
        )
        assert plan.request_errors == {}

    def test_request_errors_populated(self):
        plan = StepCapturePlan(
            gather_indices={},
            scratch_gpu={},
            scratch_dtype={},
            entries=[],
            request_errors={"r1": "bad position"},
        )
        assert plan.request_errors["r1"] == "bad position"
