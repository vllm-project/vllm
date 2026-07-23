# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive_verification
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    get_step_cost_profile_cases,
)


def test_budget_assigns_capacity_from_stale_confidence():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = 2
    manager._stale_idx = 0
    manager._staged_probs = [
        SimpleNamespace(np=np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32))
    ]
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.array([1, 1], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([1, 1], dtype=np.int32)),
    )
    manager.cost_tables = (
        np.zeros(3, dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )

    capacities = manager._compute_budget(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
        has_structured_output=False,
    )

    assert capacities == {"low": 0, "high": 1}


def test_apply_budget_keeps_metadata_on_one_capacity_vector():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager._batch_budget = {"first": 1, "second": 3}

    capacities, scheduled = manager.apply_budget(
        ["second", "first"],
        np.array([5, 5], dtype=np.int32),
        np.array([6, 6], dtype=np.int32),
    )

    np.testing.assert_array_equal(capacities, [3, 1])
    np.testing.assert_array_equal(scheduled, [4, 2])
    assert manager._batch_budget is None


def test_step_cost_profile_cases_cover_target_and_drafter_shapes():
    target_graphs = [
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 64, 64),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 128, 128),
        BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 256, None),
    ]
    draft_graphs = [
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 256, 32),
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 512, 64, num_active_loras=1),
    ]

    assert get_step_cost_profile_cases(target_graphs, draft_graphs, 8) == [
        (32, 32),
        (64, 8),
        (128, 16),
    ]


def test_step_costs_use_median_dummy_metrics(monkeypatch):
    monkeypatch.setattr(
        adaptive_verification,
        "get_tp_group",
        lambda: SimpleNamespace(broadcast_object=lambda value, src: value),
    )
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(
        max_num_reqs=2,
        max_num_batched_tokens=16,
    )
    samples = [
        (8, 1, forward_ms, drafter_ms)
        for forward_ms, drafter_ms in [
            (9.0, 2.0),
            (10.0, 3.0),
            (100.0, 30.0),
            (11.0, 4.0),
            (8.0, 1.0),
        ]
    ]
    samples += [(16, 2, 20.0, 5.0) for _ in range(5)]

    manager.set_step_costs(samples)

    assert manager.cost_tables is not None
    draft_cost, verify_cost = manager.cost_tables
    assert draft_cost[1] == 3.0
    assert draft_cost[2] == 5.0
    assert verify_cost[8] == 10.0
    assert verify_cost[16] == 20.0


def test_speculative_dummy_batch_models_real_work_with_padding():
    buffers = InputBuffers(4, 8, torch.device("cpu"))

    batch = InputBatch.make_dummy(
        2,
        6,
        buffers,
        num_reqs_after_padding=4,
        num_tokens_after_padding=8,
        num_draft_tokens_per_req=np.array([2, 2], dtype=np.int32),
        mark_all_padding=False,
    )

    assert batch.num_draft_tokens == 4
    assert batch.logits_indices.tolist() == list(range(6))
    assert batch.expanded_idx_mapping.tolist() == [0, 0, 0, 1, 1, 1]
    assert batch.expanded_local_pos.tolist() == [0, 1, 2, 0, 1, 2]
    assert batch.query_start_loc.tolist() == [0, 3, 6, 6, 6]
    assert batch.is_padding.tolist() == [False] * 6 + [True] * 2


def test_speculative_dummy_batch_models_cached_context():
    buffers = InputBuffers(4, 8, torch.device("cpu"))

    batch = InputBatch.make_dummy(2, 6, buffers, num_computed_tokens=2048)

    assert batch.query_start_loc.tolist() == [0, 3, 6]
    assert batch.seq_lens.tolist() == [2051, 2051]
    assert batch.seq_lens_cpu_upper_bound.tolist() == [2051, 2051]
    assert batch.num_computed_tokens_np.tolist() == [2048, 2048]
    assert batch.positions.tolist() == [2048, 2049, 2050] * 2
