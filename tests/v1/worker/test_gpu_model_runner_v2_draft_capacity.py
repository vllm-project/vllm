# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for draft capacity tests", allow_module_level=True)

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.decompaction import (
    prepare_sampler_decompaction_metadata,
)
from vllm.v1.worker.gpu.spec_decode.dspark.capacity import (
    CapacityBasedVerificationManager,
    compute_draft_token_capacity_from_confidence,
    get_effective_scheduled_token_counts,
)


def test_compute_draft_token_capacity_from_confidence_uses_global_prefix_order():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.90, 0.90],
            [0.95, 0.10, 0.99],
            [0.70, 0.70, 0.70],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((3,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.75,
        num_reqs=3,
        num_speculative_steps=3,
        survival_probs=survival_probs,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [2, 1, 0]


def test_compute_draft_token_capacity_uses_budgeted_global_prefix_order():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.80],
            [0.80, 0.80],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        survival_probs=survival_probs,
        budget_frac=0.5,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [2, 1]


def test_compute_draft_token_capacity_keeps_threshold_ties():
    device = torch.device("cuda")
    confidence_probs = torch.tensor(
        [
            [0.90, 0.80],
            [0.90, 0.80],
        ],
        dtype=torch.float32,
        device=device,
    )
    confidence_logits = torch.logit(confidence_probs)
    draft_token_capacity = torch.full((2,), -1, dtype=torch.int32, device=device)
    survival_probs = torch.empty_like(confidence_logits)

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        survival_probs=survival_probs,
        budget_frac=0.25,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [1, 1]


def test_capacity_based_verification_manager_updates_cpu_capacities():
    device = torch.device("cuda")
    draft_token_capacity_np = np.full(4, 3, dtype=np.int32)
    handler = CapacityBasedVerificationManager(
        max_num_tokens=16,
        max_num_reqs=4,
        draft_token_capacity_np=draft_token_capacity_np,
        last_sampled_tokens=torch.zeros((4, 1), dtype=torch.int64, device=device),
        draft_tokens=torch.zeros((4, 3), dtype=torch.int64, device=device),
        device=device,
    )
    input_batch: Any = SimpleNamespace(
        req_ids=["req0", "req1"],
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
    )
    draft_token_capacity = torch.tensor([1, 2], dtype=torch.int32, device=device)

    handler.set_draft_token_capacities(
        input_batch.req_ids,
        input_batch.idx_mapping_np,
        draft_token_capacity,
        num_draft_tokens=3,
    )
    assert handler.copy_event_pending

    torch.accelerator.synchronize()
    assert handler.try_update_draft_token_capacities(
        {"req0": 2, "req1": 0},
    )
    assert draft_token_capacity_np.tolist() == [2, 3, 1, 3]

    draft_token_capacity_np.fill(3)
    assert handler.try_update_draft_token_capacities(
        {"req1": 0},
    )
    assert draft_token_capacity_np.tolist() == [2, 3, 3, 3]


def test_effective_scheduled_token_counts_apply_capacity_before_dispatch():
    scheduler_output: Any = SimpleNamespace(
        num_scheduled_tokens={"req0": 4, "req1": 4, "req2": 3},
        total_num_scheduled_tokens=11,
        scheduled_spec_decode_tokens={
            "req0": [11, 12, 13],
            "req1": [21, 22, 23],
            "req2": [31, 32],
        },
    )

    assert get_effective_scheduled_token_counts(
        scheduler_output,
        {"req0": 0, "req1": 1, "req2": 2},
        np.array([1, 3, 7], dtype=np.int32),
    ) == (9, 4)


def test_sampler_decompaction_metadata_maps_pruned_tails_to_compact_bonus():
    device = torch.device("cuda")
    compact_cu_num_logits = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    full_cu_num_logits = torch.tensor([0, 4, 7], dtype=torch.int32, device=device)
    compact_query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    full_query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32, device=device)
    full_expanded_idx_mapping = torch.arange(7, dtype=torch.int32, device=device)
    full_expanded_local_pos = torch.arange(7, dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64, device=device)
    last_sampled_tokens = torch.tensor([[101], [201]], dtype=torch.int64, device=device)
    draft_tokens = torch.tensor(
        [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
    )

    metadata = prepare_sampler_decompaction_metadata(
        compact_cu_num_logits,
        full_cu_num_logits,
        full_query_start_loc,
        compact_query_start_loc,
        idx_mapping,
        positions,
        last_sampled_tokens,
        draft_tokens,
        total_num_logits=7,
        max_num_logits_per_req=4,
        expanded_idx_mapping=full_expanded_idx_mapping,
        expanded_local_pos=full_expanded_local_pos,
    )

    torch.accelerator.synchronize()
    assert metadata.cu_num_logits is full_cu_num_logits
    assert metadata.expanded_idx_mapping.cpu().tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert metadata.expanded_local_pos.cpu().tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert metadata.query_start_loc is full_query_start_loc
    assert metadata.target_logit_idx_mapping.cpu().tolist() == [0, 1, 1, 1, 2, 3, 4]
    assert metadata.draft_sampled.cpu().tolist() == [101, 11, -1, -1, 201, 21, 22]
    assert metadata.pos.cpu().tolist() == [10, 11, 11, 11, 20, 21, 22]


def test_capacity_cudagraph_dispatch_filters_by_max_query_len():
    manager = object.__new__(CudaGraphManager)
    manager._graphs_captured = True
    manager._resolve_effective_loras = lambda num_loras: num_loras
    regular_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=12,
        uniform_token_count=6,
    )
    capacity_desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        num_tokens=15,
        num_reqs=4,
        max_req_tokens=6,
    )
    manager._candidates = {
        (11, 0): [
            regular_desc,
            capacity_desc,
        ]
    }

    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=11,
        uniform_token_count=None,
        num_active_loras=0,
        max_req_tokens=6,
    )

    assert desc is capacity_desc

    desc = CudaGraphManager.dispatch(
        manager,
        num_reqs=4,
        num_tokens=11,
        uniform_token_count=6,
        num_active_loras=0,
    )

    assert desc is regular_desc
