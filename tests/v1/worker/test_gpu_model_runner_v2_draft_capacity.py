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

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    compute_draft_token_capacity_from_confidence,
)
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


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


def test_compute_draft_token_capacity_uses_sps_profile():
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
    # B=2 -> 2.0, B=3 -> 2.32, B=4 -> 2.59, B=5 -> 2.431, B=6 -> 2.277
    sps_profile = torch.tensor(
        [1.0, 1.0, 1.0, 0.8, 0.7, 0.55, 0.45],
        dtype=torch.float32,
        device=device,
    )

    compute_draft_token_capacity_from_confidence(
        confidence_logits,
        draft_token_capacity,
        min_survival_probability=0.0,
        num_reqs=2,
        num_speculative_steps=2,
        survival_probs=survival_probs,
        sps_profile=sps_profile,
    )

    torch.accelerator.synchronize()
    assert draft_token_capacity.cpu().tolist() == [1, 1]


def test_draft_tokens_handler_uses_capacity_placeholders():
    device = torch.device("cuda")
    handler = DraftTokensHandler(device)
    input_batch: Any = SimpleNamespace(
        req_ids=["req0", "req1"],
        has_structured_output_reqs=False,
    )
    draft_tokens = torch.zeros((2, 3), dtype=torch.int64, device=device)
    draft_token_capacity = torch.tensor([1, 3], dtype=torch.int32, device=device)

    handler.set_draft_tokens(input_batch, draft_tokens, draft_token_capacity)

    draft_token_capacity_np = np.full(4, 3, dtype=np.int32)
    handler.sync_draft_token_capacities(
        draft_token_capacity_np,
        {"req0": 2, "req1": 0},
    )
    assert draft_token_capacity_np.tolist() == [3, 3, 1, 3]

    draft_token_ids = handler.get_draft_tokens()
    assert draft_token_ids is not None
    assert draft_token_ids.draft_token_ids == [[-1], [-1, -1, -1]]
