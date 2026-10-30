# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round contract: plan types, plan layout, commit along a path.

The kernel tests run on a GPU, or on CPU under ``TRITON_INTERPRET=1`` (set before
anything imports Triton).
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import trim_draft_tokens
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    PlanShape,
    SpecResources,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = current_platform.device_type if HAS_GPU else "cpu"


def test_round_types_are_versioned_and_plans_state_their_shape():
    resources = SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4), PlanShape(None))
    )
    assert resources.max_branching == 1
    parent = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)
    plan = VerifyPlan(
        parent=parent,
        count=torch.tensor([4], dtype=torch.int32),
        tokens=torch.zeros(1, 4, dtype=torch.int32),
    )
    assert not plan.is_chain
    result = AcceptResult(
        path=torch.tensor([[0, 2]], dtype=torch.int32),
        path_len=torch.tensor([2], dtype=torch.int32),
        sampled=torch.tensor([[5, 6]]),
    )
    assert result.processed_logits is None


def test_only_a_chain_plan_may_leave_its_rows_to_the_runner():
    parent = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
    # The runner's own chain plan: counts and tokens are in its buffers.
    assert VerifyPlan(parent=parent, count=None, is_chain=True).tokens is None
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=None)
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=torch.tensor([3], dtype=torch.int32))


def _speculator_type(**overrides):
    """A class with BaseSpeculator's methods, overriding ``overrides``."""
    methods = {k: v for k, v in vars(BaseSpeculator).items() if not k.startswith("__")}
    return type("Speculator", (), {**methods, **overrides})


def test_overridden_hooks_name_what_the_speculator_overrides():
    assert _speculator_type()().overridden_hooks() == frozenset()
    planner = _speculator_type(plan=lambda self, inputs: None)()
    assert planner.overridden_hooks() == {"plan"}


def test_trim_draft_tokens_keeps_each_requests_first_verified_drafts():
    tokens, drafts = trim_draft_tokens(
        np.array([4, 1, 3]), np.array([3, 0, 2]), np.array([1, 0, 2])
    )
    assert tokens.tolist() == [2, 1, 3]
    assert drafts.tolist() == [1, 0, 2]
    with pytest.raises(ValueError, match="between 1 and its scheduled"):
        trim_draft_tokens(np.array([4]), np.array([3]), np.array([4]))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_candidates=0, capture_shapes=()), "max_candidates"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(3),)), "outside"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(0),)), "outside"),
        (
            dict(max_candidates=2, capture_shapes=(), max_branching=0),
            "max_branching",
        ),
    ],
)
def test_spec_resources_reject_shapes_they_cannot_hold(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpecResources(**kwargs)


def test_spec_resources_list_uniform_counts_and_varlen():
    resources = SpecResources(
        max_candidates=4,
        capture_shapes=(PlanShape(4), PlanShape(None), PlanShape(1)),
    )
    assert resources.uniform_counts == [1, 4]
    assert resources.varlen
