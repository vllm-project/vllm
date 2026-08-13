# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import _get_ep_group_ranks


def test_explicit_ep_groups_preserve_tp_lanes() -> None:
    all_ranks = torch.arange(8).reshape(1, 2, 1, 1, 4)

    groups = _get_ep_group_ranks(
        all_ranks,
        data_parallel_size=2,
        prefill_context_model_parallel_size=1,
        tensor_model_parallel_size=4,
        expert_parallel_size=2,
    )

    assert groups == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_explicit_ep_disables_flattened_sequence_parallel() -> None:
    config = ParallelConfig(
        tensor_parallel_size=4,
        data_parallel_size=2,
        enable_expert_parallel=True,
        expert_parallel_size=2,
    )

    assert not config.use_sequence_parallel_moe


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_expert_parallel": False},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"expert_parallel_size": 4},
        {"all2all_backend": "deepep_low_latency"},
        {"enable_eplb": True},
        {"enable_elastic_ep": True},
    ],
)
def test_explicit_ep_rejects_unsupported_topologies(overrides) -> None:
    kwargs = {
        "tensor_parallel_size": 4,
        "data_parallel_size": 2,
        "enable_expert_parallel": True,
        "expert_parallel_size": 2,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        ParallelConfig(**kwargs)
