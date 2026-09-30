# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.two_tensor import TwoTensor

from vllm.config import ParallelConfig
from vllm.model_executor import parameter
from vllm.model_executor.model_loader.weight_checksum import (
    compute_tensor_digests,
    zero_weights,
)
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = pytest.mark.cpu_test


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        nn.init.ones_(self.linear.weight)
        nn.init.ones_(self.linear.bias)
        # A strided view, which cannot be reinterpreted as bytes in place.
        self.strided = nn.Parameter(torch.arange(1.0, 7.0)[::2])
        self.scalar = nn.Parameter(torch.tensor(2.0))
        # Loading never restores buffers, so reset must leave them alone.
        self.register_buffer("k_scale", torch.tensor(2.0))


def test_digests_cover_every_parameter():
    assert set(compute_tensor_digests(_Model())) == {
        "linear.weight",
        "linear.bias",
        "strided",
        "scalar",
    }


def test_zero_weights_changes_every_parameter_and_no_buffer():
    model = _Model()
    before = compute_tensor_digests(model)
    zero_weights(model)
    after = compute_tensor_digests(model)
    assert all(before[name] != after[name] for name in before)
    assert model.k_scale.item() == 2.0


def test_shared_weight_partitions_are_hashed_and_reset(monkeypatch):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    model = nn.Module()
    model.transform = parameter.SharedWeightParameter(weight_loader=None)
    model.transform.add_partition(0, object(), 2, 2)
    model.transform.partitions[0].data.fill_(3.0)

    before = compute_tensor_digests(model)
    zero_weights(model)
    assert set(before) == {"transform.0"}
    assert compute_tensor_digests(model) != before


def test_tensor_subclass_inner_tensors_are_hashed_and_reset():
    model = nn.Module()
    model.weight = nn.Parameter(
        TwoTensor(torch.ones(2, 2), torch.full((2, 2), 2.0)), requires_grad=False
    )

    before = compute_tensor_digests(model)
    zero_weights(model)
    assert set(before) == {"weight.a", "weight.b"}
    assert all(
        before[name] != digest for name, digest in compute_tensor_digests(model).items()
    )


def test_dense_dp_replicas_get_distinct_key_prefixes(monkeypatch):
    for name in ("get_tp_group", "get_pp_group", "get_pcp_group"):
        monkeypatch.setattr(gpu_worker, name, lambda: SimpleNamespace(rank_in_group=0))
    prefixes = []
    for dp_rank in (0, 1):
        worker = object.__new__(Worker)
        worker.parallel_config = ParallelConfig(
            data_parallel_size=2, data_parallel_rank=dp_rank
        )
        # Dense DP resets data_parallel_rank to 0 in each engine.
        worker.parallel_config.reconfigure_for_independent_dp_rank()
        prefixes.append(worker._weight_checksum_prefix())
    assert prefixes == ["dp0:pp0:pcp0:tp0:", "dp1:pp0:pcp0:tp0:"]
