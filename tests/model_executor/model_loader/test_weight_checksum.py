# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig
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
        self.register_buffer("k_scale", torch.tensor(2.0))
        self.register_buffer("scratch", torch.ones(2), persistent=False)
        self.register_buffer("inv_freq", torch.ones(2))


def test_digests_cover_parameters_and_persistent_weight_buffers():
    assert set(compute_tensor_digests(_Model())) == {
        "linear.weight",
        "linear.bias",
        "k_scale",
    }


def test_zero_weights_changes_every_covered_tensor_only():
    model = _Model()
    before = compute_tensor_digests(model)
    zero_weights(model)
    after = compute_tensor_digests(model)
    assert all(before[name] != after[name] for name in before)
    assert model.scratch.eq(1).all() and model.inv_freq.eq(1).all()


def test_dense_dp_replicas_get_distinct_key_prefixes(monkeypatch):
    for name in ("get_tp_group", "get_pp_group", "get_pcp_group"):
        monkeypatch.setattr(gpu_worker, name, lambda: SimpleNamespace(rank_in_group=0))
    prefixes = []
    for dp_rank in (0, 1):
        worker = object.__new__(Worker)
        worker.model_config = SimpleNamespace(is_moe=False)
        worker.parallel_config = ParallelConfig(
            data_parallel_size=2, data_parallel_rank=dp_rank
        )
        # Dense DP resets data_parallel_rank to 0 in each engine.
        worker.parallel_config.reconfigure_for_independent_dp_rank()
        prefixes.append(worker._weight_checksum_prefix())
    assert prefixes == ["dp0:pp0:pcp0:tp0:ep0:", "dp1:pp0:pcp0:tp0:ep0:"]
