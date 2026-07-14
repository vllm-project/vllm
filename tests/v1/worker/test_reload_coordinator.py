# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.model_executor.model_loader.reload import ReloadCapabilityError
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.reload_coordinator import ReloadCoordinator


class _Owner:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        quantization: str | None = None,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self.model = model
        self.model_config = SimpleNamespace(
            model="test-model",
            quantization=quantization,
        )
        self.compilation_config = SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=cudagraph_mode,
        )
        self.load_config = SimpleNamespace(load_format="dummy")
        self.cache_events: list[str] = []

    def get_model(self) -> torch.nn.Module:
        return self.model

    def reset_encoder_cache(self) -> None:
        self.cache_events.append("encoder")

    def reset_mm_cache(self) -> None:
        self.cache_events.append("multimodal")


def test_v2_reload_coordinator_commits_and_invalidates_caches():
    model = torch.nn.Linear(4, 3, bias=False)
    owner = _Owner(model)
    coordinator = ReloadCoordinator(owner)
    original_weight = model.weight

    coordinator.reload_weights(
        weights_iterator=[("weight", torch.ones_like(model.weight))],
        is_checkpoint_format=False,
    )

    assert model.weight is original_weight
    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert owner.cache_events == ["encoder", "multimodal"]
    assert coordinator.plan_for().committed_epoch == 1


def test_v2_graph_reload_rejects_uncertified_backend_before_mutation():
    model = torch.nn.Linear(4, 3, bias=False)
    owner = _Owner(
        model,
        quantization="fp8",
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )
    coordinator = ReloadCoordinator(owner)
    original = model.weight.detach().clone()

    with pytest.raises(ReloadCapabilityError, match="fp8"):
        coordinator.reload_weights(
            weights_iterator=[("weight", torch.zeros_like(model.weight))],
            is_checkpoint_format=False,
        )

    assert torch.equal(model.weight, original)


def test_v2_runner_delegates_reload_to_coordinator():
    runner = object.__new__(GPUModelRunner)
    runner.reload_coordinator = Mock()
    weights = [("weight", torch.ones(2, 2))]

    GPUModelRunner.reload_weights(
        runner,
        weights_iterator=weights,
        is_checkpoint_format=False,
    )

    runner.reload_coordinator.reload_weights.assert_called_once_with(
        weights_iterator=weights,
        is_checkpoint_format=False,
    )
