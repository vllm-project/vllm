# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


def test_dspark_parallel_config_disables_eplb_atomically():
    target_config = _FakeParallelConfig()

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert not draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 0
    assert not draft_config.enable_elastic_ep
    assert draft_config.eplb_config is not target_config.eplb_config


@pytest.mark.parametrize("watermarking", [False, True])
def test_candidate_pruning_preserves_dense_sampling_features(monkeypatch, watermarking):
    """Unsupported combinations retain the dense estimator/watermarker path."""
    speculator = object.__new__(DSparkSpeculator)
    speculator.vllm_config = None
    speculator.markov_topk = 32
    speculator.enable_adaptive_verification = not watermarking
    speculator.use_acceptance_estimator = not watermarking
    speculator.draft_watermarker = object() if watermarking else None
    speculator.draft_logits = None
    model = SimpleNamespace(model=SimpleNamespace(confidence_head=None))
    monkeypatch.setattr(spec_module, "load_dspark_model", lambda *args: model)

    assert speculator.load_draft_model(None, set()) is model
    assert not speculator._markov_walk_enabled
    assert speculator.use_acceptance_estimator == (not watermarking)
    assert speculator.markov_topk == 0

    speculator.model = SimpleNamespace(map_draft_to_target=lambda tokens: tokens)
    speculator._step_cols = torch.arange(1)
    speculator._maybe_predict_acceptance = Mock()
    logits = torch.tensor([[0.0, 2.0, 1.0]])
    mapping = torch.tensor([0])
    tokens = speculator._sample_logits(logits, mapping, torch.tensor([1]), 0)
    torch.testing.assert_close(tokens, torch.tensor([1]))
    speculator._maybe_predict_acceptance.assert_called_once_with(
        logits, mapping, speculator._step_cols[0]
    )
