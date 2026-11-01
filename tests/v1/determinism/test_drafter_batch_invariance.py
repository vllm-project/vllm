# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config import KernelConfig, SpeculativeConfig
from vllm.config.speculative import _DRAFT_VLLM_CONFIG_OVERRIDES
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts


@dataclass
class _Target:
    kernel_config: KernelConfig = field(default_factory=KernelConfig)


class _NotBatchInvariant(TritonExperts):
    _supports_current_device = staticmethod(lambda: True)
    _supports_batch_invariance = staticmethod(lambda: False)


def _accepts_experts(kernel_config: KernelConfig) -> bool:
    moe = FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        batch_invariant=kernel_config.batch_invariant,
    )
    cls = _NotBatchInvariant
    fmt = mk.FusedMoEActivationFormat.Standard
    return cls.is_supported_config(cls, moe, None, None, fmt)[0]


@pytest.mark.parametrize("batch_invariant", [None, True, False])
def test_drafter_batch_invariant_override(monkeypatch, batch_invariant):
    """Under VLLM_BATCH_INVARIANT only a drafter set to False may pick MoE
    experts that are not batch invariant; the target never may."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    spec = SimpleNamespace(**{src: None for src, _, _ in _DRAFT_VLLM_CONFIG_OVERRIDES})
    spec.batch_invariant = batch_invariant
    target = _Target()
    draft = SpeculativeConfig.apply_draft_overrides(spec, target)
    assert _accepts_experts(draft.kernel_config) is (batch_invariant is False)
    assert not _accepts_experts(target.kernel_config)
