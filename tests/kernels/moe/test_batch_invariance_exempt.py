# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""batch_invariance_exempt() lifts the batch-invariance requirement on MoE
experts selection, e.g. for a speculative drafter."""

import pytest

import vllm.envs as envs
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.determinism.scope import (
    batch_invariance_exempt,
    requires_batch_invariant_kernels,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform


class _NonInvariantTritonExperts(TritonExperts):
    @staticmethod
    def _supports_batch_invariance() -> bool:
        return False


def test_exempt_scope_only_lifts_the_requirement_inside(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert requires_batch_invariant_kernels()
    with batch_invariance_exempt():
        assert not requires_batch_invariant_kernels()
        with batch_invariance_exempt():
            assert not requires_batch_invariant_kernels()
        assert not requires_batch_invariant_kernels()
    assert requires_batch_invariant_kernels()

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert not requires_batch_invariant_kernels()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="TritonExperts on CUDA")
def test_exempt_scope_accepts_experts_without_batch_invariance(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    config = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=512
    )

    def check(cls):
        return cls.is_supported_config(
            cls, config, None, None, mk.FusedMoEActivationFormat.Standard
        )

    assert check(TritonExperts) == (True, None)
    supported, reason = check(_NonInvariantTritonExperts)
    assert not supported
    assert "batch invariance" in reason
    with batch_invariance_exempt():
        assert check(_NonInvariantTritonExperts) == (True, None)
        assert check(TritonExperts) == (True, None)
