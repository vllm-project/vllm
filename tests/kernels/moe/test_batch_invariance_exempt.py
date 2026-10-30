# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_BI_DRAFTER=0 lets the drafter skip batch-invariant MoE experts."""

import pytest

import vllm.envs as envs
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.determinism import (
    building_drafter,
    requires_batch_invariant_kernels,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import speculator as speculator_mod
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


class _NonInvariantTritonExperts(TritonExperts):
    @staticmethod
    def _supports_batch_invariance() -> bool:
        return False


@pytest.mark.parametrize("bi_drafter", [True, False])
def test_only_drafter_is_exempt(monkeypatch, bi_drafter):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_BI_DRAFTER", bi_drafter)
    assert requires_batch_invariant_kernels()
    with building_drafter():
        assert requires_batch_invariant_kernels() == bi_drafter
    assert requires_batch_invariant_kernels()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="TritonExperts on CUDA")
def test_drafter_accepts_non_invariant_experts(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_BI_DRAFTER", False)
    config = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=512
    )

    def check(cls):
        return cls.is_supported_config(
            cls, config, None, None, mk.FusedMoEActivationFormat.Standard
        )

    supported, reason = check(_NonInvariantTritonExperts)
    assert not supported and "batch invariance" in reason
    with building_drafter():
        assert check(_NonInvariantTritonExperts) == (True, None)


def test_load_model_builds_drafter_in_scope(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_BI_DRAFTER", False)
    monkeypatch.setattr(speculator_mod, "get_layers_from_vllm_config", lambda *a: {})
    seen = []

    def load_draft_model(self, target_model, target_attn_layer_names):
        seen.append(requires_batch_invariant_kernels())
        raise StopIteration

    monkeypatch.setattr(MTPSpeculator, "load_draft_model", load_draft_model)
    speculator = object.__new__(MTPSpeculator)
    speculator.vllm_config = None
    with pytest.raises(StopIteration):
        speculator.load_model(object())
    assert seen == [False]
