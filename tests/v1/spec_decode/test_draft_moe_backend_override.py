# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model must honour ``moe_backend`` from --speculative-config.

The draft is loaded with the target's VllmConfig, so without an explicit
override it inherits the target's --moe-backend. An MTP head on a quantized
target is typically unquantized, and quantized-only backends reject it, so the
server fails to start rather than falling back.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm.config import LoadConfig, SpeculativeConfig
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _KernelConfig:
    moe_backend: str | None = None
    batch_invariant: bool = True


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    batch_invariant: bool | None = None
    draft_model_config: object = None
    draft_load_config: object = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides


@dataclass
class _VllmConfig:
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)


def _config(
    target_moe: str, draft_moe: str | None, batch_invariant: bool | None = None
) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(
            moe_backend=draft_moe, batch_invariant=batch_invariant
        ),
    )


class _Captured(Exception):
    """Stops load_eagle_model once we hold the config it would have used."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def test_draft_moe_backend_overrides_the_target():
    """A draft moe_backend must reach the draft's kernel config."""
    used = _capture_draft_config(_config("flashinfer_b12x", "flashinfer_cutlass"))
    assert used.kernel_config.moe_backend == "flashinfer_cutlass"


def test_draft_inherits_target_when_no_override():
    """No override means the draft still inherits the target, as before."""
    used = _capture_draft_config(_config("flashinfer_b12x", None))
    assert used.kernel_config.moe_backend == "flashinfer_b12x"


def test_override_does_not_mutate_the_target_config():
    """The target must keep its own backend after the draft is built."""
    cfg = _config("flashinfer_b12x", "flashinfer_cutlass")
    _capture_draft_config(cfg)
    assert cfg.kernel_config.moe_backend == "flashinfer_b12x"


@pytest.mark.parametrize("batch_invariant", [None, False, True])
def test_draft_batch_invariant_override(batch_invariant):
    """Only batch_invariant=False turns it off, and only for the draft."""
    cfg = _config("triton", None, batch_invariant)
    used = _capture_draft_config(cfg)
    assert used.kernel_config.batch_invariant is (batch_invariant is not False)
    assert cfg.kernel_config.batch_invariant


class _NonBatchInvariantExperts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return False


def _accepts_non_batch_invariant_experts(kernel_config) -> bool:
    moe_config = FusedMoEConfig(
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
    supported, reason = _NonBatchInvariantExperts.is_supported_config(
        _NonBatchInvariantExperts,
        moe_config,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported or "batch invariance" in str(reason), reason
    return supported


@pytest.mark.parametrize("batch_invariant", [None, False])
def test_batch_invariant_false_lifts_moe_check_only_for_draft(
    monkeypatch, batch_invariant
):
    """Under BI, non-BI experts are allowed only for a drafter set to False."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    cfg = _config("triton", None, batch_invariant)
    used = _capture_draft_config(cfg)
    assert _accepts_non_batch_invariant_experts(used.kernel_config) is (
        batch_invariant is False
    )
    assert not _accepts_non_batch_invariant_experts(cfg.kernel_config)
