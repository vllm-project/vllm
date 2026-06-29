# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER BatchedExperts tests require ROCm.", allow_module_level=True)

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "AITER BatchedExperts tests require supported ROCm AITER.",
        allow_module_level=True,
    )

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterBatchedExpertsFp8,
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    backend_to_kernel_cls,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Static,
)


def _make_moe_config(
    num_experts: int,
    hidden_dim: int,
    max_num_tokens: int,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_dim,
        intermediate_size=16,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.float32,
        max_num_tokens=max_num_tokens,
    )


def _parallel_config(all2all_backend: str) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=2,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend=all2all_backend,
        enable_eplb=False,
    )


def test_activation_formats():
    assert (
        AiterBatchedExpertsFp8.activation_format()
        == mk.FusedMoEActivationFormat.BatchedExperts
    )
    assert AiterExperts.activation_format() == mk.FusedMoEActivationFormat.Standard


@pytest.mark.parametrize(
    ("all2all_backend", "expected"),
    [
        ("deepep_low_latency", True),
        ("nixl_ep", True),
        ("flashinfer_nvlink_one_sided", False),
        ("flashinfer_nvlink_two_sided", False),
        ("mori_high_throughput", False),
    ],
)
def test_supported_parallel_configs(all2all_backend: str, expected: bool):
    parallel_config = _parallel_config(all2all_backend)

    assert parallel_config.use_batched_experts_activation_format is expected
    assert AiterBatchedExpertsFp8._supports_parallel_config(parallel_config) is expected


@pytest.mark.parametrize(
    ("weight_key", "activation_key", "expected"),
    [
        (kFp8Static128BlockSym, kFp8Dynamic128Sym, True),
        (kFp8StaticTensorSym, kFp8StaticTensorSym, True),
        (kFp8StaticTensorSym, kFp8DynamicTensorSym, True),
        (kFp8StaticChannelSym, kFp8DynamicTokenSym, True),
        (None, None, False),
        (kMxfp4Static, None, False),
    ],
)
def test_quant_scheme_scope(weight_key, activation_key, expected: bool):
    assert (
        AiterBatchedExpertsFp8._supports_quant_scheme(weight_key, activation_key)
        is expected
    )


@pytest.mark.parametrize(
    ("num_experts", "tokens_per_expert", "hidden_dim", "scale_dim"),
    [
        (1, 4, 8, None),
        (2, 3, 4, 2),
        (3, 2, 5, 1),
    ],
)
def test_apply_flattens_batched_layout(
    monkeypatch: pytest.MonkeyPatch,
    num_experts: int,
    tokens_per_expert: int,
    hidden_dim: int,
    scale_dim: int | None,
):
    hidden_states = torch.arange(
        num_experts * tokens_per_expert * hidden_dim,
        dtype=torch.float32,
    ).reshape(num_experts, tokens_per_expert, hidden_dim)
    output = torch.empty_like(hidden_states)
    a2_scale = torch.tensor([0.5])
    a1q_scale = None
    expected_a1q_scale = None
    if scale_dim is not None:
        a1q_scale = torch.arange(
            num_experts * tokens_per_expert * scale_dim,
            dtype=torch.float32,
        ).reshape(num_experts, tokens_per_expert, scale_dim)
        expected_a1q_scale = a1q_scale.reshape(
            num_experts * tokens_per_expert,
            scale_dim,
        )
    captured = {}

    def fake_rocm_aiter_fused_experts(**kwargs):
        captured.update(kwargs)
        routed_ids = kwargs["topk_ids"].to(dtype=kwargs["hidden_states"].dtype)
        return kwargs["hidden_states"] + routed_ids

    monkeypatch.setattr(
        rocm_aiter_moe,
        "rocm_aiter_fused_experts",
        fake_rocm_aiter_fused_experts,
    )
    wrapper = AiterBatchedExpertsFp8(
        _make_moe_config(num_experts, hidden_dim, tokens_per_expert),
        FUSED_MOE_UNQUANTIZED_CONFIG,
        max_num_tokens=tokens_per_expert,
        num_dispatchers=1,
    )

    wrapper.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(num_experts, 1, 1),
        w2=torch.empty(num_experts, 1, 1),
        topk_weights=torch.empty(1, 1),
        topk_ids=torch.empty(1, 1, dtype=torch.int64),
        activation=MoEActivation.SILU,
        global_num_experts=99,
        expert_map=torch.arange(num_experts),
        a1q_scale=a1q_scale,
        a2_scale=a2_scale,
        workspace13=torch.empty(0),
        workspace2=torch.empty(0),
        expert_tokens_meta=mk.ExpertTokensMetadata.make_from_list(
            [tokens_per_expert] * num_experts,
            "cpu",
        ),
        apply_router_weight_on_input=True,
    )

    expected_ids = (
        torch.arange(num_experts, dtype=torch.int32)
        .repeat_interleave(tokens_per_expert)
        .unsqueeze(-1)
    )
    expected_hidden = hidden_states.reshape(
        num_experts * tokens_per_expert,
        hidden_dim,
    )

    assert torch.equal(captured["hidden_states"], expected_hidden)
    assert torch.equal(captured["topk_ids"], expected_ids)
    assert torch.equal(
        captured["topk_weights"],
        torch.ones(num_experts * tokens_per_expert, 1),
    )
    assert captured["moe_config"].num_experts == num_experts
    assert captured["expert_map"] is None
    assert captured["num_local_tokens"] is None
    assert captured["apply_router_weight_on_input"] is False
    if expected_a1q_scale is None:
        assert captured["a1q_scale"] is None
    else:
        assert torch.equal(captured["a1q_scale"], expected_a1q_scale)

    expected_output = expected_hidden + expected_ids.float()
    assert torch.equal(
        output,
        expected_output.reshape(num_experts, tokens_per_expert, hidden_dim),
    )


def test_oracle_registers_batched_aiter_backend():
    assert backend_to_kernel_cls(Fp8MoeBackend.BATCHED_AITER) == [
        AiterBatchedExpertsFp8
    ]


def test_oracle_routes_batched_aiter_env(monkeypatch: pytest.MonkeyPatch):
    def is_set(name: str):
        return name in {"VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"}

    def is_supported(cls, config, weight_key, activation_key, activation_format):
        assert activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        return True, None

    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=True,
            use_deepep_v2_kernels=False,
            ep_size=1,
        ),
    )
    monkeypatch.setattr(fp8_oracle.envs, "is_set", is_set)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_ROCM_USE_AITER_MOE", True)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)
    monkeypatch.setattr(
        AiterBatchedExpertsFp8,
        "is_supported_config",
        staticmethod(is_supported),
    )

    backend, experts_cls = select_fp8_moe_backend(config, None, None)

    assert backend == Fp8MoeBackend.BATCHED_AITER
    assert experts_cls is AiterBatchedExpertsFp8


def test_oracle_routes_explicit_aiter_to_batched_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    def is_supported(cls, config, weight_key, activation_key, activation_format):
        assert activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        return True, None

    config = SimpleNamespace(
        moe_backend="aiter",
        moe_parallel_config=SimpleNamespace(
            use_batched_activation_format=True,
            use_deepep_v2_kernels=False,
            ep_size=1,
        ),
    )
    monkeypatch.setattr(fp8_oracle.envs, "is_set", lambda name: False)
    monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)
    monkeypatch.setattr(
        AiterBatchedExpertsFp8,
        "is_supported_config",
        staticmethod(is_supported),
    )

    backend, experts_cls = select_fp8_moe_backend(config, None, None)

    assert backend == Fp8MoeBackend.BATCHED_AITER
    assert experts_cls is AiterBatchedExpertsFp8
