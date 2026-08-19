# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)


def _config(
    *,
    routing_method: RoutingMethodType = RoutingMethodType.DeepSeekV3,
    use_ep: bool = False,
    shared_expert_weight: float = 1.0,
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    if use_ep:
        parallel_config = replace(
            parallel_config,
            use_ep=True,
            ep_size=2,
            all2all_backend="allgather_reducescatter",
        )
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=7168,
        intermediate_size=2048,
        num_local_experts=128 if use_ep else 256,
        num_logical_experts=256,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=routing_method,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        num_fused_shared_experts=1,
        fused_shared_expert_weight=shared_expert_weight,
    )


def _is_supported(monkeypatch, config: FusedMoEConfig) -> tuple[bool, str | None]:
    monkeypatch.setattr(
        TrtLlmNvFp4ExpertsMonolithic,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    return TrtLlmNvFp4ExpertsMonolithic.is_supported_config(
        TrtLlmNvFp4ExpertsMonolithic,
        config,
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )


def test_native_shared_experts_are_selectable_for_deepseek_v3(monkeypatch) -> None:
    supported, reason = _is_supported(monkeypatch, _config())

    assert supported, reason
    assert TrtLlmNvFp4ExpertsMonolithic.supports_native_fused_shared_experts()


def test_native_shared_experts_reject_non_deepseek_routing(monkeypatch) -> None:
    supported, reason = _is_supported(
        monkeypatch,
        _config(routing_method=RoutingMethodType.Renormalize),
    )

    assert not supported
    assert reason is not None and "DeepSeekV3 routing" in reason


def test_native_shared_experts_reject_modular_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        TrtLlmNvFp4ExpertsModular,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    supported, reason = TrtLlmNvFp4ExpertsModular.is_supported_config(
        TrtLlmNvFp4ExpertsModular,
        _config(),
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )

    assert not supported
    assert reason is not None and "monolithic backend" in reason


def test_native_shared_experts_reject_ep(monkeypatch) -> None:
    supported, reason = _is_supported(monkeypatch, _config(use_ep=True))

    assert not supported
    assert reason is not None and "do not support EP" in reason


def test_native_shared_experts_require_unit_weight(monkeypatch) -> None:
    supported, reason = _is_supported(
        monkeypatch,
        _config(shared_expert_weight=0.4),
    )

    assert not supported
    assert reason is not None and "weight 1.0" in reason


def test_native_shared_experts_disable_routing_replay() -> None:
    experts = object.__new__(TrtLlmNvFp4ExpertsMonolithic)
    experts.moe_config = _config()

    assert not experts.supports_routing_replay_capture()


def test_native_shared_experts_extend_global_quant_metadata(monkeypatch) -> None:
    config = _config()
    config.num_local_experts += config.num_fused_shared_experts
    quant_method = Mock()
    quant_method.maybe_roundup_sizes.return_value = (
        config.hidden_dim,
        config.intermediate_size_per_partition,
    )
    quant_method.supports_eplb = False
    monkeypatch.setattr(RoutedExperts, "update_expert_map_info", lambda self: None)
    monkeypatch.setattr(
        RoutedExperts,
        "_get_quant_method",
        lambda *args, **kwargs: quant_method,
    )

    experts = RoutedExperts(
        layer_name="model.layers.3.mlp.experts",
        params_dtype=torch.bfloat16,
        moe_config=config,
        quant_config=None,
        expert_map_manager=Mock(),
    )

    assert experts.kernel_global_num_experts == 257
    assert quant_method.create_weights.call_args.kwargs["num_experts"] == 257
    assert quant_method.create_weights.call_args.kwargs["global_num_experts"] == 257
