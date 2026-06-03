# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe as fused_moe
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.model_executor.models.utils import WeightsMapper


class DummyLayer:
    pass


class _FakeQuantMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, extra_weight_attrs
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bias
        return torch.zeros(
            *x.shape[:-1],
            layer.output_size,
            dtype=x.dtype,
            device=x.device,
        )


class _FakeQuantConfig:
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> LinearMethodBase:
        del layer, prefix
        return _FakeQuantMethod()


def _make_inc_config(extra_config: dict[str, dict[str, int]]) -> INCConfig:
    config = INCConfig(
        weight_bits=4,
        group_size=128,
        sym=True,
        extra_config=extra_config,
    )
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }
    return config


def test_inc_extra_config_maps_regex_keys_for_fused_qkv() -> None:
    config = _make_inc_config(
        {
            r".*model\.language_model\.layers\.\d+\.self_attn\..*": {
                "bits": 8,
            },
            "model.language_model.layers.5.self_attn.q_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.k_proj": {"bits": 8},
        }
    )
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
        }
    )

    config.apply_vllm_mapper(mapper)

    assert config.get_layer_config(
        DummyLayer(), "language_model.model.layers.5.self_attn.qkv_proj"
    ) == (8, 128, True)


def test_inc_fused_qkv_still_rejects_real_mixed_configs() -> None:
    config = _make_inc_config(
        {
            r".*model\.language_model\.layers\.\d+\.self_attn\..*": {
                "bits": 8,
            },
            "model.language_model.layers.5.self_attn.q_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.k_proj": {"bits": 8},
            "model.language_model.layers.5.self_attn.v_proj": {"bits": 4},
        }
    )
    mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
        }
    )

    config.apply_vllm_mapper(mapper)

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.get_layer_config(
            DummyLayer(), "language_model.model.layers.5.self_attn.qkv_proj"
        )


def test_gate_linear_accepts_quant_config_for_router_weights() -> None:
    layer = GateLinear(
        4,
        3,
        bias=False,
        out_dtype=torch.float32,
        quant_config=_FakeQuantConfig(),
        prefix="layers.0.router.proj",
        disable_tp=True,
    )

    assert hasattr(layer, "qweight")
    assert not hasattr(layer, "weight")
    assert not layer.allow_specialized_router_gemm
    assert not layer.allow_dsv3_router_gemm
    assert not layer.allow_cublas_router_gemm

    output, bias = layer(torch.ones(2, 4))
    assert output.shape == (2, 3)
    assert output.dtype == torch.float32
    assert bias is None


def test_moe_wna16_forwards_layer_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del w1, w2
        captured.update(kwargs)
        return torch.empty_like(hidden_states)

    monkeypatch.setattr(fused_moe, "fused_experts", fake_fused_experts)

    quant_config = MoeWNA16Config.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "lm_head": False,
        }
    )
    method = MoeWNA16Method(quant_config, moe=SimpleNamespace(disable_inplace=True))
    method.moe.disable_inplace = True

    layer = SimpleNamespace(
        activation=MoEActivation.GELU_TANH,
        w13_qweight=torch.empty(1, 4, 4, dtype=torch.uint8),
        w2_qweight=torch.empty(1, 4, 4, dtype=torch.uint8),
        apply_router_weight_on_input=False,
        global_num_experts=1,
        expert_map=None,
    )

    output = method.apply(
        layer,
        torch.ones(2, 4),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert output.shape == (2, 4)
    assert captured["activation"] == MoEActivation.GELU_TANH
