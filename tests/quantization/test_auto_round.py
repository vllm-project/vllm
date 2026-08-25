# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import (
    INCMxfp4Scheme,
    INCMxfp8Scheme,
    INCWna16Scheme,
    resolve_scheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_linear import (
    INCMxfp8LinearScheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import (
    INCLinearScheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_w4a8_linear import (
    INCXPUW4A8LinearMethod,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear import (
    INCARKLinearMethod,
    INCWNA16LinearScheme,
    INCXPULinearMethod,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_scheme import (
    _resolve_awq_moe,
    _resolve_gptq_moe,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

MODELS = [
    pytest.param(
        "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",
        id="auto_round:auto_gptq",
    ),
    pytest.param(
        "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",
        marks=pytest.mark.skipif(
            not (current_platform.is_cuda() or current_platform.is_xpu()),
            reason="AWQ AutoRound model only supports CUDA/XPU backend for now.",
        ),
        id="auto_round:auto_awq",
    ),
    pytest.param(
        "Intel/Qwen3-8B-w2g64-for-ut",
        marks=pytest.mark.skipif(
            not (current_platform.is_xpu()),
            reason="INC int2 on XPU requires the ARK backend.",
        ),
        id="auto_round:auto_gptq_int2_tp2",
    ),
    pytest.param(
        "INC4AI/Qwen3-8B-MXFP8-AR",
        marks=pytest.mark.skipif(
            not (current_platform.is_cuda() or current_platform.is_xpu()),
            reason="MXFP8 AutoRound model only supports CUDA/XPU backend for now.",
        ),
        id="auto_round:llm_compressor_mxfp8",
    ),
]

QWEN3_AUTOROUND_MODELS = [
    pytest.param(
        "INCModel/Qwen3-1.7B-AutoRound-MXFP4-W4A4",
        marks=pytest.mark.skipif(
            not (current_platform.is_cuda() or current_platform.is_xpu()),
            reason="Qwen3-1.7B MXFP4 AutoRound model requires CUDA/XPU.",
        ),
        id="auto_round:mxfp4:qwen3-1p7b",
    ),
    pytest.param(
        "INCModel/Qwen3-30B-A3B-12L-W4A16-test",
        marks=pytest.mark.skipif(
            not (current_platform.is_cuda() or current_platform.is_xpu()),
            reason="Qwen3-30B-A3B W4A16 AutoRound model requires CUDA/XPU.",
        ),
        id="auto_round:w4a16:qwen3-30b-a3b",
    ),
    pytest.param(
        "INCModel/Qwen3-30B-A3B-12L-MXFP4-test",
        marks=pytest.mark.skipif(
            not (current_platform.is_cuda() or current_platform.is_xpu()),
            reason="Qwen3-30B-A3B MXFP4 AutoRound model requires CUDA/XPU.",
        ),
        id="auto_round:mxfp4:qwen3-30b-a3b",
    ),
]

MODEL_RUNNER_KWARGS: dict[str, dict[str, Any]] = {
    "INCModel/Qwen3-1.7B-AutoRound-MXFP4-W4A4": {"enforce_eager": True},
    "INCModel/Qwen3-30B-A3B-12L-MXFP4-test": {"enforce_eager": True},
    "Intel/Qwen3-8B-w2g64-for-ut": {
        "block_size": 64,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 512,
    },
    "INC4AI/Qwen3-8B-MXFP8-AR": {
        "block_size": 64,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 512,
    },
}


@pytest.mark.skipif(
    not (
        current_platform.is_cpu()
        or current_platform.is_xpu()
        or current_platform.is_cuda()
    ),
    reason="Only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS + QWEN3_AUTOROUND_MODELS)
def test_auto_round_model(vllm_runner, model):
    with vllm_runner(model, **MODEL_RUNNER_KWARGS.get(model, {})) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)

    assert output
    print(output[0][1])


# ---------------------------------------------------------------------------
# Unit tests for INCConfig and related classes
# ---------------------------------------------------------------------------


class DummyLayer:
    pass


class DummyFusedMoE:
    pass


def make_config(**overrides) -> INCConfig:
    kwargs = {
        "weight_bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
        "block_name_to_quantize": None,
        "extra_config": None,
        "data_type": "int",
        "backend": "auto",
    }
    kwargs.update(overrides)
    return INCConfig(**kwargs)


def make_layer_config(**overrides) -> INCLayerConfig:
    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
        "backend": "auto",
        "data_type": "int",
        "quantized": True,
    }
    kwargs.update(overrides)
    return INCLayerConfig(**kwargs)


def make_qwen3_autoround_config(kind: str) -> INCConfig:
    configs = {
        "qwen3_1p7b_mxfp4": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "packing_format": "auto_round:llm_compressor",
            "data_type": "mx_fp",
            "extra_config": {
                "model.layers.0.self_attn.q_proj": {
                    "bits": 16,
                    "data_type": "float",
                },
            },
        },
        "qwen3_30b_a3b_w4a16": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "packing_format": "auto_round:auto_gptq",
            "data_type": "int",
            "extra_config": {
                "model.layers.0.mlp.gate": {
                    "bits": 16,
                    "data_type": "float",
                },
            },
        },
        "qwen3_30b_a3b_mxfp4": {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "packing_format": "auto_round:llm_compressor",
            "data_type": "mx_fp",
            "act_bits": 4,
            "act_group_size": 32,
            "act_data_type": "mx_fp",
            "extra_config": {
                "model.layers.0.mlp.gate": {
                    "bits": 16,
                    "data_type": "float",
                },
                "model.layers.0.self_attn.q_proj": {
                    "bits": 16,
                    "data_type": "float",
                },
            },
        },
    }
    try:
        config = configs[kind]
    except KeyError as err:
        raise AssertionError(f"unknown qwen3 autoround config: {kind}") from err
    return INCConfig.from_config(config)


def test_inc_config_parser_exact_match() -> None:
    config = make_config(
        extra_config={
            "layers.0.self_attn.q_proj": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.config_parser.resolve(
        DummyLayer(), "layers.0.self_attn.q_proj"
    )

    assert layer_config.bits == 8
    assert layer_config.group_size == 64
    assert layer_config.sym is False
    assert layer_config.quantized is True


def test_inc_model_prefix_early_exit() -> None:
    """extra_config keys with model. prefix trigger early unquantized return."""
    config = make_config(
        extra_config={
            "model.layers.1.mlp.gate_proj": {
                "bits": 16,
            },
        }
    )

    # get_quant_method checks model. prefix for unquantized early-exit
    result = config.get_quant_method(DummyLayer(), "layers.1.mlp.gate_proj")
    assert isinstance(result, UnquantizedLinearMethod)


def test_inc_config_parser_regex_match() -> None:
    config = make_config(
        extra_config={
            r"layers\.\d+\.self_attn\.(q|k|v)_proj": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.config_parser.resolve(
        DummyLayer(), "layers.3.self_attn.q_proj"
    )

    assert layer_config.bits == 8
    assert layer_config.group_size == 64
    assert layer_config.sym is False


def test_inc_config_parser_invalid_regex_ignored() -> None:
    config = make_config(
        extra_config={
            "[invalid": {
                "bits": 8,
                "group_size": 64,
                "sym": False,
            }
        }
    )

    layer_config = config.config_parser.resolve(
        DummyLayer(), "layers.0.self_attn.q_proj"
    )

    assert layer_config.bits == 4
    assert layer_config.group_size == 128
    assert layer_config.sym is True


def test_inc_config_parser_block_name_to_quantize_marks_unquantized() -> None:
    config = make_config(block_name_to_quantize=["layers.1"])

    layer_config = config.config_parser.resolve(
        DummyLayer(), "layers.0.self_attn.q_proj"
    )

    assert layer_config.bits == 16
    assert layer_config.group_size == -1
    assert layer_config.sym is True
    assert layer_config.quantized is False


def test_inc_config_parser_parallel_lm_head_defaults_to_unquantized() -> None:
    layer = object.__new__(ParallelLMHead)
    config = make_config()

    layer_config = config.config_parser.resolve(layer, "lm_head")

    assert layer_config.quantized is False
    assert layer_config.bits == 16


def test_inc_config_parser_suffix_match_for_lm_head() -> None:
    """Short extra_config key should match fully-qualified lm_head layer name."""
    layer = object.__new__(ParallelLMHead)
    config = make_config(
        extra_config={
            "lm_head": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            }
        }
    )

    layer_config = config.config_parser.resolve(layer, "model.language_model.lm_head")

    assert layer_config.quantized is True
    assert layer_config.bits == 4
    assert layer_config.group_size == 128
    assert layer_config.sym is True


def test_inc_config_parser_fused_moe_requires_consistent_configs() -> None:
    config = make_config(
        extra_config={
            "layers.0.block_sparse_moe.experts.0.w1": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.block_sparse_moe.experts.0.w2": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
            },
        }
    )

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.config_parser.resolve(DummyFusedMoE(), "layers.0.block_sparse_moe")


def test_inc_config_parser_fused_module_requires_consistent_configs() -> None:
    config = make_config(
        extra_config={
            "layers.0.self_attn.q_proj": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.self_attn.k_proj": {
                "bits": 8,
                "group_size": 128,
                "sym": True,
            },
            "layers.0.self_attn.v_proj": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            },
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    with pytest.raises(ValueError, match="requires consistent quant config"):
        config.config_parser.resolve(DummyLayer(), "layers.0.self_attn.qkv_proj")


def test_inc_mxfp8() -> None:
    config = make_config(
        weight_bits=8,
        group_size=32,
        sym=True,
        packing_format="auto_round:llm_compressor",
        data_type="mx_fp",
    )

    assert config.weight_bits == 8
    assert config.group_size == 32
    assert config.data_type == "mx_fp"
    assert config.packing_format == "auto_round:llm_compressor"


def test_inc_config_rejects_invalid_mxfp8_activation_config() -> None:
    with pytest.raises(AssertionError, match="act_dynamic=True"):
        INCConfig.from_config(
            {
                "bits": 8,
                "group_size": 32,
                "sym": True,
                "packing_format": "auto_round:llm_compressor",
                "data_type": "mx_fp",
                "act_bits": 8,
                "act_data_type": "mx_fp",
                "act_group_size": 32,
                "act_sym": True,
                "act_dynamic": False,
            }
        )


def test_inc_layer_config_mx_fp_helpers() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=32,
        sym=True,
        packing_format="",
        backend="",
        data_type="mx_fp",
        quantized=True,
    )

    assert layer_config.is_mxfp4 is True
    assert layer_config.is_mxfp8 is False


def test_inc_resolve_scheme_selects_wna16() -> None:
    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    scheme = resolve_scheme(layer_config)

    assert isinstance(scheme, INCWna16Scheme)


def test_inc_config_accepts_mxfp_family_llm_compressor() -> None:
    config = INCConfig.from_config(
        {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "packing_format": "auto_round:llm_compressor",
            "data_type": "mx_fp4e2m1",
        }
    )

    layer_config = config.config_parser.resolve(
        DummyLayer(), "model.layers.0.mlp.down_proj"
    )

    assert config.sym is True
    assert layer_config.is_mxfp4 is True
    assert isinstance(resolve_scheme(layer_config), INCMxfp4Scheme)


def test_qwen3_1p7b_mxfp4_autoround_uses_mxfp4_linear_scheme(
    monkeypatch,
) -> None:
    class DummyKernel:
        pass

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes."
        "inc_mxfp4_linear.init_mxfp4_linear_kernel",
        lambda **kwargs: DummyKernel(),
    )

    from vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_linear import (  # noqa: E501
        INCMxfp4LinearMethod,
    )

    config = make_qwen3_autoround_config("qwen3_1p7b_mxfp4")

    assert (
        INCConfig.override_quantization_method(
            {"quant_method": "auto-round"}, user_quant=None
        )
        == "inc"
    )
    ignored_method = config.get_quant_method(
        object.__new__(LinearBase), "model.layers.0.self_attn.q_proj"
    )
    layer_config = config.config_parser.resolve(
        DummyLayer(), "model.layers.0.mlp.gate_proj"
    )
    method = INCMxfp4Scheme().get_linear_method(
        config,
        object.__new__(LinearBase),
        "model.layers.0.mlp.gate_proj",
        layer_config,
    )

    assert isinstance(ignored_method, UnquantizedLinearMethod)
    assert layer_config.bits == 4
    assert layer_config.group_size == 32
    assert layer_config.is_mxfp4 is True
    assert isinstance(resolve_scheme(layer_config), INCMxfp4Scheme)
    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCMxfp4LinearMethod)
    assert isinstance(method.scheme.kernel, DummyKernel)


def test_qwen3_30b_a3b_w4a16_autoround_routes_to_gptq_moe(
    monkeypatch,
) -> None:
    captured = {}
    expected_method = object()

    class DummyMoeConfig:
        pass

    def fake_resolve_gptq_moe(layer, layer_config):
        captured["layer"] = layer
        captured["layer_config"] = layer_config
        return expected_method

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes."
        "inc_wna16_scheme._resolve_gptq_moe",
        fake_resolve_gptq_moe,
    )

    config = make_qwen3_autoround_config("qwen3_30b_a3b_w4a16")
    layer = object.__new__(RoutedExperts)
    layer.moe_config = DummyMoeConfig()

    method = config.get_quant_method(layer, "model.layers.0.mlp")

    assert method is expected_method
    assert captured["layer"] is layer
    assert captured["layer_config"].bits == 4
    assert captured["layer_config"].group_size == 32
    assert captured["layer_config"].is_gptq is True
    assert captured["layer_config"].is_wna16_int is True


def test_qwen3_30b_a3b_mxfp4_autoround_routes_to_mxfp4_moe(
    monkeypatch,
) -> None:
    class DummyMoeConfig:
        pass

    class DummyMxfp4MoEMethod:
        def __init__(self, moe_config) -> None:
            self.moe_config = moe_config

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe.INCMxfp4MoEMethod",
        DummyMxfp4MoEMethod,
    )

    config = make_qwen3_autoround_config("qwen3_30b_a3b_mxfp4")
    layer = object.__new__(RoutedExperts)
    layer.moe_config = DummyMoeConfig()

    ignored_method = config.get_quant_method(
        object.__new__(LinearBase), "model.layers.0.self_attn.q_proj"
    )
    method = config.get_quant_method(layer, "model.layers.0.mlp")
    layer_config = config.config_parser.resolve(DummyLayer(), "model.layers.0.mlp")

    assert isinstance(ignored_method, UnquantizedLinearMethod)
    assert layer_config.bits == 4
    assert layer_config.group_size == 32
    assert layer_config.is_mxfp4 is True
    assert isinstance(resolve_scheme(layer_config), INCMxfp4Scheme)
    assert isinstance(method, DummyMxfp4MoEMethod)
    assert method.moe_config is layer.moe_config


def test_inc_mxfp4_linear_method_registers_and_processes_weights(
    monkeypatch,
) -> None:
    captured = {}

    class DummyKernel:
        def process_weights_after_loading(self, layer) -> None:
            captured["processed_layer"] = layer

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes."
        "inc_mxfp4_linear.init_mxfp4_linear_kernel",
        lambda **kwargs: DummyKernel(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    from vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_linear import (  # noqa: E501
        INCMxfp4LinearMethod,
    )

    layer = torch.nn.Module()
    method = INCMxfp4LinearMethod(
        make_layer_config(group_size=32, data_type="mx_fp4e2m1")
    )

    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[16, 32],
        input_size=64,
        output_size=48,
        params_dtype=torch.bfloat16,
    )

    assert layer.weight_packed.shape == (48, 32)
    assert layer.weight_packed.dtype is torch.uint8
    assert layer.weight_scale.shape == (48, 2)
    assert layer.weight_scale.dtype is torch.uint8
    assert layer.logical_widths == [16, 32]
    assert layer.input_size_per_partition == 64
    assert layer.output_size_per_partition == 48

    packed_data = layer.weight_packed.data
    method.process_weights_after_loading(layer)

    assert layer.weight.data.data_ptr() == packed_data.data_ptr()
    assert not hasattr(layer, "weight_packed")
    assert captured["processed_layer"] is layer


@pytest.mark.parametrize(
    ("moe_backend", "is_xpu", "mxfp4_backend"),
    [
        ("auto", True, Mxfp4MoeBackend.XPU),
        ("b12x", False, Mxfp4MoeBackend.B12X_MXFP4_MXFP8),
    ],
)
def test_inc_mxfp4_moe_method_preserves_checkpoint_packing(
    monkeypatch,
    moe_backend: str,
    is_xpu: bool,
    mxfp4_backend: Mxfp4MoeBackend,
) -> None:
    captured = {}
    expected_quant_config = object()
    expected_kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(
            process_weights_after_loading=lambda layer: captured.update(
                {"processed_layer": layer}
            )
        )
    )
    expected_experts_cls = object()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe."
        "CutlassExpertsMxfp4._supports_current_device",
        lambda: False,
    )
    monkeypatch.setattr(current_platform, "is_xpu", lambda: is_xpu)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe."
        "select_mxfp4_moe_backend",
        lambda moe: (mxfp4_backend, expected_experts_cls),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe."
        "prepare_moe_fp4_layer_for_marlin",
        lambda layer: pytest.fail("packed backends must not use Marlin packing"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe."
        "make_mxfp4_moe_quant_config",
        lambda **kwargs: captured.update({"quant_config_kwargs": kwargs})
        or expected_quant_config,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe."
        "make_mxfp4_moe_kernel",
        lambda **kwargs: captured.update({"kernel_kwargs": kwargs}) or expected_kernel,
    )

    from vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp4_moe import (
        INCMxfp4MoEMethod,
    )

    expected_moe_config = SimpleNamespace(
        w13_num_shards=2,
        moe_backend=moe_backend,
    )
    method = INCMxfp4MoEMethod(moe=cast(Any, expected_moe_config))
    layer = torch.nn.Module()
    layer._expert_routing_tables = lambda: "routing-tables"

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    assert method.experts_cls is expected_experts_cls
    assert layer.w13_weight_packed.shape == (2, 64, 32)
    assert layer.w2_weight_packed.shape == (2, 64, 16)
    assert layer.w13_weight_scale.shape == (2, 64, 2)
    assert layer.w2_weight_scale.shape == (2, 64, 1)

    w13_packed_data = layer.w13_weight_packed.data
    w2_packed_data = layer.w2_weight_packed.data
    method.process_weights_after_loading(layer)

    assert layer.w13_weight.data.data_ptr() == w13_packed_data.data_ptr()
    assert layer.w2_weight.data.data_ptr() == w2_packed_data.data_ptr()
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
    assert captured["quant_config_kwargs"]["w1_scale"] is layer.w13_weight_scale
    assert captured["quant_config_kwargs"]["w2_scale"] is layer.w2_weight_scale
    assert captured["kernel_kwargs"]["moe_quant_config"] is expected_quant_config
    assert captured["kernel_kwargs"]["moe_config"] is expected_moe_config
    assert captured["kernel_kwargs"]["experts_cls"] is expected_experts_cls
    assert captured["kernel_kwargs"]["routing_tables"] == "routing-tables"
    assert captured["processed_layer"] is layer
    assert method.moe_kernel is expected_kernel


def test_wna16_xpu_moe_routes_to_gptq_moe(monkeypatch) -> None:
    captured = {}
    expected_method = object()

    class DummyMoeConfig:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes."
        "inc_wna16_scheme._resolve_gptq_moe",
        lambda layer, layer_config: captured.update(
            {"layer": layer, "layer_config": layer_config}
        )
        or expected_method,
    )

    layer = object.__new__(RoutedExperts)
    layer.moe_config = DummyMoeConfig()
    method = INCWna16Scheme().get_moe_method(
        make_config(),
        layer,
        "model.layers.0.mlp",
        make_layer_config(group_size=32),
    )

    assert method is expected_method
    assert captured["layer"] is layer
    assert captured["layer_config"].is_gptq is True


def test_inc_resolve_scheme_selects_mxfp8() -> None:
    layer_config = INCLayerConfig(
        bits=8,
        group_size=32,
        sym=True,
        packing_format="auto_round:llm_compressor",
        backend="auto",
        data_type="mx_fp",
        quantized=True,
    )

    scheme = resolve_scheme(layer_config)

    assert isinstance(scheme, INCMxfp8Scheme)


class DummyLinearScheme(INCLinearScheme):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(self, *args, **kwargs) -> None:
        self.calls.append(("create_weights", args, kwargs))

    def process_weights_after_loading(self, layer) -> None:
        self.calls.append(("process_weights_after_loading", layer))

    def apply_weights(self, layer, x, bias=None):
        self.calls.append(("apply_weights", layer, x, bias))
        return "applied"


def test_inc_mxfp8_linear_scheme_delegates_to_kernel(monkeypatch) -> None:
    class DummyKernel:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def process_weights_after_loading(self, layer) -> None:
            self.calls.append(("process", layer))

        def apply_weights(self, layer, x, bias=None):
            self.calls.append(("apply", layer, x, bias))
            return "applied"

    kernel = DummyKernel()
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_linear.init_mxfp8_linear_kernel",
        lambda: kernel,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_linear.ModelWeightParameter",
        lambda **kwargs: torch.nn.Parameter(kwargs["data"], requires_grad=False),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_linear.GroupQuantScaleParameter",
        lambda **kwargs: torch.nn.Parameter(kwargs["data"], requires_grad=False),
    )

    scheme = INCMxfp8LinearScheme()
    layer = torch.nn.Module()

    scheme.create_weights(
        layer=layer,
        input_size_per_partition=64,
        output_partition_sizes=[48, 16],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.weight.shape == (64, 64)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (64, 2)
    assert layer.weight_scale.dtype == torch.uint8

    scheme.process_weights_after_loading(layer)
    result = scheme.apply_weights(layer, torch.randn(1, 64), None)

    assert result == "applied"
    assert [call[0] for call in kernel.calls] == ["process", "apply"]


def test_inc_mxfp8_linear_scheme_requires_block_32_input(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_mxfp8_linear.init_mxfp8_linear_kernel",
        lambda: object(),
    )
    scheme = INCMxfp8LinearScheme()

    with pytest.raises(ValueError, match="divisible by 32"):
        scheme.create_weights(
            layer=torch.nn.Module(),
            input_size_per_partition=48,
            output_partition_sizes=[32],
            input_size=48,
            output_size=32,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )


def test_inc_linear_method_delegates() -> None:
    scheme = DummyLinearScheme()
    method = INCLinearMethod(scheme)
    layer = DummyLayer()

    method.create_weights(
        layer,
        input_size_per_partition=1,
        output_partition_sizes=[2],
        input_size=1,
        output_size=2,
        params_dtype=None,
    )
    method.process_weights_after_loading(layer)
    result = method.apply(layer, "x", "b")

    assert result == "applied"
    assert [call[0] for call in scheme.calls] == [
        "create_weights",
        "process_weights_after_loading",
        "apply_weights",
    ]


def test_wna16_xpu_prefers_ark_when_available(monkeypatch) -> None:
    class DummyQuantLinear:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (True, None, object(), DummyQuantLinear),
    )

    method = INCWna16Scheme().get_linear_method(
        make_config(),
        object(),
        "layer",
        make_layer_config(),
    )

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCARKLinearMethod)


def test_inc_config_from_config_accepts_xpu_int2() -> None:
    def _make_int2_raw_config(**overrides) -> dict[str, object]:
        kwargs = {
            "bits": 2,
            "group_size": 64,
            "sym": True,
            "data_type": "int",
            "quant_method": "auto-round",
        }
        kwargs.update(overrides)

        return kwargs

    config = INCConfig.from_config(_make_int2_raw_config())

    assert config.weight_bits == 2
    assert config.group_size == 64
    assert config.sym is True
    assert config.data_type == "int"
    assert config.packing_format == "auto_round:auto_gptq"
    assert config.backend == "auto"


def test_wna16_xpu_int2_prefers_ark_when_available(monkeypatch) -> None:
    class DummyQuantLinear:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (True, None, object(), DummyQuantLinear),
    )

    method = INCWna16Scheme().get_linear_method(
        make_config(weight_bits=2, group_size=64),
        object(),
        "layer",
        make_layer_config(bits=2, group_size=64),
    )

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCARKLinearMethod)


def test_wna16_xpu_int2_requires_ark_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (False, "missing", None, None),
    )

    with pytest.raises(
        NotImplementedError,
        match="INC int2 on XPU requires the ARK backend",
    ):
        INCWna16Scheme().get_linear_method(
            make_config(weight_bits=2, group_size=64),
            object(),
            "layer",
            make_layer_config(bits=2, group_size=64),
        )


def test_wna16_xpu_int2_unsupported_config_still_raises(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)

    with pytest.raises(NotImplementedError, match="unsupported config"):
        INCWna16Scheme().get_linear_method(
            make_config(weight_bits=2, sym=False),
            object(),
            "layer",
            make_layer_config(bits=2, sym=False),
        )


def test_inc_ark_linear_method_xpu_int2_create_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    class DummyQuantLinear:
        pass

    class DummyLayer(torch.nn.Module):
        pass

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (True, None, object(), DummyQuantLinear),
    )

    layer = DummyLayer()
    method = INCARKLinearMethod(make_layer_config(bits=2, group_size=64))

    method.create_weights(
        layer=layer,
        input_size_per_partition=64,
        output_partition_sizes=[32, 32],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert method.pack_factor == 16
    assert layer.qweight.shape == (4, 64)
    assert layer.qweight.dtype == torch.int32
    assert layer.scales.shape == (1, 64)
    assert layer.scales.dtype == torch.bfloat16
    assert layer.qzeros.shape == (1, 4)
    assert layer.qzeros.dtype == torch.int32
    assert layer.g_idx.shape == (64,)
    assert layer.g_idx.dtype == torch.int32
    assert layer.in_features == 64
    assert layer.out_features == 64
    assert layer.params_dtype == torch.bfloat16


def test_wna16_xpu_falls_back_when_ark_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (False, "missing", None, None),
    )

    method = INCWna16Scheme().get_linear_method(
        make_config(),
        object(),
        "layer",
        make_layer_config(),
    )

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCXPULinearMethod)


def test_wna16_cpu_gptq_prefers_ark_when_available(monkeypatch) -> None:
    class DummyQuantLinear:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (True, None, object(), DummyQuantLinear),
    )

    method = INCWna16Scheme().get_linear_method(
        make_config(),
        object(),
        "layer",
        make_layer_config(),
    )

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCARKLinearMethod)


def test_wna16_cpu_gptq_raises_when_ark_and_marlin_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state",
        lambda: (False, "missing", None, None),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear.check_marlin_supported",
        lambda *args, **kwargs: False,
    )

    with pytest.raises(NotImplementedError, match="Only 4-bit and 8-bit symmetric"):
        INCWna16Scheme().get_linear_method(
            make_config(),
            object(),
            "layer",
            make_layer_config(),
        )


def test_wna16_linear_gptq_uses_auto_gptq_when_supported(monkeypatch) -> None:
    captured = {}

    class DummyMethod:
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_linear."
        "check_marlin_supported",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_gptq.AutoGPTQLinearMethod",
        DummyMethod,
    )

    scheme = INCWNA16LinearScheme(make_layer_config())

    assert isinstance(scheme.inner_method, DummyMethod)
    assert isinstance(captured["cfg"], AutoGPTQConfig)
    assert captured["cfg"].weight_bits == 4
    assert captured["cfg"].group_size == 128
    assert captured["cfg"].is_sym is True


def test_wna16_linear_gptq_unsupported_config_raises() -> None:
    with pytest.raises(NotImplementedError, match="Only 4-bit and 8-bit symmetric"):
        INCWNA16LinearScheme(make_layer_config(sym=False))


def test_wna16_xpu_unsupported_config_still_raises(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)

    with pytest.raises(NotImplementedError, match="unsupported config"):
        INCWna16Scheme().get_linear_method(
            make_config(weight_bits=2, sym=False),
            object(),
            "layer",
            make_layer_config(bits=2, sym=False),
        )


def test_inc_get_quant_method_unquantized_linear_returns_unquantized() -> None:
    config = make_config(extra_config={"layer": {"bits": 16}})
    layer = object.__new__(LinearBase)

    method = config.get_quant_method(layer, "layer")

    assert isinstance(method, UnquantizedLinearMethod)


def test_inc_get_quant_method_unquantized_moe_returns_unquantized(
    monkeypatch,
) -> None:
    """Early-exit returns UnquantizedFusedMoEMethod for FusedMoE layers
    when extra_config has bits >= 16."""
    config = make_config(extra_config={"layer": {"bits": 16}})
    layer = object.__new__(RoutedExperts)
    layer.moe_config = None  # UnquantizedFusedMoEMethod accepts moe_config

    class DummyUnquantizedFusedMoEMethod:
        def __init__(self, moe_config) -> None:
            self.moe_config = moe_config

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.inc.UnquantizedFusedMoEMethod",
        DummyUnquantizedFusedMoEMethod,
    )

    method = config.get_quant_method(layer, "layer")

    assert isinstance(method, DummyUnquantizedFusedMoEMethod)
    assert method.moe_config is None


def test_inc_get_quant_method_linear_uses_resolved_scheme(monkeypatch) -> None:
    config = make_config()
    layer = object.__new__(LinearBase)
    sentinel = object()

    class DummyScheme:
        def get_linear_method(self, _config, _layer, _prefix, _layer_config):
            return sentinel

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.factory.resolve_scheme",
        lambda _layer_config: DummyScheme(),
    )

    method = config.get_quant_method(layer, "layer")

    assert method is sentinel


def test_inc_get_quant_method_lm_head_uses_suffix_match(monkeypatch) -> None:
    """lm_head extra_config should apply to fully-qualified prefix."""
    config = make_config(
        extra_config={
            "lm_head": {
                "bits": 4,
                "group_size": 128,
                "sym": True,
            }
        }
    )
    layer = object.__new__(ParallelLMHead)
    sentinel = object()

    class DummyScheme:
        def get_linear_method(self, _config, _layer, _prefix, _layer_config):
            return sentinel

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.factory.resolve_scheme",
        lambda _layer_config: DummyScheme(),
    )

    method = config.get_quant_method(layer, "model.language_model.lm_head")

    assert method is sentinel


def test_inc_get_quant_method_moe_uses_resolved_scheme(monkeypatch) -> None:
    config = make_config()
    layer = object.__new__(RoutedExperts)
    sentinel = object()

    class DummyScheme:
        def get_moe_method(self, _config, _layer, _prefix, _layer_config):
            return sentinel

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.inc.schemes.factory.resolve_scheme",
        lambda _layer_config: DummyScheme(),
    )

    method = config.get_quant_method(layer, "layer")

    assert method is sentinel


def test_resolve_gptq_moe_falls_back_to_moe_wna16(monkeypatch) -> None:
    captured = {}

    class DummyMoeConfig:
        pass

    class DummyLayer:
        moe_config = DummyMoeConfig()

    class DummyBuiltConfig:
        pass

    built_config = DummyBuiltConfig()

    class DummyMethod:
        def __init__(self, cfg, moe):
            captured["cfg"] = cfg
            captured["moe"] = moe

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.marlin_utils."
        "check_moe_marlin_supports_layer",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Config.from_config",
        lambda cfg: captured.update({"from_config": cfg}) or built_config,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.moe_wna16.MoeWNA16Method",
        DummyMethod,
    )

    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=True,
        packing_format="auto_round:auto_gptq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    _resolve_gptq_moe(DummyLayer(), layer_config)

    assert captured["from_config"] == {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "lm_head": False,
    }
    assert captured["cfg"] is built_config
    assert captured["moe"] is DummyLayer.moe_config


def test_resolve_gptq_moe_uses_auto_gptq_when_supported(monkeypatch) -> None:
    captured = {}

    class DummyMoeConfig:
        pass

    class DummyLayer:
        moe_config = DummyMoeConfig()

    class DummyMethod:
        def __init__(self, cfg, moe):
            captured["cfg"] = cfg
            captured["moe"] = moe

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.marlin_utils."
        "check_moe_marlin_supports_layer",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_gptq.AutoGPTQMoEMethod",
        DummyMethod,
    )

    _resolve_gptq_moe(DummyLayer(), make_layer_config())

    assert isinstance(captured["cfg"], AutoGPTQConfig)
    assert captured["cfg"].weight_bits == 4
    assert captured["cfg"].group_size == 128
    assert captured["cfg"].is_sym is True
    assert captured["moe"] is DummyLayer.moe_config


def test_resolve_awq_moe_uses_marlin_when_supported(monkeypatch) -> None:
    captured = {}

    class DummyMoeConfig:
        pass

    class DummyLayer:
        moe_config = DummyMoeConfig()

    class DummyMethod:
        def __init__(self, cfg, moe):
            captured["cfg"] = cfg
            captured["moe"] = moe

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.utils.marlin_utils.check_moe_marlin_supports_layer",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.verify_marlin_supported",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.auto_awq.AutoAWQMoEMethod",
        DummyMethod,
    )

    layer_config = INCLayerConfig(
        bits=4,
        group_size=128,
        sym=False,
        packing_format="auto_round:auto_awq",
        backend="auto",
        data_type="int",
        quantized=True,
    )

    _resolve_awq_moe(DummyLayer(), layer_config)

    assert captured["cfg"].weight_bits == 4
    assert captured["cfg"].zero_point is True
    assert captured["moe"] is DummyLayer.moe_config


# ---------------------------------------------------------------------------
# Tests for get_layer_config step 4 (fused QKV / packed_modules_mapping)
# ---------------------------------------------------------------------------


class TestGetLayerConfigFusedQKV:
    """Tests for step-4 (fused QKV / packed_modules_mapping) logic.

    Focused on preventing false-positive substring matches.
    """

    def test_exact_fusion_key_match(self):
        """A layer whose name contains 'qkv' maps to its extra_config entry."""
        config = make_config(
            extra_config={
                "model.layers.0.self_attn.qkv_proj": {"bits": 8},
            }
        )
        config.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        bits, _, _ = config.get_layer_config(
            DummyLayer(), "model.layers.0.self_attn.qkv_proj"
        )
        assert bits == 8

    def test_false_substring_match_does_not_override(self):
        """Regression test for the false-substring-match bug.

        Scenario (Qwen3.6-35B-A3B VLM):
        - packed_modules_mapping has "qkv" → ["qkv"] (from vision encoder).
        - The GDN text-attention layer is named "in_proj_qkvz".
        - "qkv" is a substring of "in_proj_qkvz", so old code would enter
          step 4 and generate sub_name "in_proj_qkvz" (replacing "qkv" with
          "qkv"). That name is NOT in extra_config, so get_config() falls
          back to the global default (bits=4), even though correct is 16.
        - Fix: skip the fusion key when none of the generated sub_names
          actually exist in extra_config.
        """
        config = make_config(
            extra_config={
                "model.layers.0.in_proj_qkv": {"bits": 16},
                "model.layers.0.in_proj_z": {"bits": 16},
            }
        )
        config.packed_modules_mapping = {
            "qkv": ["qkv"],
        }
        bits, _, _ = config.get_layer_config(
            DummyLayer(), "model.layers.0.in_proj_qkvz"
        )
        # bits should be the global default (4) – no erroneous fusion match
        assert bits == 4

    def test_real_qkv_fusion_key_still_resolves(self):
        """The true "qkv" fusion (vision encoder) still resolves correctly."""
        config = make_config(
            extra_config={
                "vision_model.encoder.layers.0.self_attn.qkv": {"bits": 8},
            }
        )
        config.packed_modules_mapping = {
            "qkv": ["qkv"],
        }
        bits, _, _ = config.get_layer_config(
            DummyLayer(), "vision_model.encoder.layers.0.self_attn.qkv"
        )
        assert bits == 8

    def test_mixed_fp16_and_int4_fused_layer(self):
        """All sub-keys must agree; inconsistent configs raise ValueError."""
        config = make_config(
            extra_config={
                "model.layers.0.self_attn.q_proj": {"bits": 16},
                "model.layers.0.self_attn.k_proj": {"bits": 4},
                "model.layers.0.self_attn.v_proj": {"bits": 4},
            }
        )
        config.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        with pytest.raises(ValueError, match="consistent quant config"):
            config.get_layer_config(DummyLayer(), "model.layers.0.self_attn.qkv_proj")

    def test_fusion_triggered_by_regex_configured_sub_name(self):
        """Fusion step 4 is still triggered when sub_names match via regex.

        Ensures the guard does not regress when extra_config uses regex
        patterns instead of exact keys to configure sub-modules.
        """
        config = make_config(
            extra_config={
                r"model\.layers\.\d+\.self_attn\.(q|k|v)_proj": {"bits": 8},
            }
        )
        config.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        bits, _, _ = config.get_layer_config(
            DummyLayer(), "model.layers.0.self_attn.qkv_proj"
        )
        assert bits == 8


# ---------------------------------------------------------------------------
# INC XPU w4a8: int4 weights with dynamic per-token int8 activations
#
# The w4a8 path keeps int4 weights but dynamically quantizes activations to
# per-token symmetric int8, so the GEMM stays on the int8 datapath instead of
# upconverting weights to the activation dtype. It is opt-in through
# VLLM_XPU_INC_WNA16_BACKEND because which kernel is fastest depends on the
# device: ARK can use XMX int8 on some XPUs, and the default ("auto")
# preference order is deliberately left unchanged.
#
# The backend-selection tests stub out ARK and the platform; the weight-creation
# tests only need the TP-rank stubs. Both run anywhere. The apply_weights tests
# stub out int4_gemm_w4a8 — what needs guarding there is the calling convention
# (scale dtypes, argument order, weight reuse, batch-size gate), which is where
# this path is easy to get silently wrong, since the kernel accepts bf16 scales
# and returns plausible-looking garbage rather than raising. The ones that cross
# the token threshold reach ``vllm._xpu_ops`` for the activation quantizer, so
# they are XPU-image only; the small-batch fallback test returns before that
# import and must stay unmarked to keep covering the gate off-XPU. The
# end-to-end tests load a real int4 checkpoint and additionally need the op.
# ---------------------------------------------------------------------------

_BACKEND_ENV = "VLLM_XPU_INC_WNA16_BACKEND"
_ARK_STATE = (
    "vllm.model_executor.layers.quantization.inc.schemes.inc_ark_ops.get_ark_state"
)
_QUANT_REF = "vllm._xpu_ops.xpu_ops.dynamic_per_token_int8_quant_ref"

IN_FEATURES = 128
OUT_FEATURES = 64

# auto-round int4/sym/g128 checkpoints, which is what this backend serves. The
# GPTQ- and AWQ-packed variants take different repack branches in
# ``process_weights_after_loading``, so both are worth loading.
E2E_MODELS = [
    pytest.param("OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc", id="auto_round:auto_gptq"),
    pytest.param(
        "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound", id="auto_round:auto_awq"
    ),
]


def _has_xpu_ops() -> bool:
    """True when ``vllm._xpu_ops`` is importable.

    ``apply_weights`` imports it for the activation quantizer, and the stubs
    below patch through it, so tests that exercise that path cannot even be set
    up on a build without vllm-xpu-kernels installed.
    """
    if not current_platform.is_xpu():
        return False
    try:
        import vllm._xpu_ops  # noqa: F401
    except ImportError:
        return False
    return True


def _has_w4a8_kernel() -> bool:
    """True when this build can actually run the w4a8 GEMM.

    The op is registered as a side effect of importing ``vllm._xpu_ops``, so it
    is not visible on ``torch.ops._xpu_C`` until that import happens.
    """
    return _has_xpu_ops() and hasattr(torch.ops._xpu_C, "int4_gemm_w4a8")


requires_xpu_ops = pytest.mark.skipif(
    not _has_xpu_ops(),
    reason="needs an XPU image with vllm._xpu_ops available",
)

requires_w4a8_kernel = pytest.mark.skipif(
    not _has_w4a8_kernel(),
    reason="needs an XPU with the int4_gemm_w4a8 op",
)


def _fake_xpu(monkeypatch, ark_available: bool = True) -> None:
    class DummyQuantLinear:
        pass

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(
        _ARK_STATE,
        lambda: (
            (True, None, object(), DummyQuantLinear)
            if ark_available
            else (False, "missing", None, None)
        ),
    )


def _with_w4a8_kernel(monkeypatch) -> None:
    """Present the op so the w4a8 support check passes off-XPU."""

    class FakeOps:
        int4_gemm_w4a8 = object()

    monkeypatch.setattr(torch.ops, "_xpu_C", FakeOps, raising=False)


def _dispatch(layer_config=None):
    return INCWna16Scheme().get_linear_method(
        object(), object(), "layer", layer_config or make_layer_config()
    )


@pytest.fixture
def single_tp_rank(monkeypatch):
    """Parameter creation reads the TP rank/size; there is no distributed init."""
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _create_weights(
    method,
    input_size_per_partition: int = IN_FEATURES,
    output_partition_sizes: list[int] | None = None,
) -> torch.nn.Module:
    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    output_partition_sizes = (
        [OUT_FEATURES] if output_partition_sizes is None else output_partition_sizes
    )
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size_per_partition,
        output_partition_sizes=output_partition_sizes,
        input_size=input_size_per_partition,
        output_size=sum(output_partition_sizes),
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    return layer


@pytest.fixture
def w4a8_layer(single_tp_rank):
    """A w4a8 method plus a layer whose weights are created but not processed."""
    method = INCXPUW4A8LinearMethod(make_layer_config())
    return method, _create_weights(method)


def _stub_quant(monkeypatch, captured=None):
    def fake_quant(x, use_sym, bits):
        if captured is not None:
            captured["quant_args"] = (tuple(x.shape), use_sym, bits)
        m, k = x.shape
        return (
            torch.zeros(m, k, dtype=torch.int8),
            torch.ones(m, 1, dtype=x.dtype),
            torch.zeros(m, 1, dtype=torch.int32),
        )

    monkeypatch.setattr(_QUANT_REF, fake_quant, raising=False)


def _stub_gemm(monkeypatch, captured=None):
    def fake_gemm(*args):
        if captured is not None:
            captured["gemm_args"] = args
        # The real kernel always emits float16, whatever the model dtype.
        return torch.zeros(args[0].shape[0], OUT_FEATURES, dtype=torch.float16)

    monkeypatch.setattr(torch.ops._xpu_C, "int4_gemm_w4a8", fake_gemm, raising=False)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_default_backend_keeps_ark_preference(monkeypatch) -> None:
    """The "auto" default must not change upstream behaviour.

    ARK still wins whenever it is importable.
    """
    monkeypatch.delenv(_BACKEND_ENV, raising=False)
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch()

    assert isinstance(method, INCLinearMethod)
    assert isinstance(method.scheme, INCARKLinearMethod)


def test_w4a8_backend_overrides_available_ark(monkeypatch) -> None:
    """An explicit request wins even though ARK would otherwise be chosen."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch()

    assert isinstance(method.scheme, INCXPUW4A8LinearMethod)
    assert not isinstance(method.scheme, INCARKLinearMethod)
    # w4a8 subclasses w4a16 so it can fall back for small batches.
    assert isinstance(method.scheme, INCXPULinearMethod)


def test_w4a16_backend_overrides_available_ark(monkeypatch) -> None:
    monkeypatch.setenv(_BACKEND_ENV, "w4a16")
    _fake_xpu(monkeypatch, ark_available=True)

    method = _dispatch()

    assert isinstance(method.scheme, INCXPULinearMethod)
    assert not isinstance(method.scheme, INCXPUW4A8LinearMethod)
    assert not isinstance(method.scheme, INCARKLinearMethod)


def test_ark_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    """Explicitly asking for ARK must fail loudly, not fall back silently."""
    monkeypatch.setenv(_BACKEND_ENV, "ark")
    _fake_xpu(monkeypatch, ark_available=False)

    with pytest.raises(NotImplementedError, match="auto_round_kernel is unavailable"):
        _dispatch()


@pytest.mark.parametrize("backend", ["w4a16", "w4a8"])
def test_onednn_backends_reject_int2(monkeypatch, backend) -> None:
    """The oneDNN int4 GEMMs cannot serve int2; ARK is the only int2 path."""
    monkeypatch.setenv(_BACKEND_ENV, backend)
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    with pytest.raises(NotImplementedError, match="only supports int4"):
        _dispatch(make_layer_config(bits=2))


@pytest.mark.parametrize("group_size", [48, -1, 0])
def test_w4a8_rejects_unaligned_group_size(monkeypatch, group_size) -> None:
    """The kernel needs a group size that is a positive multiple of 32."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    with pytest.raises(NotImplementedError, match="multiple of 32"):
        _dispatch(make_layer_config(group_size=group_size))


@pytest.mark.parametrize("group_size", [128, 64, 32])
def test_w4a8_accepts_aligned_group_size(monkeypatch, group_size) -> None:
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)
    _with_w4a8_kernel(monkeypatch)

    method = _dispatch(make_layer_config(group_size=group_size))

    assert isinstance(method.scheme, INCXPUW4A8LinearMethod)


def test_w4a8_requires_the_kernel(monkeypatch) -> None:
    """Older vllm-xpu-kernels builds lack the op entirely."""
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    _fake_xpu(monkeypatch, ark_available=True)

    class FakeOpsNoW4A8:
        pass

    monkeypatch.setattr(torch.ops, "_xpu_C", FakeOpsNoW4A8, raising=False)

    with pytest.raises(NotImplementedError, match="int4_gemm_w4a8"):
        _dispatch()


def test_invalid_backend_value_raises(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.setenv(_BACKEND_ENV, "onednn")
    with pytest.raises(ValueError, match=_BACKEND_ENV):
        _ = envs.VLLM_XPU_INC_WNA16_BACKEND


# ---------------------------------------------------------------------------
# Partition shape
# ---------------------------------------------------------------------------
#
# ``XPUW4A8IntLinearKernel.can_implement`` rejects partition shapes whose in/out
# dims are not multiples of 8, but that check is unreachable from the INC path:
# the shapes only exist once ``create_weights`` is called, well after backend
# selection. Unaligned dims must not reach the kernel — the int4 packing divides
# both dims by 8, so an unaligned size is silently truncated into a weight tensor
# that is too small rather than rejected.


def test_rejects_unaligned_input_partition(single_tp_rank) -> None:
    method = INCXPUW4A8LinearMethod(make_layer_config(group_size=32))

    with pytest.raises(NotImplementedError, match=r"multiples of 8.*input=132"):
        _create_weights(method, input_size_per_partition=132)


def test_rejects_unaligned_output_partition(single_tp_rank) -> None:
    method = INCXPUW4A8LinearMethod(make_layer_config())

    with pytest.raises(NotImplementedError, match=r"multiples of 8.*output=60"):
        _create_weights(method, output_partition_sizes=[60])


def test_rejects_unaligned_sum_of_output_partitions(single_tp_rank) -> None:
    """A fused QKV/MLP layer is only as aligned as the sum of its shards."""
    method = INCXPUW4A8LinearMethod(make_layer_config())

    with pytest.raises(NotImplementedError, match=r"multiples of 8.*output=76"):
        _create_weights(method, output_partition_sizes=[OUT_FEATURES, 12])


def test_reports_both_unaligned_dims(single_tp_rank) -> None:
    """The message names every offending dim, not just the first."""
    method = INCXPUW4A8LinearMethod(make_layer_config(group_size=32))

    with pytest.raises(NotImplementedError) as excinfo:
        _create_weights(
            method, input_size_per_partition=132, output_partition_sizes=[60]
        )

    assert "input=132" in str(excinfo.value)
    assert "output=60" in str(excinfo.value)


def test_accepts_aligned_partition_shape(single_tp_rank) -> None:
    """Multiples of 8 that are not powers of two are still fine."""
    method = INCXPUW4A8LinearMethod(make_layer_config(group_size=32))

    layer = _create_weights(
        method, input_size_per_partition=96, output_partition_sizes=[24, 48]
    )

    assert layer.qweight.shape == (96 // 8, 72)


def test_w4a16_accepts_unaligned_partition_shape(single_tp_rank) -> None:
    """The constraint is a w4a8 kernel requirement; w4a16 must stay unaffected."""
    method = INCXPULinearMethod(make_layer_config(group_size=32))

    layer = _create_weights(method, output_partition_sizes=[60])

    assert layer.scales.shape[1] == 60


# ---------------------------------------------------------------------------
# Weight processing
# ---------------------------------------------------------------------------


def test_keeps_fp16_scales(w4a8_layer) -> None:
    """The kernel reads scales as fp16 regardless of activation dtype.

    Passing bf16 scales does not raise — it silently returns wrong results — so
    an fp16 copy must exist, while the bf16 ``scales`` survive for the
    small-batch w4a16 fallback.
    """
    method, layer = w4a8_layer
    layer.scales.data.fill_(0.125)  # exactly representable in both dtypes

    method.process_weights_after_loading(layer)

    assert layer.scales_fp16.dtype is torch.float16
    assert layer.scales_fp16.shape == layer.scales.shape
    assert torch.equal(layer.scales_fp16.data.float(), layer.scales.data.float())
    # The w4a16 fallback still needs the original activation-dtype scales.
    assert layer.scales.dtype is torch.bfloat16


def test_reuses_w4a16_weight_layout(w4a8_layer) -> None:
    """w4a8 and w4a16 accept bit-identical weights, so there is no repacking."""
    method, layer = w4a8_layer
    qweight_before = layer.qweight.data.clone()

    method.process_weights_after_loading(layer)

    assert layer.qweight.dtype is torch.int32
    assert layer.qweight.shape == (IN_FEATURES // 8, OUT_FEATURES)
    assert torch.equal(layer.qweight.data, qweight_before)
    # Symmetric int4: the zero point is the constant 8, as for w4a16.
    assert layer.qzeros.dtype is torch.int8
    assert layer.qzeros.tolist() == [8]


# ---------------------------------------------------------------------------
# apply_weights
# ---------------------------------------------------------------------------


def test_falls_back_to_w4a16_for_small_batches(monkeypatch, w4a8_layer) -> None:
    """Below the token threshold, per-token quant overhead dominates."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    calls = []

    def record_fallback(self, layer, x, bias=None):
        calls.append(tuple(x.shape))
        return torch.zeros(x.shape[:-1] + (OUT_FEATURES,), dtype=x.dtype)

    monkeypatch.setattr(INCXPULinearMethod, "apply_weights", record_fallback)
    monkeypatch.setattr(
        torch.ops._xpu_C,
        "int4_gemm_w4a8",
        lambda *args: pytest.fail("small batches must not call the w4a8 kernel"),
        raising=False,
    )

    tokens = method._MIN_TOKENS_FOR_INT8 - 1
    x = torch.zeros(tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert calls == [(tokens, IN_FEATURES)]
    assert out.shape == (tokens, OUT_FEATURES)


@requires_xpu_ops
def test_calls_kernel_at_threshold(monkeypatch, w4a8_layer) -> None:
    """At/above the threshold, activations are int8-quantized per token."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    monkeypatch.setattr(
        INCXPULinearMethod,
        "apply_weights",
        lambda self, layer, x, bias=None: pytest.fail(
            "large batches must not take the w4a16 fallback"
        ),
    )
    captured: dict[str, tuple] = {}
    _stub_quant(monkeypatch, captured)
    _stub_gemm(monkeypatch, captured)

    tokens = method._MIN_TOKENS_FOR_INT8
    x = torch.zeros(tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    # Symmetric 8-bit per-token quantization over the flattened activations.
    assert captured["quant_args"] == ((tokens, IN_FEATURES), True, 8)

    (
        quant_x,
        x_scale,
        x_zero,
        qweight,
        w_scale,
        w_zp,
        group_size,
        g_idx,
        bias,
    ) = captured["gemm_args"]
    assert quant_x.dtype is torch.int8
    # Both scales must reach the kernel as fp16, whatever the model dtype.
    assert x_scale.dtype is torch.float16
    assert w_scale.dtype is torch.float16
    assert w_scale is layer.scales_fp16
    assert x_zero.dtype is torch.int32
    assert qweight is layer.qweight
    assert w_zp is layer.qzeros
    assert group_size == 128
    assert g_idx is None
    assert bias is None

    # The kernel emits fp16; the result is cast back to the activation dtype.
    assert out.dtype is torch.bfloat16
    assert out.shape == (tokens, OUT_FEATURES)


@requires_xpu_ops
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
def test_scales_are_fp16_for_any_activation_dtype(
    monkeypatch, w4a8_layer, act_dtype
) -> None:
    """fp16 scales are required for bf16 *and* fp16 activations alike."""
    method, layer = w4a8_layer
    layer.scales.data = layer.scales.data.to(act_dtype)
    method.process_weights_after_loading(layer)

    captured: dict[str, tuple] = {}
    _stub_quant(monkeypatch, captured)
    _stub_gemm(monkeypatch, captured)

    x = torch.zeros(method._MIN_TOKENS_FOR_INT8, IN_FEATURES, dtype=act_dtype)
    out = method.apply_weights(layer, x)

    assert captured["gemm_args"][1].dtype is torch.float16
    assert captured["gemm_args"][4].dtype is torch.float16
    assert out.dtype is act_dtype


@requires_xpu_ops
def test_forwards_bias_to_kernel(monkeypatch, w4a8_layer) -> None:
    """Bias is applied by the kernel, not added afterwards."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    captured: dict[str, tuple] = {}
    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch, captured)

    bias = torch.zeros(OUT_FEATURES, dtype=torch.bfloat16)
    x = torch.zeros(method._MIN_TOKENS_FOR_INT8, IN_FEATURES, dtype=torch.bfloat16)
    method.apply_weights(layer, x, bias)

    assert captured["gemm_args"][8] is bias


@requires_xpu_ops
def test_preserves_leading_dims(monkeypatch, w4a8_layer) -> None:
    """3D activations must round-trip through the flatten/reshape."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch)

    tokens = method._MIN_TOKENS_FOR_INT8
    x = torch.zeros(4, tokens, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert out.shape == (4, tokens, OUT_FEATURES)
    assert out.dtype is torch.bfloat16


@requires_xpu_ops
def test_batch_gate_uses_flattened_token_count(monkeypatch, w4a8_layer) -> None:
    """The gate counts total tokens, not the leading dimension."""
    method, layer = w4a8_layer
    method.process_weights_after_loading(layer)

    monkeypatch.setattr(
        INCXPULinearMethod,
        "apply_weights",
        lambda self, layer, x, bias=None: pytest.fail(
            "flattened token count is at the threshold; must use w4a8"
        ),
    )
    _stub_quant(monkeypatch)
    _stub_gemm(monkeypatch)

    tokens = method._MIN_TOKENS_FOR_INT8
    # Each leading slice is below the threshold, but 4 * (tokens // 4) is not.
    x = torch.zeros(4, tokens // 4, IN_FEATURES, dtype=torch.bfloat16)
    out = method.apply_weights(layer, x)

    assert out.shape == (4, tokens // 4, OUT_FEATURES)


# ---------------------------------------------------------------------------
# End to end on real weights
# ---------------------------------------------------------------------------


@requires_w4a8_kernel
@pytest.mark.parametrize("model", E2E_MODELS)
def test_e2e_generates_with_real_weights(vllm_runner, monkeypatch, model) -> None:
    """The whole path — real int4 checkpoint, real kernel, real output.

    The stubbed tests above cannot catch a wrong-but-plausible calling
    convention, because the fake kernel accepts anything. Greedy decoding of a
    factual prompt does: wrongly set scales or a bad weight layout turn the
    answer into noise rather than raising.
    """
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")

    with vllm_runner(
        model, max_model_len=512, enforce_eager=True, gpu_memory_utilization=0.55
    ) as vllm_model:
        out = vllm_model.generate_greedy(["The capital of France is"], max_tokens=8)

    assert "Paris" in out[0][1]


@requires_w4a8_kernel
def test_e2e_selects_w4a8_for_every_linear(vllm_runner, monkeypatch) -> None:
    """Guard against the validation silently becoming dead code.

    If backend selection stopped routing here, the generation test above would
    still pass on the w4a16 path, so assert the method is actually in use and
    that every partition shape it accepted honours the alignment rule.
    """
    monkeypatch.setenv(_BACKEND_ENV, "w4a8")
    # apply_model pickles the closure below.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        E2E_MODELS[0].values[0],
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.55,
    ) as vllm_model:

        def check_model(model):
            shapes = []
            for module in model.modules():
                scheme = getattr(getattr(module, "quant_method", None), "scheme", None)
                if isinstance(scheme, INCXPUW4A8LinearMethod):
                    # qweight is the packed [in // 8, out] layout.
                    packed_in, out = module.qweight.shape
                    shapes.append((packed_in * 8, out))

            assert shapes, "no linear layer used the w4a8 method"
            assert all(i % 8 == 0 and o % 8 == 0 for i, o in shapes), shapes

        vllm_model.apply_model(check_model)
