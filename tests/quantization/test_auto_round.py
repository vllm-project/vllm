# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.inc.config_parser import INCLayerConfig
from vllm.model_executor.layers.quantization.inc.inc_linear import INCLinearMethod
from vllm.model_executor.layers.quantization.inc.schemes import (
    INCWna16Scheme,
    resolve_scheme,
)
from vllm.model_executor.layers.quantization.inc.schemes.inc_scheme import (
    INCLinearScheme,
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
            not (current_platform.is_cuda() or current_platform.is_xpu())
            or current_platform.device_count() < 2,
            reason="72B INT2 AutoRound model requires XPU with at least 2 devices.",
        ),
        id="auto_round:auto_gptq_int2_tp2",
    ),
]

MODEL_RUNNER_KWARGS = {
    "Intel/Qwen3-8B-w2g64-for-ut": {
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
@pytest.mark.parametrize("model", MODELS)
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
        "vllm.model_executor.layers.quantization.utils.marlin_utils.check_marlin_supported",
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
        "vllm.model_executor.layers.quantization.utils.marlin_utils.check_marlin_supported",
        lambda *args, **kwargs: True,
    )
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
        "vllm.model_executor.layers.quantization.utils.marlin_utils.check_marlin_supported",
        lambda *args, **kwargs: True,
    )
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
