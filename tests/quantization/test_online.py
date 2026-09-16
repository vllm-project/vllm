# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests online quantization."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from torch.distributed import ProcessGroup

from tests.quantization.utils import (
    _test_online_quant_peak_mem_impl,
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm._custom_ops import scaled_fp4_quant
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.quantization import (
    QuantizationConfigArgs,
    resolve_quantization_config,
)
from vllm.config.vllm import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMxFp8Config,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    _fp8_channel_scale,
    _fp8_quant_per_channel,
    _fp8_scale,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.online.int8 import Int8OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
    _quantize_moe_weight_to_nvfp4,
)
from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig,
    QuarkLinearMethod,
)
from vllm.model_executor.layers.quantization.utils import quant_utils
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    amax_for_tp_weight_quant,
    kMxfp8Dynamic,
    weight_amax,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.base_loader import log_online_quantization
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.models.granitemoe import (
    GraniteMoeModel,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942, on_gfx950
else:

    def on_gfx950() -> bool:
        return False

    def on_gfx942() -> bool:
        return False


DEVICE = current_platform.device_type
GRANITE_MODEL_NAME = "ibm-granite/granite-3.0-1b-a400m-base"
PARTIALLY_PREQUANTIZED_MODEL_NAME = (
    "nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable"
)


def test_online_nvfp4_reuses_kernel_when_weights_are_reprocessed(
    monkeypatch,
) -> None:
    method = object.__new__(Nvfp4OnlineMoEMethod)
    method.moe = SimpleNamespace(is_act_and_mul=True)
    method.nvfp4_backend = object()
    method.experts_cls = object
    method.moe_quant_config = None
    method.moe_kernel = None

    layer = Mock()
    converted_weights = tuple(object() for _ in range(8))
    convert_weights = Mock(return_value=converted_weights)
    process_weights = Mock()
    kernel = SimpleNamespace(
        fused_experts=SimpleNamespace(
            process_weights_after_loading=process_weights,
        )
    )
    make_kernel = Mock(return_value=kernel)
    get_quant_config = Mock(return_value=object())
    method.get_fused_moe_quant_config = get_quant_config

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.nvfp4."
        "convert_to_nvfp4_moe_kernel_format",
        convert_weights,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.nvfp4.replace_parameter",
        Mock(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.nvfp4.make_nvfp4_moe_kernel",
        make_kernel,
    )

    method._setup_kernel(layer)
    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert convert_weights.call_count == 2
    make_kernel.assert_called_once()
    get_quant_config.assert_called_once()
    assert process_weights.call_count == 2


def _fully_quantized_quark_config() -> QuarkConfig:
    return QuarkConfig(
        {
            "exclude": [],
            "global_quant_config": {
                "weight": {
                    "dtype": "int8",
                    "qscheme": "per_tensor",
                    "is_dynamic": False,
                    "symmetric": True,
                },
                "input_tensors": {
                    "dtype": "int8",
                    "qscheme": "per_tensor",
                    "is_dynamic": False,
                    "symmetric": True,
                },
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
        }
    )


def _fully_quantized_modelopt_config() -> ModelOptFp8Config:
    return ModelOptFp8Config(
        quant_method="FP8",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )


def _moe_only_compressed_tensors_config() -> CompressedTensorsConfig:
    return CompressedTensorsConfig(
        target_scheme_map={"RoutedExperts": {}},
        ignore=[],
        quant_format="pack-quantized",
    )


def _write_minimal_llama_config(
    model_path: Path, quantization_config: dict[str, Any] | None = None
) -> None:
    config: dict[str, Any] = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "max_position_embeddings": 32,
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (model_path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize(
    "checkpoint_config_factory,raises_conflict",
    [
        pytest.param(_fully_quantized_quark_config, True, id="quark"),
        pytest.param(_fully_quantized_modelopt_config, True, id="modelopt"),
        pytest.param(
            _moe_only_compressed_tensors_config,
            False,
            id="compressed_tensors",
        ),
    ],
)
def test_online_prequantized_compatibility(
    checkpoint_config_factory,
    raises_conflict: bool,
    default_vllm_config,
    dist_init,
) -> None:
    """Online weights replace only layers left unquantized by a checkpoint."""
    default_vllm_config.model_config = ModelConfig()
    checkpoint_config = checkpoint_config_factory()

    checkpoint_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear="mxfp8")
    )
    config = checkpoint_config

    layer_kwargs = {
        "input_size": 32,
        "output_size": 32,
        "bias": False,
        "params_dtype": torch.bfloat16,
        "quant_config": config,
        "prefix": "model.layers.0.self_attn.o_proj",
        "disable_tp": True,
    }

    if raises_conflict:
        with pytest.raises(ValueError, match="pre-quantized layer"):
            ColumnParallelLinear(**layer_kwargs)
    else:
        layer = ColumnParallelLinear(**layer_kwargs)
        assert isinstance(layer.quant_method, Mxfp8OnlineLinearMethod)


def test_online_target_rejects_prequantized_layer(
    default_vllm_config, dist_init
) -> None:
    """A targets match participates in checkpoint compatibility checks."""
    default_vllm_config.model_config = ModelConfig()
    prefix = "model.layers.0.self_attn.o_proj"
    quant_config = _fully_quantized_quark_config()
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={prefix: "mxfp8"})
    )

    with pytest.raises(ValueError, match="pre-quantized layer"):
        ColumnParallelLinear(
            input_size=32,
            output_size=32,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
        )


def test_online_ignore_keeps_checkpoint_quantization_linear(
    default_vllm_config, dist_init, monkeypatch
):
    """Ignoring online quantization does not replace a checkpoint method (linear)."""
    default_vllm_config.model_config = ModelConfig()
    quant_config = _fully_quantized_quark_config()
    prefix = "model.layers.0.self_attn.o_proj"
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear="mxfp8", ignore=[prefix])
    )
    monkeypatch.setattr(
        quant_config.online_quantization_config,
        "get_quant_method",
        lambda *args: pytest.fail("online method should not be constructed"),
    )

    layer = ColumnParallelLinear(
        input_size=32,
        output_size=32,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
        disable_tp=True,
    )

    assert isinstance(layer.quant_method, QuarkLinearMethod)


def test_online_quantization_rejects_prequantized_moe(
    default_vllm_config, dist_init
) -> None:
    """Online linear and MoE quantization reject a pre-quantized MoE layer."""
    default_vllm_config.model_config = ModelConfig()
    prefix = "model.layers.0.mlp.experts"
    quant_config = _fully_quantized_quark_config()
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear="mxfp4", moe="mxfp4")
    )

    with pytest.raises(ValueError, match="pre-quantized layer"):
        FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=32,
            intermediate_size=64,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=prefix,
        )


def test_activation_only_override_applies_to_checkpoint_method(
    default_vllm_config, tmp_path
) -> None:
    """Activation-only overrides remain available to the checkpoint method."""
    _write_minimal_llama_config(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        quantization="compressed-tensors",
        quantization_config=QuantizationConfigArgs(moe={"activation": "mxfp8"}),
        hf_overrides={
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {},
            }
        },
    )

    result = weight_utils.get_quant_config(model_config, LoadConfig())

    assert isinstance(result, CompressedTensorsConfig)
    assert isinstance(result.online_quantization_config, OnlineQuantizationConfig)

    default_vllm_config.model_config = model_config
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _resolve_activation_key,
    )

    assert _resolve_activation_key(None) == kMxfp8Dynamic


def test_online_overlay_requires_normalized_quantization_config(tmp_path) -> None:
    """The loader rejects a raw config dict instead of silently ignoring it."""
    _write_minimal_llama_config(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        quantization="compressed-tensors",
        quantization_config={"linear": "mxfp8"},
        hf_overrides={
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {},
            }
        },
    )

    # assert isinstance(online_args, QuantizationConfigArgs)
    # in maybe_compose_online_quantization
    with pytest.raises(AssertionError):
        weight_utils.get_quant_config(model_config, LoadConfig())


def test_online_overlay_loads_checkpoint_config_file(tmp_path) -> None:
    """An online overlay must not bypass checkpoint config file loading."""
    _write_minimal_llama_config(tmp_path)
    (tmp_path / "quantize_config.json").write_text(
        '{"w_bit": 4, "q_group_size": 128, "zero_point": true}'
    )
    model_config = ModelConfig(
        model=str(tmp_path),
        quantization="awq",
        quantization_config=QuantizationConfigArgs(linear="mxfp8"),
    )

    result = weight_utils.get_quant_config(model_config, LoadConfig())

    assert isinstance(result, AutoAWQConfig)
    assert result.weight_bits == 4
    assert result.group_size == 128
    assert result.zero_point
    assert isinstance(result.online_quantization_config, OnlineQuantizationConfig)
    assert result.online_quantization_config.args is model_config.quantization_config


@pytest.mark.parametrize(
    "quantization,checkpoint_quantization_config,is_config_file",
    [
        pytest.param("fp8_per_block", None, False, id="fp8-per-block-online"),
        pytest.param("mxfp4", None, False, id="mxfp4-online"),
        pytest.param("mxfp8", None, False, id="mxfp8-online"),
        pytest.param(
            "mxfp8",
            {"quant_method": "mxfp8", "ignored_layers": []},
            False,
            id="mxfp8-inline-checkpoint-config",
        ),
        pytest.param(
            "mxfp8",
            {"quant_method": "mxfp8", "ignored_layers": []},
            True,
            id="mxfp8-checkpoint-config-file",
        ),
    ],
)
def test_online_shorthand_selects_checkpoint_or_online_config(
    tmp_path, quantization, checkpoint_quantization_config, is_config_file
) -> None:
    """Online shorthands defer to checkpoint metadata when available."""

    if is_config_file:
        (tmp_path / "hf_quant_config.json").write_text(
            '{"quant_method": "mxfp8", "ignored_layers": []}'
        )

    _write_minimal_llama_config(
        tmp_path,
        None if is_config_file else checkpoint_quantization_config,
    )
    model_config = ModelConfig(model=str(tmp_path), quantization=quantization)

    result = weight_utils.get_quant_config(model_config, LoadConfig())

    if checkpoint_quantization_config is not None:
        assert isinstance(result, ModelOptMxFp8Config)
        assert result.online_quantization_config is None
    else:
        assert isinstance(result, OnlineQuantizationConfig)
        assert result.online_quantization_config is None


def test_log_online_quantization_for_composable_config(monkeypatch) -> None:
    """Composable configs log their nested online quantization results."""
    online_config = OnlineQuantizationConfig(QuantizationConfigArgs(linear="mxfp8"))
    online_config.quantized_layers = {
        "model.layers.0.self_attn.o_proj": ("linear", "mxfp8", None),
        "model.layers.1.self_attn.o_proj": ("linear", "mxfp8", None),
    }
    vllm_config = VllmConfig(quant_config=online_config)
    log_args = []
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info",
        lambda *args: log_args.append(args),
    )

    log_online_quantization(vllm_config)

    assert log_args == [
        (
            "Quantized %d layers of types: %s",
            2,
            "; ".join(online_config.quantized_layer_summaries),
        )
    ]


def test_checkpoint_quantization_rejects_online_shorthand(tmp_path) -> None:
    """A checkpoint quant method cannot be replaced by an online shorthand."""
    _write_minimal_llama_config(tmp_path, {"quant_method": "mxfp4"})

    with pytest.raises(ValueError, match="does not match the quantization"):
        ModelConfig(model=str(tmp_path), quantization="fp8_per_channel")


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    ("model_name,quant_scheme,online_quant_args,expected_linear_cls,expected_moe_cls"),
    [
        # simple case - quantization='fp8_per_tensor'
        (
            GRANITE_MODEL_NAME,
            "fp8_per_tensor",
            None,
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # simple case - quantization='fp8_per_block'
        (
            GRANITE_MODEL_NAME,
            "fp8_per_block",
            None,
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerBlockOnlineMoEMethod,
        ),
        (
            GRANITE_MODEL_NAME,
            "fp8_per_channel",
            None,
            Fp8PtpcOnlineLinearMethod,
            Fp8PtpcOnlineMoEMethod,
        ),
        # quantization='online' with per-layer-kind overrides
        (
            GRANITE_MODEL_NAME,
            "online",
            {
                "linear": "fp8_per_block",
                "moe": "fp8_per_tensor",
            },
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # quantization='online' with per-layer target patterns
        (
            GRANITE_MODEL_NAME,
            "online",
            {
                "targets": {
                    r"re:.*self_attn\.o_proj": "fp8_per_block",
                    r"re:.*block_sparse_moe\.experts": "fp8_per_tensor",
                }
            },
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # ignore with direct layer name
        (
            GRANITE_MODEL_NAME,
            "fp8_per_tensor",
            # qkv_proj is fused from q_proj/k_proj/v_proj. The shard regex
            # remains supported alongside direct fused-name regexes.
            {"ignore": ["model.layers.1.self_attn.o_proj", "re:.*[qkv]_proj"]},
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        (
            GRANITE_MODEL_NAME,
            "mxfp4",
            None,
            Mxfp4OnlineLinearMethod,
            Mxfp4OnlineMoEMethod,
        ),
        pytest.param(
            PARTIALLY_PREQUANTIZED_MODEL_NAME,
            None,
            {"targets": {"model.layers.1.self_attn.o_proj": "mxfp8"}},
            Mxfp8OnlineLinearMethod,
            CompressedTensorsMoEMethod,
            id="partially_prequantized_checkpoint",
        ),
    ],
    ids=[
        "fp8_per_tensor",
        "fp8_per_block",
        "fp8_per_channel",
        "per_layer_kind_overrides",
        "targets",
        "ignore",
        "mxfp4",
        "partially_prequantized_checkpoint",
    ],
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    model_name: str,
    quant_scheme: str | None,
    online_quant_args: dict | None,
    expected_linear_cls,
    expected_moe_cls,
    use_rocm_aiter: bool,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    """
    Tests that online quantization frontend configuration works -
    selecting quant schemes, overriding quant schemes by type, ignoring
    layers.

    Does not test performance, peak memory usage, etc.
    """

    # TODO: Relax this condition once there is a native MXFP4_MXFP4
    # linear/moe backend supported on cuda.
    if quant_scheme == "mxfp4" and not (on_gfx950() or on_gfx942()):
        pytest.skip("mxfp4 online quantization is only tested on AMD gfx942, gfx950.")

    if current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_rocm_aiter else "0")
        rocm_aiter_ops.refresh_env_variables()

    if current_platform.is_xpu() and quant_scheme in (
        "fp8_per_block",
        "fp8_per_channel",
    ):
        pytest.skip(f"Skip test for online {quant_scheme} on XPU platform.")

    if model_name == PARTIALLY_PREQUANTIZED_MODEL_NAME:
        model, vllm_config = load_model_without_vllm_runner(
            model_name,
            model_config_kwargs={
                "quantization_config": resolve_quantization_config(
                    None, online_quant_args
                )
            },
            model_loader_cls=DummyModelLoader,
        )
        assert isinstance(
            model.model.layers[1].self_attn.o_proj.quant_method,
            expected_linear_cls,
        )
        assert isinstance(
            model.model.layers[0].mlp.experts._quant_method,
            expected_moe_cls,
        )
        assert isinstance(
            model.model.layers[0].self_attn.o_proj.quant_method,
            CompressedTensorsLinearMethod,
        )
    else:
        model, vllm_config = load_model_without_vllm_runner(
            model_name,
            dtype="bfloat16",
            quantization=quant_scheme,
            model_config_kwargs={
                "quantization_config": resolve_quantization_config(
                    quant_scheme, online_quant_args
                ),
                "hf_overrides": {
                    "num_hidden_layers": 3,
                    "vocab_size": 256,
                    "hidden_size": 256,
                    "intermediate_size": 512,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "max_position_embeddings": 64,
                    "num_local_experts": 4,
                    "num_experts_per_tok": 2,
                },
            },
            model_loader_cls=DummyModelLoader,
        )

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())

    if model_name == PARTIALLY_PREQUANTIZED_MODEL_NAME:
        o_proj = model.model.layers[1].self_attn.o_proj
        moe = model.model.layers[0].mlp.experts
    else:
        o_proj = model.model.layers[0].self_attn.o_proj
        moe = model.model.layers[0].block_sparse_moe.experts
    assert isinstance(o_proj.quant_method, expected_linear_cls)
    assert isinstance(moe._quant_method, expected_moe_cls)

    if model_name == PARTIALLY_PREQUANTIZED_MODEL_NAME and isinstance(
        o_proj.quant_method.kernel, MarlinMxfp8LinearKernel
    ):
        assert o_proj.weight.dtype == torch.int32
    elif model_name == PARTIALLY_PREQUANTIZED_MODEL_NAME and isinstance(
        o_proj.quant_method.kernel, EmulationMxfp8LinearKernel
    ):
        assert o_proj.weight.dtype == torch.bfloat16
    elif model_name == PARTIALLY_PREQUANTIZED_MODEL_NAME:
        assert o_proj.weight.dtype == MXFP8_VALUE_DTYPE
    elif quant_scheme == "mxfp4":
        assert o_proj.weight.dtype == torch.uint8
    elif current_platform.is_cuda() or current_platform.is_xpu():
        assert o_proj.weight.dtype == torch.float8_e4m3fn
    elif current_platform.is_rocm():
        assert o_proj.weight.dtype == current_platform.fp8_dtype()
    else:
        pytest.skip("Only runs on CUDA and ROCm.")

    if quant_scheme == "fp8_per_channel":
        assert o_proj.weight_scale.ndim == 2
        assert o_proj.weight_scale.shape[-1] == 1
        assert o_proj.input_scale is None

    if isinstance(online_quant_args, dict) and "ignore" in online_quant_args:
        for layer_idx in range(len(model.model.layers)):
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            if layer_idx == 1:
                assert isinstance(o_proj.quant_method, UnquantizedLinearMethod)
            else:
                assert isinstance(o_proj.quant_method, expected_linear_cls)

        for layer in model.model.layers:
            assert isinstance(
                layer.self_attn.qkv_proj.quant_method, UnquantizedLinearMethod
            )

    if (
        model_name == GRANITE_MODEL_NAME
        and isinstance(online_quant_args, dict)
        and "targets" in online_quant_args
    ):
        for layer in model.model.layers:
            assert isinstance(
                layer.self_attn.qkv_proj.quant_method, UnquantizedLinearMethod
            )

    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE)
    positions = torch.arange(input_ids.numel(), device=DEVICE)
    with set_forward_context(None, vllm_config, num_tokens=input_ids.numel()):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "model_name", [GRANITE_MODEL_NAME, PARTIALLY_PREQUANTIZED_MODEL_NAME]
)
def test_online_quantization_loads_real_weights(
    model_name, vllm_runner, monkeypatch
) -> None:
    """Verify online quantization loads real checkpoints end to end."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    original_load_weights = GraniteMoeModel.load_weights

    def load_weights(self, weights):
        weights = (
            (name, weight)
            for name, weight in weights
            if not name.startswith("layers.") or int(name.split(".")[1]) < 3
        )
        return original_load_weights(self, weights)

    monkeypatch.setattr(GraniteMoeModel, "load_weights", load_weights)

    if model_name == GRANITE_MODEL_NAME:
        runner_kwargs = {
            "quantization": "fp8_per_tensor",
            "dtype": "bfloat16",
            "enforce_eager": True,
            "hf_overrides": {"num_hidden_layers": 3},
            "max_model_len": 16,
            "max_num_seqs": 1,
        }
    else:
        runner_kwargs = {
            "quantization_config": {
                "targets": {"model.layers.1.self_attn.o_proj": "mxfp8"}
            },
            "enforce_eager": True,
        }

    with vllm_runner(model_name, **runner_kwargs) as llm:
        if model_name == GRANITE_MODEL_NAME:

            def check_model(model):
                layer = model.model.layers[0]
                assert isinstance(
                    layer.self_attn.o_proj.quant_method,
                    Fp8PerTensorOnlineLinearMethod,
                )
                assert isinstance(
                    layer.block_sparse_moe.experts._quant_method,
                    Fp8PerTensorOnlineMoEMethod,
                )

        else:

            def check_model(model):
                assert isinstance(
                    model.model.layers[1].self_attn.o_proj.quant_method,
                    Mxfp8OnlineLinearMethod,
                )
                assert isinstance(
                    model.model.layers[0].self_attn.o_proj.quant_method,
                    CompressedTensorsLinearMethod,
                )
                assert isinstance(
                    model.model.layers[0].mlp.experts._quant_method,
                    CompressedTensorsMoEMethod,
                )

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=1)
        assert outputs


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "targets,prefix,expected_method_cls,unmatched_prefix,expected_metadata",
    [
        (
            {r"re:.*self_attn\.o_proj": "fp8_per_block"},
            "model.layers.0.self_attn.o_proj",
            Fp8PerBlockOnlineLinearMethod,
            "model.layers.0.self_attn.qkv_proj",
            ("targets", "fp8_per_block", r"re:.*self_attn\.o_proj"),
        ),
        (
            {r"re:.*qkv_proj.*": "fp8_per_tensor"},
            "model.layers.0.self_attn.qkv_proj",
            Fp8PerTensorOnlineLinearMethod,
            "model.layers.0.self_attn.o_proj",
            ("targets", "fp8_per_tensor", r"re:.*qkv_proj.*"),
        ),
        (
            {r"re:.*[qkv]_proj": "fp8_per_tensor"},
            "model.layers.0.self_attn.qkv_proj",
            Fp8PerTensorOnlineLinearMethod,
            "model.layers.0.self_attn.o_proj",
            ("targets", "fp8_per_tensor", r"re:.*[qkv]_proj"),
        ),
    ],
    ids=["linear_regex", "direct_fused_regex", "legacy_fused_regex"],
)
def test_online_quantization_targets(
    default_vllm_config,
    dist_init,
    targets: dict[str, str],
    prefix: str,
    expected_method_cls,
    unmatched_prefix: str,
    expected_metadata: tuple[str, str, str],
) -> None:
    """Target patterns select the real online linear methods."""
    default_vllm_config.model_config = ModelConfig()
    config = OnlineQuantizationConfig(QuantizationConfigArgs(targets=targets))
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    layer = ColumnParallelLinear(
        input_size=1,
        output_size=1,
        bias=False,
        disable_tp=True,
    )

    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, expected_method_cls)
    assert config.quantized_layers == {prefix: expected_metadata}

    unmatched_method = config.get_quant_method(layer, unmatched_prefix)
    assert isinstance(unmatched_method, UnquantizedLinearMethod)
    assert config.quantized_layers == {prefix: expected_metadata}


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quantization_records_global_config(
    default_vllm_config, dist_init
) -> None:
    default_vllm_config.model_config = ModelConfig()
    config = OnlineQuantizationConfig(QuantizationConfigArgs(linear="fp8_per_block"))
    prefix = "model.layers.0.self_attn.o_proj"
    layer = ColumnParallelLinear(
        input_size=1,
        output_size=1,
        bias=False,
        disable_tp=True,
    )

    method = config.get_quant_method(layer, prefix)

    assert isinstance(method, Fp8PerBlockOnlineLinearMethod)
    assert config.quantized_layers == {
        prefix: ("linear", str(config.args.linear), None)
    }


def test_online_quantization_targets_ignore_collision() -> None:
    """A targets/ignore collision is reported when the layer is dispatched."""
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={"model.layers.0.self_attn.o_proj": "fp8_per_tensor"},
            ignore=["model.layers.0.self_attn.o_proj"],
        )
    )
    with pytest.raises(ValueError, match="matches both quantization_config.ignore"):
        config.resolve_quant_method_cls(
            Mock(spec=LinearBase), "model.layers.0.self_attn.o_proj"
        )


def test_online_quantization_targets_reject_unsupported_layer() -> None:
    """A targets match on a non-linear, non-MoE layer is rejected."""
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"lm_head": "fp8_per_tensor"})
    )
    lm_head = VocabParallelEmbedding(
        num_embeddings=1,
        embedding_dim=1,
        disable_tp=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Layer lm_head was matched by quantization_config.targets "
            r"\(lm_head\), but online quantization is not supported for "
            "VocabParallelEmbedding."
        ),
    ):
        config.get_quant_method(lm_head, "lm_head")


def test_log_online_quantization(default_vllm_config, monkeypatch) -> None:
    config = OnlineQuantizationConfig(QuantizationConfigArgs(linear="fp8_per_tensor"))
    config.quantized_layers = {
        "model.layers.0.mlp.down_proj": ("linear", "fp8_per_tensor", None),
        "model.layers.1.mlp.down_proj": ("linear", "fp8_per_tensor", None),
        "model.layers.0.self_attn.qkv_proj": (
            "targets",
            "mxfp4",
            r"re:.*qkv_proj.*",
        ),
    }
    default_vllm_config.quant_config = config

    logged_messages: list[str] = []

    def record_info(message: str, *args: object) -> None:
        logged_messages.append(message % args)

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info", record_info
    )
    log_online_quantization(default_vllm_config)

    assert logged_messages == [
        "Quantized 3 layers of types: mlp.down_proj: 2 (from linear: "
        "fp8_per_tensor); self_attn.qkv_proj: 1 (from targets: "
        "re:.*qkv_proj.*, mxfp4)"
    ]


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer_trtllm_fused_moe()
    ),
    reason="nvfp4_per_token needs a Blackwell (SM100) GPU + FlashInfer TRTLLM MoE.",
)
def test_online_nvfp4_per_token_moe(vllm_runner, monkeypatch) -> None:
    """Online NVFP4 quantizes the MoE and leaves dense layers unquantized."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="nvfp4_per_token",
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            assert isinstance(
                layer.block_sparse_moe.experts._quant_method, Nvfp4OnlineMoEMethod
            )
            assert isinstance(
                layer.self_attn.o_proj.quant_method, UnquantizedLinearMethod
            )

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


def _patch_max_reduce(monkeypatch, full_amax) -> None:
    """Stand in for the TP/EP MAX all-reduce, returning the unsharded amax."""
    expected = cast(ProcessGroup, object())
    stub = SimpleNamespace(device_group=expected)
    monkeypatch.setattr(quant_utils, "get_tp_group", lambda: stub)
    monkeypatch.setattr(quant_utils, "get_ep_group", lambda: stub)

    def fake_all_reduce(tensor, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is expected
        tensor.copy_(full_amax)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)


def test_is_tp_sharded_false_when_scale_is_already_global() -> None:
    """Replicated and column-parallel-with-channel-scales need no collective."""
    replicated = SimpleNamespace(
        tp_size=4,
        input_size=64,
        output_size=32,
        input_size_per_partition=64,
        output_size_per_partition=32,
    )
    assert not _is_tp_sharded(replicated)

    column = SimpleNamespace(
        tp_size=4,
        input_size=64,
        output_size=32,
        input_size_per_partition=64,
        output_size_per_partition=8,
    )
    assert not _is_tp_sharded(column, reduces_output_dim=False)
    assert _is_tp_sharded(column)


def _quantize_linear(weight, scheme, is_sharded):
    if scheme == "per_tensor":
        amax = weight_amax(weight).reshape(1)
        scale = _fp8_scale(amax_for_tp_weight_quant(amax, is_sharded))
        return ops.scaled_fp8_quant(weight, scale=scale)[0], scale
    amax = weight_amax(weight, dim=-1, keepdim=True)
    scale = _fp8_channel_scale(amax_for_tp_weight_quant(amax, is_sharded))
    return _fp8_quant_per_channel(weight, scale), scale


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("scheme", ["per_tensor", "per_channel"])
@pytest.mark.parametrize("shard_dim", [0, 1])
def test_online_linear_tp_weight_quant_matches_unsharded(
    monkeypatch, scheme: str, shard_dim: int
) -> None:
    """TP shards pack the same FP8 values and scales as the unsharded weight."""
    torch.manual_seed(0)
    weight = torch.randn(32, 64, device=DEVICE, dtype=torch.bfloat16)
    weight[-1, -1] = 64.0

    full_weight, full_scale = _quantize_linear(weight, scheme, False)

    # Per-channel scales reduce only the input dim, so a column (dim 0) shard
    # already matches without a collective.
    is_sharded = scheme == "per_tensor" or shard_dim == 1
    if is_sharded:
        full_amax = (
            weight_amax(weight).reshape(1)
            if scheme == "per_tensor"
            else weight_amax(weight, dim=-1, keepdim=True)
        )
        _patch_max_reduce(monkeypatch, full_amax)

    shard_size = weight.shape[shard_dim] // 2
    shard = weight.narrow(shard_dim, 0, shard_size).contiguous()
    tp_weight, tp_scale = _quantize_linear(shard, scheme, is_sharded)

    assert torch.equal(tp_weight, full_weight.narrow(shard_dim, 0, shard_size))
    if scheme == "per_channel" and shard_dim == 0:
        assert torch.equal(tp_scale, full_scale.narrow(0, 0, shard_size))
    else:
        assert torch.equal(tp_scale, full_scale)


def _quantize_moe(weight, scheme, moe_tp_size):
    if scheme == "nvfp4":
        return _quantize_moe_weight_to_nvfp4(weight, moe_tp_size)
    if scheme == "per_tensor":
        amax = weight_amax(weight.flatten(1), dim=-1)
        scale = _fp8_scale(amax_for_moe_weight_quant(amax, moe_tp_size))
        quant = lambda w, s: ops.scaled_fp8_quant(w, scale=s)[0]  # noqa: E731
    else:
        amax = weight_amax(weight, dim=-1, keepdim=True)
        scale = _fp8_channel_scale(amax_for_moe_weight_quant(amax, moe_tp_size))
        quant = _fp8_quant_per_channel
    qweight = torch.stack([quant(w, s) for w, s in zip(weight, scale)])
    return qweight, scale


@pytest.mark.parametrize("scheme", ["per_tensor", "per_channel", "nvfp4"])
def test_online_moe_tp_weight_quant_matches_ep(monkeypatch, scheme: str) -> None:
    """TP shards of w2 pack the same values and scales as full experts."""
    if scheme == "nvfp4":
        if (
            not (
                current_platform.is_cuda()
                and current_platform.is_device_capability_family(100)
            )
            or current_platform.is_xpu()
        ):
            pytest.skip("NVFP4 weight quantization needs a Blackwell (SM100) GPU.")
    elif not is_quant_method_supported("fp8"):
        pytest.skip("FP8 is not supported on this GPU type.")

    torch.manual_seed(0)
    weight = torch.randn(2, 32, 32, device=DEVICE, dtype=torch.bfloat16)
    weight[:, -1, -1] = torch.tensor([32.0, 64.0], device=DEVICE)

    ep_out = _quantize_moe(weight, scheme, 1)

    full_amax = (
        weight_amax(weight, dim=-1, keepdim=True)
        if scheme == "per_channel"
        else weight_amax(weight.flatten(1), dim=-1).to(torch.float32)
    )
    _patch_max_reduce(monkeypatch, full_amax)

    # w2 is sharded along its last (intermediate) dim.
    shard_size = weight.shape[2] // 2
    tp_out = _quantize_moe(weight[:, :, :shard_size], scheme, 2)

    packing = 2 if scheme == "nvfp4" else 1
    assert torch.equal(tp_out[0], ep_out[0][:, :, : shard_size // packing])
    if scheme == "nvfp4":
        assert torch.equal(tp_out[1], ep_out[1][:, :, : shard_size // 16])
    assert torch.equal(tp_out[-1], ep_out[-1])


def test_online_int8_moe_w2_scale_matches_unsharded(monkeypatch) -> None:
    """Int8 MoE w2 reduces over the sharded intermediate dim."""
    torch.manual_seed(0)
    w13 = torch.randn(2, 16, 8, dtype=torch.bfloat16)
    w2 = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    w2[:, -1, -1] = 64.0

    def quantize(w2_in, moe_tp_size):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_in, requires_grad=False)
        layer.num_experts = layer.local_num_experts = w13.shape[0]
        method = SimpleNamespace(moe=SimpleNamespace(tp_size=moe_tp_size))
        Int8OnlineMoEMethod._quantize_weights(method, layer)
        return layer.w2_weight, layer.w2_scale

    full_weight, full_scale = quantize(w2, 1)

    _patch_max_reduce(monkeypatch, weight_amax(w2, dim=-1))

    shard_size = w2.shape[2] // 2
    tp_weight, tp_scale = quantize(w2[:, :, :shard_size].contiguous(), 2)

    assert torch.equal(tp_weight, full_weight[:, :, :shard_size])
    assert torch.equal(tp_scale, full_scale)


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="NVFP4 weight quantization needs a Blackwell (SM100) GPU.",
)
def test_online_nvfp4_quantizes_original_expert_weights() -> None:
    torch.manual_seed(0)
    weight = torch.randn(2, 32, 32, device="cuda", dtype=torch.bfloat16)

    quantized, block_scale, global_decode_scale = _quantize_moe_weight_to_nvfp4(weight)
    global_encode_scale = 1.0 / global_decode_scale
    expected = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_encode_scale,
            strict=True,
        )
    ]

    assert torch.equal(
        quantized,
        torch.stack([expert_weight for expert_weight, _ in expected]),
    )
    assert torch.equal(
        block_scale,
        torch.stack([expert_scale for _, expert_scale in expected]),
    )


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_peak_mem(
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch,
) -> None:
    _test_online_quant_peak_mem_impl(
        "fp8_per_tensor", vllm_runner, caplog_mp_spawn, monkeypatch
    )


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_load_format_dummy(
    vllm_runner,
    monkeypatch,
    caplog,
) -> None:
    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="fp8_per_tensor",
        enforce_eager=True,
        load_format="dummy",
    ) as llm:
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])
