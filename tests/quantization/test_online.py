# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests online quantization."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.distributed import ProcessGroup

from tests.quantization.utils import (
    _test_online_quant_peak_mem_impl,
    is_quant_method_supported,
)
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm._custom_ops import scaled_fp4_quant
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    _fp8_channel_scale,
    _fp8_quant_per_channel,
    _fp8_scale,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.online.int8 import Int8OnlineMoEMethod
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    amax_for_tp_weight_quant,
    weight_amax,
)
from vllm.model_executor.model_loader import weight_utils
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


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

    checkpoint_config.set_online_quantization(QuantizationConfigArgs(linear="mxfp8"))
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
        with pytest.raises(ValueError, match="checkpoint-quantized layer"):
            ColumnParallelLinear(**layer_kwargs)
    else:
        layer = ColumnParallelLinear(**layer_kwargs)
        assert isinstance(layer.quant_method, Mxfp8OnlineLinearMethod)


def test_online_ignore_keeps_checkpoint_quantization(default_vllm_config, dist_init):
    """Ignoring online quantization does not replace a checkpoint method."""
    default_vllm_config.model_config = ModelConfig()
    quant_config = _fully_quantized_quark_config()
    prefix = "model.layers.0.self_attn.o_proj"
    quant_config.set_online_quantization(
        QuantizationConfigArgs(linear="mxfp8", ignore=[prefix])
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


def test_activation_only_override_keeps_checkpoint_config(monkeypatch) -> None:
    """Activation-only overrides do not attach an online quantization overlay."""
    checkpoint_config = _moe_only_compressed_tensors_config()
    quant_cls = SimpleNamespace(from_config=lambda _: checkpoint_config)
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            quantization="compressed-tensors",
            quantization_config=QuantizationConfigArgs(moe={"activation": "mxfp8"}),
            hf_config=SimpleNamespace(
                quantization_config={"quant_method": "compressed-tensors"},
                text_config=None,
            ),
            hf_overrides={},
        ),
    )
    monkeypatch.setattr(weight_utils, "get_quantization_config", lambda _: quant_cls)

    result = weight_utils.get_quant_config(
        model_config, cast(LoadConfig, SimpleNamespace())
    )

    assert result is checkpoint_config
    assert result.online_quant_config is None


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "model_name,quant_scheme,online_quant_args,expected_linear_cls,expected_moe_cls,linear_layer_idx",
    [
        # simple case - quantization='fp8_per_tensor'
        (
            "ibm-granite/granite-3.0-1b-a400m-base",
            "fp8_per_tensor",
            None,
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
            0,
        ),
        # simple case - quantization='fp8_per_block'
        (
            "ibm-granite/granite-3.0-1b-a400m-base",
            "fp8_per_block",
            None,
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerBlockOnlineMoEMethod,
            0,
        ),
        # quantization='online' with per-layer-kind overrides
        (
            "ibm-granite/granite-3.0-1b-a400m-base",
            "online",
            {
                "linear": "fp8_per_block",
                "moe": "fp8_per_tensor",
            },
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
            0,
        ),
        # ignore with direct layer name
        (
            "ibm-granite/granite-3.0-1b-a400m-base",
            "fp8_per_tensor",
            # qkv_proj is fused from q_proj/k_proj/v_proj, so currently the
            # ignore regex must match the unfused shard names
            # TODO(future PR): also make 're:.*qkv_proj.*' work
            {"ignore": ["model.layers.1.self_attn.o_proj", "re:.*[qkv]_proj"]},
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
            0,
        ),
        (
            "nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable",
            None,
            {
                "linear": "mxfp8",
                "ignore": [
                    # layer 0 self_attn is prequantized
                    "re:model\\.layers\\.0\\.self_attn\\..*",
                    # layer 0 gate is preexcluded
                    "model.layers.0.mlp.gate",
                    # Checkpoint has "targets": ["Linear"]
                    # and shared_experts not excluded
                    "re:model\\.layers\\.\\d+\\.mlp\\.shared_expert\\..*",
                ],
            },
            Mxfp8OnlineLinearMethod,
            CompressedTensorsMoEMethod,
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    vllm_runner,
    model_name: str,
    quant_scheme: str | None,
    online_quant_args: dict | None,
    expected_linear_cls,
    expected_moe_cls,
    linear_layer_idx: int,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    """
    Tests that online quantization frontend configuration works -
    selecting quant schemes, overriding quant schemes by type, ignoring
    layers.

    Does not test performance, peak memory usage, etc.
    """

    if current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_rocm_aiter else "0")
        rocm_aiter_ops.refresh_env_variables()

    if current_platform.is_xpu() and quant_scheme == "fp8_per_block":
        pytest.skip("Skip test for online fp8_per_block on XPU platform.")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    runner_kwargs: dict[str, Any] = dict(
        quantization=quant_scheme,
        enforce_eager=True,
    )
    if online_quant_args is not None:
        runner_kwargs["quantization_config"] = online_quant_args

    with vllm_runner(
        model_name,
        **runner_kwargs,
    ) as llm:

        def check_model(model):
            o_proj = model.model.layers[linear_layer_idx].self_attn.o_proj
            moe = getattr(model.model.layers[0], "block_sparse_moe", None)
            moe = model.model.layers[0].mlp.experts if moe is None else moe.experts

            # o_proj and moe in layer 0 are always quantized (never ignored)
            # because of how we craft the test case inputs
            assert isinstance(o_proj.quant_method, expected_linear_cls)
            if moe is not None:
                assert isinstance(moe._quant_method, expected_moe_cls)

            if model_name == "nm-testing/tinysmokeqwen3moe-W4A16-first-only-CTstable":
                assert isinstance(
                    model.model.layers[1].self_attn.o_proj.quant_method,
                    Mxfp8OnlineLinearMethod,
                )
                layer_0 = model.model.layers[0]
                for ignored_layer in (
                    layer_0.self_attn.qkv_proj,
                    layer_0.self_attn.o_proj,
                    layer_0.mlp.gate,
                    layer_0.mlp.shared_expert.gate_up_proj,
                    layer_0.mlp.shared_expert.down_proj,
                ):
                    assert isinstance(
                        ignored_layer.quant_method,
                        CompressedTensorsLinearMethod,
                    )

            if current_platform.is_cuda() or current_platform.is_xpu():
                assert o_proj.weight.dtype == torch.float8_e4m3fn
            elif current_platform.is_rocm():
                assert o_proj.weight.dtype == current_platform.fp8_dtype()
            else:
                pytest.skip("Only runs on CUDA and ROCm.")

            # Verify ignored layers are unquantized.
            if (
                model_name == "ibm-granite/granite-3.0-1b-a400m-base"
                and isinstance(online_quant_args, dict)
                and "ignore" in online_quant_args
            ):
                # only .*1.self_attn_o_proj is skipped
                for layer_idx in range(len(model.model.layers)):
                    o_proj = model.model.layers[layer_idx].self_attn.o_proj
                    if layer_idx == 1:
                        assert isinstance(o_proj.quant_method, UnquantizedLinearMethod)
                    else:
                        assert isinstance(o_proj.quant_method, expected_linear_cls)

                # every .*self_attn.qkv_proj is skipped
                for layer_idx in range(len(model.model.layers)):
                    qkv_proj = model.model.layers[layer_idx].self_attn.qkv_proj
                    assert isinstance(qkv_proj.quant_method, UnquantizedLinearMethod)

        llm.apply_model(check_model)

        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


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
    weight = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
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
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            pytest.skip("NVFP4 weight quantization needs a Blackwell (SM100) GPU.")
    elif not is_quant_method_supported("fp8"):
        pytest.skip("FP8 is not supported on this GPU type.")

    torch.manual_seed(0)
    weight = torch.randn(2, 32, 32, device="cuda", dtype=torch.bfloat16)
    weight[:, -1, -1] = torch.tensor([32.0, 64.0], device="cuda")

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
