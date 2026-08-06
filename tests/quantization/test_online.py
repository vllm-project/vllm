# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests online quantization."""

from unittest.mock import Mock

import pytest
import torch

from tests.quantization.utils import (
    _test_online_quant_peak_mem_impl,
    is_quant_method_supported,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm._custom_ops import scaled_fp4_quant
from vllm.config import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
    _quantize_moe_weight_to_nvfp4,
)
from vllm.model_executor.model_loader.base_loader import log_online_quantization
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "quant_scheme,online_quant_args,expected_linear_cls,expected_moe_cls",
    [
        # simple case - quantization='fp8_per_tensor'
        (
            "fp8_per_tensor",
            None,
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # simple case - quantization='fp8_per_block'
        (
            "fp8_per_block",
            None,
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerBlockOnlineMoEMethod,
        ),
        # quantization='online' with per-layer-kind overrides
        (
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
            "fp8_per_tensor",
            # qkv_proj is fused from q_proj/k_proj/v_proj. The shard regex
            # remains supported alongside direct fused-name regexes.
            {"ignore": ["model.layers.1.self_attn.o_proj", "re:.*[qkv]_proj"]},
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
    ],
    ids=[
        "fp8_per_tensor",
        "fp8_per_block",
        "per_layer_kind_overrides",
        "targets",
        "ignore",
    ],
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    vllm_runner,
    quant_scheme: str,
    online_quant_args: dict | None,
    expected_linear_cls,
    expected_moe_cls,
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

    # a tiny model with both dense and MoE layers
    model_name = "ibm-granite/granite-3.0-1b-a400m-base"

    runner_kwargs = dict(
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
            # checks further down in the test case are hardcoded for this
            # model
            assert model_name == "ibm-granite/granite-3.0-1b-a400m-base"

            o_proj = model.model.layers[0].self_attn.o_proj
            moe = model.model.layers[0].block_sparse_moe.experts

            # o_proj and moe in layer 0 are always quantized (never ignored)
            # because of how we craft the test case inputs
            assert isinstance(o_proj.quant_method, expected_linear_cls)
            if moe is not None:
                assert isinstance(moe._quant_method, expected_moe_cls)

            if current_platform.is_cuda() or current_platform.is_xpu():
                assert o_proj.weight.dtype == torch.float8_e4m3fn
            elif current_platform.is_rocm():
                assert o_proj.weight.dtype == current_platform.fp8_dtype()
            else:
                pytest.skip("Only runs on CUDA and ROCm.")

            # Verify ignored layers are unquantized.
            if isinstance(online_quant_args, dict) and "ignore" in online_quant_args:
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

            if isinstance(online_quant_args, dict) and "targets" in online_quant_args:
                # qkv_proj matches neither target pattern and must remain in
                # full precision when targets are used instead of global specs.
                for layer_idx in range(len(model.model.layers)):
                    qkv_proj = model.model.layers[layer_idx].self_attn.qkv_proj
                    assert isinstance(qkv_proj.quant_method, UnquantizedLinearMethod)

        llm.apply_model(check_model)

        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


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
        config._dispatch_target(
            "model.layers.0.self_attn.o_proj", Mock(spec=LinearBase)
        )


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
