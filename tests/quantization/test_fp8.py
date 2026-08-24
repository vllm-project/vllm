# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type

MODELS = [
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    # The checkpoint below was removed from the HF.
    # TODO: add a small replacement checkpoint.
    pytest.param(
        "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
        marks=pytest.mark.skip(reason="Checkpoint removed from HF."),
    ),
]


def test_static_fp8_moe_input_scales_remain_scalar() -> None:
    a1_scale, a2_scale = process_fp8_input_tensor_strategy_moe(
        torch.tensor([0.25, 0.5]),
        torch.tensor([0.75, 0.6]),
        enable_eplb=False,
    )

    assert a1_scale.ndim == a2_scale.ndim == 0


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    "force_marlin", [True, False] if current_platform.is_cuda() else [False]
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_model_load_and_run(
    vllm_runner, model_id: str, force_marlin: bool, use_rocm_aiter: bool, monkeypatch
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    kwargs = {}
    if force_marlin:
        kwargs["linear_backend"] = "marlin"
        kwargs["moe_backend"] = "marlin"

    with vllm_runner(model_id, enforce_eager=True, **kwargs) as llm:
        # note: this does not test accuracy, just that we can run through
        # see lm-eval tests for accuracy
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:
    def quantize_ref(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(current_platform.fp8_dtype())
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(current_platform.fp8_dtype())
        return qweight

    def per_tensor_dequantize(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 4 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 4.
    x = (torch.randn(size=(11, 11), device=DEVICE_TYPE) * 13).to(dtype)

    # Dynamic quantization
    ref_y, inv_scale = ops.scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)

    # Reference dynamic quantization
    y = quantize_ref(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Static quantization
    y, _ = ops.scaled_fp8_quant(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Padding
    y, _ = ops.scaled_fp8_quant(x, inv_scale, num_token_padding=17)
    assert y.shape[0] == 17
    torch.testing.assert_close(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale, dtype),
    )

    # non-contiguous input with padding
    m, n, padded_stride = 975, 512, 576
    padded_tensor = (torch.randn(size=(m, padded_stride), device=DEVICE_TYPE) * 13).to(
        dtype
    )
    x_nc = padded_tensor[:, :n]  # shape (m, n) with stride (padded_stride, 1)

    assert not x_nc.is_contiguous()
    assert x_nc.stride(0) == padded_stride

    # dynamic quantization
    ref_y_nc, inv_scale_nc = ops.scaled_fp8_quant(x_nc, None)
    ref_y_nc = per_tensor_dequantize(ref_y_nc, inv_scale_nc, dtype)

    # reference dynamic quantization
    y_nc = quantize_ref(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # static quantization
    y_nc, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # padding after non-contiguous input quantization
    y_nc_pad, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc, num_token_padding=m + 10)
    assert y_nc_pad.shape[0] == m + 10
    torch.testing.assert_close(
        ref_y_nc,
        per_tensor_dequantize(
            torch.narrow(y_nc_pad, 0, 0, x_nc.shape[0]), inv_scale_nc, dtype
        ),
    )


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="FP8 e4m3fn weight reloading is not supported on e4m3fnuz platforms",
)
@pytest.mark.parametrize("method_cls", [Fp8LinearMethod, Fp8MoEMethod])
# FP8 weight reloading does not support online quantization
@pytest.mark.parametrize("weight_block_size", [None, [128, 128]])
# any postprocessing that is applied to the weights such as padding and repacking
# (excluding device sharding) must also be applied to the reloaded weights
#
# this is the case for marlin as well as per-tensor Fp8MoEMethod
@pytest.mark.parametrize("use_marlin", [False])  # skip True
def test_fp8_reloading(
    default_vllm_config,
    method_cls,
    weight_block_size,
    use_marlin,
    dist_init,
    monkeypatch,
):
    # NOTE(rob): this test fails when using DeepGEMM because the
    # shapes are invalid. Previously the test was passing because
    # we set fp8_backend to None, which sidestepped the issue.
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")

    if method_cls is Fp8MoEMethod and weight_block_size is None:
        pytest.skip(
            "FP8 Tensor weight reloading does not support fusing w13_weight_scale. "
            "If this is your use case, consider using a restore function like #26327"
        )

    # Set model config as model_config.dtype is required in Fp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = "triton"
    layer_size = 128 if weight_block_size is not None else 1
    with torch.device(f"{DEVICE_TYPE}:0"):
        config = Fp8Config(
            weight_block_size=weight_block_size,
        )

        if method_cls is Fp8LinearMethod:
            layer = torch.nn.Linear(layer_size, layer_size)
            method = method_cls(config)
            method.create_weights(
                layer=layer,
                input_size_per_partition=layer_size,
                output_partition_sizes=[layer_size],
                input_size=layer_size,
                output_size=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
            method.use_marlin = use_marlin

        else:
            layer = FusedMoEFactory(
                num_experts=1,
                top_k=1,
                hidden_size=layer_size,
                intermediate_size=layer_size,
            )
            layer = layer.routed_experts
            method = method_cls(config, layer)
            method.create_weights(
                layer=layer,
                num_experts=1,
                hidden_size=layer_size,
                intermediate_size_per_partition=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )

    # capture weights format during loading
    original_metadata = [
        (name, param.shape, getattr(param, "weight_loader", default_weight_loader))
        for name, param in layer.named_parameters()
    ]

    # test loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)

    # test reloading works after loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)


def test_kv_cache_scale_sync_to_host_copies():
    """Test device-to-host sync of the k/v quantization scales, for both the
    checkpoint-load and runtime-calc paths that produce them.
    """
    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.kv_cache_dtype = "fp8"

    method = BaseKVCacheMethod(quant_config=None)
    method.create_weights(layer)
    # 0.3 stays != 1.0 even after the fp8_fnuz x2 rescale.
    checkpoint_scale = torch.tensor(0.3, dtype=torch.float32)
    layer.k_scale.weight_loader(layer.k_scale, checkpoint_scale)
    layer.v_scale.weight_loader(layer.v_scale, checkpoint_scale)
    method.process_weights_after_loading(layer)

    assert layer._k_scale_float != 1.0
    assert layer._v_scale_float != 1.0
    # Host copy must mirror both the float and the device scale tensor.
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale_float)
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale_float)
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale.item())
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale.item())


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_kv_cache_dtype_skip_layers(vllm_runner, monkeypatch):
    """Test that kv_cache_dtype_skip_layers skips quantization for specified layers."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        "facebook/opt-125m",
        kv_cache_dtype="fp8",
        kv_cache_dtype_skip_layers=["0", "2"],
        enforce_eager=True,
    ) as llm:

        def check_layers(model):
            for i, layer in enumerate(model.model.decoder.layers):
                expected = "auto" if str(i) in ["0", "2"] else "fp8"
                assert layer.self_attn.attn.kv_cache_dtype == expected

        llm.apply_model(check_layers)
