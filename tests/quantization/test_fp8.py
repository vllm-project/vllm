# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod,
    Fp8MoeBackend,
    Fp8MoEMethod,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

MODELS = [
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    # The checkpoint below was removed from the HF.
    # TODO: add a small replacement checkpoint.
    pytest.param(
        "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
        marks=pytest.mark.skip(reason="Checkpoint removed from HF."),
    ),
]


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("force_marlin", [False, True])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_model_load_and_run(
    vllm_runner, model_id: str, force_marlin: bool, use_rocm_aiter: bool, monkeypatch
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    if force_marlin:
        monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")

    with vllm_runner(model_id, enforce_eager=True) as llm:
        # note: this does not test accuracy, just that we can run through
        # see lm-eval tests for accuracy
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


KV_CACHE_MODELS = [
    # AutoFP8 format using separate .k_scale and .v_scale
    # The original checkpoint below was removed from the Hub. To unblock CI and
    # until a small replacement with split K/V scales is found, skip this case.
    # See PR #27717 for context.
    pytest.param(
        "nm-testing/Qwen2-1.5B-Instruct-FP8-K-V",
        marks=pytest.mark.skip(
            reason=(
                "Checkpoint removed from HF; temporarily disabling this "
                "AutoFP8 split K/V case (PR #27717)."
            )
        ),
    ),
]


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", KV_CACHE_MODELS)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_kv_cache_model_load_and_run(
    vllm_runner, model_id: str, use_rocm_aiter: bool, monkeypatch
):
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(model_id, kv_cache_dtype="fp8", enforce_eager=True) as llm:

        def check_model(model):
            attn = model.model.layers[0].self_attn.attn

            assert isinstance(attn.quant_method, Fp8KVCacheMethod)

            if not current_platform.is_rocm():
                # NOTE: This code path requires validation on Non-CUDA platform
                # NOTE: it is valid for scales to be 1.0 (default value), but
                # we know these checkpoints have scales < 1.0
                assert 0.0 < attn._k_scale < 1.0
                assert 0.0 < attn._v_scale < 1.0
            else:
                # NOTE: This code path is for ROCm platform
                # NOTE: it is valid for scales to be 1.0 (default value), but
                # we know these checkpoints have scales < 1.0
                # However on ROCm platform, the _k_scale and _v_scale will be
                # scaled by a factor of 2 as described in
                # vllm/model_executor/layers/quantization/kv_cache.py
                assert 0.0 < attn._k_scale < (1.0 * 2.0)
                assert 0.0 < attn._v_scale < (1.0 * 2.0)

        llm.apply_model(check_model)

        # note: this does not test accuracy, just that we can run through
        # see lm-eval tests for accuracy
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("force_marlin", [False, True])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_load_fp16_model(
    vllm_runner,
    kv_cache_dtype: str,
    force_marlin: bool,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    if force_marlin:
        monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")

    with vllm_runner(
        "facebook/opt-125m",
        quantization="fp8",
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
    ) as llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, Fp8LinearMethod)
            if kv_cache_dtype == "fp8":
                attn = model.model.decoder.layers[0].self_attn.attn
                assert isinstance(attn.quant_method, Fp8KVCacheMethod)
                assert attn._k_scale == 1.0
                assert attn._v_scale == 1.0

            if current_platform.is_cuda():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == torch.float8_e4m3fn
                else:
                    # For GPUs without hardware support, we pack the fp8 weights
                    # for weight-only quantization using Marlin kernels
                    assert fc1.weight.dtype == torch.int32
            elif current_platform.is_rocm():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == current_platform.fp8_dtype()
                else:  # unsupported ROCm platform
                    pytest.skip(
                        "Skip `test_load_fp16_model`. "
                        "It only runs on ROCm platform with FP8 compute."
                        " e.g. MI300X and above."
                    )
            else:  # unsupported platform
                pytest.skip(
                    "Skip `test_load_fp16_model`. "
                    "It only runs on CUDA and ROCm platform."
                )

        llm.apply_model(check_model)


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:
    def quantize_ref(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(torch.float8_e4m3fn)
        return qweight

    def per_tensor_dequantize(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 4 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 4.
    x = (torch.randn(size=(11, 11), device="cuda") * 13).to(dtype)

    # Dynamic quantization
    ref_y, inv_scale = ops.scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)

    # Reference dynamic quantizaton
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
    padded_tensor = (torch.randn(size=(m, padded_stride), device="cuda") * 13).to(dtype)
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


@pytest.mark.parametrize("method_cls", [Fp8LinearMethod, Fp8MoEMethod])
# FP8 weight reloading does not support online quantization
@pytest.mark.parametrize("is_checkpoint_fp8_serialized", [True])  # skip False
@pytest.mark.parametrize("weight_block_size", [None, [1, 1]])
# any postprocessing that is applied to the weights such as padding and repacking
# (excluding device sharding) must also be applied to the reloaded weights
#
# this is the case for marlin as well as per-tensor Fp8MoEMethod
@pytest.mark.parametrize("use_marlin", [False])  # skip True
def test_fp8_reloading(
    method_cls, is_checkpoint_fp8_serialized, weight_block_size, use_marlin, dist_init
):
    if is_checkpoint_fp8_serialized is False:
        pytest.skip("FP8 weight reloading does not support online quantization")

    if method_cls is Fp8MoEMethod and weight_block_size is None:
        pytest.skip(
            "FP8 Tensor weight reloading does not support fusing w13_weight_scale. "
            "If this is your use case, consider using a restore function like #26327"
        )

    with torch.device("cuda:0"):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            weight_block_size=weight_block_size,
        )

        if method_cls is Fp8LinearMethod:
            layer = torch.nn.Linear(1, 1)
            method = method_cls(config)
            method.create_weights(
                layer=layer,
                input_size_per_partition=1,
                output_partition_sizes=[1],
                input_size=1,
                output_size=1,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )

        else:
            layer = FusedMoE(
                num_experts=1,
                top_k=1,
                hidden_size=1,
                intermediate_size=1,
            )
            method = method_cls(config, layer)
            method.create_weights(
                layer=layer,
                num_experts=1,
                hidden_size=1,
                intermediate_size_per_partition=1,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )

        # Fp8LinearMethod uses use_marlin
        # Fp8MoEMethod uses fp8_backend
        method.use_marlin = use_marlin
        method.fp8_backend = Fp8MoeBackend.MARLIN if use_marlin else None

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
    # assuming that no reshaping occurred
    for name, shape, original_weight_loader in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        assert weight_loader is original_weight_loader
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)
