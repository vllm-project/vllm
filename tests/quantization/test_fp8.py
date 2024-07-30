"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.fp8 import (Fp8KVCacheMethod,
                                                         Fp8LinearMethod)

MODELS = [
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    "nm-testing/Phi-3-mini-128k-instruct-FP8",
    "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
]


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("model_id", MODELS)
def test_model_load_and_run(vllm_runner, model_id: str):
    with vllm_runner(model_id) as llm:
        # note: this does not test accuracy, just that we can run through
        # see lm-eval tests for accuracy
        outputs = llm.generate_greedy(prompts=["Hello my name is"],
                                      max_tokens=10)
        print(outputs[0][1])


KV_CACHE_MODELS = [
    # Deprecated AutoFP8 format using .kv_scale
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    # AutoFP8 format using separate .k_scale and .v_scale
    "nm-testing/Qwen2-1.5B-Instruct-FP8-K-V",
]


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("model_id", KV_CACHE_MODELS)
def test_kv_cache_model_load_and_run(vllm_runner, model_id: str):
    with vllm_runner(model_id, kv_cache_dtype="fp8") as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        attn = model.model.layers[0].self_attn.attn
        assert isinstance(attn.quant_method, Fp8KVCacheMethod)
        # NOTE: it is valid for scales to be 1.0 (default value), but we know
        # these checkpoints have scales < 1.0
        assert 0.0 < attn._k_scale < 1.0
        assert 0.0 < attn._v_scale < 1.0

        # note: this does not test accuracy, just that we can run through
        # see lm-eval tests for accuracy
        outputs = llm.generate_greedy(prompts=["Hello my name is"],
                                      max_tokens=10)
        print(outputs[0][1])


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_load_fp16_model(vllm_runner, kv_cache_dtype: str) -> None:
    with vllm_runner("facebook/opt-125m",
                     quantization="fp8",
                     kv_cache_dtype=kv_cache_dtype) as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        fc1 = model.model.decoder.layers[0].fc1
        assert isinstance(fc1.quant_method, Fp8LinearMethod)
        if kv_cache_dtype == "fp8":
            attn = model.model.decoder.layers[0].self_attn.attn
            assert isinstance(attn.quant_method, Fp8KVCacheMethod)
            assert attn._k_scale == 1.0
            assert attn._v_scale == 1.0

        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability >= 89:
            # For GPUs with hardware support, we keep weights in fp8
            assert fc1.weight.dtype == torch.float8_e4m3fn
        else:
            # For GPUs without hardware support, we pack the fp8 weights
            # for weight-only quantization using Marlin kernels
            assert fc1.weight.dtype == torch.int32


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:

    def quantize_ref(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min,
                                                           max=finfo.max)
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
    assert torch.allclose(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Static quantization
    y, _ = ops.scaled_fp8_quant(x, inv_scale)
    assert torch.allclose(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Padding
    y, _ = ops.scaled_fp8_quant(x, inv_scale, num_token_padding=17)
    assert y.shape[0] == 17
    assert torch.allclose(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale,
                              dtype))
