"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""
import pytest
import torch

from vllm._custom_ops import scaled_fp8_quant
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

capability = torch.cuda.get_device_capability()
capability = capability[0] * 10 + capability[1]


@pytest.mark.skipif(
    capability < QUANTIZATION_METHODS["fp8"].get_min_capability(),
    reason="FP8 is not supported on this GPU type.")
def test_load_fp16_model(vllm_runner) -> None:
    with vllm_runner("facebook/opt-125m", quantization="fp8") as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        fc1 = model.model.decoder.layers[0].fc1
        assert isinstance(fc1.quant_method, Fp8LinearMethod)
        assert fc1.weight.dtype == torch.float8_e4m3fn


@pytest.mark.skipif(
    capability < QUANTIZATION_METHODS["fp8"].get_min_capability(),
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
    ref_y, inv_scale = scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)

    # Reference dynamic quantizaton
    y = quantize_ref(x, inv_scale)
    assert torch.allclose(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Static quantization
    y, _ = scaled_fp8_quant(x, inv_scale)
    assert torch.allclose(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Padding
    y, _ = scaled_fp8_quant(x, inv_scale, batch_dim_padding=17)
    assert y.shape[0] == 17
    assert torch.allclose(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale,
                              dtype))
