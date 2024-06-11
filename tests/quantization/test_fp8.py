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
@pytest.mark.parameterize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:

    def fp8_quant_ref(tensor) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = finfo.max / tensor.abs().max()
        qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
        return qweight.to(torch.float8_e4m3fn), scale

    # Note that we use a shape % 4 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 4.
    x = torch.randn(size=(10, 10), dtype=torch.float16,
                    device="cuda").to(dtype)
    ref_y, ref_scale = fp8_quant_ref(x)
    ref_y = ref_y.to(dtype)  # Convert back for comparison

    # Dynamic quantization
    y, scale = scaled_fp8_quant(x, None)
    assert ref_scale.cpu().item() == scale.cpu().item()
    assert torch.allclose(ref_y, y.to(dtype))

    # Static quantization
    y, _ = scaled_fp8_quant(x, scale)
    assert torch.allclose(ref_y, y.to(dtype))

    # Padding
    y, _ = scaled_fp8_quant(x, scale, batch_dim_padding=17)
    assert y.shape[0] == 17
    assert torch.allclose(ref_y, torch.narrow(y, 0, 0, x.shape[0]).to(dtype))
