"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
def test_load_fp16_model(vllm_runner) -> None:
    with vllm_runner("facebook/opt-125m", quantization="fp8") as llm:

        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        fc1 = model.model.decoder.layers[0].fc1
        assert isinstance(fc1.quant_method, Fp8LinearMethod)
        assert fc1.weight.dtype == torch.float8_e4m3fn


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
    y, _ = ops.scaled_fp8_quant(x, inv_scale, batch_dim_padding=17)
    assert y.shape[0] == 17
    assert torch.allclose(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale,
                              dtype))


@pytest.mark.skipif(not is_quant_method_supported("fp8"),
                    reason="FP8 is not supported on this GPU type.")
def test_fp8_pack_unpack_roundtrip() -> None:

    def pack_fp8_to_int32_ref(fp8_tensor: torch.Tensor) -> torch.Tensor:
        assert fp8_tensor.dtype == torch.float8_e4m3fn
        assert fp8_tensor.shape[0] % 4 == 0

        # Reshape to prepare for packing
        reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

        # Convert fp8 to uint8 (byte) representation
        byte_tensor = reshaped.view(torch.uint8)

        # Pack 4 uint8 values into one int32
        packed = (byte_tensor[:, 0].to(torch.int32) |
                (byte_tensor[:, 1].to(torch.int32) << 8) |
                (byte_tensor[:, 2].to(torch.int32) << 16) |
                (byte_tensor[:, 3].to(torch.int32) << 24))

        return packed.view(fp8_tensor.shape[0] // 4, *fp8_tensor.shape[1:])

    def unpack_int32_to_fp8_ref(packed_tensor: torch.Tensor) -> torch.Tensor:
        assert packed_tensor.dtype == torch.int32

        # Extract individual bytes
        b0 = (packed_tensor & 0xFF).to(torch.uint8)
        b1 = ((packed_tensor >> 8) & 0xFF).to(torch.uint8)
        b2 = ((packed_tensor >> 16) & 0xFF).to(torch.uint8)
        b3 = ((packed_tensor >> 24) & 0xFF).to(torch.uint8)

        # Stack the bytes
        stacked = torch.stack([b0, b1, b2, b3], dim=1)

        # Reshape to the original shape
        unpacked = stacked.reshape(packed_tensor.shape[0] * 4, *packed_tensor.shape[1:])

        # Convert back to Float8_e4m3fn
        return unpacked.view(torch.float8_e4m3fn)

    # Create a random tensor and quantize it to fp8
    x = torch.randn(size=(512, 1024), device="cuda")
    fp8_tensor, inv_scale = ops.scaled_fp8_quant(x, None)

    # Pack the fp8 tensor to int32
    packed_tensor = ops.pack_fp8_to_int32(fp8_tensor)
    ref_packed_tensor = pack_fp8_to_int32_ref(fp8_tensor)
    assert torch.all(packed_tensor == ref_packed_tensor)

    # Unpack the int32 tensor back to fp8
    unpacked_tensor = unpack_int32_to_fp8_ref(packed_tensor)
    ref_unpacked_tensor = unpack_int32_to_fp8_ref(ref_packed_tensor)

    # Check that the shapes are correct
    assert fp8_tensor.shape == unpacked_tensor.shape
    assert ref_unpacked_tensor.shape == unpacked_tensor.shape

    # Check that the dtypes are correct
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert unpacked_tensor.dtype == torch.float8_e4m3fn
    assert ref_unpacked_tensor.dtype == torch.float8_e4m3fn

    # Check that the values are identical
    assert torch.all(fp8_tensor == unpacked_tensor)
    assert torch.all(ref_unpacked_tensor == unpacked_tensor)
