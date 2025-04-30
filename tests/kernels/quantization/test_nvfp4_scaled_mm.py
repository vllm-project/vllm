# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if not current_platform.has_device_capability(100):
    pytest.skip(reason="Nvfp4 Requires compute capability of 10 or above.",
                allow_module_level=True)

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ['cuda:0']

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1fn.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

kE2M1ToFloat = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.],
                            dtype=torch.float32)


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(tensor_fp4,
                        tensor_sf,
                        global_scale,
                        dtype,
                        device,
                        block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


def get_ref_results(a_fp4, b_fp4, a_sf, b_sf, a_global_scale, b_global_scale,
                    m, n, dtype, block_size, device):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert (m_k == n_k)
    a_in_dtype = dequantize_to_dtype(a_fp4,
                                     a_sf,
                                     a_global_scale,
                                     dtype=dtype,
                                     device=device,
                                     block_size=block_size)
    b_in_dtype = dequantize_to_dtype(b_fp4,
                                     b_sf,
                                     b_global_scale,
                                     dtype=dtype,
                                     device=device,
                                     block_size=block_size)
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                      torch.amax(a_dtype.flatten(), dim=-1)).to(torch.float32)
    b_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                      torch.amax(b_dtype.flatten(), dim=-1)).to(torch.float32)
    alpha = 1. / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    expected_out = get_ref_results(a_fp4, b_fp4, a_scale_interleaved,
                                   b_scale_interleaved, a_global_scale,
                                   b_global_scale, m, n, dtype, block_size,
                                   device)
    out = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_scale_interleaved,
                                    b_scale_interleaved, alpha, dtype)

    torch.testing.assert_close(out,
                               expected_out.to(dtype=dtype),
                               atol=1e-1,
                               rtol=1e-1)
