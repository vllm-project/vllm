import pytest
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="Nvfp4 currently only supported on CUDA.",
                allow_module_level=True)

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
SEEDS = [42]
CUDA_DEVICES = ['cuda:0']

logger = init_logger(__name__)

kE2M1ToFloatArray = [
    0.,
    0.5,
    1.,
    1.5,
    2.,
    3.,
    4.,
    6.,
]


def e2M1ToFloat(int4_value):
    signBit = (int4_value & 0x8)
    int4_absValue = int4_value & 0x7
    float_result = kE2M1ToFloatArray[int4_absValue]
    if (signBit):
        float_result = -float_result
    return float_result


def break_fp4_bytes(a, dtype, device='cuda'):
    assert (a.dtype == torch.uint8)
    m, n = a.shape
    a = a.flatten()
    highHalfByte = (a & 0xF0) >> 4  # Get upper 4 bits
    lowHalfByte = a & 0x0F
    fH = torch.tensor([e2M1ToFloat(x) for x in highHalfByte]).to(device)
    fL = torch.tensor([e2M1ToFloat(x) for x in lowHalfByte]).to(device)
    # [0xAB, 0xCD] -> [0xB, 0xA, 0xD, 0xC]
    out = torch.stack((fL, fH), dim=-1).reshape(m, n * 2)
    return out


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor,
                               m,
                               k,
                               block_size=16):
    sf_m, sf_k = a_sf_swizzled.shape
    # 0: 1 ; 1: m//128 ; 2: k // block_size // 4  ; 3: 32  ; 4: 4  ; 5: 4
    tmp = torch.reshape(a_sf_swizzled,
                        (1, m // 128, k // block_size // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m, k // block_size)
    out_m, out_k = out.shape
    assert (out_m == sf_m and out_k == sf_k)
    return out


def dequantize_to_dtype(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale,
    unswizzle: bool,
    dtype,
    device,
    block_size=16,
):
    """
  Original tensor shape: [m, k]
  Quantized tensor shape: [m, k // 2]
  Scaling factor Shape: [m, k // block_size]
  Steps for dequantize:
  
  Scaling Factor (Dtype: e4m3 or int32):
    (1) If the scaling factor is in int32, view to fp8-e4m3
    (2) sf_dtype = Dequantize fp8-e4m3 to fp16(or higher 
                 precision) by casting i.e. tensor.to(dtype)
    (3) Unscale sf_dec_perblock such that 
             sf_unscale_dec_perblock = sf_dtype * global_factor
    
  """
    assert tensor_fp4.dtype == torch.uint8
    #before scaling
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype=dtype, device=device)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    if unswizzle:
        tensor_sf = convert_swizzled_to_linear(tensor_sf,
                                               m,
                                               k,
                                               block_size=block_size)
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
                                     unswizzle=True,
                                     dtype=dtype,
                                     device=device,
                                     block_size=block_size)
    b_in_dtype = dequantize_to_dtype(b_fp4,
                                     b_sf,
                                     b_global_scale,
                                     unswizzle=True,
                                     dtype=dtype,
                                     device=device,
                                     block_size=block_size)
    #high precision matmul should work as a reference
    ref_d = torch.matmul(a_in_dtype, b_in_dtype.t())
    assert (m == ref_d.shape[0] and n == ref_d.shape[1]
            ), f"Expected: [{m}, {n}], observed: {ref_d.shape}"
    return ref_d


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    if logger.getEffectiveLevel() == "INFO":
        ref_f16 = torch.matmul(a_dtype, b_dtype.t().contiguous())
        logger.info("expected fp16 mm result: ", ref_f16)

    a_global_scale = ((448.0 * 6.0) /
                      torch.amax(a_dtype.flatten(), dim=-1)).to(torch.float32)
    b_global_scale = ((448.0 * 6.0) /
                      torch.amax(b_dtype.flatten(), dim=-1)).to(torch.float32)
    alpha = 1 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = ops.quantize_to_fp4(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.quantize_to_fp4(b_dtype, b_global_scale)

    expected_out = get_ref_results(a_fp4, b_fp4, a_scale_interleaved,
                                   b_scale_interleaved, a_global_scale,
                                   b_global_scale, m, n, dtype, block_size,
                                   device)
    out = ops.cutlass_fp4_gemm(a_fp4, b_fp4, a_scale_interleaved,
                               b_scale_interleaved, alpha, dtype)

    torch.testing.assert_close(out,
                               expected_out.to(dtype=dtype),
                               atol=1e-1,
                               rtol=1e-1)
