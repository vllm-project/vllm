# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.scalar_type import scalar_types

__all__ = [
    "break_fp4_bytes",
    "dequantize_to_dtype",
    "ref_nvfp4_quant",
]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()

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
    abs_vals = (combined & 0x07).to(torch.long)
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
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype)


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def ref_nvfp4_quant(x, global_scale, block_size):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // block_size, block_size))
    vec_max = torch.max(torch.abs(x), dim=-1,
                        keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = torch.clamp(scale, max=448, min=-448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    # both outputs are float32
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def run_nvfp4_emulations(x: torch.Tensor, input_global_scale: torch.Tensor,
                         weight: torch.Tensor,
                         weight_scale_swizzled: torch.Tensor,
                         weight_global_scale: torch.Tensor):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(w_fp4, weight_scale_swizzled.data,
                               weight_global_scale, output_dtype, x.device,
                               group_size)

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out
