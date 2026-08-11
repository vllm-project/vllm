# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
from torch import Tensor

from vllm.utils.import_utils import has_helion

# Skip entire module if helion is not available
if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm import _custom_ops as ops
from vllm.kernels.helion.ops.nvfp4_gemm import (
    FP4_E2M1_LUT,
    _fp4_storage,
    nvfp4_gemm_w4a4,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def unpack_and_dequantize_fp4(packed: Tensor, last_dim: bool = False) -> Tensor:
    """Unpack and dequantize packed FP4 E2M1 values along dim 0 to float32."""
    packed_storage = _fp4_storage(packed)
    lo = (packed_storage & 0xF).to(torch.long)
    hi = ((packed_storage >> 4) & 0xF).to(torch.long)
    lut = FP4_E2M1_LUT.to(device=packed_storage.device)
    lo_f = lut[lo]
    hi_f = lut[hi]
    if last_dim:
        M, K_half = packed_storage.shape
        stacked = torch.stack([lo_f, hi_f], dim=-1)
        return stacked.reshape(M, K_half * 2)
    stacked = torch.stack([lo_f, hi_f], dim=1)
    return stacked.reshape(packed_storage.shape[0] * 2, packed_storage.shape[1])


def swizzled_scale_offsets(row: Tensor, col: Tensor, cols: int) -> Tensor:
    num_col_tiles = _ceil_div(cols, 4)
    tile_offset = ((row // 128) * num_col_tiles + col // 4) * 512
    return tile_offset + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4


def reference_nvfp4_w4a4_matmul(
    A_packed: Tensor,
    B_packed: Tensor,
    act_scale: Tensor,
    weight_scale: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Pure PyTorch mathematical reference for NVFP4-in, NVFP4-weight GEMV."""
    A_dequant = unpack_and_dequantize_fp4(A_packed, last_dim=True)
    B_dequant = unpack_and_dequantize_fp4(B_packed)
    M, K = A_dequant.shape
    _, N = B_dequant.shape
    K_groups = K // 16

    group_idx = torch.arange(K, device=A_packed.device) // 16

    row_idx_a = torch.arange(M, device=A_packed.device)[:, None]
    a_scale_offsets = swizzled_scale_offsets(row_idx_a, group_idx[None, :], K_groups)
    a_scale_vals = act_scale.reshape(-1)[a_scale_offsets].to(torch.float32)
    A_scaled = A_dequant * a_scale_vals

    col_idx_b = torch.arange(N, device=A_packed.device)[None, :]
    b_scale_offsets = swizzled_scale_offsets(col_idx_b, group_idx[:, None], K_groups)
    b_scale_vals = weight_scale.reshape(-1)[b_scale_offsets].to(torch.float32)
    B_scaled = B_dequant * b_scale_vals

    return (torch.matmul(A_scaled, B_scaled) * alpha).to(torch.bfloat16)


if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
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
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)

    # Helion FP4×FP4 GEMV
    alpha_helion = float(1.0 / (a_global_scale * b_global_scale))
    b_helion = b_fp4.view(torch.uint8).t().contiguous()

    helion_out = nvfp4_gemm_w4a4(
        a_fp4,
        b_helion,
        a_scale_interleaved,
        b_scale_interleaved,
        alpha=alpha_helion,
    )

    # Compare to CUTLASS
    torch.testing.assert_close(helion_out, out, atol=1e-1, rtol=1e-1)
    print(f"Helion FP4×FP4 GEMV M={m}, N={n}, K={k}s")
