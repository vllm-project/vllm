# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16]
# m, n, k
SHAPES = [
    (1, 128, 64),
    (1, 256, 128),
    (1, 128, 4096),
    (1, 4096, 4096),
    (1, 2048, 7168),
    (64, 128, 64),
    (64, 256, 128),
    (128, 128, 64),
    (128, 128, 128),
    (256, 128, 64),
    (128, 256, 128),
]


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
    print(f"Cutlass FP4×FP4 GEMM out{out.shape} expected_out{expected_out.shape}")
    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)

    # Helion FP4×FP4 GEMV otherwise use Cutlass
    helion_out = torch.empty(
        (a_fp4.shape[0], b_fp4.shape[0]), dtype=dtype, device=device
    )
    alpha_helion = float(1.0 / (a_global_scale * b_global_scale))
    torch.ops.vllm.nvfp4_gemv_fp4in(
        helion_out,
        b_fp4,
        a_fp4,
        b_scale_interleaved,
        a_scale_interleaved,
        alpha=alpha,
        alpha_scalar=alpha_helion,
    )

    # Compare to CUTLASS
    torch.testing.assert_close(helion_out, out, atol=1e-1, rtol=1e-1)
    print(f"Helion FP4×FP4 GEMV M={m}, N={n}, K={k}")
