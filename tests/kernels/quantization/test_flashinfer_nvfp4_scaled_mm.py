# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [
    (128, 128, 64),
    (128, 128, 128),
    (256, 128, 64),
    (128, 256, 128),
    (1, 128, 128),
]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96), (2, 128, 64), (3, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _linearize_8x4_scale(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    block_size = 16
    scale_n = n // block_size
    rounded_m = _round_up(m, 8)
    rounded_n = _round_up(scale_n, 4)
    return (
        scale.view(torch.uint8)
        .reshape(rounded_m // 8, rounded_n // 4, 8, 4)
        .permute(0, 2, 1, 3)
        .reshape(rounded_m, rounded_n)
    )


@pytest.mark.parametrize(
    "shape", [(1, 64), (7, 80), (8, 64), (9, 80), (31, 96), (32, 80)]
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_trtllm_8x4_scale_padding_zeroed(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    device: str,
) -> None:
    m, n = shape
    scale_n = n // 16
    rounded_m = _round_up(m, 8)
    rounded_n = _round_up(scale_n, 4)
    poisoned_scale = torch.full(
        (rounded_m * rounded_n,), 0x7F, dtype=torch.uint8, device=device
    )

    def fake_quant(
        input: torch.Tensor, input_global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty(
            (input.shape[0], input.shape[1] // 2),
            dtype=torch.uint8,
            device=input.device,
        )
        return output, poisoned_scale.clone()

    monkeypatch.setattr(ops, "flashinfer_quant_nvfp4_8x4_sf_layout", fake_quant)

    x = torch.ones((m, n), dtype=torch.bfloat16, device=device)
    global_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    _, output_scale = ops.scaled_fp4_quant(x, global_scale, backend="trtllm")
    linear_scale = _linearize_8x4_scale(output_scale, m, n)

    torch.testing.assert_close(
        linear_scale[:m, :scale_n],
        torch.full((m, scale_n), 0x7F, dtype=torch.uint8, device=device),
    )
    assert torch.count_nonzero(linear_scale[m:, :]) == 0
    assert torch.count_nonzero(linear_scale[:, scale_n:]) == 0


@pytest.mark.parametrize("shape", [(1, 64), (7, 80), (9, 80), (31, 96), (32, 80)])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_trtllm_scaled_fp4_quant_8x4_padding_zeroed(
    shape: tuple[int, int],
    device: str,
) -> None:
    set_random_seed(42)
    m, n = shape
    scale_n = n // 16
    x = torch.randn((m, n), dtype=torch.bfloat16, device=device)
    global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(x.flatten(), dim=-1)
    ).to(torch.float32)

    _, output_scale = ops.scaled_fp4_quant(x, global_scale, backend="trtllm")
    linear_scale = _linearize_8x4_scale(output_scale, m, n)

    assert torch.count_nonzero(linear_scale[m:, :]) == 0
    assert torch.count_nonzero(linear_scale[:, scale_n:]) == 0


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
    is_sf_128x4_layout,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_sf,
        a_global_scale,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_128x4_layout=is_sf_128x4_layout,
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", ["cute-dsl", "cutlass", "cudnn", "trtllm", "b12x"])
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
    backend: str,
    autotune: bool,
) -> None:
    if "trtllm" in backend and dtype == torch.float16:
        pytest.skip("Only torch.bfloat16 is supported for TRTLLM FP4 GEMM operations")
    if backend == "cute-dsl" and not current_platform.is_device_capability_family(100):
        pytest.skip("FlashInfer cutedsl backend is only supported on SM10x")
    if backend == "b12x" and not current_platform.has_device_capability(120):
        pytest.skip("b12x FP4 GEMM requires SM120+ (CC 12.0+)")
    if backend == "b12x" and not has_flashinfer_b12x_gemm():
        pytest.skip("b12x FP4 GEMM backend not available in installed FlashInfer")

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
    # cutlass and b12x use swizzled scales directly; trtllm needs them unswizzled.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(
        a_dtype, a_global_scale, is_sf_swizzled_layout=True, backend=backend
    )
    is_sf_128x4_layout = not (backend == "trtllm" and m <= 32)

    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(
        b_dtype, b_global_scale, is_sf_swizzled_layout=True
    )

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
        is_sf_128x4_layout,
    )

    import flashinfer

    if "trtllm" in backend:
        epilogue_tile_m = 128
        b_fp4 = flashinfer.shuffle_matrix_a(b_fp4.view(torch.uint8), epilogue_tile_m)
        b_scale_interleaved = convert_swizzled_to_linear(
            b_scale_interleaved, n, k, block_size
        )
        b_scale_interleaved = (
            flashinfer.shuffle_matrix_sf_a(
                b_scale_interleaved.view(torch.uint8), epilogue_tile_m
            )
            .reshape(b_scale_interleaved.shape)
            .view(torch.float8_e4m3fn)
        )

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            a_scale_interleaved,
            b_scale_interleaved,
            alpha,
            dtype,
            backend=backend,
        )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)
