# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Flashinfer FP8 gemms requires compute capability of 10.0 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
) -> None:
    set_random_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((n, k), dtype=dtype, device=device) / k

    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)

    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=dtype)

    if use_bias:
        bias = torch.randn((n,), dtype=dtype, device=device)
        expected_out = expected_out + bias
    else:
        bias = None

    import flashinfer

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp8_mm(
            a_fp8,
            b_fp8.t(),
            a_scale,
            b_scale,
            dtype,
            bias=bias,
        )

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM12x small-M CUTLASS dispatch",
)
@pytest.mark.parametrize("m_offset", [None, 0, 1])
@pytest.mark.parametrize("use_bias", [True, False])
@torch.inference_mode()
def test_sm12x_fp8_scaled_mm_dispatch(m_offset: int | None, use_bias: bool) -> None:
    """Both branches of the SM12x dispatch (CUTLASS up to the threshold M,
    FlashInfer bmm_fp8 above it) match the reference."""
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        SM12X_FP8_CUTLASS_MAX_M,
    )

    set_random_seed(42)
    m = 1 if m_offset is None else SM12X_FP8_CUTLASS_MAX_M + m_offset
    n, k = 256, 128
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") / k
    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda") if use_bias else None
    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=torch.bfloat16)
    if bias is not None:
        expected_out = expected_out + bias

    out = flashinfer_scaled_fp8_mm(
        a_fp8,
        b_fp8.t(),
        a_scale,
        b_scale,
        torch.bfloat16,
        bias,
        cutlass_max_m=SM12X_FP8_CUTLASS_MAX_M,
    )

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM12x small-M CUTLASS dispatch",
)
@torch.inference_mode()
def test_sm12x_fp8_scaled_mm_dispatch_rejects_batched() -> None:
    """cutlass_max_m is only valid for the unbatched call."""
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        SM12X_FP8_CUTLASS_MAX_M,
    )

    a = torch.zeros((2, 8, 128), device="cuda").to(torch.float8_e4m3fn)
    b = torch.zeros((2, 256, 128), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError):
        torch.ops.vllm.bmm_fp8(
            a,
            b.transpose(1, 2),
            scale,
            scale,
            torch.bfloat16,
            "auto",
            SM12X_FP8_CUTLASS_MAX_M,
        )
