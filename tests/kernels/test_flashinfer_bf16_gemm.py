# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    FLASHINFER_BF16_GEMM_BACKENDS,
    FLASHINFER_BF16_GEMM_BACKENDS_WITHOUT_BIAS,
    flashinfer_bf16_mm,
    get_flashinfer_bf16_supported_backends,
    has_flashinfer_bf16_gemm,
)
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cuda() or not has_flashinfer_bf16_gemm():
    pytest.skip(
        reason="FlashInfer BF16 GEMM requires CUDA with FlashInfer installed.",
        allow_module_level=True,
    )

SUPPORTED_BACKENDS = get_flashinfer_bf16_supported_backends()
if not SUPPORTED_BACKENDS:
    pytest.skip(
        reason="FlashInfer BF16 GEMM has no supported backend on this device.",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16]
SHAPES = [
    (48, 96, 128),
    (128, 128, 64),
    (128, 256, 128),
    (150, 128, 96),
]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", FLASHINFER_BF16_GEMM_BACKENDS)
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_bf16_gemm_matches_linear(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    backend: str,
    autotune: bool,
) -> None:
    if backend not in SUPPORTED_BACKENDS:
        pytest.skip(f"FlashInfer BF16 GEMM backend {backend!r} is not supported.")
    if use_bias and backend in FLASHINFER_BF16_GEMM_BACKENDS_WITHOUT_BIAS:
        pytest.skip(f"FlashInfer BF16 GEMM backend {backend!r} does not support bias.")

    set_random_seed(seed)
    m, n, k = shape
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)

    bias = torch.randn((n,), dtype=dtype, device=device) if use_bias else None

    expected = torch.nn.functional.linear(x, weight, bias)

    import flashinfer

    try:
        with flashinfer.autotune(autotune):
            out = flashinfer_bf16_mm(
                x,
                weight.t(),
                bias=bias,
                backend=backend,
            )
    except RuntimeError as exc:
        if "Unsupported gpu architecture 'compute_100f'" in str(exc):
            pytest.skip(
                "FlashInfer TGV BF16 GEMM requires nvcc support for compute_100f."
            )
        raise

    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
