# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_bf16_mm,
    get_flashinfer_bf16_supported_backends,
    has_flashinfer_bf16_gemm,
)

if not current_platform.is_cuda() or not has_flashinfer_bf16_gemm():
    pytest.skip(
        reason="FlashInfer BF16 GEMM requires CUDA with flashinfer installed.",
        allow_module_level=True,
    )

SUPPORTED_BACKENDS = get_flashinfer_bf16_supported_backends()
if not SUPPORTED_BACKENDS:
    pytest.skip(
        reason="FlashInfer BF16 GEMM has no supported backend on this device.",
        allow_module_level=True,
    )

@torch.inference_mode()
def test_flashinfer_bf16_gemm_matches_linear() -> None:
    import flashinfer

    backend = next(
        backend
        for backend in ("cudnn", "cutlass", "tgv")
        if backend in SUPPORTED_BACKENDS
    )

    x = torch.randn((48, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((96, 128), dtype=torch.bfloat16, device="cuda")
    bias = None if backend == "cutlass" else torch.randn(
        (96,), dtype=torch.bfloat16, device="cuda"
    )

    expected = torch.nn.functional.linear(x, weight, bias)

    try:
        with flashinfer.autotune(True):
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
