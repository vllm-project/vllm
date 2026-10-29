# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# Odd token counts, non-power-of-2 and large hidden, plus an empty batch.
SHAPES = [(1, 512), (3, 1792), (17, 4096), (128, 8192), (1023, 2048), (0, 1024)]
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_sigmoid_gate_mul Triton kernel requires a CUDA/ROCm device",
)


_TOLS = {
    torch.float32: (1e-6, 1e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_correctness(shape: tuple[int, int], dtype: torch.dtype):
    """The fused kernel must match ``x * torch.sigmoid(gate)`` and must not
    write in place."""
    set_random_seed(13)
    num_tokens, hidden = shape
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    # Scale the gate to exercise both tails of the sigmoid.
    gate = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda") * 4

    result = fused_sigmoid_gate_mul(x, gate)
    reference = x * torch.sigmoid(gate)

    if result.numel() > 0:
        assert result.data_ptr() != x.data_ptr()
    atol, rtol = _TOLS[dtype]
    torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)


@requires_cuda
@torch.inference_mode()
def test_fused_sigmoid_gate_mul_offset_row_view():
    """A row-offset (contiguous but misaligned-pointer) view must compile and
    match eager -- this exercises the unaligned pointer specialization."""
    set_random_seed(13)
    base_x = torch.randn(65, 1023, dtype=torch.bfloat16, device="cuda")
    base_g = torch.randn(65, 1023, dtype=torch.bfloat16, device="cuda")
    x, gate = base_x[1:], base_g[1:]

    result = fused_sigmoid_gate_mul(x, gate)
    reference = x * torch.sigmoid(gate)

    torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_fused_sigmoid_gate_mul_fallback_noncontiguous():
    """Non-CUDA or non-contiguous inputs take the eager fallback (CPU-safe)."""
    x = torch.randn(64, 256, dtype=torch.float32).t()
    gate = torch.randn(64, 256, dtype=torch.float32).t()
    assert not x.is_contiguous()

    result = fused_sigmoid_gate_mul(x, gate)
    expected = x * torch.sigmoid(gate)

    torch.testing.assert_close(result, expected, atol=0.0, rtol=0.0)
