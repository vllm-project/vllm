# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
HIDDENS = [2048, 2560, 4096, 5120, 8192]
NUM_TOKENS = [0, 1, 3, 64, 257, 1024]
SEEDS = [13]

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="GemmaRMSNorm Triton kernels require a CUDA/ROCm device",
)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-2, 1e-2


def _make_norm(hidden: int, dtype: torch.dtype = torch.float32) -> GemmaRMSNorm:
    norm = GemmaRMSNorm(hidden, eps=1e-6)
    norm.weight.data = torch.randn(hidden, dtype=dtype, device="cuda") * 0.1
    return norm


@requires_cuda
@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gemma_rms_norm_triton_matches_native(
    default_vllm_config,
    hidden: int,
    num_tokens: int,
    dtype: torch.dtype,
    seed: int,
):
    """forward_cuda (Triton) must match forward_native and stay out-of-place."""
    set_random_seed(seed)
    norm = _make_norm(hidden)
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")

    ref = norm.forward_native(x.clone())
    assert isinstance(ref, torch.Tensor)
    out = norm.forward_cuda(x)
    assert isinstance(out, torch.Tensor)

    if out.numel() > 0:
        assert out.data_ptr() != x.data_ptr()
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gemma_fused_add_rms_norm_triton_matches_native(
    default_vllm_config,
    hidden: int,
    num_tokens: int,
    dtype: torch.dtype,
    seed: int,
):
    """forward_cuda (Triton) must match forward_native, update residual in
    place, and leave x untouched."""
    set_random_seed(seed)
    norm = _make_norm(hidden)
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    x_orig = x.clone()

    ref_out, ref_residual = norm.forward_native(x.clone(), residual.clone())
    out, new_residual = norm.forward_cuda(x, residual)

    assert new_residual.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(x, x_orig, atol=0.0, rtol=0.0)
    atol, rtol = _tolerances(dtype)
    # The residual update is an exact fp32 add + downcast in both paths.
    torch.testing.assert_close(new_residual, ref_residual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_gemma_rms_norm_triton_strided_and_offset_rows(
    default_vllm_config, dtype: torch.dtype
):
    """Non-contiguous rows (stride > hidden) and offset (contiguous but
    misaligned-pointer) views must match native."""
    set_random_seed(13)
    hidden, pad = 2560, 128
    norm = _make_norm(hidden)

    base = torch.randn(33, hidden + pad, dtype=dtype, device="cuda")
    x_strided = base[:, :hidden]
    assert not x_strided.is_contiguous()
    x_offset = torch.randn(65, hidden, dtype=dtype, device="cuda")[1:]
    residual_offset = torch.randn(65, hidden, dtype=dtype, device="cuda")[1:]

    for x in (x_strided, x_offset):
        ref = norm.forward_native(x.clone())
        out = norm.forward_cuda(x)
        atol, rtol = _tolerances(dtype)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    ref_out, ref_residual = norm.forward_native(
        x_offset.clone(), residual_offset.clone()
    )
    out, new_residual = norm.forward_cuda(x_offset, residual_offset)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(new_residual, ref_residual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)


@requires_cuda
@torch.inference_mode()
def test_gemma_rms_norm_triton_low_precision_weight(default_vllm_config):
    """A bf16 weight (module cast) must match native, which always computes
    ``weight.float() + 1.0``."""
    set_random_seed(13)
    hidden, num_tokens = 4096, 64
    norm = _make_norm(hidden, dtype=torch.bfloat16)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")

    ref = norm.forward_native(x.clone())
    out = norm.forward_cuda(x)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
