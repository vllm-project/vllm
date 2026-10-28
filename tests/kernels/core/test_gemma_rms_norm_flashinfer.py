# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3_5RMSNorm (flashinfer Gemma kernels) must match the native path.

The NVIDIA Qwen3.5 implementation routes its norms through flashinfer's
``gemma_rmsnorm``/``gemma_fused_add_rmsnorm`` (the +1 weight offset folded
into the kernel) for fp16/bf16 inputs; fp32, empty, or strided inputs fall
back to ``GemmaRMSNorm.forward_native``. These tests pin both paths' numerics
against ``forward_native``.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="flashinfer Gemma norm kernels require a CUDA device",
)

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
HIDDENS = [256, 2048, 4096]
NUM_TOKENS = [0, 1, 17, 128]
SEEDS = [0, 1]


def _make_norm(hidden: int, dtype: torch.dtype):
    from vllm.models.qwen3_5.nvidia.model import Qwen3_5RMSNorm

    # The NVIDIA norm keeps its weight in the model dtype; checkpoint values
    # are bf16-exact, so seed the (already cast) parameter directly.
    norm = Qwen3_5RMSNorm(hidden, eps=1e-6, dtype=dtype)
    norm.weight.data = torch.randn(hidden, dtype=dtype) * 0.2
    return norm.cuda()


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-5, 1e-5


def _uses_flashinfer(num_tokens: int, dtype: torch.dtype) -> bool:
    from vllm.models.qwen3_5.nvidia import model as nvidia_model

    return (
        nvidia_model.gemma_rmsnorm is not None
        and num_tokens > 0
        and dtype != torch.float32
    )


@requires_cuda
@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gemma_rms_norm_flashinfer_matches_native(
    default_vllm_config, hidden: int, num_tokens: int, dtype: torch.dtype, seed: int
):
    set_random_seed(seed)
    norm = _make_norm(hidden, dtype)
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")

    ref = norm.forward_native(x.clone())
    out = norm.forward_cuda(x.clone())

    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gemma_fused_add_rms_norm_flashinfer_matches_native(
    default_vllm_config, hidden: int, num_tokens: int, dtype: torch.dtype, seed: int
):
    set_random_seed(seed)
    norm = _make_norm(hidden, dtype)
    x = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
    residual = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")

    ref_out, ref_residual = norm.forward_native(x.clone(), residual.clone())
    out, new_residual = norm.forward_cuda(x, residual)

    # The flashinfer kernel mutates x and residual in place; fallbacks
    # (empty input, fp32, or flashinfer unavailable) go through
    # forward_native, which allocates new outputs.
    if _uses_flashinfer(num_tokens, dtype):
        assert new_residual.data_ptr() == residual.data_ptr()
        assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(new_residual, ref_residual, atol=0.0, rtol=0.0)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
