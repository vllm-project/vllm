# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused dual-GemmaRMSNorm + concat kernel used by the
Qwen3.5 MTP drafter.

The kernel must match the eager 3-op reference: two Gemma-style RMSNorms
(scale by ``1 + weight``, fp32 accumulation) followed by a last-dim concat.
"""

import pytest
import torch

from vllm.kernels.triton.fused_pre_fc_norms import fused_pre_fc_norms
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

EPS = 1e-6
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
NUM_TOKENS = [1, 129]
HIDDEN_SIZES = [2048, 4096]
# (rtol, atol) per dtype; the kernel accumulates in fp32 like the reference,
# so the gap is output rounding and reduction order only.
TOLS = {
    torch.bfloat16: (1e-2, 1e-3),
    torch.float16: (1e-3, 1e-4),
    torch.float32: (1e-5, 1e-5),
}

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="fused pre-fc norms Triton kernel requires a CUDA device",
)


def _gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * (w.float() + 1.0)).to(dtype)


@requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_pre_fc_norms_correctness(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
):
    """Fused kernel == cat([norm_e(embeds), norm_h(hidden)]) for Gemma-style
    (1 + w) norms with non-zero weights."""
    set_random_seed(0)
    embeds = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    emb_w = torch.randn(hidden_size, dtype=dtype, device="cuda") * 0.2
    hid_w = torch.randn(hidden_size, dtype=dtype, device="cuda") * 0.2

    fused = fused_pre_fc_norms(embeds, hidden, emb_w, hid_w, EPS)
    ref = torch.cat(
        [_gemma_rmsnorm(embeds, emb_w, EPS), _gemma_rmsnorm(hidden, hid_w, EPS)],
        dim=-1,
    )

    assert fused.shape == (num_tokens, 2 * hidden_size)
    rtol, atol = TOLS[dtype]
    torch.testing.assert_close(fused.float(), ref.float(), rtol=rtol, atol=atol)
