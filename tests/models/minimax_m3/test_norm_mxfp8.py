# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 Triton input norm against the FlashInfer norm (plus the
reference MXFP8 quantize for MXFP8 output): residual and UE8M0 scales bitwise,
E4M3 codes within one quantization step, BF16 output within one BF16 ulp."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

HIDDEN = 6144


def _rows(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Per-32-block magnitudes over 2^-20..2^20, plus special rows."""
    x = torch.randn(n, k, generator=gen, device="cuda")
    e = torch.randint(-20, 21, (n, k // 32, 1), generator=gen, device="cuda")
    x = (x.view(n, -1, 32) * torch.pow(2.0, e.float())).view(n, k)
    if n >= 4:
        x[0] = 0.0
        x[1] = 1e-40
        x[2] = float("inf")
        x[3] = 448.0
    return x.to(torch.bfloat16)


def _code_steps(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-element code distance in value order; NaN codes must match exactly.

    ``dtype`` is the value dtype: float8_e4m3fn (E4M3 codes) or bfloat16."""
    n = 256 if dtype == torch.float8_e4m3fn else 65536
    int_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else torch.uint16
    codes = torch.arange(n, dtype=torch.int32, device=a.device).to(int_dtype)
    vals = codes.view(dtype).float()
    rank = torch.empty(n, dtype=torch.int32, device=a.device)
    rank[vals.argsort()] = torch.arange(n, dtype=torch.int32, device=a.device)
    a_i = a.contiguous().view(int_dtype).long()
    b_i = b.contiguous().view(int_dtype).long()
    a_nan, b_nan = vals.isnan()[a_i], vals.isnan()[b_i]
    assert torch.equal(a_nan, b_nan), "NaN pattern mismatch"
    return torch.where(a_nan, 0, (rank[a_i] - rank[b_i]).abs())


@pytest.mark.parametrize("mxfp8", [True, False])
@pytest.mark.parametrize("fused_add", [True, False])
@pytest.mark.parametrize(
    ("num_tokens", "strided"), [(1, False), (129, True), (1024, False)]
)
def test_input_norm_matches_reference(num_tokens, strided, fused_add, mxfp8):
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.minimax_m3.nvidia.ops.norm_mxfp8 import gemma_rmsnorm_mxfp8

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = _rows(num_tokens, HIDDEN, gen)
    residual = _rows(num_tokens, HIDDEN, gen)
    weight = (torch.randn(HIDDEN, generator=gen, device="cuda") * 0.2).to(x.dtype)
    eps = 1e-6

    x_ref, res_ref = x.clone(), residual.clone()
    if fused_add:
        gemma_fused_add_rmsnorm(x_ref, res_ref, weight, eps)
        y_ref = x_ref
    else:
        y_ref = gemma_rmsnorm(x_ref, weight, eps)

    if strided:
        # Row-contiguous views with row stride 2*H.
        x_buf = torch.empty(num_tokens, 2 * HIDDEN, dtype=x.dtype, device="cuda")
        res_buf = torch.empty_like(x_buf)
        x_buf[:, :HIDDEN], res_buf[:, :HIDDEN] = x, residual
        x_new, res_new = x_buf[:, :HIDDEN], res_buf[:, :HIDDEN]
    else:
        x_new, res_new = x.clone(), residual.clone()
    out, s = gemma_rmsnorm_mxfp8(
        x_new, weight, eps, residual=res_new if fused_add else None, mxfp8=mxfp8
    )

    if fused_add:
        assert torch.equal(res_new.contiguous(), res_ref)
    assert torch.equal(x_new.contiguous(), x)
    if mxfp8:
        q_ref, s_ref = mxfp8_e4m3_quantize(y_ref, is_sf_swizzled_layout=True)
        assert torch.equal(s, s_ref)
        steps = _code_steps(out, q_ref, torch.float8_e4m3fn)
    else:
        assert s is None
        steps = _code_steps(out, y_ref, torch.bfloat16)
    assert steps.max().item() <= 1


def test_norm_module_accepts_3d_per_head():
    """The dspark drafter norms per-head context K with (tokens, heads,
    head_dim); the module routes non-2-D inputs to the FlashInfer norm."""
    from flashinfer.norm import gemma_rmsnorm

    from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm

    gen = torch.Generator(device="cuda").manual_seed(0)
    norm = MiniMAXGemmaRMSNorm(128)
    norm.weight.data = (torch.randn(128, generator=gen, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    x = torch.randn(129, 4, 128, generator=gen, device="cuda").to(torch.bfloat16)

    y_ref = gemma_rmsnorm(x.clone(), norm.weight, norm.variance_epsilon)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.equal(out, y_ref)
