# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-scale layouts accepted by the aiter A8W8 block-scale preshuffled
GEMM (rocm_aiter_ops.triton_gemm_a8w8_blockscale_preshuffle).

The scales reach it as a row-major (M, G) tensor, as a column-major view
(QuantFP8 on gfx1250), or straight from group_fp8_quant(transpose_scale=True)
after QuantFP8's re-view. All three must give the unshuffled reference.
"""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

BLOCK = 128


def _preshuffle_backend_available() -> bool:
    if not current_platform.is_rocm() or not torch.cuda.is_available():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return bool(rocm_aiter_ops.is_linear_enabled()) and get_cdna_version() >= 4


pytestmark = pytest.mark.skipif(
    not _preshuffle_backend_available(),
    reason="needs aiter's preshuffled block-scale GEMM (CDNA4+ with aiter linear)",
)


def _reference(a_fp8, a_scale, b_fp8, b_scale) -> torch.Tensor:
    a = a_fp8.float() * a_scale.repeat_interleave(BLOCK, dim=1)
    b = b_fp8.float() * b_scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return a @ b.t()


@pytest.mark.parametrize("m", [1, 33, 256])
@pytest.mark.parametrize(("n", "k"), [(1536, 7168), (7168, 2048)])
def test_preshuffle_gemm_scale_layouts(m: int, n: int, k: int) -> None:
    torch.manual_seed(m)
    fp8 = current_platform.fp8_dtype()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_scale = rocm_aiter_ops.group_fp8_quant(x, BLOCK)  # row-major (M, G)
    b_fp8 = (torch.randn(n, k, device="cuda") * 0.05).to(fp8)
    b_scale = torch.rand(n // BLOCK, k // BLOCK, device="cuda") + 0.5
    b_shuf = rocm_aiter_ops.shuffle_weight(b_fp8, layout=(16, 16))
    ref = _reference(a_fp8, a_scale, b_fp8, b_scale)

    def gemm(scale: torch.Tensor) -> torch.Tensor:
        return rocm_aiter_ops.triton_gemm_a8w8_blockscale_preshuffle(
            a_fp8, b_shuf, scale, b_scale, [BLOCK, BLOCK], torch.bfloat16
        ).float()

    row_major = a_scale.contiguous()
    col_major = a_scale.t().contiguous().t()
    # A size-1 dim has no meaningful stride, so only M > 1 is column-major.
    assert m == 1 or col_major.stride(0) == 1
    # QuantFP8.forward_hip on gfx1250: M-major bytes, re-viewed to (M, G).
    a_fp8_t, raw = rocm_aiter_ops.group_fp8_quant(x, BLOCK, transpose_scale=True)
    torch.testing.assert_close(a_fp8_t.float(), a_fp8.float())
    requant = raw.view(raw.shape[1], raw.shape[0]).transpose(-2, -1)
    torch.testing.assert_close(requant, a_scale)

    for scale in (row_major, col_major, requant):
        torch.testing.assert_close(gemm(scale), ref, atol=0.5, rtol=2e-2)
