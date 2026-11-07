# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the gfx1250 router GEMM (aiter tgemm behind
fp32_router_gemm_dispatch) for BF16 router weights."""

import pytest
import torch

from vllm.platforms import current_platform


def _on_gfx1250() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1250

    return on_gfx1250()


pytestmark = pytest.mark.skipif(
    not _on_gfx1250(), reason="the tgemm router GEMM path is gfx1250-only"
)


@pytest.mark.parametrize(("hidden_size", "num_experts"), [(7168, 384), (4096, 256)])
@pytest.mark.parametrize("num_tokens", [1, 7, 32, 33, 256, 4096])
def test_gfx1250_router_gemm_matches_reference(
    hidden_size: int, num_experts: int, num_tokens: int
) -> None:
    # Importing gate_linear registers the dispatch custom op.
    from vllm.model_executor.layers.fused_moe.router import gate_linear  # noqa: F401
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        can_use_rocm_fp32_router_gemm,
    )

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(num_experts, hidden_size, device="cuda") * 0.02).to(torch.bfloat16)
    # gfx1250 takes the tgemm path at every token count, not just M <= 32.
    assert can_use_rocm_fp32_router_gemm(x, w)

    out = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
    ref = x.double() @ w.double().T
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.double(), ref, atol=1e-3, rtol=1e-3)
    # The selected experts must not change, except on near-ties at the cutoff.
    ref_top = ref.topk(7, dim=-1).values
    clear = (ref_top[:, 5] - ref_top[:, 6]) > 1e-3
    got = out.topk(6, dim=-1).indices.sort(dim=-1).values
    want = ref.topk(6, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(got[clear], want[clear])


def test_gfx1250_router_gemm_rejects_fp32_weight() -> None:
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        can_use_rocm_fp32_router_gemm,
    )

    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 1024, device="cuda", dtype=torch.float32)
    assert not can_use_rocm_fp32_router_gemm(x, w)
