# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.models.kimi_k3.amd.ops.kda_fb_gemm import (
    KDA_FB_GEMM_MAX_TOKENS,
    kda_fb_gemm,
    kda_fb_gemm_supported,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD KDA f_b GEMM requires ROCm",
)

# Kimi-K3.1 KDA at TP8: head_dim 128, 96 heads -> 12 local heads. The in_proj
# row is [qkv 4608 | g 1536 | f_a 128 | beta 12 | pad 4] = 6288 columns.
HEAD_DIM = 128
LOCAL_PROJ = 1536
ROW = 6288
F_A_OFFSET = 4 * LOCAL_PROJ


def _strided_f_a(num_tokens: int, row: int = ROW, offset: int = F_A_OFFSET):
    projected = torch.randn(num_tokens, row, device="cuda", dtype=torch.bfloat16)
    return projected, projected[:, offset : offset + HEAD_DIM]


def _weight(n: int = LOCAL_PROJ, k: int = HEAD_DIM) -> torch.Tensor:
    return torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 5, 8, 16, 33, 64, 128, 192, 256, 336, 1024]
)
def test_kda_fb_gemm_matches_generic_gemm(num_tokens: int):
    torch.manual_seed(num_tokens)
    _, f_a = _strided_f_a(num_tokens)
    weight = _weight()
    assert num_tokens == 1 or not f_a.is_contiguous()
    assert kda_fb_gemm_supported(f_a, weight)

    new = kda_fb_gemm(f_a, weight)
    old = rocm_unquantized_gemm_impl(f_a, weight)
    ref = f_a.float() @ weight.float().T
    assert new.shape == (num_tokens, LOCAL_PROJ)
    torch.testing.assert_close(new.float(), ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(new, old, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [4, 192, 336, KDA_FB_GEMM_MAX_TOKENS + 1, 4096])
def test_kda_fb_proj_op(num_tokens: int):
    # The custom op takes the strided path up to the token cap and the generic
    # GEMM above it; both must give the same answer.
    torch.manual_seed(0)
    _, f_a = _strided_f_a(num_tokens)
    weight = _weight()
    out = torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, weight)
    ref = f_a.float() @ weight.float().T
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


def test_kda_fb_proj_reads_in_place():
    # The point of the strided path: no copy kernel before the GEMM.
    from torch.profiler import ProfilerActivity, profile

    _, f_a = _strided_f_a(4)
    weight = _weight()
    torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, weight)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, weight)
        torch.accelerator.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert names, "no kernels recorded"
    assert all("_kda_fb_gemm_kernel" in name for name in names), names


def test_kda_fb_proj_compiled_cuda_graph():
    # Mirrors KimiK3DeltaAttention.forward: split the in_proj output under
    # torch.compile with a dynamic batch, then replay a captured graph.
    torch.manual_seed(0)
    weight = _weight()
    split_sizes = [3 * LOCAL_PROJ, LOCAL_PROJ, HEAD_DIM, 12, 4]

    def fn(projected: torch.Tensor) -> torch.Tensor:
        f_a = projected.split(split_sizes, dim=-1)[2]
        return torch.ops.vllm.kimi_k3_kda_fb_proj(f_a, weight) * 2

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    for num_tokens in (4, 192):
        projected, f_a = _strided_f_a(num_tokens)
        static_in = projected.clone()
        torch.testing.assert_close(compiled(static_in), fn(static_in))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = compiled(static_in)
        static_in.copy_(torch.randn_like(static_in))
        graph.replay()
        ref = (static_in[:, F_A_OFFSET : F_A_OFFSET + HEAD_DIM].float()) @ (
            weight.float().T
        )
        torch.testing.assert_close(static_out.float(), ref * 2, atol=4e-2, rtol=1e-2)


def test_kda_fb_gemm_unsupported_layouts():
    weight = _weight()
    _, f_a = _strided_f_a(4)
    assert not kda_fb_gemm_supported(f_a.t().contiguous().t(), weight)
    assert not kda_fb_gemm_supported(f_a.float(), weight)
    assert not kda_fb_gemm_supported(f_a[:, :96], weight[:, :96].contiguous())
