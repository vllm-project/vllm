# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's attention input projection (``DecodeProjection``: warp
split-K, cluster split-K or cuBLAS per row count) against an FP32 matmul of the same
BF16 operands, at the TP8 K3.1 shapes and every engine's row range, eager and
replayed from a CUDA graph."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (  # noqa: E402
    decode,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (  # noqa: E402
    projection_policies,
)

# (N, K): TP8 K3.1 KDA input projection, DSA packed projection, DSA indexer wq_b.
SHAPES = [(6288, 7168), (3808, 7168), (4096, 1536)]
ROWS = [1, 2, 5, 16, 20, 64, 65, 160, 320]


def reference(x, weight):
    return x.float() @ weight.float().T


def check(out, ref):
    """BF16 rounding of an FP32-accumulated dot product (K <= 7168)."""
    assert torch.isfinite(out).all()
    err = (out.float() - ref).norm() / ref.norm()
    assert err < 4e-3, err
    torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=2e-2)


@pytest.mark.parametrize("shape", SHAPES)
def test_decode_projection_rows(shape):
    n, k = shape
    gen = torch.Generator(device="cuda").manual_seed(n)
    weight = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    projection = decode.DecodeProjection(weight, ROWS, projection_policies(8, n, ROWS))
    engines = {}
    for rows in ROWS:
        x = torch.randn(rows, k, device="cuda", generator=gen).bfloat16()
        out = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        projection.run(x, out)
        check(out, reference(x, weight))
        engines[rows] = projection.engine(rows)
    # Each engine serves its documented row range.
    assert engines[1] == engines[16] == "warp"
    assert engines[20] == engines[64] == "split_k"
    assert engines[65] == engines[320] == "cublas"


def test_decode_projection_graph_and_pitch():
    """Packed projection buffers with more rows than the step (the providers'
    ``projected[:b]`` rows) and CUDA-graph replay with new activations, every engine
    (warp split-K, cluster split-K, cuBLAS) checked after each replay."""
    n, k, cap = 3808, 7168, 320
    weight = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    rows = [5, 32, 160]
    projection = decode.DecodeProjection(weight, rows, projection_policies(8, n, rows))
    assert [projection.engine(b) for b in rows] == ["warp", "split_k", "cublas"]
    buffers = {
        b: torch.zeros(cap, n, device="cuda", dtype=torch.bfloat16) for b in rows
    }
    xs = {b: torch.randn(b, k, device="cuda").bfloat16() for b in rows}
    for b in rows:
        projection.run(xs[b], buffers[b][:b])  # warm up / compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for b in rows:
            projection.run(xs[b], buffers[b][:b])
    for _ in range(2):
        for b in rows:
            xs[b].copy_(torch.randn(b, k, device="cuda"))
            buffers[b].fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        for b in rows:
            check(buffers[b][:b], reference(xs[b], weight))
            # Rows past the step stay untouched.
            assert buffers[b][b:].isnan().all()


@pytest.mark.parametrize("m", [1, 8, 9, 16])
def test_warp_tactic_residency(m):
    """Warp split-K tactics keep every wave's CTAs resident (one SM's shared memory
    holds ``ceil(ctas / SMs)`` rings) and fit the kernel's limits."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for n, k in SHAPES:
        t = decode.warp_tactic(m, n, k, sms)
        ctas = (n // t.output_tile) * -(-m // t.token_tile)
        resident = -(-ctas // sms)
        need = resident * (decode._smem_bytes(t) + decode._CTA_RESERVED_SMEM)
        assert need <= decode._SM_SMEM_BYTES or t.stages == 2
        assert t.token_tile >= min(m, 16)


def test_decode_projection_strided_operands():
    """Activations the warp split-K cannot take (a row pitch wider than K, an
    offset not 32 B aligned) fall back to cuBLAS with the same result."""
    n, k, b = 3808, 7168, 5
    weight = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    projection = decode.DecodeProjection(weight, [b], projection_policies(8, n, [b]))
    assert projection.engine(b) == "warp"
    wide = torch.randn(b, k + 64, device="cuda").bfloat16()
    for x in (wide[:, :k], wide.view(-1)[8 : 8 + b * k].view(b, k)):
        out = torch.empty(b, n, device="cuda", dtype=torch.bfloat16)
        projection.run(x, out)
        check(out, reference(x, weight))
