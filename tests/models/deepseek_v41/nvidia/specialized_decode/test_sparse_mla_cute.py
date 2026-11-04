# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``sparse_mla_decode`` (``sparse_mla_cute.py``) against an FP32 reference.

Each token attends to its own columns: the first 128 index the sliding-window
pool and the rest the compressed pool, as storage offsets. The cases cover the
widths of 1 to 5 chunks (128, a partial second chunk, 384, 512, 640), masked
columns (-1 and columns past a token's length, including a token with no active
column, whose output is zero), batches from one token to the most the kernel
takes, sinks and none, and scales other than one. The tolerance allows for the
probabilities' FP8 rounding (TRTLLM-gen's own error is of the same size).
"""

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.sparse_mla_cute import (  # noqa: E501
    _max_tokens,
    sparse_mla_decode,
    supports_sparse_mla_cute,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="needs an SM100-family GPU",
)

DEVICE = "cuda"
H, D, SWA = 16, 512, 128
SWA_ROWS, COMP_ROWS, PAGE = 1024, 8192, 64


def fp8(shape: tuple[int, ...], std: float, g: torch.Generator) -> torch.Tensor:
    """Normal values of standard deviation ``std`` as FP8 E4M3."""
    return (torch.randn(shape, device=DEVICE, generator=g) * std).to(
        torch.float8_e4m3fn
    )


def make_case(tokens: int, width: int, seed: int) -> dict:
    """Pools, indices with masked columns, and lengths for ``tokens`` tokens."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    swa_pool = fp8((SWA_ROWS // PAGE, PAGE, D), 1.0, g)
    comp_pool = fp8((COMP_ROWS // PAGE, PAGE, D), 1.0, g)
    swa = torch.randint(
        0, SWA_ROWS, (tokens, min(width, SWA)), device=DEVICE, generator=g
    )
    columns = [swa]
    if width > SWA:
        columns.append(
            torch.randint(
                0, COMP_ROWS, (tokens, width - SWA), device=DEVICE, generator=g
            )
        )
    indices = torch.cat(columns, dim=1).to(torch.int32)
    drop = torch.rand(indices.shape, device=DEVICE, generator=g) < 0.05
    indices[drop] = -1
    lens = torch.full((tokens,), width, dtype=torch.int32, device=DEVICE)
    if tokens > 1:
        lens[1] = max(width - 77, 1)  # a partial last chunk
    if tokens > 2:
        lens[2] = 0  # no active column
    return dict(
        q=fp8((tokens, H, D), 0.5, g),
        swa_pool=swa_pool,
        comp_pool=comp_pool,
        indices=indices,
        lens=lens,
        sinks=torch.randn(H, device=DEVICE, generator=g),
    )


def reference(
    c: dict, bmm1_scale: float, bmm2_scale: float, sinks: torch.Tensor | None
) -> torch.Tensor:
    """FP32 sparse attention over each token's active columns (K = V)."""
    swa = c["swa_pool"].reshape(-1, D).float()
    comp = c["comp_pool"].reshape(-1, D).float()
    q = c["q"].float()
    out = torch.zeros(q.shape, device=DEVICE)
    for t in range(q.shape[0]):
        n = int(c["lens"][t])
        idx = c["indices"][t, :n].long()
        cols = torch.arange(n, device=DEVICE)
        keep = idx >= 0
        if not bool(keep.any()):
            continue
        rows = torch.where(
            (cols < SWA)[:, None],
            swa[idx.clamp(0, swa.shape[0] - 1)],
            comp[idx.clamp(0, comp.shape[0] - 1)],
        )[keep]
        s = (q[t] @ rows.T) * bmm1_scale
        m = s.max(dim=1).values
        if sinks is not None:
            m = torch.maximum(m, sinks)
        p = torch.exp(s - m[:, None])
        denom = p.sum(dim=1)
        if sinks is not None:
            denom = denom + torch.exp(sinks - m)
        out[t] = (p @ rows) / denom[:, None] * bmm2_scale
    return out


@pytest.mark.parametrize("width", [128, 200, 384, 512, 640])
@pytest.mark.parametrize("tokens", [1, 6, 48])
@pytest.mark.parametrize("with_sinks", [True, False])
def test_matches_reference(width: int, tokens: int, with_sinks: bool) -> None:
    c = make_case(tokens, width, seed=width * 1000 + tokens)
    # The kernel takes only tokens whose clusters are all resident at once,
    # which depends on the device's SM count (read once the case's tensors
    # have made the device's context current).
    tokens = min(tokens, _max_tokens(c["q"].device.index, -(-width // 128)))
    c.update({key: c[key][:tokens] for key in ("q", "indices", "lens")})
    sinks = c["sinks"] if with_sinks else None
    bmm1_scale, bmm2_scale = D**-0.5 * 1.25, 0.75
    out = torch.full((tokens, H, D), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    args = (c["q"], c["swa_pool"], c["comp_pool"], c["indices"], c["lens"], out)
    assert supports_sparse_mla_cute(*args, bmm1_scale, bmm2_scale, sinks)
    sparse_mla_decode(*args, bmm1_scale, bmm2_scale, sinks)
    torch.accelerator.synchronize()
    ref = reference(c, bmm1_scale, bmm2_scale, sinks)
    got = out.float()
    assert not torch.isnan(got).any()
    torch.testing.assert_close(got, ref, atol=0.03, rtol=0.05)
    if tokens > 2:
        assert bool((got[2] == 0).all())


def test_cuda_graph_replay_matches_eager() -> None:
    c = make_case(12, 640, seed=7)
    args_in = (c["q"], c["swa_pool"], c["comp_pool"], c["indices"], c["lens"])
    eager = torch.empty((12, H, D), dtype=torch.bfloat16, device=DEVICE)
    sparse_mla_decode(*args_in, eager, D**-0.5, 1.0, c["sinks"])
    graphed = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla_decode(*args_in, graphed, D**-0.5, 1.0, c["sinks"])
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(graphed, eager)


def test_declines_unsupported() -> None:
    c = make_case(4, 640, seed=3)
    out = torch.empty((4, H, D), dtype=torch.bfloat16, device=DEVICE)
    base = [c["q"], c["swa_pool"], c["comp_pool"], c["indices"], c["lens"], out]
    assert supports_sparse_mla_cute(*base, 0.1, 1.0, c["sinks"])
    # Tensor scales, BF16 sinks, a 768-wide set, a BF16 query.
    assert not supports_sparse_mla_cute(*base, torch.tensor(0.1), 1.0, c["sinks"])
    assert not supports_sparse_mla_cute(*base, 0.1, 1.0, c["sinks"].bfloat16())
    wide = torch.zeros((4, 768), dtype=torch.int32, device=DEVICE)
    assert not supports_sparse_mla_cute(*base[:3], wide, *base[4:], 0.1, 1.0, None)
    bf16_q = c["q"].bfloat16()
    assert not supports_sparse_mla_cute(bf16_q, *base[1:], 0.1, 1.0, None)
    # More tokens than one wave of clusters.
    many = _max_tokens(torch.accelerator.current_device_index(), 5) + 1
    big = make_case(many, 640, seed=5)
    big_out = torch.empty((many, H, D), dtype=torch.bfloat16, device=DEVICE)
    big_args = (
        big["q"],
        big["swa_pool"],
        big["comp_pool"],
        big["indices"],
        big["lens"],
    )
    assert not supports_sparse_mla_cute(*big_args, big_out, 0.1, 1.0, None)


def test_index_rows_wider_than_the_table() -> None:
    # A [tokens, 640] view of a wider buffer, as when the builder's table is
    # sliced: the columns past 640 hold indices the kernel must not read.
    c = make_case(6, 640, seed=11)
    wide = torch.full((6, 768), 5, dtype=torch.int32, device=DEVICE)
    wide[:, :640] = c["indices"]
    view = wide[:, :640]
    args_in = (c["q"], c["swa_pool"], c["comp_pool"])
    dense = torch.empty((6, H, D), dtype=torch.bfloat16, device=DEVICE)
    strided = torch.empty_like(dense)
    sparse_mla_decode(*args_in, c["indices"], c["lens"], dense, 0.05, 1.0, c["sinks"])
    sparse_mla_decode(*args_in, view, c["lens"], strided, 0.05, 1.0, c["sinks"])
    torch.accelerator.synchronize()
    assert torch.equal(strided, dense)
