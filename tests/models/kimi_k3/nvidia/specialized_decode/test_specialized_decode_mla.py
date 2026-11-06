# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode (decode_backend=latency_specialized): the MLA layers'
precision against the stock TOKENSPEED_MLA path."""

import pytest
import torch

from vllm.platforms import current_platform

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

HEADS, EPS = 12, 1e-6  # TP8


def _pair_inputs(gen, batch, pitch):
    """Packed projection rows whose q_a pairs (2m, 2m + 1) are equal values +-k/16
    (k in 16..31: every sum of squares exact in any order), and a q_a RMSNorm weight
    whose odd element is the next BF16 above its even one, so q_c[2m] - q_c[2m + 1]
    is a few BF16 ulps: exact in FP32, BF16 and E4M3, so any q_c rounding the norm
    form moves shows in it."""
    dev, bf = "cuda", torch.bfloat16
    k = torch.randint(16, 32, (batch, 768), generator=gen, device=dev).float() / 16
    k *= torch.randint(0, 2, (batch, 768), generator=gen, device=dev) * 2 - 1
    projected = torch.zeros(batch, pitch, device=dev, dtype=bf)
    projected[:, :1536] = k.repeat_interleave(2, dim=1).to(bf)
    even = (torch.rand(768, generator=gen, device=dev) + 0.5).to(bf)
    odd = (even.view(torch.int16) + 1).view(bf)
    weight = torch.stack([even, odd], dim=1).flatten()
    return projected, weight


def _pair_differences(q_c):
    """q_c[:, 2m] - q_c[:, 2m + 1] (exact) as E4M3 bytes."""
    diff = q_c[:, 0::2].float() - q_c[:, 1::2].float()
    return diff.to(torch.float8_e4m3fn).view(torch.uint8)


@requires_sm100
@pytest.mark.parametrize("batch", [3, 64])
def test_k3_mla_query_matches_stock_q_norm(batch):
    """K3 MLA: the query kernel PreparedMla compiles (mla.prepared.compile_core,
    MlaQuery's scalar form below 4 rows, tiled from 4) normalizes q_a as the stock
    fused_q_kv_rmsnorm (Triton) does, bit for bit, seen through its FP8 query:
    Q_B's rope rows pick q_c[2m] - q_c[2m + 1] (_pair_inputs), each m once over the
    12 heads x 64 rope rows, and W_UK is zero (a zero latent). The vllm_c form
    (IEEE divide) differs on these rows, so a query compiled without the Triton
    form fails."""
    import cutlass

    from vllm.models.common.ops import fused_q_kv_rmsnorm
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_norm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.prepared import (
        compile_core,
    )

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    bf, f32 = torch.bfloat16, torch.float32
    pitch = 2112 + HEADS * 128
    gen = torch.Generator(device=dev).manual_seed(batch)
    q_b = torch.zeros(HEADS * 192, 1536, device=dev, dtype=bf)
    m = torch.arange(HEADS * 64, device=dev)
    rows = (m // 64) * 192 + 128 + m % 64
    q_b[rows, 2 * m], q_b[rows, 2 * m + 1] = 1.0, -1.0
    w_uk = torch.zeros(HEADS, 128, 512, device=dev, dtype=bf)
    w_uv = torch.zeros(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    kv_norm = torch.ones(512, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev, dtype=f32)
    cache = torch.zeros(2, 64, 576, device=dev, dtype=torch.uint8)
    slots = torch.full((batch,), -1, device=dev, dtype=torch.int64)  # no KV insert
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    mismatched = ieee_differs = 0
    kernel, ieee_norm = None, compile_norm(batch, HEADS, pitch)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    for _ in range(4096 // batch):
        projected, q_norm = _pair_inputs(gen, batch, pitch)
        weights = (q_norm, kv_norm, q_b, w_uk, w_uv, unit, unit, cache)
        if kernel is None:
            specs = tuple(layout(w) for w in weights)
            kernel = compile_core(batch, HEADS, pitch, specs, dev.index)[0]
        views = [to_cute(w, 4 if w.dtype == f32 else 16) for w in weights[:5]]
        args = (to_cute(projected[:, :2112]), *views[:4], to_cute(cache))
        args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
        kernel(*args, cutlass.Float32(EPS), to_cute(query), stream(dev))
        stock, _ = fused_q_kv_rmsnorm(
            projected[:, :1536], projected[:, 1536:2048], q_norm, kv_norm, EPS
        )
        ref = _pair_differences(stock).view(batch, HEADS, 64)
        mismatched += int((query[..., 512:] != ref).sum())
        assert not query[..., :512].any()  # zero W_UK: a zero latent
        args = (to_cute(projected), to_cute(q_norm), to_cute(q_c))
        ieee_norm(*args, cutlass.Float32(EPS), stream(dev))
        ieee = _pair_differences(q_c)
        ieee_differs += int((ieee.view(batch, HEADS, 64) != ref).sum())
    assert mismatched == 0
    assert ieee_differs > 0  # the inputs tell the two norm forms apart


@requires_sm100
@pytest.mark.parametrize(
    "reqs,q,heads,dcp", [(1, 1, 12, 1), (64, 1, 12, 1), (12, 5, 12, 1), (12, 5, 96, 8)]
)
def test_mla_workspace_reserved_at_construction(reqs, q, heads, dcp):
    """Tokenspeed's split-KV workspace of a prepared shape is reserved by
    reserve_workspace (PreparedMla's constructor, before vLLM profiles its memory
    for the KV cache): the decode of that shape (stock_decode, as PreparedMla runs
    it: DCP verify packs (row, head) tiles) takes it and allocates nothing."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp as mla

    dev = torch.device("cuda", torch.accelerator.current_device_index())
    batch = reqs * q
    max_seq = mla.local_max(1 << 17, dcp) if dcp > 1 else 1 << 17
    split = mla.fill_splits(batch, dev)
    pack = dcp > 1 and q > 1
    mla._WORKSPACE.clear()
    mla.reserve_workspace(reqs, q, heads, dev, max_seq, split, pack)
    held = mla._WORKSPACE[dev, heads]
    cache = torch.zeros(4, 64, 576, device=dev, dtype=torch.uint8)
    table = torch.arange(2, device=dev, dtype=torch.int32).repeat(reqs, 1)
    lengths = torch.full((reqs,), 100, device=dev, dtype=torch.int32)
    query = torch.zeros(reqs, q, heads, 576, device=dev, dtype=torch.uint8)
    out = torch.zeros(reqs, q, heads, 512, device=dev, dtype=torch.bfloat16)
    lse, cp = None, {}
    if dcp > 1:
        lse = torch.zeros(reqs, q, heads, device=dev)
        cp = dict(cp=(dcp, 0, 1), totals=lengths * dcp, pack=pack)
    before = torch.accelerator.memory_allocated(dev)
    query = query.view(torch.float8_e4m3fn)
    args = (query, cache, table, lengths, max_seq, (0.1, 1.0), out, lse, split)
    mla.stock_decode(*args, **cp)
    torch.accelerator.synchronize()
    assert mla._WORKSPACE[dev, heads] is held
    assert torch.accelerator.memory_allocated(dev) == before
