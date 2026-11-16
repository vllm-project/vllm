# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 grouped-GEMM routed MoE of DeepSeek-V4.1-Flash's large decode
steps (`vllm/models/deepseek_v41/amd/largestep`): stage 1, the h
quantization, stage 2 and the combine, on one GPU, for one TP rank's share,
against the torch reference with the same rounding points
(`moe_host.ref_routed`).

The share is the intermediate width: 288 at TP 8 (MI300X), 576 at TP 4
(MI325X). Steps of 16, 48, 96 and 144 rows (tokens), random routing of 6 of
384 experts a token. The weights are random plain MXFP4 tensors in the
layout vLLM loads, turned into the gfx942 copies by the mono kernels'
`weights942.moe_copies`, as the model does at load.

Pass: the relative L2 error of the step's output under 2e-2, no NaN. Both
sides round h and each pick's output to bf16 and quantize the same values to
MXFP8; the f32 sums run in different orders, so a value near a rounding
boundary can land on the other side.

    pytest -s tests/kernels/models/amd/test_dsv41_largestep_moe.py
"""

import pytest
import torch

from vllm.platforms import current_platform

E, TOPK, HID = 384, 6, 5120


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 3


def _plain_weights(inter, dev, g):
    np_ = -(-inter // 128) * 128
    u8 = torch.uint8

    def ri(lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=g, device=dev, dtype=u8)

    w13 = ri(0, 256, (E, 2 * np_, HID // 2))
    w13_s = ri(118, 123, (E, 2 * np_, HID // 32))
    w2 = ri(0, 256, (E, HID, np_ // 2))
    w2_s = ri(118, 123, (E, HID, np_ // 32))
    for half in (0, np_):
        w13[:, half + inter : half + np_] = 0
    w2[..., inter // 2 :] = 0
    return w13, w13_s, w2, w2_s


@pytest.mark.skipif(not _on_gfx942(), reason="the gfx942 large-step kernels")
@pytest.mark.parametrize("tp", [8, 4])
@pytest.mark.parametrize("rows", [16, 48, 96, 144])
def test_routed_moe_matches_reference(tp: int, rows: int):
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep import moe_host as H
    from vllm.models.deepseek_v41.amd.largestep.moe_down import build_moe_down
    from vllm.models.deepseek_v41.amd.largestep.moe_small import (
        build_combine,
        build_quant_h,
    )
    from vllm.models.deepseek_v41.amd.largestep.moe_ug import build_moe_ug
    from vllm.models.deepseek_v41.amd.mono.weights942 import moe_copies

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows + tp)
    inter = 2304 // tp
    w13, w13_s, w2, w2_s = _plain_weights(inter, dev, g)
    c13, c13_s, c2, c2_s = moe_copies(w13.clone(), w13_s, w2.clone(), w2_s, inter)
    x = (torch.randn(rows, HID, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    topk = torch.stack(
        [torch.randperm(E, generator=g, device=dev)[:TOPK] for _ in range(rows)]
    )
    tw = torch.rand(rows, TOPK, generator=g, device=dev) * 0.5
    picks = rows * TOPK
    xb, xs = H.quant_rows(x)
    table, ng = H.pick_groups(topk, E)
    count = torch.tensor([ng], dtype=torch.int32, device=dev)
    h = torch.zeros(picks, inter, dtype=torch.bfloat16, device=dev)
    hq = torch.zeros(picks, inter, dtype=torch.uint8, device=dev)
    hqs = torch.zeros(picks, inter // 32, dtype=torch.uint8, device=dev)
    y = torch.zeros(picks, HID, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(rows, HID, dtype=torch.bfloat16, device=dev)
    wts = tw.reshape(-1).float().contiguous()

    build_moe_ug(tp)(
        c13.data_ptr(), c13_s.data_ptr(), xb.data_ptr(), xs.data_ptr(),
        table.data_ptr(), count.data_ptr(), h.data_ptr(),
    )  # fmt: skip
    build_quant_h(inter)(
        h.data_ptr(), hq.data_ptr(), hqs.data_ptr(), picks * (inter // 32)
    )
    build_moe_down(tp)(
        c2.data_ptr(), c2_s.data_ptr(), hq.data_ptr(), hqs.data_ptr(),
        wts.data_ptr(), table.data_ptr(), count.data_ptr(), y.data_ptr(),
    )  # fmt: skip
    build_combine()(y.data_ptr(), out.data_ptr(), rows)
    torch.accelerator.synchronize()

    ref = H.ref_routed(x, w13, w13_s, w2, w2_s, topk, tw, inter)
    got = out.float()
    assert not torch.isnan(got).any()
    err = ((got - ref).norm() / ref.norm()).item()
    assert err < 2e-2, f"TP{tp}, {rows} rows: relative L2 error {err:.3e}"


@pytest.mark.skipif(not _on_gfx942(), reason="the gfx942 large-step kernels")
@pytest.mark.parametrize("rows", [1, 16, 96, 256])
def test_pick_sort_matches_host(rows: int):
    """The GPU pick sort (`moe_sort.py`) writes the table and group count of
    `moe_host.pick_groups` exactly, including a step where many tokens pick
    the same expert (more than one group for it)."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep import moe_host as H
    from vllm.models.deepseek_v41.amd.largestep.moe_sort import build_moe_sort

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows)
    # Every token picks expert 5 first, then 5 others, so expert 5 has
    # `rows` picks and more than one group from 17 rows on. A token never
    # picks an expert twice.
    others = [
        torch.randperm(E - 1, generator=g, device=dev)[: TOPK - 1] for _ in range(rows)
    ]
    topk = torch.stack(
        [
            torch.cat([torch.tensor([5], device=dev), o + (o >= 5).long()])
            for o in others
        ]
    )
    topk = topk.to(torch.int32).contiguous()
    want, ng = H.pick_groups(topk, E)
    table = torch.full(
        (E + rows * TOPK // 16 + 1, 4 + 16), -7, dtype=torch.int32, device=dev
    )
    count = torch.zeros(1, dtype=torch.int32, device=dev)
    build_moe_sort()(topk.data_ptr(), rows * TOPK, table.data_ptr(), count.data_ptr())
    torch.accelerator.synchronize()
    assert int(count) == ng
    assert torch.equal(table[:ng], want.to(torch.int32))


@pytest.mark.skipif(not _on_gfx942(), reason="the gfx942 large-step kernels")
@pytest.mark.parametrize("tp", [8, 4])
@pytest.mark.parametrize("rows", [16, 96, 144])
def test_shared_expert_matches_reference(tp: int, rows: int):
    """The shared expert's two launches (`shared.py`) with the h quantization
    between them, against `moe_host.ref_shared`, on random MXFP8 weights made
    into the gfx942 copies by `weights942.linear_copy` as at model load."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep import moe_host as H
    from vllm.models.deepseek_v41.amd.largestep.moe_small import build_quant_h
    from vllm.models.deepseek_v41.amd.largestep.shared import (
        build_shared_down,
        build_shared_gate_up,
    )
    from vllm.models.deepseek_v41.amd.mono.weights942 import linear_copy

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows + 10 * tp)
    inter = 2304 // tp

    def mxfp8(n, k):
        w = torch.randn(n, k, generator=g, device=dev).clamp(-6, 6)
        w8 = w.to(torch.float8_e4m3fn).view(torch.uint8)
        sc = torch.randint(
            118, 123, (n // 32, k // 32), generator=g, device=dev, dtype=torch.uint8
        )
        return w8, sc

    gu8, gu_s = mxfp8(2 * inter, HID)
    d8, d_s = mxfp8(HID, inter)
    cgu, cgu_s = linear_copy(gu8.view(torch.float8_e4m3fn), gu_s)
    cd, cd_s = linear_copy(d8.view(torch.float8_e4m3fn), d_s)
    x = (torch.randn(rows, HID, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    xb, xs = H.quant_rows_natural(x)
    sh = torch.zeros(rows, inter, dtype=torch.bfloat16, device=dev)
    hb = torch.zeros(rows, inter, dtype=torch.uint8, device=dev)
    hs = torch.zeros(rows, inter // 32, dtype=torch.uint8, device=dev)
    out = torch.zeros(rows, HID, dtype=torch.bfloat16, device=dev)
    build_shared_gate_up(tp)(
        cgu.data_ptr(), cgu_s.data_ptr(), xb.data_ptr(), xs.data_ptr(), rows,
        sh.data_ptr(),
    )  # fmt: skip
    build_quant_h(inter, even_odd=False)(
        sh.data_ptr(), hb.data_ptr(), hs.data_ptr(), rows * (inter // 32)
    )
    build_shared_down(tp)(
        cd.data_ptr(), cd_s.data_ptr(), hb.data_ptr(), hs.data_ptr(), rows,
        out.data_ptr(),
    )  # fmt: skip
    torch.accelerator.synchronize()
    ref = H.ref_shared(x, gu8, gu_s, d8, d_s)
    got = out.float()
    assert not torch.isnan(got).any()
    err = ((got - ref).norm() / ref.norm()).item()
    assert err < 2e-2, f"TP{tp}, {rows} rows: relative L2 error {err:.3e}"


@pytest.mark.skipif(not _on_gfx942(), reason="the gfx942 large-step kernels")
@pytest.mark.parametrize("rows", [1, 48, 256])
def test_quant_x_matches_host(rows: int):
    """`moe_xq.build_quant_x` writes the bytes and codes of
    `moe_host.quant_rows` and `quant_rows_natural` exactly."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep import moe_host as H
    from vllm.models.deepseek_v41.amd.largestep.moe_xq import build_quant_x

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows)
    x = (torch.randn(rows, HID, generator=g, device=dev) * 3).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block: the 1e-10 floor
    bufs = [torch.zeros(rows, n, dtype=torch.uint8, device=dev)
            for n in (HID, HID // 32, HID, HID // 32)]  # fmt: skip
    build_quant_x()(x.data_ptr(), *[b.data_ptr() for b in bufs], rows)
    torch.accelerator.synchronize()
    xb, xs = H.quant_rows(x)
    xbn, xsn = H.quant_rows_natural(x)
    for got, want in zip(bufs, (xb, xs, xbn, xsn), strict=True):
        assert torch.equal(got, want)
