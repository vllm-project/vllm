# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 large-step routed MoE of DeepSeek-V4.1-Flash
(`vllm/models/deepseek_v41/amd/largestep`), one GPU, one TP rank's share.

The reference is `dsv41_largestep_reference.py` in this folder: torch only,
float64, no activation quantization, with the exact values of the MXFP4
weights. The block quantizes x and the intermediate h to MXFP8 and rounds h
and each pick's output to bf16, so it is close to the reference but not
equal. The tolerances below are set from that rounding, and each says why.

Stage tests (`test_stage_*`) feed each kernel the exact input of the
previous stage, so a stage's error is measured alone. Layout tests
(`test_pick_sort_*`, `test_quant_x_*`) check the bytes and tables bit for
bit. The vLLM MoE against the same reference, on the same weights, is in
`tests/models/test_dsv41_mono_numerics.py::test_largestep_hook_matches_moe`
(it needs the layer, so 4 GPUs).

    pytest -s tests/kernels/models/amd/test_dsv41_largestep_moe.py
"""

import pytest
import torch

from tests.kernels.models.amd import dsv41_largestep_reference as R
from vllm.platforms import current_platform

E, TOPK, HID = R.EXPERTS, R.TOPK, R.HIDDEN

# Tolerances, as relative L2 error against the float64 reference, each about
# 1.3x the error measured on MI300X (24 Nov 2026, every M and TP alike to 2%).
# MXFP8 (e4m3, 3 mantissa bits, one power-of-two scale per 32 values) gives a
# relative error of a few percent per value, and on random data a GEMM output
# keeps about the relative error of its quantized input. A wrong expert, row or
# scale gives a relative error near 1, so every bound is far from a real bug.
STAGE1_RTOL = 5e-2  # x to MXFP8, swiglu, h to bf16: measured 3.8e-2
STAGE2_RTOL = 3.5e-2  # h to MXFP8, down projection, a pick to bf16: 2.7e-2
COMBINE_RTOL = 2.5e-3  # bf16 picks summed in f32, the token to bf16: 1.7e-3
BLOCK_RTOL = 6.5e-2  # all of the above, x and h both quantized: 4.7e-2

ROWS = [
    pytest.param(16, id="M16"),
    pytest.param(48, id="M48-8req"),
    pytest.param(96, id="M96-16req"),
    pytest.param(144, id="M144-24req"),
    pytest.param(256, id="M256-max"),
]
TPS = [pytest.param(8, id="TP8-inter288"), pytest.param(4, id="TP4-inter576")]


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 3


needs_gfx942 = pytest.mark.skipif(
    not _on_gfx942(), reason="the gfx942 large-step kernels"
)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.double(), want.double()
    return ((got - want).norm() / want.norm()).item()


def _plain_weights(inter: int, dev, g):
    """Random MXFP4 expert weights in the layout vLLM loads: w13 [E, 2 Np,
    H / 2] (gate rows, then up rows, each padded to Np with zeros), w2 [E, H,
    Np / 2], E8M0 codes 118..122 (scales 2^-9..2^-5)."""
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


class _Experts:
    """Stands in for the routed experts as the mono decode layer's load hook
    leaves them: only `mono942`, the gfx942 copies, is read."""

    def __init__(self, copies):
        self.mono942 = copies


class _Case:
    """One TP share's weights, inputs and routing, plus `LargeStepMoE` built
    the way the model builds it."""

    def __init__(self, tp: int, rows: int, seed: int):
        pytest.importorskip("flydsl")
        from vllm.models.deepseek_v41.amd.largestep.moe import LargeStepMoE
        from vllm.models.deepseek_v41.amd.mono.weights942 import moe_copies

        self.dev = dev = torch.device("cuda:0")
        g = torch.Generator(device=dev).manual_seed(seed)
        self.tp, self.rows, self.inter = tp, rows, 2304 // tp
        self.w13, self.w13_s, self.w2, self.w2_s = _plain_weights(self.inter, dev, g)
        copies = moe_copies(
            self.w13.clone(), self.w13_s, self.w2.clone(), self.w2_s, self.inter
        )
        self.block = LargeStepMoE(_Experts(copies), tp, dev)
        self.x = (torch.randn(rows, HID, generator=g, device=dev) * 0.5).to(
            torch.bfloat16
        )
        self.topk = torch.stack(
            [torch.randperm(E, generator=g, device=dev)[:TOPK] for _ in range(rows)]
        )
        self.tw = torch.rand(rows, TOPK, generator=g, device=dev) * 0.5
        self.picks = rows * TOPK


@needs_gfx942
@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("rows", ROWS)
def test_block_matches_fp64_reference(tp: int, rows: int):
    """`LargeStepMoE` as the model calls it (pick sort, x quantization, both
    stages, combine, shared scratch) against the float64 reference."""
    c = _Case(tp, rows, seed=1000 + rows + tp)
    got = c.block(c.x, c.topk, c.tw)
    torch.accelerator.synchronize()
    want = R.routed_fp64(c.x, c.w13, c.w13_s, c.w2, c.w2_s, c.topk, c.tw, c.inter)
    assert torch.isfinite(got).all()
    err = _rel(got, want)
    print(f"block {tp=} {rows=}: rel L2 {err:.3e}")
    assert err < BLOCK_RTOL, f"relative L2 {err:.3e}"


@needs_gfx942
@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("rows", ROWS)
def test_stage_ug_matches_fp64_reference(tp: int, rows: int):
    """Stage 1 alone (`moe_ug`): h = swiglu(W_gate x, W_up x) of each pick.
    The kernel reads x as MXFP8 bytes, so its quantization is part of this
    stage's error."""
    c = _Case(tp, rows, seed=2000 + rows + tp)
    b = c.block
    s = b.scratch
    st = torch.cuda.current_stream()
    ids = c.topk.to(torch.int32).contiguous()
    b.sort(ids.data_ptr(), c.picks, s.table.data_ptr(), s.count.data_ptr(), st)
    b.qx(c.x.data_ptr(), s.xb.data_ptr(), s.xs.data_ptr(), c.rows, st)
    b.ug(
        b.w13.data_ptr(), b.w13_s.data_ptr(), s.xb.data_ptr(), s.xs.data_ptr(),
        s.table.data_ptr(), s.count.data_ptr(), s.h.data_ptr(), st,
    )  # fmt: skip
    torch.accelerator.synchronize()
    got = s.h[: c.picks]
    want = R.stage1_fp64(c.x, c.w13, c.w13_s, c.topk, c.inter)
    assert torch.isfinite(got).all()
    err = _rel(got, want)
    print(f"stage ug {tp=} {rows=}: rel L2 {err:.3e}")
    assert err < STAGE1_RTOL, f"relative L2 {err:.3e}"


@needs_gfx942
@pytest.mark.parametrize("tp", TPS)
@pytest.mark.parametrize("rows", ROWS)
def test_stage_down_matches_fp64_reference(tp: int, rows: int):
    """The h quantization and stage 2 (`quant_h`, `moe_down`) on a given h: each
    pick's routing weight x W_down h. The reference takes the same bf16 h, so
    only this stage's error is measured."""
    c = _Case(tp, rows, seed=3000 + rows + tp)
    b = c.block
    s = b.scratch
    st = torch.cuda.current_stream()
    g = torch.Generator(device=c.dev).manual_seed(rows)
    h = (torch.randn(c.picks, c.inter, generator=g, device=c.dev) * 2).to(
        torch.bfloat16
    )
    s.h[: c.picks].copy_(h)
    ids = c.topk.to(torch.int32).contiguous()
    wts = c.tw.float().reshape(-1).contiguous()
    b.sort(ids.data_ptr(), c.picks, s.table.data_ptr(), s.count.data_ptr(), st)
    b.qh(
        s.h.data_ptr(), s.hq.data_ptr(), s.hqs.data_ptr(),
        c.picks * (c.inter // 32), st,
    )  # fmt: skip
    b.down(
        b.w2.data_ptr(), b.w2_s.data_ptr(), s.hq.data_ptr(), s.hqs.data_ptr(),
        wts.data_ptr(), s.table.data_ptr(), s.count.data_ptr(), s.y.data_ptr(),
        st,
    )  # fmt: skip
    torch.accelerator.synchronize()
    got = s.y[: c.picks]
    want = R.stage2_fp64(h, c.w2, c.w2_s, c.topk, c.tw, c.inter)
    assert torch.isfinite(got).all()
    err = _rel(got, want)
    print(f"stage down {tp=} {rows=}: rel L2 {err:.3e}")
    assert err < STAGE2_RTOL, f"relative L2 {err:.3e}"


@needs_gfx942
@pytest.mark.parametrize("rows", ROWS)
def test_stage_combine_matches_fp64_reference(rows: int):
    """The combine (`combine`): each token's 6 pick rows summed, to bf16."""
    c = _Case(8, rows, seed=4000 + rows)
    b = c.block
    s = b.scratch
    g = torch.Generator(device=c.dev).manual_seed(rows)
    y = torch.randn(c.picks, HID, generator=g, device=c.dev).to(torch.bfloat16)
    s.y[: c.picks].copy_(y)
    b.combine(s.y.data_ptr(), s.routed.data_ptr(), rows, torch.cuda.current_stream())
    torch.accelerator.synchronize()
    want = y.double().view(rows, TOPK, HID).sum(1)
    err = _rel(s.routed[:rows], want)
    print(f"combine {rows=}: rel L2 {err:.3e}")
    assert err < COMBINE_RTOL, f"relative L2 {err:.3e}"


@needs_gfx942
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(1, id="M1"),
        pytest.param(16, id="M16-one-group-per-expert"),
        pytest.param(96, id="M96-several-groups"),
        pytest.param(256, id="M256-max"),
    ],
)
def test_pick_sort_matches_reference_table(rows: int):
    """The GPU pick sort (`moe_sort`) writes the table and group count of
    `pick_groups` exactly. Every token picks expert 5 first, so expert 5 has
    `rows` picks and more than one group of 16 from 17 rows on."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep.moe_sort import build_moe_sort

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows)
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
    want, ng = R.pick_groups(topk, E)
    table = torch.full(
        (E + rows * TOPK // 16 + 1, 4 + 16), -7, dtype=torch.int32, device=dev
    )
    count = torch.zeros(1, dtype=torch.int32, device=dev)
    build_moe_sort()(topk.data_ptr(), rows * TOPK, table.data_ptr(), count.data_ptr())
    torch.accelerator.synchronize()
    assert int(count) == ng
    assert torch.equal(table[:ng], want.to(torch.int32))


@needs_gfx942
@pytest.mark.parametrize(
    ("rows", "pad"),
    [
        pytest.param(43, 5, id="M43-pad5"),
        pytest.param(78, 18, id="M78-pad18"),
        pytest.param(250, 6, id="M250-pad6-to-max"),
    ],
)
def test_pick_sort_skips_padding_rows(rows: int, pad: int):
    """CUDA graph padding rows at the end of a captured step come from the
    router with expert -1 in every pick (VLLM_MOE_SKIP_PADDING). The sort
    leaves them out: the table and count equal those of the real rows alone,
    and no word past the table's groups is written."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep.moe_sort import build_moe_sort

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows)
    real = torch.stack(
        [torch.randperm(E, generator=g, device=dev)[:TOPK] for _ in range(rows)]
    ).to(torch.int32)
    topk = torch.cat(
        [real, torch.full((pad, TOPK), -1, dtype=torch.int32, device=dev)]
    ).contiguous()
    want, ng = R.pick_groups(real, E)
    groups = E + (rows + pad) * TOPK // 16 + 1
    table = torch.full((groups, 4 + 16), -7, dtype=torch.int32, device=dev)
    count = torch.zeros(1, dtype=torch.int32, device=dev)
    build_moe_sort()(
        topk.data_ptr(), (rows + pad) * TOPK, table.data_ptr(), count.data_ptr()
    )
    torch.accelerator.synchronize()
    assert int(count) == ng
    assert torch.equal(table[:ng], want.to(torch.int32))
    assert bool((table[ng:] == -7).all())


@needs_gfx942
@pytest.mark.parametrize(
    "rows", [pytest.param(1, id="M1"), pytest.param(48, id="M48"),
             pytest.param(256, id="M256")]
)  # fmt: skip
def test_quant_x_matches_reference_bytes(rows: int):
    """`moe_xq.build_quant_x` writes the bytes and codes of `quant_rows`
    exactly, including an all-zero 32-value block (amax floor 1e-10)."""
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.largestep.moe_xq import build_quant_x

    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(rows)
    x = (torch.randn(rows, HID, generator=g, device=dev) * 3).to(torch.bfloat16)
    x[0, :32] = 0
    bufs = [torch.zeros(rows, n, dtype=torch.uint8, device=dev)
            for n in (HID, HID // 32)]  # fmt: skip
    build_quant_x()(x.data_ptr(), *[b.data_ptr() for b in bufs], rows)
    torch.accelerator.synchronize()
    xb, xs = R.quant_rows(x)
    for got, want in zip(bufs, (xb, xs), strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "rows,prefills,decodes,decode_tokens,takes",
    [
        pytest.param(48, 0, 8, 48, True, id="decode-only-verify-step"),
        pytest.param(256, 0, 42, 252, True, id="largest-step-graph-padded-to-256"),
        pytest.param(42, 0, 7, 42, False, id="mono-decode-layer-step"),
        pytest.param(257, 0, 43, 257, False, id="above-block-limit"),
        pytest.param(96, 1, 0, 0, False, id="prefill-chunk"),
        pytest.param(96, 1, 8, 48, False, id="mixed-batch"),
    ],
)
def test_takes_decode_only_steps(rows, prefills, decodes, decode_tokens, takes):
    """`LargeStepHook.takes` takes decode-only steps of 43 to 256 rows and
    nothing else, by the same attention-metadata check as the mono decode layer
    (`MonoDecodeLayer._decode_step`). Needs no GPU: the hook's block is not
    built and the forward context carries only the layer's SWA metadata."""
    import types

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.amd.largestep.moe import LargeStepHook

    swa = types.SimpleNamespace(
        num_prefills=prefills, num_decodes=decodes, num_decode_tokens=decode_tokens
    )
    attn = types.SimpleNamespace(
        swa_cache_layer=types.SimpleNamespace(prefix="l.swa"),
        compressed_cache_prefix=None,
    )
    layer = types.SimpleNamespace(attn=attn)
    hook = LargeStepHook.__new__(LargeStepHook)
    hook._layer = lambda: layer
    fc = ForwardContext.__new__(ForwardContext)
    fc.attn_metadata = {"l.swa": swa}  # type: ignore[dict-item]
    fc.cudagraph_runtime_mode = CUDAGraphMode.FULL
    x = torch.empty(rows, 8)
    with override_forward_context(fc):
        assert hook.takes(x) is takes
    # No forward context (a profile or dummy run): the original path.
    with override_forward_context(None):
        assert hook.takes(x) is False
