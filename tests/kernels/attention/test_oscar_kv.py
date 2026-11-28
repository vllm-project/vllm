# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR KV cache kernels (vllm/v1/attention/ops/oscar_kv.py) against a torch reference.

The reference re-implements the format (Hadamard rotation, per-row |x| clip,
uniform codes, fp16 scale/min with row-norm correction, int8 recent window) so
the kernels are checked for what they compute, not just that they run.
"""

import math

import pytest
import torch

from vllm.v1.attention.ops import oscar_kv as ok

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = torch.device("cuda")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


PAGE = 16
FORMATS = {"k3v3": (3, 3, 256), "k3v2": (3, 2, 1024), "k2v2": (2, 2, 1024)}


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(DEV)


def ref_quant(xr: torch.Tensor, clip: float, bits: int) -> torch.Tensor:
    d = xr.shape[-1]
    idx = min(d - 1, int(clip * d))
    thr = xr.abs().sort(-1).values[..., idx : idx + 1]
    xc = torch.maximum(torch.minimum(xr, thr), -thr)
    lv = 2**bits - 1
    mn, mx = xc.amin(-1, keepdim=True), xc.amax(-1, keepdim=True)
    sc = (mx - mn).clamp_min(1e-8) / lv
    q = ((xc - mn) / sc).round().clamp(0, lv)
    deq = q * sc + mn
    f = xr.norm(dim=-1, keepdim=True) / deq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return q * (sc * f).half().float() + (mn * f).half().float()


def ref_int8(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (x / s).round() * s


def ref_attention(q, kq, vq, kw, vw, L, recent, scale, sink=0):
    """Q (Hq, D); kq/vq quantized and kw/vw window values (N, Hk, D); all rotated."""
    g = q.shape[0] // kq.shape[1]
    pos = torch.arange(L, device=DEV)
    hi = ((pos >= L - recent) | (pos < sink))[:, None, None]
    k = torch.where(hi, kw[:L], kq[:L]).repeat_interleave(g, 1)
    v = torch.where(hi, vw[:L], vq[:L]).repeat_interleave(g, 1)
    p = torch.softmax(torch.einsum("hd,nhd->hn", q, k) * scale, -1)
    return torch.einsum("hn,nhd->hd", p, v)


def make_kv(n, hk, d, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(n, hk, d, device=DEV, generator=g)
    k[..., :: max(1, d // 16)] *= 6  # outlier channels, as in real keys
    v = torch.randn(n, hk, d, device=DEV, generator=g)
    return k.bfloat16(), v.bfloat16()


def rows(fmt, max_reqs, hk, free_bytes=1 << 40):
    return ok.window_rows(fmt, PAGE, max_reqs, 1, hk, free_bytes)


def new_cache(fmt, pages, hk, layout):
    if layout == "position":
        return torch.zeros(pages, PAGE, hk, fmt.slot, device=DEV, dtype=torch.uint8)
    return torch.zeros(pages, hk, 1, PAGE * fmt.slot, device=DEV, dtype=torch.uint8)


def setup(fmt, n, hk, layout, seed=0, reqs=1):
    d = fmt.head_dim
    k, v = make_kv(n, hk, d, seed)
    pages = math.ceil(n / PAGE)
    bt = (
        (torch.randperm(pages, device=DEV).int() + 1)[None]
        .expand(reqs, -1)
        .contiguous()
    )
    cache = new_cache(fmt, pages + 1, hk, layout)
    win = ok.OscarWindow(fmt, cache.shape[0], PAGE, rows(fmt, 8, hk), DEV)
    stage = win.alloc_stage(hk, DEV)
    return k, v, bt, cache, win, stage


def store_all(k, v, bt, cache, win, stage, lens):
    win.update(bt, torch.tensor(lens, device=DEV, dtype=torch.int32))
    pos = torch.arange(k.shape[0], device=DEV)
    slots = (bt[0][pos // PAGE] * PAGE + pos % PAGE).int()
    ok.oscar_store(k.contiguous(), v.contiguous(), cache, slots, win, stage)


def run_decode(q, cache, bt, lens, win, stage, scale):
    b, hq, d = q.shape
    specs = ok.decode_workspace(win.fmt, b, hq, stage[0].shape[1], q.dtype)
    mid, qr, out = (torch.empty(s, dtype=t, device=DEV) for s, t in specs)
    sl = torch.tensor(lens, device=DEV, dtype=torch.int32)
    return ok.oscar_decode(q, cache, bt, sl, win, stage, scale, mid, qr, out).float()


def reference(fmt, k, v):
    h = hadamard(fmt.head_dim)
    kr, vr = k.float() @ h, v.float() @ h
    kq = ref_quant(kr, ok.CLIP_K[fmt.kbits], fmt.kbits)
    vq = ref_quant(vr, ok.CLIP_V[fmt.vbits], fmt.vbits)
    return h, kq, vq, ref_int8(kr), ref_int8(vr)


@pytest.mark.parametrize("name", list(FORMATS))
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("layout", ["head", "position"])
def test_decode_matches_reference(name, head_dim, layout):
    kb, vb, recent = FORMATS[name]
    fmt = ok.OscarFormat(kb, vb, recent, head_dim)
    hq, hk = 12, 2
    lens = [1, 5, 63, 64, 65, recent - 1, recent, recent + 1, recent + 300, 3000]
    n = max(lens)
    k, v, bt, cache, win, stage = setup(fmt, n, hk, layout, reqs=len(lens))
    store_all(k, v, bt, cache, win, stage, lens)
    q = torch.randn(len(lens), hq, head_dim, device=DEV).bfloat16()
    scale = head_dim**-0.5
    out = run_decode(q, cache, bt, lens, win, stage, scale)
    h, kq, vq, kw, vw = reference(fmt, k, v)
    errs = []
    for i, L in enumerate(lens):
        exp = (
            ref_attention(q[i].float() @ h, kq, vq, kw, vw, L, recent, scale, fmt.sink)
            @ h
        )
        assert torch.isfinite(out[i]).all(), f"non-finite output at L={L}"
        errs.append(((out[i] - exp).norm() / exp.norm()).item())
    assert max(errs) < 1e-2, errs


@pytest.mark.parametrize("head_dim", [128, 256])
def test_dequant_uses_window_then_codes(head_dim):
    fmt = ok.OscarFormat(2, 2, 1024, head_dim)
    n, hk = 2500, 2
    k, v, bt, cache, win, stage = setup(fmt, n, hk, "position", seed=3)
    store_all(k, v, bt, cache, win, stage, [n])
    ko = torch.empty(hk, n, head_dim, device=DEV, dtype=torch.float16)
    vo = torch.empty_like(ko)
    ok.oscar_dequant(cache, bt[0], n, win, stage, ko, vo)
    h, kq, vq, kw, vw = reference(fmt, k, v)
    # pages of the sink and of the last 1024 tokens are staged; the rest are codes
    pos = torch.arange(n, device=DEV)
    win_tok = ((pos >= (n - 1024) // PAGE * PAGE) | (pos < fmt.sink))[:, None, None]
    exp_k = torch.where(win_tok, kw, kq) @ h
    rel = (ko.float().transpose(0, 1) - exp_k).flatten(1).norm(dim=1) / exp_k.flatten(
        1
    ).norm(dim=1)
    assert (rel < 1e-2).float().mean() > 0.999, rel.max()


@pytest.mark.parametrize("head_dim", [128, 256])
def test_stale_window_is_never_used(head_dim):
    """Overwrite the quantized cache without a store (KV offload restore / block
    reuse): the staged window still holds the old content and must be ignored."""
    fmt = ok.OscarFormat(3, 3, 256, head_dim)
    n, hq, hk = 1000, 12, 2
    k, v, bt, cache, win, stage = setup(fmt, n, hk, "head", seed=1)
    store_all(k, v, bt, cache, win, stage, [n])
    k2, v2 = make_kv(n, hk, head_dim, seed=2)
    other = torch.zeros_like(cache)
    win2 = ok.OscarWindow(fmt, cache.shape[0], PAGE, rows(fmt, 8, hk), DEV)
    store_all(k2, v2, bt, other, win2, win2.alloc_stage(hk, DEV), [n])
    cache.copy_(other)
    q = torch.randn(1, hq, head_dim, device=DEV).bfloat16()
    scale = head_dim**-0.5
    out = run_decode(q, cache, bt, [n], win, stage, scale)[0]
    h, kq, vq, _, _ = reference(fmt, k2, v2)
    exp = (
        ref_attention(q[0].float() @ h, kq, vq, kq, vq, n, 0, scale) @ h
    )  # all from codes
    assert ((out - exp).norm() / exp.norm()).item() < 1e-2


def test_window_rows_follow_the_memory_budget():
    fmt = ok.OscarFormat(2, 2, 1024, 128)
    ppw = ok.pages_per_window(fmt, PAGE)
    assert rows(fmt, 16, 8) == (ok.WINDOW_REQS + 2) * ppw  # plenty of memory
    assert rows(fmt, 2, 8) == 4 * ppw  # few sequences
    row_bytes = 8 * PAGE * (2 * 128 + 12)
    tight = rows(
        fmt, 16, 8, free_bytes=int(3 * ppw * row_bytes / ok.WINDOW_MEMORY_FRACTION)
    )
    assert tight == 3 * ppw
    assert rows(fmt, 16, 8, free_bytes=0) == ppw  # never below one window


def test_window_rows_are_capped_without_duplicates():
    fmt = ok.OscarFormat(2, 2, 1024, 256)
    win = ok.OscarWindow(fmt, 100_000, PAGE, rows(fmt, 16, 2), DEV)
    bt = torch.randperm(99_000, device=DEV)[: 16 * 200].view(16, 200).int()
    for step in range(3):
        win.update(bt, torch.full((16,), 3000 + step, device=DEV, dtype=torch.int32))
        used = win.row_owner[: win.rows]
        used = used[used >= 0]
        assert used.numel() == used.unique().numel()
        assert used.numel() <= win.rows
