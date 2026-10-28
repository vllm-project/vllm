# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 QKV+indexer MXFP8 GEMM with fused epilogue
(ops/qkv_index_fused_epilogue.py) vs the production chain: mxfp8_e4m3_quantize
-> flashinfer mm_mxfp8 -> fused_minimax_m3_qknorm_rope_kv_insert.

Each case runs both tile configs (1-SM and 2-SM), which must match bitwise,
and compares them to the chain: bf16 q within tolerances, the quantized
outputs (fp8 q / index_q, NVFP4 caches) code by code. The chain rounds the
GEMM output to bf16 first, so values near a rounding boundary land one code
apart.
"""

import pytest
import torch

import vllm._custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="QKV+indexer MXFP8 GEMM is SM100+ CUDA only",
)

HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64
EPS = 1e-6
MAX_POS = 8192
Q_FP8_SCALE = 2.0
# per-rank (q, kv, index_q) heads; N = (nq + 2 * nkv + niq + 1) * HEAD_DIM
TP_HEADS = {1: (64, 4, 4), 2: (32, 2, 2), 4: (16, 1, 1)}
# Caches start as 0xFF: never a scale byte (E4M3 scales saturate at 0x7E), so
# a stray or missing write shows up in the code comparison.
CACHE_FILL = 0xFF

BF16_RTOL = 2**-6
BF16_ATOL = 0.05
# fp8 values near zero (RoPE cancellation, where an E4M3 step is 2^-7..2^-9)
# can land a few codes apart but stay within 2^-6.
FP8_MAX_MISMATCH_FRAC = 0.05
FP8_SMALL_ABS = 2**-6
NVFP4_MAX_MISMATCH_FRAC = 0.02
NVFP4_SF_MAX_MISMATCH_FRAC = 0.05


def _make_inputs(nq, nkv, niq, M, nb, bs, slots):
    """MXFP8 activations / weight, norm weights, positions, the RoPE cache and
    the KV / index-cache slots: random unique with ~3% -1 (no insert), the
    last M of the cache, or all -1."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
        swizzle_mxfp8_scale,
    )

    # CPU RNG: the CUDA one's output depends on the GPU's SM count
    gen = torch.Generator().manual_seed(M)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).cuda()

    n = (nq + 2 * nkv + niq + 1) * HEAD_DIM
    # activations with a few outlier channels, like real hidden states
    x = randn(M, HIDDEN) * 0.5
    x[:, torch.randperm(HIDDEN, generator=gen)[:16].cuda()] *= 20
    x = x.bfloat16()
    w = (randn(n, HIDDEN) * 0.05).bfloat16()
    xq, xsf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    wq, wsf = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    wsf = swizzle_mxfp8_scale(wsf, M=n, K=HIDDEN)
    # Gemma norm weights (scale 1 + w) as in the checkpoint: q / k near 0,
    # index_q down to -1 (scale near 0), index_k up to ~2.5
    offset_std = torch.tensor(
        [[0.1, 0.1], [0.2, 0.1], [-0.4, 0.25], [0.9, 0.5]], device="cuda"
    )
    norm_weights = offset_std[:, :1] + offset_std[:, 1:] * randn(4, HEAD_DIM)
    norm_weights = norm_weights.bfloat16()
    positions = torch.randint(0, MAX_POS, (M,), generator=gen).cuda()
    inv_freq = 1.0 / (
        5e6 ** (torch.arange(0, ROTARY_DIM, 2, device="cuda").float() / ROTARY_DIM)
    )
    freqs = torch.outer(torch.arange(MAX_POS, device="cuda").float(), inv_freq)
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).bfloat16()

    if slots == "none":
        sm = ism = torch.full((M,), -1, device="cuda")
    elif slots == "last_page":
        sm = ism = torch.arange(nb * bs - M, nb * bs, device="cuda")
    else:
        sm, ism = (torch.randperm(nb * bs, generator=gen)[:M] for _ in range(2))
        sm[torch.rand(M, generator=gen) < 0.03] = -1
        ism[torch.rand(M, generator=gen) < 0.03] = -1
        sm, ism = sm.cuda(), ism.cuda()
    return xq, xsf, wq, wsf, norm_weights, positions, cos_sin, sm, ism


def _ordered(codes, sign_bit):
    """Sign-magnitude codes -> integers ordered like their values (+0 == -0),
    so adjacent representable values differ by 1."""
    codes = codes.to(torch.int16)
    mag = codes & (sign_bit - 1)
    return torch.where(codes & sign_bit != 0, -mag, mag)


def _check_fp8(name, got, want):
    """E4M3 codes: NaN masks exact, at most FP8_MAX_MISMATCH_FRAC differ, each
    within one code or FP8_SMALL_ABS of the chain's value."""
    a, b = got.float(), want.float()
    assert (a.isnan() == b.isnan()).all(), f"{name}: NaN masks differ"
    fin = ~a.isnan()
    d = _ordered(got.view(torch.uint8), 0x80) - _ordered(want.view(torch.uint8), 0x80)
    d = d.abs()[fin]
    mism = (d > 0).float().mean()
    assert mism <= FP8_MAX_MISMATCH_FRAC, f"{name}: {mism:.2%} E4M3 codes differ"
    far = (d > 1) & ((a - b).abs()[fin] > FP8_SMALL_ABS)
    assert not far.any(), f"{name}: {far.sum()} E4M3 codes off by > 1 code"


def _check_cache(name, got, want, slots, bs):
    """NVFP4 cache [nb, (nslots,) bs, 72] vs the chain's: pages without an
    inserted token are byte-identical; in the others, E2M1 codes and E4M3
    scale bytes are each within one code, and few differ. A scale one code
    apart re-codes its whole group, so E2M1 mismatches only count in groups
    with equal scales."""
    pages = torch.zeros(got.shape[0], dtype=torch.bool, device=got.device)
    pages[slots[slots >= 0] // bs] = True
    changed = (got != want).flatten(1).any(1)
    assert not (changed & ~pages).any(), f"{name}: writes to pages with no token"

    def split(c):
        """Each (page, slot) holds bs tokens' E2M1 data (8 groups of 8B), then
        their E4M3 group scales (8B; V's swizzled in 4-token quads) -> codes
        [rows, bs, 8, 16] and scales [rows, bs, 8]."""
        c = c[pages].reshape(-1, bs * 72)
        data = c[:, : bs * 64].reshape(-1, bs, 8, 8)
        codes = _ordered(torch.stack((data & 0xF, data >> 4), -1), 0x8)
        sf = c[:, bs * 64 :].reshape(-1, bs, 8).to(torch.int16)
        if got.dim() == 4:  # K/V cache: odd slots are V
            # byte (t // 4 * 4 + g // 2) * 8 + g % 2 * 4 + t % 4 -> [t, g]
            k, v = sf.reshape(-1, 2, bs, 8).unbind(1)
            v = v.reshape(-1, bs // 4, 4, 2, 4).permute(0, 1, 4, 2, 3)
            sf = torch.stack((k, v.reshape(-1, bs, 8)), 1).reshape(-1, bs, 8)
        return codes.flatten(-2), sf

    (g, g_sf), (w, w_sf) = split(got), split(want)
    d, d_sf = (g - w).abs(), (g_sf - w_sf).abs()
    assert (d <= 1).all() and (d_sf <= 1).all(), f"{name}: codes off by > 1 code"
    written = (g_sf != CACHE_FILL) | (w_sf != CACHE_FILL)
    for what, mism, frac in (
        ("E2M1 codes", d[written & (d_sf == 0)] > 0, NVFP4_MAX_MISMATCH_FRAC),
        ("scale bytes", d_sf[written] > 0, NVFP4_SF_MAX_MISMATCH_FRAC),
    ):
        if mism.numel() == 0:
            continue
        # 5 sigma of binomial slack: small M writes only a few groups
        rate = mism.float().mean().item()
        bound = frac + 5 * (frac / mism.numel()) ** 0.5
        assert rate <= bound, f"{name}: {rate:.2%} of written {what} differ"


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(dict(tp=1, M=1), id="tp1-m1"),
        pytest.param(dict(tp=1, M=37), id="tp1-m37"),
        pytest.param(dict(tp=1, M=1000), id="tp1-m1000"),
        pytest.param(dict(tp=1, M=4096, q="fp8"), id="tp1-m4096-fp8q"),
        pytest.param(dict(tp=1, M=32768, q="fp8"), id="tp1-m32768-fp8q"),
        pytest.param(dict(tp=2, M=1024), id="tp2-m1024"),
        pytest.param(dict(tp=4, M=1024, q="bf16"), id="tp4-m1024-bf16q"),
        pytest.param(dict(tp=4, M=4100, q="fp8"), id="tp4-m4100-fp8q"),
        pytest.param(dict(tp=4, M=1024, kv_scales=(0.5, 1.7)), id="tp4-kv-scales"),
        pytest.param(dict(tp=4, M=1024, slots="none"), id="tp4-no-insert"),
        pytest.param(dict(tp=1, M=1024, layers=3), id="tp1-cross-layer"),
        pytest.param(dict(tp=4, M=1024, block_size=64), id="tp4-bs64"),
        # > 2**31 bytes: writers in the last page exercise int64 byte offsets
        pytest.param(
            dict(tp=1, M=128, nb=30000, slots="last_page"), id="tp1-big-cache"
        ),
    ],
)
def test_qkv_index_fused_epilogue(case):
    import flashinfer

    from vllm.models.minimax_m3.nvidia.ops.qkv_index_fused_epilogue import (
        minimax_m3_qkv_index_fused_epilogue,
    )

    tp, M = case["tp"], case["M"]
    q_mode = case.get("q", "both")
    kv_scales = case.get("kv_scales", (1.0, 1.0))
    bs, layers = case.get("block_size", 128), case.get("layers", 1)
    nb = case.get("nb", (M + M // 2 + bs - 1) // bs + 1)
    nq, nkv, niq = TP_HEADS[tp]

    xq, xsf, wq, wsf, norm_weights, positions, cos_sin, sm, ism = _make_inputs(
        nq, nkv, niq, M, nb, bs, case.get("slots", "random")
    )

    def caches():
        # the last layer of [nb, layers, *page] buffers: page-strided if layers > 1
        kv = torch.full((nb, layers, 2 * nkv, bs, 72), CACHE_FILL, dtype=torch.uint8)
        ic = torch.full((nb, layers, bs, 72), CACHE_FILL, dtype=torch.uint8)
        return kv.cuda()[:, -1], ic.cuda()[:, -1]

    # the production chain
    kv_ref, ic_ref = caches()
    q_ref = torch.empty(M, nq * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    q8_ref = torch.empty(M, nq * HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    iq_ref = torch.empty(M, niq * HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    qkv = flashinfer.mm_mxfp8(
        xq, wq.t(), xsf, wsf, out_dtype=torch.bfloat16, backend="cute-dsl"
    )
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, norm_weights[0], norm_weights[1], cos_sin, positions, nq, nkv,
        ROTARY_DIM, EPS, norm_weights[2], norm_weights[3], niq, sm, ism,
        kv_ref, ic_ref, bs, q_ref, iq_ref, "nvfp4",
        q_fp8_out=q8_ref, q_fp8_scale=Q_FP8_SCALE,
        kv_k_scale=torch.tensor([kv_scales[0]], device="cuda"),
        kv_v_scale=torch.tensor([kv_scales[1]], device="cuda"),
    )  # fmt: skip

    outs = []
    for cta_group in (1, 2):
        kv, ic = caches()
        q = torch.empty_like(q_ref) if q_mode != "fp8" else None
        q8 = torch.empty_like(q8_ref) if q_mode != "bf16" else None
        iq = torch.empty_like(iq_ref)
        minimax_m3_qkv_index_fused_epilogue(
            xq, xsf, wq, wsf, positions, cos_sin, *norm_weights, EPS,
            slot_mapping=sm, index_slot_mapping=ism, kv_cache=kv, index_cache=ic,
            q_out=q, index_q_out=iq, q_fp8_out=q8, q_fp8_scale=Q_FP8_SCALE,
            k_scale=kv_scales[0], v_scale=kv_scales[1], cta_group=cta_group,
        )  # fmt: skip
        outs.append((q, q8, iq, kv, ic))

    # both tile configs write identical bytes
    for a, b in zip(*outs):
        if a is not None:
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    q, q8, iq, kv, ic = outs[0]
    if q is not None:
        torch.testing.assert_close(
            q, q_ref, rtol=BF16_RTOL, atol=BF16_ATOL, equal_nan=True
        )
    if q8 is not None:
        _check_fp8("q_fp8", q8, q8_ref)
    _check_fp8("index_q", iq, iq_ref)
    _check_cache("kv_cache", kv, kv_ref, sm, bs)
    _check_cache("index_cache", ic, ic_ref, ism, bs)
    # the other layers of a cross-layer buffer stay untouched
    for c in (kv, ic):
        storage = torch.empty(0, dtype=torch.uint8, device="cuda")
        storage.set_(c.untyped_storage())
        outside = torch.ones_like(storage, dtype=torch.bool)
        outside.as_strided(c.size(), c.stride(), c.storage_offset()).fill_(False)
        assert (storage[outside] == CACHE_FILL).all(), "writes outside the view"
