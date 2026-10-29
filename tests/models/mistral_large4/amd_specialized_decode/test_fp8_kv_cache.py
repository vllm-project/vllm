# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests of the Mistral Large 4 specialized decode on an fp8 latent KV cache.

One TP4 rank's shard of one MoE layer per kind runs on a single GPU through
``Glm5MonoKernel`` (npes=1), with random weights, a randomly paged cache and one
DSpark verify step of 4 rows whose last row is padding (slot -1: KV-sharing
layers used to load it ~4 GiB past the source cache, a GPU memory fault).

The cache holds the same content as bf16 and as fp8 (written by vLLM's fp8
cache writer, which fills it on every non-mono step). KV-sharing layers never
write the cache, so their outputs match exactly; the other layers must store
what vLLM's writer stores for the bf16 rows, saturating at +-448.
"""

import functools

import pytest
import torch

from vllm.platforms import current_platform

FP8 = torch.float8_e4m3fn
HIDDEN, NH, KVL, PE, VD, E = 6144, 16, 512, 64, 128, 128
QK = KVL + PE
PAGE, PAGES, S, POS0, TOPK = 64, 64, 4, 3000, 2048


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the ML4 mono kernels need gfx950 (MI355X)"
)


def _randn(*shape, scale=1.0):
    return torch.randn(*shape, device="cuda") * scale


@functools.cache
def build_layer(kind: str):
    """(kernel variant, LayerWeights) of one random layer, packed as
    ``runtime.weights.build_moe_layer`` packs a loaded one."""
    from types import SimpleNamespace as NS

    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.config import (  # noqa: E501
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )
    from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.layer import (  # noqa: E501
        LayerWeights,
    )
    from vllm.models.mistral_large4.amd.specialized_decode.runtime import weights
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.variants import (
        load_variant,
    )

    torch.manual_seed(0)
    q128 = weights._q128
    nope_layer, kv_share, _ = weights.KINDS[kind]
    v = load_variant(nope_layer, kv_share)
    C = v.geometry_config.glm5_tp_config(4)
    bf, f32 = torch.bfloat16, torch.float32
    t = {}
    t["w_q_b"], t["s_q_b"] = q128(_randn(NH * 128, HIDDEN, scale=HIDDEN**-0.5))
    wkva = _randn(KVL + PE, HIDDEN, scale=HIDDEN**-0.5)
    if nope_layer:
        wkva[KVL:] = 0
    t["w_qkv_a"], t["s_qkv_a"] = q128(wkva * (not kv_share))
    t["g_kv"] = (torch.rand(KVL, device="cuda") + 0.5).to(bf)
    wkvb = _randn(NH, C.nope_dim + VD, KVL, scale=KVL**-0.5)
    w_uk = wkvb[:, : C.nope_dim].transpose(1, 2).reshape(NH * KVL, C.nope_dim)
    t["w_uk"], t["s_uk"] = q128(w_uk, 64)
    t["w_uv"], t["s_uv"] = q128(wkvb[:, C.nope_dim :].reshape(NH * VD, KVL))
    t["w_o"], t["s_o"] = q128(_randn(HIDDEN, NH * VD, scale=(NH * VD) ** -0.5))
    t["g_in"] = t["g_post"] = torch.ones(HIDDEN, dtype=bf, device="cuda")
    t["w_r"] = _randn(E, HIDDEN, scale=HIDDEN**-0.5).to(bf)
    t["bias"] = torch.zeros(E, dtype=f32, device="cuda")
    gc, ih, idim = v.geometry_config, weights.INDEX_HEADS, weights.INDEX_DIM
    prm = torch.zeros(gc.ML4_P_WORDS, dtype=f32, device="cuda")
    prm[gc.ML4_P_SS_S : gc.ML4_P_SS_S + NH] = 1.0
    prm[gc.ML4_P_IQ_W : gc.ML4_P_IQ_W + HIDDEN] = 1.0
    prm[gc.ML4_P_ISS_S : gc.ML4_P_ISS_S + ih] = 1.0
    t["g_q"] = prm
    t["w_index_q"], t["s_index_q"] = q128(_randn(ih * idim, HIDDEN, scale=HIDDEN**-0.5))
    kw = _randn(idim + ih, HIDDEN, scale=HIDDEN**-0.5)
    t["w_index_k"], t["s_index_k"] = q128(kw[:idim])
    t["w_index_w"] = kw[idim:].to(bf).contiguous()
    t["g_index_k"] = torch.ones(idim, dtype=f32, device="cuda")
    t["b_index_k"] = torch.zeros(idim, dtype=f32, device="cuda")
    # routed experts: random MXFP4 values / E8M0 scales in vLLM's AITER a4w4
    # storage (the kernels' ATOM layout); their content is irrelevant here
    u8 = torch.uint8
    for w, s, rows, cols in (("w_ug", "s_ug", 2 * C.inter, HIDDEN),
                             ("w_dn", "s_dn", HIDDEN, C.inter)):  # fmt: skip
        values = torch.randint(0, 256, (E, rows, cols // 2), dtype=u8, device="cuda")
        scales = torch.randint(
            118, 122, (E * rows, cols // 32), dtype=u8, device="cuda"
        )
        t[s] = e8m0_shuffle(scales).view(u8)
        t[w] = values
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = rocm_aiter_ops.shuffle_weights(t["w_ug"].view(fp4), t["w_dn"].view(fp4))
    t["w_ug"], t["w_dn"] = w13.view(u8), w2.view(u8)
    t["w_ug"].is_shuffled = t["w_dn"].is_shuffled = True
    # shared expert: block fp8, re-quantized to MXFP4 as the runtime does
    linear = lambda n, k: NS(  # noqa: E731
        **dict(zip(("weight", "weight_scale_inv"), q128(_randn(n, k, scale=k**-0.5))))
    )
    shared = NS(
        gate_up_proj=linear(2 * C.inter, HIDDEN), down_proj=linear(HIDDEN, C.inter)
    )
    t.update(weights._shared_expert_mxfp4(shared))
    W = LayerWeights(
        NH, t, C, 0, 1,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=E,
    )  # fmt: skip
    return v, W


def vllm_fp8_rows(rows: torch.Tensor) -> torch.Tensor:
    """[n, 576] latent rows as vLLM's fp8 cache writer stores them (scale 1)."""
    from vllm import _custom_ops as ops

    rows, n = rows.to(torch.bfloat16), rows.shape[0]
    cache = torch.zeros(n, 1, QK, dtype=torch.uint8, device=rows.device)
    slots, one = torch.arange(n, device=rows.device), torch.ones(1, device=rows.device)
    ops.concat_and_cache_mla(rows[:, :KVL], rows[:, KVL:], cache, slots, "fp8", one)
    return cache.view(n, QK).view(FP8)


def make_step():
    """Cache content (e4m3 values), fp8 index cache, one 4-row step whose last
    row is padding."""
    torch.manual_seed(1)
    base = vllm_fp8_rows(_randn(PAGES * PAGE, QK, scale=0.5))
    keys = _randn(PAGES, PAGE * 128, scale=0.5).to(FP8).view(torch.uint8)
    scales = torch.ones(PAGES, PAGE, device="cuda").view(torch.uint8)
    index_cache = torch.cat([keys, scales], 1).view(PAGES, PAGE, 132).contiguous()
    pages = torch.randperm(PAGES, device="cuda", dtype=torch.int32)
    positions = torch.arange(POS0, POS0 + S, device="cuda")
    slots = pages[positions // PAGE].long() * PAGE + positions % PAGE
    slots[-1] = -1  # vLLM's PAD_SLOT_ID
    nkeys = torch.where(slots >= 0, torch.clamp(positions + 1, max=TOPK), 0)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    freqs = 10000 ** (-torch.arange(32, device="cuda") / 32)
    ang = torch.arange(PAGES * PAGE, device="cuda")[:, None] * freqs[None]
    step = dict(
        h=_randn(S, HIDDEN).to(torch.bfloat16),
        cur_pos=torch.tensor([POS0], dtype=torch.int32, device="cuda"),
        cos=torch.cos(ang).contiguous(), sin=torch.sin(ang).contiguous(),
        block_tables=pages[None].contiguous(), positions=positions,
        slots=slots, indptr=indptr,
    )  # fmt: skip
    return base, index_cache, step


def run_layer(kind, kv_cache_dtype, kv, index_cache, step, W=None):
    v, W0 = build_layer(kind)
    kernel = v.op.Glm5MonoKernel(
        W or W0, S, rank=0, npes=1, group=None, topk=TOPK, launches_per_step=1,
        with_indexer=True, index_max_seq=PAGES * PAGE, index_request_width=S,
        index_page_size=PAGE, index_block_table_stride=PAGES,
        index_cache_dtype="fp8", attention_weight="fp8_block128",
        kv_cache_layout="atom", kv_cache_dtype=kv_cache_dtype,
        native_fp4_mfma=True, rope_dtype="fp32", separate_shared=True,
    )  # fmt: skip
    try:
        out = kernel.forward(
            step["h"], step["cur_pos"], kv, kv, None, step["cos"], step["sin"],
            index_cache=index_cache.clone(), block_tables=step["block_tables"],
            positions=step["positions"], slot_mapping=step["slots"],
            sparse_kv_indptr=step["indptr"],
        )[: S - 1]  # fmt: skip
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        return out.clone()
    finally:
        kernel.close()


def run_both(kind, history_scale=1, W=None):
    """The step on the same cache content as bf16 and as fp8:
    (out_bf16, out_fp8, cache_bf16, cache_fp8, cache content, written slots)."""
    base, index_cache, step = make_step()
    base = (base.float() * history_scale).to(FP8)  # power of two: exact
    kv_bf, kv_f8 = base.to(torch.bfloat16), base.clone()
    out_bf = run_layer(kind, "bf16", kv_bf, index_cache, step, W)
    out_f8 = run_layer(kind, "fp8", kv_f8, index_cache, step, W)
    return out_bf, out_f8, kv_bf, kv_f8, base, step["slots"][:-1]


@pytest.mark.parametrize("kind", ("rope", "nope", "nope_kvs"))
# history_scale 4: the old rows dominate attention, so the new rows' fp8
# rounding does not show and every kind matches bf16 exactly
@pytest.mark.parametrize("history_scale", (1, 4))
@torch.inference_mode()
def test_fp8_cache_matches_bf16(kind, history_scale):
    out_bf, out_f8, kv_bf, kv_f8, base, slots = run_both(kind, history_scale)
    written = torch.zeros(PAGES * PAGE, dtype=torch.bool, device="cuda")
    written[slots] = kind != "nope_kvs"  # KV-sharing layers never write
    u8 = torch.uint8
    assert torch.equal(kv_f8[~written].view(u8), base[~written].view(u8))
    if kind != "nope_kvs":
        new = vllm_fp8_rows(kv_bf[written])
        assert torch.equal(kv_f8[written].view(u8), new.view(u8))
    if kind == "nope_kvs" or history_scale > 1:
        assert torch.equal(out_f8, out_bf)
    else:  # attends to this step's rows rounded to fp8
        rel = (out_f8.float() - out_bf.float()).norm() / out_bf.float().norm()
        assert rel < 0.1, rel


@pytest.mark.parametrize("kind", ("rope", "nope"))
@torch.inference_mode()
def test_fp8_cache_saturates(kind):
    """Latent and rope key values beyond +-448 saturate, as vLLM's writer does,
    instead of becoming NaN."""
    from dataclasses import replace

    _, W = build_layer(kind)
    t = dict(W.t, g_kv=W.t["g_kv"] * 1000, s_qkv_a=W.t["s_qkv_a"] * 1000)
    _, _, kv_bf, kv_f8, _, slots = run_both(kind, W=replace(W, t=t))
    new_bf, new_f8 = kv_bf[slots], kv_f8[slots]
    assert (new_bf[:, :KVL].float().abs() > 448).any()
    assert (new_bf[:, KVL:].float().abs() > 448).any() == (kind == "rope")
    assert not torch.isnan(new_f8.float()).any()
    assert torch.equal(
        new_f8.view(torch.uint8), vllm_fp8_rows(new_bf).view(torch.uint8)
    )
