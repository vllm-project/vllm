# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the horizontally-fused deepseek_v32 (NVIDIA SM100) Triton
kernels used by the specialized DSA model:

  fused_norm_rope
    - q  : q_lora RMSNorm
    - kv : kv_lora RMSNorm + (interleaved) RoPE on k_pe + MLA cache insert
           (bf16 or per-tensor fp8)
    - idx: indexer-K LayerNorm + RoPE (interleaved or NeoX) + UE8M0 fp8 quant +
           packed indexer cache insert; plus the top-k buffer (-1) fill
  fused_q
    - mqa: ql_nope + (interleaved) RoPE'd q_pe, concat-quantized to the fp8 MQA
           query
    - idx: indexer-Q RoPE (interleaved or NeoX) + UE8M0 fp8 quant + folded
           index weights
  fused_eh_norm (MTP): zero-at-pos-0 + enorm RMSNorm(embeds) + hnorm
           RMSNorm(prev), concatenated side-by-side

Each kernel is compared against a PyTorch reference. The kernel keeps the whole
pipeline in fp32 and rounds once, so it can land on the opposite side of a
round-to-nearest tie from the reference for a few elements: deterministic fp8
outputs are checked within 1 representable-step (ULP); bf16 norm/RoPE outputs use
rtol/atol=1e-2 (the tolerance the sibling deepseek_v4 fused-kernel test uses).
"""

from typing import NamedTuple

import pytest
import torch

from vllm.models.deepseek_v32.common import kernels as K
from vllm.platforms import current_platform

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

EPS = 1e-6


class ModelConfig(NamedTuple):
    """Per-model attention dims (num_heads is the TP8 local count)."""

    q_lora: int
    kv_lora: int
    rope_dim: int
    num_heads: int
    index_heads: int
    index_head_dim: int
    hidden: int


MODEL_CONFIGS = [
    ModelConfig(2048, 512, 64, 8, 32, 128, 6144),  # GLM-5.2
    ModelConfig(1536, 512, 64, 16, 64, 128, 7168),  # DeepSeek-V3.2
]
MODEL_IDS = ["glm5.2", "dsv3.2"]

pytestmark = pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="deepseek_v32 fused kernels require fp8 support",
)


# ── reference helpers ────────────────────────────────────────────────────────


def make_cos_sin(max_pos: int, rot_dim: int, device) -> torch.Tensor:
    """cos||sin cache: row[pos] = [cos(theta)(rot/2), sin(theta)(rot/2)]."""
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RMSNorm matching kernels._rms_norm (fp32, eps inside rsqrt). Returns fp32."""
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(ms + EPS) * w.float()


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + EPS) * w.float() + b.float()


def rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    """Apply RoPE to the first ``rot_dim`` elements of x's last dim.

    x: [..., head_dim] fp32. ``cos_sin`` is [max_pos, rot_dim]. ``interleave``
    selects adjacent-pair (GLM) vs split-half NeoX (DeepSeek-V3.2) layout.
    """
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()]
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def ue8m0_quant(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last dim) UE8M0 fp8 quant matching kernels._fp8_ue8m0_quantize."""
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def _bf16_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)

    return int((key(a) - key(b)).abs().max().item())


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def assert_bf16(got: torch.Tensor, ref_fp32: torch.Tensor, msg: str):
    # Kernel keeps RMSNorm/RoPE in fp32 and rounds to bf16 once; the fp32
    # reduction/FMA order differs from torch, so a few elements land on the
    # opposite side of a round-to-nearest tie. Use the same tolerance the
    # sibling deepseek_v4 fused-kernel test uses for this bf16 norm+rope class.
    torch.testing.assert_close(
        got.float(), ref_fp32.float(), rtol=1e-2, atol=1e-2, msg=lambda m: f"{msg}: {m}"
    )


def assert_fp8(got: torch.Tensor, ref: torch.Tensor, msg: str):
    assert _fp8_ulp(got, ref) <= 1, f"{msg}: >1 fp8 ULP"


# ── fused_norm_rope ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512, 4096])
@pytest.mark.parametrize("index_interleave", [True, False])
@pytest.mark.parametrize("mla_fp8", [False, True])
def test_fused_norm_rope(
    num_tokens: int, index_interleave: bool, mla_fp8: bool, cfg: ModelConfig
):
    torch.manual_seed(0)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    ik = torch.randn(num_tokens, cfg.index_head_dim, device=dev, dtype=torch.bfloat16)
    ikw = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)
    ikb = torch.randn(cfg.index_head_dim, device=dev, dtype=torch.float32)

    mla_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)  # MLA k_pe: interleaved
    idx_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    bs = max_pos  # single block covering all tokens
    mla_dim = cfg.kv_lora + cfg.rope_dim
    if mla_fp8:
        mla_cache = torch.zeros(1, bs, mla_dim, device=dev, dtype=torch.uint8)
        mla_dtype = "fp8"
        mla_k_scale = torch.tensor([0.3], device=dev, dtype=torch.float32)
    else:
        mla_cache = torch.zeros(1, bs, mla_dim, device=dev, dtype=torch.bfloat16)
        mla_dtype = "auto"
        mla_k_scale = None
    idx_row = cfg.index_head_dim + cfg.index_head_dim // 128 * 4  # 132
    idx_cache = torch.zeros(1, bs, idx_row, device=dev, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        ik,
        ikw,
        ikb,
        EPS,
        idx_cos_sin,
        topk,
        slot_mapping=slot,
        indexer_k_cache=idx_cache,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype=mla_dtype,
        mla_k_scale=mla_k_scale,
        has_indexer=True,
        index_rope_interleave=index_interleave,
    )

    # q_lora RMSNorm
    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm")

    # MLA cache: [kv_c_normed | k_pe_roped(interleaved)]
    kv_ref = rms_norm(kv_c, kvw)
    kpe_ref = rope(k_pe.float(), pos, mla_cos_sin, interleave=True)
    if mla_fp8:
        cache = mla_cache.view(FP8)[0, :num_tokens]
        s = mla_k_scale.item()
        assert_fp8(cache[:, : cfg.kv_lora], (kv_ref / s).to(FP8), "MLA kv fp8")
        assert_fp8(cache[:, cfg.kv_lora :], (kpe_ref / s).to(FP8), "MLA k_pe fp8")
    else:
        cache = mla_cache[0, :num_tokens]
        assert_bf16(cache[:, : cfg.kv_lora], kv_ref, "MLA kv bf16")
        assert_bf16(cache[:, cfg.kv_lora :], kpe_ref, "MLA k_pe bf16")

    # Indexer-K cache (packed [bs*head_dim fp8 | bs*4 fp32 scale]).
    ik_ref = layer_norm(ik, ikw, ikb)
    ik_ref = rope(ik_ref, pos, idx_cos_sin, interleave=index_interleave)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    flat = idx_cache[0].reshape(-1)
    vals = flat[: bs * cfg.index_head_dim].view(FP8).reshape(bs, cfg.index_head_dim)
    scales = flat[bs * cfg.index_head_dim :].view(torch.float32)
    assert_fp8(vals[:num_tokens], q_ref, "indexer-K fp8")
    torch.testing.assert_close(scales[:num_tokens], s_ref, rtol=0, atol=0)

    # Top-k buffer cleared to -1 on indexer layers.
    assert (topk == -1).all(), "topk buffer not cleared on indexer layer"


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_fused_norm_rope_no_indexer(num_tokens: int, cfg: ModelConfig):
    """Shared (no-indexer) layer: q + kv/MLA only; top-k buffer untouched."""
    torch.manual_seed(1)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)

    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    mla_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    bs = max_pos
    mla_cache = torch.zeros(
        1, bs, cfg.kv_lora + cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        None,
        None,
        None,
        EPS,
        None,
        topk,
        slot_mapping=slot,
        indexer_k_cache=None,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="auto",
        mla_k_scale=None,
        has_indexer=False,
        index_rope_interleave=False,
    )

    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm (no-indexer)")
    cache = mla_cache[0, :num_tokens]
    assert_bf16(cache[:, : cfg.kv_lora], rms_norm(kv_c, kvw), "MLA kv (no-indexer)")
    assert_bf16(
        cache[:, cfg.kv_lora :],
        rope(k_pe.float(), pos, mla_cos_sin, interleave=True),
        "MLA k_pe (no-indexer)",
    )
    # Shared layers reuse the previous indexer's top-k: buffer must be untouched.
    assert (topk == 7).all(), "topk buffer should be untouched on shared layer"


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512])
def test_fused_norm_rope_ds_mla(num_tokens: int, cfg: ModelConfig):
    """fp8_ds_mla MLA cache layout (FlashMLA sparse, bf16-query path; SM90/SM100).

    Per-token 656-byte entry: 512 fp8 NoPE (4 per-128 tiles, dynamic float32
    scale) | 4 float32 scales | 64 bf16 (unquantized) RoPE.
    """
    torch.manual_seed(5)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_c = torch.randn(num_tokens, cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(num_tokens, cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(num_tokens, cfg.rope_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(cfg.q_lora, device=dev, dtype=torch.bfloat16)
    kvw = torch.randn(cfg.kv_lora, device=dev, dtype=torch.bfloat16)
    mla_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    bs = max_pos
    mla_cache = torch.zeros(1, bs, 656, device=dev, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    topk = torch.full((num_tokens, 2048), 7, device=dev, dtype=torch.int32)

    q_out = K.fused_norm_rope(
        pos,
        q_c,
        qw,
        EPS,
        kv_c,
        kvw,
        EPS,
        k_pe,
        mla_cos_sin,
        None,
        None,
        None,
        EPS,
        None,
        topk,
        slot_mapping=slot,
        indexer_k_cache=None,
        mla_kv_cache=mla_cache,
        mla_kv_cache_dtype="fp8_ds_mla",
        mla_k_scale=None,
        has_indexer=False,
        index_rope_interleave=False,
    )

    assert_bf16(q_out, rms_norm(q_c, qw), "q_c rmsnorm (ds_mla)")

    kv_ref = rms_norm(kv_c, kvw)  # [N, 512] fp32
    kpe_ref = rope(k_pe.float(), pos, mla_cos_sin, interleave=True)  # [N, 64]
    tiles = kv_ref.view(num_tokens, 4, 128)
    ref_scale = torch.clamp(tiles.abs().amax(dim=-1) / FP8_MAX, min=1.1754944e-38)
    ref_nope = (tiles / ref_scale[..., None]).reshape(num_tokens, cfg.kv_lora).to(FP8)

    cache = mla_cache[0, :num_tokens]  # [N, 656] uint8
    nope = cache[:, : cfg.kv_lora].view(FP8)
    scales = cache.view(torch.float32)[:, cfg.kv_lora // 4 : cfg.kv_lora // 4 + 4]
    rope_off = cfg.kv_lora // 2 + 8
    rope_vals = cache.view(torch.bfloat16)[:, rope_off : rope_off + cfg.rope_dim]

    torch.testing.assert_close(scales, ref_scale, rtol=1e-2, atol=1e-6)
    assert_fp8(nope, ref_nope, "ds_mla NoPE fp8")
    assert_bf16(rope_vals, kpe_ref, "ds_mla RoPE bf16")
    # No indexer on this call: top-k buffer must be untouched.
    assert (topk == 7).all(), "topk buffer should be untouched (no indexer)"


# ── fused_q ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512, 4096])
@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_q(num_tokens: int, index_interleave: bool, cfg: ModelConfig):
    torch.manual_seed(2)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_pe = torch.randn(
        num_tokens, cfg.num_heads, cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    index_q = torch.randn(
        num_tokens,
        cfg.index_heads,
        cfg.index_head_dim,
        device=dev,
        dtype=torch.bfloat16,
    )
    index_w = torch.randn(num_tokens, cfg.index_heads, device=dev, dtype=torch.float32)
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    softmax_scale = cfg.index_head_dim**-0.5
    head_scale = cfg.index_heads**-0.5
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)  # q_pe: interleaved
    idx_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    iq_fp8, iw_out, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        ql_nope,
        q_scale,
        index_w,
        softmax_scale,
        head_scale,
        has_indexer=True,
        index_rope_interleave=index_interleave,
    )

    s = q_scale.item()
    # MQA query: [ql_nope | q_pe RoPE'd (interleaved)], per-tensor fp8.
    mqa_nope_ref = (ql_nope.float() / s).to(FP8)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    mqa_pe_ref = (qpe_ref / s).to(FP8)
    assert_fp8(mqa[:, :, : cfg.kv_lora], mqa_nope_ref, "mqa ql_nope")
    assert_fp8(mqa[:, :, cfg.kv_lora :], mqa_pe_ref, "mqa q_pe")

    # Indexer-Q: RoPE + UE8M0 fp8 quant; index weights fold in q-scale.
    iq_ref = rope(
        index_q.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.index_heads),
        idx_cos_sin,
        interleave=index_interleave,
    )
    q_ref, scale_ref = ue8m0_quant(iq_ref)
    assert_fp8(iq_fp8, q_ref, "indexer-Q fp8")
    iw_ref = index_w * scale_ref * softmax_scale * head_scale
    torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_fused_q_no_indexer(num_tokens: int, cfg: ModelConfig):
    torch.manual_seed(3)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_pe = torch.randn(
        num_tokens, cfg.num_heads, cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.5], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    _, _, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        None,
        None,
        ql_nope,
        q_scale,
        None,
        0.0,
        0.0,
        has_indexer=False,
        index_rope_interleave=False,
    )
    s = q_scale.item()
    assert_fp8(mqa[:, :, : cfg.kv_lora], (ql_nope.float() / s).to(FP8), "mqa ql_nope")
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    assert_fp8(mqa[:, :, cfg.kv_lora :], (qpe_ref / s).to(FP8), "mqa q_pe")


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
@pytest.mark.parametrize("has_indexer", [True, False])
def test_fused_q_bf16_query(num_tokens: int, has_indexer: bool, cfg: ModelConfig):
    """bf16-query path (FlashMLA sparse, SM90/SM100): only the RoPE'd q_pe is
    produced (bf16, unquantized); ql_nope is consumed directly by the caller."""
    torch.manual_seed(6)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_pe = torch.randn(
        num_tokens, cfg.num_heads, cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    index_q = index_w = idx_cos_sin = None
    if has_indexer:
        index_q = torch.randn(
            num_tokens,
            cfg.index_heads,
            cfg.index_head_dim,
            device=dev,
            dtype=torch.bfloat16,
        )
        index_w = torch.randn(
            num_tokens, cfg.index_heads, device=dev, dtype=torch.float32
        )
        idx_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    iq_fp8, iw_out, q_pe_out = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        ql_nope,
        q_scale,
        index_w,
        cfg.index_head_dim**-0.5,
        cfg.index_heads**-0.5,
        has_indexer=has_indexer,
        index_rope_interleave=False,
        quantize_mqa=False,
    )

    # MQA query: only the RoPE'd q_pe, bf16, unquantized.
    assert q_pe_out.dtype == torch.bfloat16
    assert q_pe_out.shape == (num_tokens, cfg.num_heads, cfg.rope_dim)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    assert_bf16(q_pe_out, qpe_ref, "bf16 q_pe RoPE")

    # Indexer-Q is unchanged on this path (still UE8M0 fp8 + folded weights).
    if has_indexer:
        assert index_q is not None
        iq_ref = rope(
            index_q.float(),
            pos.unsqueeze(-1).expand(num_tokens, cfg.index_heads),
            idx_cos_sin,
            interleave=False,
        )
        q_ref, scale_ref = ue8m0_quant(iq_ref)
        assert_fp8(iq_fp8, q_ref, "indexer-Q fp8 (bf16-query path)")
        iw_ref = (
            index_w * scale_ref * (cfg.index_head_dim**-0.5) * (cfg.index_heads**-0.5)
        )
        torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


# ── fused_eh_norm (MTP) ──────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512])
def test_fused_eh_norm(num_tokens: int, cfg: ModelConfig):
    torch.manual_seed(4)
    dev = "cuda"
    # Mix in a position-0 token to exercise the embeds-zeroing branch.
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    pos[0] = 0
    embeds = torch.randn(num_tokens, cfg.hidden, device=dev, dtype=torch.bfloat16)
    prev = torch.randn(num_tokens, cfg.hidden, device=dev, dtype=torch.bfloat16)
    ew = torch.randn(cfg.hidden, device=dev, dtype=torch.bfloat16)
    hw = torch.randn(cfg.hidden, device=dev, dtype=torch.bfloat16)

    out = K.fused_eh_norm(pos, embeds, prev, ew, hw, EPS)

    masked = torch.where(pos.unsqueeze(-1) == 0, torch.zeros_like(embeds), embeds)
    ref = torch.cat([rms_norm(masked, ew), rms_norm(prev, hw)], dim=-1)
    assert out.shape == (num_tokens, 2 * cfg.hidden)
    assert_bf16(out, ref, "eh_norm")
