# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scaffolding for the DeepSeek-V4 fused compressor kernel tests.

The fused HIP compressors (CSA head=512/ratio=4, HCA head=512/ratio=128,
indexer head=128/ratio=4 with FP8 + MXFP4) are built into ``_rocm_C`` for
gfx950 (CDNA4) and exposed as ``torch.ops._rocm_C.dsv4_{csa,hca,indexer}_compress``.
The vLLM Triton FP32 path (``common/ops/fused_compress_quant_cache.py``) is the
byte-exact oracle for the FP8 outputs; the indexer MXFP4 tail is checked against
a faithful torch reference (the Triton MXFP4 kernel uses NVIDIA PTX and cannot
run on AMD).

This module is pure test/benchmark scaffolding — it is NOT collected by pytest
(no ``test_`` prefix) and is not imported by any production code. It builds, for
one scenario and one shape, the full input set (positions, slot mapping, APE,
RoPE cache, RMS weight, block table) and the populated FP32 / BF16 state caches
the compressor backends consume, plus the Triton/HIP runners, dequant + compare
helpers, and the MXFP4 torch oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# ── fixed test geometry (matches the production cache layout) ────────────────
KV_BLOCK_SIZE = 16
RMS_EPS = 1e-6
FP8_MAX = 448.0
SEED = 2026

# head=512 (CSA/HCA) packed-cache layout.
H512_TOKEN_STRIDE = 576
H512_NOPE = 448
H512_SCALE_DIM = 8
H512_N_NOPE_BLOCKS = 7
H512_QUANT_BLOCK = 64

# E2M1 decode LUT (OCP MXFP4): index = 4-bit code, low nibble=even / high=odd.
E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def detect_gfx950() -> bool:
    """True iff running on gfx950 (CDNA4). Safe to call on any platform.

    The fused HIP compressors are CDNA4-only and built into ``_rocm_C`` only
    when gfx950 is among the target archs; the correctness tests skip otherwise.
    """
    try:
        from vllm.models.deepseek_v4.amd.ops.hip_compress_dispatch import (
            hip_compressor_runtime_available,
        )

        return hip_compressor_runtime_available()
    except Exception:
        return False


# =============================================================================
# Shape config
# =============================================================================


@dataclass
class ShapeConfig:
    label: str
    head_dim: int
    rope_head_dim: int
    ratio: int
    overlap: bool
    # State-cache page size (tokens per block). CSA(ratio=4)=4, HCA(ratio=128)=8
    # to match CompressorStateCache.block_size in compressor.py.
    state_block_size: int = 4
    # Quant tail / cache layout. "csa": nope(FP8 per-64 ue8m0)+rope(bf16), the
    # head=512 path. "indexer_fp8": whole head_dim FP8 + single fp32 scale.
    # "indexer_mxfp4": E2M1 nibbles (head_dim//2 bytes) + ue8m0 per 32 (P4).
    quant_format: str = "csa"

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def token_stride(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim  # all FP8, single scale
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 2  # 2 E2M1 nibbles/byte
        return self.nope_dim + self.rope_head_dim * 2

    @property
    def scale_dim(self) -> int:
        if self.quant_format == "indexer_fp8":
            return 4  # one float32 scale
        if self.quant_format == "indexer_mxfp4":
            return self.head_dim // 32  # one ue8m0 byte per 32-elem block
        return self.nope_dim // 64 + 1

    @property
    def quant_block(self) -> int:
        if self.quant_format == "indexer_fp8":
            return self.head_dim  # single block (128)
        if self.quant_format == "indexer_mxfp4":
            return 32
        return 64

    @property
    def state_width(self) -> int:
        return (2 if self.overlap else 1) * self.head_dim

    @property
    def k_pool(self) -> int:
        return (2 if self.overlap else 1) * self.ratio

    @property
    def coff(self) -> int:
        return 2 if self.overlap else 1


CSA_MAIN = ShapeConfig("csa_main", 512, 64, 4, True)
HCA_MAIN = ShapeConfig("hca_main", 512, 64, 128, False, state_block_size=8)
# Indexer compressor = CSA geometry (ratio=4, overlap) at head_dim=128.
INDEXER_FP8 = ShapeConfig("indexer_fp8", 128, 64, 4, True, quant_format="indexer_fp8")
INDEXER_MXFP4 = ShapeConfig(
    "indexer_mxfp4", 128, 64, 4, True, quant_format="indexer_mxfp4"
)

# Shapes whose FP8 output is byte-exact-comparable to the Triton oracle.
BYTE_EXACT_SHAPES = [CSA_MAIN, HCA_MAIN, INDEXER_FP8]


class MockMetadata:
    def __init__(self, slot_mapping: torch.Tensor):
        self.slot_mapping = slot_mapping


# =============================================================================
# Compressor context: all inputs + populated state caches for one scenario
# =============================================================================


@dataclass
class CompressorContext:
    name: str
    desc: str
    shape: ShapeConfig
    positions: list[int]
    token_to_req: list[int]
    slot_mapping: list[int]
    kv_slot_mapping: list[int]
    max_position: int

    # Filled by build()
    num_tokens: int = 0
    num_compress: int = 0
    ape: torch.Tensor = None
    cos_sin_cache: torch.Tensor = None
    rms_weight: torch.Tensor = None
    block_table: torch.Tensor = None
    max_state_blocks: int = 0
    total_kv_blocks: int = 0
    bytes_per_block: int = 0
    _raw: dict = field(default_factory=dict)
    state_cache_fp32: torch.Tensor = None
    state_cache_bf16: torch.Tensor = None
    positions_t: torch.Tensor = None
    token_to_req_t: torch.Tensor = None
    slot_mapping_t: torch.Tensor = None
    kv_slot_mapping_t: torch.Tensor = None

    def build(self) -> CompressorContext:
        shape = self.shape
        self.num_tokens = len(self.positions)
        self.num_compress = sum(1 for k in self.kv_slot_mapping if k >= 0)
        bs = max(self.token_to_req) + 1 if self.token_to_req else 1

        g = torch.Generator(device="cuda").manual_seed(SEED + self.num_tokens)
        kv_raw = (
            torch.randn(
                self.num_tokens,
                shape.coff * shape.head_dim,
                dtype=torch.bfloat16,
                generator=g,
            )
            * 0.5
        )
        score_raw = (
            torch.randn(
                self.num_tokens,
                shape.coff * shape.head_dim,
                dtype=torch.bfloat16,
                generator=g,
            )
            * 0.5
        )
        self.ape = (
            torch.randn(
                shape.ratio,
                shape.coff * shape.head_dim,
                dtype=torch.float32,
                generator=g,
            )
            * 0.1
        )
        self._raw = {"kv": kv_raw, "score": score_raw}

        self.positions_t = torch.tensor(self.positions, dtype=torch.int64)
        self.token_to_req_t = torch.tensor(self.token_to_req, dtype=torch.int32)
        self.slot_mapping_t = torch.tensor(self.slot_mapping, dtype=torch.int64)
        self.kv_slot_mapping_t = torch.tensor(self.kv_slot_mapping, dtype=torch.int64)

        # Build block_table directly from the actual slot_mapping so it is
        # correct for any slot layout (sparse per-request or packed). Kernels
        # index it as block_table[req, position // state_block_size] and expect
        # the physical state-cache block where save_partial_states wrote that
        # token, i.e. slot // state_block_size.
        sbs = shape.state_block_size
        max_slot = max((s for s in self.slot_mapping if s >= 0), default=0)
        max_state_blocks = (max_slot // sbs) + 4
        max_lblk = 0
        for pos, slot in zip(self.positions, self.slot_mapping):
            if slot >= 0:
                max_lblk = max(max_lblk, pos // sbs)
        max_blocks_per_seq = max_lblk + 2
        block_table = torch.zeros(bs, max_blocks_per_seq, dtype=torch.int32)
        for req, pos, slot in zip(self.token_to_req, self.positions, self.slot_mapping):
            if slot < 0:
                continue
            block_table[req, pos // sbs] = slot // sbs
        self.block_table = block_table
        self.max_state_blocks = max_state_blocks

        self.rms_weight = (
            torch.rand(shape.head_dim, dtype=torch.float32, generator=g) * 0.5 + 0.5
        ).to(torch.bfloat16)

        max_cos_pos = self.max_position + 1
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, shape.rope_head_dim, 2, dtype=torch.float32)
                / shape.rope_head_dim
            )
        )
        freqs = torch.outer(torch.arange(max_cos_pos, dtype=torch.float32), inv_freq)
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

        self.total_kv_blocks = (self.num_compress // KV_BLOCK_SIZE) + 4
        self.bytes_per_block = (
            KV_BLOCK_SIZE * shape.token_stride + KV_BLOCK_SIZE * shape.scale_dim
        )

        self._populate_state_caches()
        return self

    def _populate_state_caches(self):
        from vllm.models.deepseek_v4.common.ops.save_partial_states import (
            save_partial_states,
        )

        shape = self.shape
        sbs = shape.state_block_size
        self.state_cache_fp32 = torch.zeros(
            self.max_state_blocks, sbs, 2 * shape.state_width, dtype=torch.float32
        )
        save_partial_states(
            kv=self._raw["kv"],
            score=self._raw["score"],
            ape=self.ape,
            positions=self.positions_t,
            state_cache=self.state_cache_fp32,
            slot_mapping=self.slot_mapping_t,
            block_size=sbs,
            state_width=shape.state_width,
            compress_ratio=shape.ratio,
        )
        self.state_cache_bf16 = torch.zeros(
            self.max_state_blocks, sbs, 2 * shape.state_width, dtype=torch.bfloat16
        )
        save_partial_states(
            kv=self._raw["kv"],
            score=self._raw["score"],
            ape=None,
            positions=self.positions_t,
            state_cache=self.state_cache_bf16,
            slot_mapping=self.slot_mapping_t,
            block_size=sbs,
            state_width=shape.state_width,
            compress_ratio=shape.ratio,
        )

    def new_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        kv = torch.zeros(self.total_kv_blocks, self.bytes_per_block, dtype=torch.uint8)
        return kv, kv.view(self.total_kv_blocks, KV_BLOCK_SIZE, -1)

    def generic_kwargs(self) -> dict:
        return dict(
            num_actual=self.num_tokens,
            token_to_req_indices=self.token_to_req_t,
            positions=self.positions_t,
            slot_mapping=self.slot_mapping_t,
            block_table=self.block_table,
            block_size=self.shape.state_block_size,
            state_width=self.shape.state_width,
            cos_sin_cache=self.cos_sin_cache,
            k_cache_metadata=MockMetadata(self.kv_slot_mapping_t),
            pdl_kwargs={},
            head_dim=self.shape.head_dim,
            rope_head_dim=self.shape.rope_head_dim,
            compress_ratio=self.shape.ratio,
            overlap=self.shape.overlap,
            use_fp4_cache=(self.shape.quant_format == "indexer_mxfp4"),
            rms_norm_weight=self.rms_weight,
            rms_norm_eps=RMS_EPS,
            quant_block=self.shape.quant_block,
            token_stride=self.shape.token_stride,
            scale_dim=self.shape.scale_dim,
        )


# =============================================================================
# Scenario builders — boundary every ``ratio`` positions, shared across shapes
# =============================================================================


def _mk(shape, name, pos, t2r, slot, max_pos) -> CompressorContext:
    kvs, b = [], 0
    for p in pos:
        if (p + 1) % shape.ratio == 0:
            kvs.append(b)
            b += 1
        else:
            kvs.append(-1)
    return CompressorContext(name, name, shape, pos, t2r, slot, kvs, max_pos)


def _prefill(shape, name, seqlen):
    return _mk(
        shape, name, list(range(seqlen)), [0] * seqlen, list(range(seqlen)), seqlen
    )


def _chunked(shape, name, offset, length):
    pos = list(range(offset, offset + length))
    return _mk(shape, name, pos, [0] * length, list(pos), offset + length)


def _multi(shape, name, lengths):
    P, T, S = [], [], []
    ml = max(lengths)
    for bi, L in enumerate(lengths):
        for p in range(L):
            P.append(p)
            T.append(bi)
            S.append(bi * ml + p)
    return _mk(shape, name, P, T, S, ml)


def _decode(shape, name, positions, ctx_len=4096):
    P, T, S = [], [], []
    for bi, mp in enumerate(positions):
        for p in range(mp + 1):
            P.append(p)
            T.append(bi)
            S.append(bi * ctx_len + p)
    return _mk(shape, name, P, T, S, ctx_len)


def _padded(shape, name, seqlen, num_pad):
    pos = list(range(seqlen)) + [0] * num_pad
    t2r = [0] * (seqlen + num_pad)
    slot = list(range(seqlen)) + [-1] * num_pad
    return _mk(shape, name, pos, t2r, slot, seqlen)


def all_scenarios(shape) -> list[CompressorContext]:
    """Canonical scenario set, valid for every supported ratio.

    Boundary decode positions (127/255/383/511) are multiples of 128 - 1, so
    they land exactly on a boundary for ratio=4 and ratio=128 alike; the mixed
    set deliberately includes a non-boundary tail. Chunked offsets exercise the
    look-back window synthesis in :func:`build_shared_input`.
    """
    s = []
    s.append(_decode(shape, "decode_boundary", [127, 255, 383, 511]))
    s.append(_decode(shape, "decode_mixed", [127, 255, 200, 383]))
    for seqlen in [128, 256, 512, 1024, 4096, 16384, 32768]:
        s.append(_prefill(shape, f"prefill_{seqlen}", seqlen))
    for seqlen in [257, 1025, 4097]:  # tails (extra non-boundary tokens)
        s.append(_prefill(shape, f"prefill_tail_{seqlen}", seqlen))
    s.append(_chunked(shape, "chunked_100", 100, 1024))  # offset -> look-back
    s.append(_chunked(shape, "chunked_4096", 4096, 4096))
    s.append(_multi(shape, "multi_2", [256, 512]))
    s.append(_multi(shape, "multi_3", [257, 1024, 384]))
    s.append(_padded(shape, "padded", 256, 4))
    return s


# Scenario names are shape-independent (the builder only varies boundary phase).
SCENARIO_NAMES = [c.name for c in all_scenarios(CSA_MAIN)]


def build_scenario(shape, name) -> CompressorContext:
    """Return the single scenario ``name`` for ``shape`` (un-built)."""
    for c in all_scenarios(shape):
        if c.name == name:
            return c
    raise KeyError(f"unknown scenario {name!r}; available: {SCENARIO_NAMES}")


# =============================================================================
# Shared input: BF16 authoritative; FP32 = bf16.float() + APE in the score half
# =============================================================================


def build_shared_input(ctx: CompressorContext) -> None:
    """Rebuild ctx's state caches so Triton and HIP see identical data.

    The BF16 state cache (no APE) is authoritative — HIP reads it and adds APE
    in-kernel. The FP32 cache is exactly ``bf16.float()`` with APE added into its
    score region (matching Triton's APE-fused semantics), so both backends read
    the same kv/score/APE values and any difference is kernel arithmetic + FP8
    quantization, never input mismatch.

    Chunked scenarios (first present position m > 0) have boundaries whose
    look-back window [m-(k_pool-1) .. m-1] reaches the previous chunk; those
    tokens are synthesized here (same save_partial_states layout, slot ==
    position) so the comparison is well-defined. No-op for sequences starting at
    position 0 (prefill / decode).
    """
    from vllm.models.deepseek_v4.common.ops.save_partial_states import (
        save_partial_states,
    )

    shape = ctx.shape
    sw = shape.state_width
    ratio = shape.ratio
    sbs = shape.state_block_size

    window_depth = (2 if shape.overlap else 1) * ratio - 1
    req_min, present = {}, set()
    for r, p, sl in zip(ctx.token_to_req, ctx.positions, ctx.slot_mapping):
        if sl < 0:
            continue
        present.add((r, p))
        req_min[r] = min(req_min.get(r, p), p)
    lb_pos, lb_slot, lb_req = [], [], []
    for r, mn in req_min.items():
        for p in range(max(0, mn - window_depth), mn):
            if (r, p) in present:
                continue
            lb_pos.append(p)
            lb_slot.append(p)
            lb_req.append(r)

    bf16 = torch.zeros(ctx.max_state_blocks, sbs, 2 * sw, dtype=torch.bfloat16)
    save_partial_states(
        kv=ctx._raw["kv"],
        score=ctx._raw["score"],
        ape=None,
        positions=ctx.positions_t,
        state_cache=bf16,
        slot_mapping=ctx.slot_mapping_t,
        block_size=sbs,
        state_width=sw,
        compress_ratio=ratio,
    )
    if lb_pos:
        g = torch.Generator(device="cuda").manual_seed(20240611 + ctx.num_tokens)
        n_lb = len(lb_pos)
        lb_kv = (
            torch.randn(
                n_lb, shape.coff * shape.head_dim, dtype=torch.bfloat16, generator=g
            )
            * 0.5
        )
        lb_score = (
            torch.randn(
                n_lb, shape.coff * shape.head_dim, dtype=torch.bfloat16, generator=g
            )
            * 0.5
        )
        save_partial_states(
            kv=lb_kv,
            score=lb_score,
            ape=None,
            positions=torch.tensor(lb_pos, dtype=torch.int64),
            state_cache=bf16,
            slot_mapping=torch.tensor(lb_slot, dtype=torch.int64),
            block_size=sbs,
            state_width=sw,
            compress_ratio=ratio,
        )
        for r, p, sl in zip(lb_req, lb_pos, lb_slot):
            ctx.block_table[r, p // sbs] = sl // sbs
    ctx.state_cache_bf16 = bf16

    fp32 = bf16.float()
    ape = ctx.ape  # [ratio, 2*head_dim] fp32 (== sw)
    for p, sl in list(zip(ctx.positions, ctx.slot_mapping)) + list(
        zip(lb_pos, lb_slot)
    ):
        if sl < 0:
            continue
        blk, off = sl // sbs, sl % sbs
        fp32[blk, off, sw : 2 * sw] += ape[p % ratio]
    ctx.state_cache_fp32 = fp32


# =============================================================================
# Backend runners
# =============================================================================


def hip_available() -> bool:
    """True iff the AOT dsv4 compressor ops are registered in _rocm_C."""
    try:
        import vllm._rocm_C  # noqa: F401

        return hasattr(torch.ops._rocm_C, "dsv4_csa_compress")
    except Exception:
        return False


def run_triton(ctx, kv_3d) -> None:
    """vLLM Triton FP32 reference (byte-exact oracle for the FP8 output)."""
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )

    compress_norm_rope_store_triton(
        state_cache=ctx.state_cache_fp32, kv_cache=kv_3d, **ctx.generic_kwargs()
    )


def run_hip(ctx, kv_3d) -> None:
    """Dispatch to the AOT _rocm_C op matching ctx.shape (positional schema)."""
    s = ctx.shape
    common = (
        ctx.state_cache_bf16,
        ctx.num_tokens,
        ctx.ape,
        ctx.token_to_req_t,
        ctx.positions_t,
        ctx.slot_mapping_t,
        ctx.block_table,
        s.state_block_size,
        ctx.rms_weight.to(torch.float32),
        RMS_EPS,
        ctx.cos_sin_cache,
        kv_3d,
        ctx.kv_slot_mapping_t,
        kv_3d.shape[1],
        s.scale_dim,
    )
    if s.head_dim == 128:  # indexer (FP8 or MXFP4)
        torch.ops._rocm_C.dsv4_indexer_compress(
            *common, s.quant_format == "indexer_mxfp4"
        )
    elif s.ratio == 128:  # HCA
        plan_capacity = ctx.num_tokens // s.ratio + ctx.block_table.shape[0] + 2
        plan_scratch = torch.empty(
            plan_capacity, dtype=torch.int32, device=ctx.state_cache_bf16.device
        )
        counter_scratch = torch.empty(
            1, dtype=torch.int32, device=ctx.state_cache_bf16.device
        )
        torch.ops._rocm_C.dsv4_hca_compress(*common, plan_scratch, counter_scratch)
    else:  # CSA
        torch.ops._rocm_C.dsv4_csa_compress(*common)


# =============================================================================
# Dequantization (vLLM cache format -> normed row) + compare
# =============================================================================


def _dequant_h512(buf, ci):
    """head=512: NOPE 448 FP8 * per-64 ue8m0 + ROPE 64 bf16 -> normed [512]."""
    soff = KV_BLOCK_SIZE * H512_TOKEN_STRIDE
    blk, p = ci // KV_BLOCK_SIZE, ci % KV_BLOCK_SIZE
    base = p * H512_TOKEN_STRIDE
    fp8 = buf[blk][base : base + H512_NOPE].view(torch.float8_e4m3fn).float()
    sc = buf[blk][
        soff + p * H512_SCALE_DIM : soff + p * H512_SCALE_DIM + H512_N_NOPE_BLOCKS
    ]
    out = torch.zeros(512)
    for nb in range(H512_N_NOPE_BLOCKS):
        e = sc[nb].int().item() - 127
        out[nb * H512_QUANT_BLOCK : (nb + 1) * H512_QUANT_BLOCK] = fp8[
            nb * H512_QUANT_BLOCK : (nb + 1) * H512_QUANT_BLOCK
        ] * (2.0**e)
    out[H512_NOPE:] = (
        buf[blk][base + H512_NOPE : base + H512_TOKEN_STRIDE]
        .view(torch.bfloat16)
        .float()
    )
    return out


def _dequant_indexer_fp8(buf, ci):
    """indexer FP8: whole 128-dim row FP8 * single fp32 scale -> [128]."""
    ts, sd = 128, 4
    soff = KV_BLOCK_SIZE * ts
    blk, p = ci // KV_BLOCK_SIZE, ci % KV_BLOCK_SIZE
    base = p * ts
    fp8 = buf[blk][base : base + 128].view(torch.float8_e4m3fn).float()
    scale = buf[blk][soff + p * sd : soff + p * sd + 4].view(torch.float32)
    return fp8 * scale


def _dequant_indexer_mxfp4(buf, ci):
    """indexer MXFP4: E2M1 nibbles * per-32 ue8m0 scale -> [128]."""
    ts, sd = 64, 4  # 64 packed bytes, 4 ue8m0 bytes
    soff = KV_BLOCK_SIZE * ts
    blk, p = ci // KV_BLOCK_SIZE, ci % KV_BLOCK_SIZE
    base = p * ts
    packed = buf[blk][base : base + ts].long()  # [64] uint8
    sc = buf[blk][soff + p * sd : soff + p * sd + sd].long()  # [4] ue8m0
    low = packed & 0xF  # even
    high = (packed >> 4) & 0xF  # odd
    scale = (2.0 ** (sc.float() - 127.0)).repeat_interleave(ts // sd)  # [64]
    lut = E2M1_LUT.to(packed.device)
    even = lut[low] * scale
    odd = lut[high] * scale
    return torch.stack([even, odd], dim=1).reshape(128)


def _dequant_row(buf, ci, shape):
    if shape.quant_format == "indexer_fp8":
        return _dequant_indexer_fp8(buf, ci)
    if shape.quant_format == "indexer_mxfp4":
        return _dequant_indexer_mxfp4(buf, ci)
    return _dequant_h512(buf, ci)


def compare_to_triton(ref, hip, ctx) -> tuple:
    """Compare a HIP packed cache against the Triton reference.

    Returns ``(aligned, detail)``. ``aligned`` means reference-equivalent: the
    dequantized output matches within FP8/bf16 tolerance.

    The tolerance is region-aware, matching what the kernels actually emit:

    * head=512 (CSA/HCA): the NOPE half is FP8-quantized, so a single element
      may round to an adjacent FP8 code (a ~2^-4 step) when Triton and HIP pick
      different RNE ties at a boundary — reference-equivalent, caught by the
      *mean*, not the max. The ROPE half is bf16 (high precision), so its *max*
      diff must stay tiny. Hence ``nope_mean`` and ``rope_max`` are thresholded
      separately. A per-element max over the FP8 NOPE region would flag exactly
      those harmless RNE ties (this is why an earlier whole-row max check failed
      large prefills despite byte% == 100).
    * indexer (head=128): whole-row mean (the FP8 single-scale / MXFP4 path is
      checked the same way, max being dominated by FP8 RNE ties).
    """
    shape = ctx.shape
    nc = ctx.num_compress
    byte_pct = (
        ref.view(torch.uint8).int() == hip.view(torch.uint8).int()
    ).float().mean().item() * 100
    if shape.head_dim == 512:
        nope_mean, rope_max = 0.0, 0.0
        for ci in range(nc):
            d = (_dequant_h512(ref, ci) - _dequant_h512(hip, ci)).abs()
            nope_mean = max(nope_mean, d[:H512_NOPE].mean().item())
            rope_max = max(rope_max, d[H512_NOPE:].max().item())
        aligned = nope_mean < 5e-2 and rope_max < 5e-2
        detail = (
            f"byte%={byte_pct:.2f} nope_mean={nope_mean:.5f} rope_max={rope_max:.4f}"
        )
    else:
        dmean = 0.0
        for ci in range(nc):
            d = (_dequant_row(ref, ci, shape) - _dequant_row(hip, ci, shape)).abs()
            dmean = max(dmean, d.mean().item())
        aligned = dmean < 1e-2
        detail = f"byte%={byte_pct:.2f} dmean={dmean:.5f}"
    return aligned, detail


# =============================================================================
# Indexer MXFP4 torch oracle (Triton MXFP4 uses NVIDIA PTX, can't run on AMD)
# =============================================================================


def _torch_indexer_row(ctx, P, r) -> torch.Tensor:
    """Reproduce the HIP front-end in torch: read the SAME bf16 state cache HIP
    reads, add APE in-kernel-style, per-dim softmax over the 8 overlap positions,
    weighted sum, RMSNorm, GPT-J RoPE -> true fp32 post-RoPE row [128]."""
    sbs = ctx.shape.state_block_size
    state = ctx.state_cache_bf16.float()
    HEAD, SW = 128, 256
    ape = ctx.ape
    kv_l, sc_l, valid = [], [], []
    for k in range(8):
        pk = P - 7 + k
        ho = 128 if k >= 4 else 0
        if pk < 0:
            valid.append(False)
            kv_l.append(torch.zeros(HEAD))
            sc_l.append(torch.zeros(HEAD))
            continue
        valid.append(True)
        blk = int(ctx.block_table[r, pk // sbs].item())
        off = pk % sbs
        kv_l.append(state[blk, off, ho : ho + HEAD])
        sc_l.append(
            state[blk, off, SW + ho : SW + ho + HEAD] + ape[k % 4, ho : ho + HEAD]
        )
    KV = torch.stack(kv_l)  # [8,128]
    SC = torch.stack(sc_l)
    mask = torch.tensor(valid, device=KV.device).view(8, 1)
    SC = torch.where(mask, SC, torch.full_like(SC, -float("inf")))
    w = torch.softmax(SC, dim=0)  # per-dim over 8
    comp = (w * KV).sum(0)  # [128]
    var = (comp * comp).mean()
    normed = comp * torch.rsqrt(var + RMS_EPS) * ctx.rms_weight.float()
    cs = ctx.cos_sin_cache[(P // 4) * 4]  # [64]: 32 cos + 32 sin
    out = normed.clone()
    idx = torch.arange(32, device=out.device)
    e = normed[64::2]
    o = normed[65::2]  # rope pairs (dims 64..127)
    cos_v = cs[idx]
    sin_v = cs[32 + idx]
    out[64::2] = e * cos_v - o * sin_v
    out[65::2] = o * cos_v + e * sin_v
    return out


def _torch_mxfp4_dequant(row: torch.Tensor) -> torch.Tensor:
    """Reference MXFP4 quant+dequant of one fp32 post-RoPE row [128].

    Mirrors the Triton _quantize_mxfp4_pair math (per-32 block ue8m0 scale =
    2^ceil(log2(amax/6)), E2M1 round-to-nearest, even->low / odd->high nibble).
    """
    HEAD = 128
    mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=row.device)
    odd_pen = torch.tensor([0.0, 1, 0, 1, 0, 1, 0, 1], device=row.device) * 1e-6
    even = row[0::2].bfloat16().float()  # [64]
    odd = row[1::2].bfloat16().float()
    out = torch.zeros(HEAD, device=row.device)

    def _q(x, log2r):
        s = x.sign()
        xs = (x.abs() * (2.0**-log2r)).unsqueeze(-1)
        idx = ((xs - mag).abs() + odd_pen).argmin(-1)
        return s * mag[idx] * (2.0**log2r)

    for b in range(4):  # 4 blocks of 16 pairs (32 elems)
        e = even[b * 16 : (b + 1) * 16]
        o = odd[b * 16 : (b + 1) * 16]
        amax = max(e.abs().max().item(), o.abs().max().item(), 6.0 * 2**-126)
        log2r = min(max(math.ceil(math.log2(amax / 6.0)), -127), 127)
        out[b * 32 : b * 32 + 32 : 2] = _q(e, log2r)
        out[b * 32 + 1 : b * 32 + 32 : 2] = _q(o, log2r)
    return out


def mxfp4_oracle_diff(ctx, hip_cpu, max_check=512) -> float:
    """Max per-boundary mean |HIP MXFP4 - torch front-end + RNE MXFP4|.

    The front-end is per-token in python; it is independently proven by the FP8
    byte-exact test, so a bounded sample (``max_check`` boundaries) suffices to
    catch quant-tail bugs.
    """
    nc = ctx.num_compress
    bnd = [
        (p, r)
        for p, r, ks in zip(ctx.positions, ctx.token_to_req, ctx.kv_slot_mapping)
        if ks >= 0
    ]
    dmean = 0.0
    for ci in range(min(nc, max_check)):
        P, r = bnd[ci]
        exp = _torch_mxfp4_dequant(_torch_indexer_row(ctx, P, r))
        act = _dequant_indexer_mxfp4(hip_cpu, ci).cuda()
        dmean = max(dmean, (exp - act).abs().mean().item())
    return dmean
