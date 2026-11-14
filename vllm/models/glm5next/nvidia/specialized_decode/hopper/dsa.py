# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA (NoPE sparse MLA) layer of a planned decode step on Hopper.

`HopperDsaLayer` runs all of `Glm5NextMLAAttention.forward` for a decode or
uniform verify step in nine kernels (stock: about 33):

1. G1: `x @ [q_a | kv_a | gate | wk | weights_proj]` (one GEMM, split-K 2,
   FP32 out);
2. q/kv a-norms and the BF16 MLA KV insert; indexer key LayerNorm, tail stash
   and pool compression into the indexer cache;
3. G2: `qc @ [q_b | wq_b]`;
4. G3: the W_UK absorb;
5. indexer query quant, pool logits, top-512 pools, expansion and the MLA KV
   rows of the selection;
6. sparse MLA over the selected rows, split over the rows;
7. merge of the splits;
8. G4: W_UV;
9. G5: o_proj, left TP-partial.

Differences from the GB300 stages (`dsa_attention`, `indexer`, `kpool`,
`linear`): the KV cache and the absorbed query stay BF16 (the Hopper recipe),
the sparse MLA is our own split kernel (no FlashInfer plan or index pack is
read), the indexer cache keeps FP32 power-of-two scales (Hopper's DeepGEMM
layout, where main's `kpool_update` writes UE8M0 bytes), and the BF16
projections are Triton GEMMs (main's split-K GEMM is tcgen05). Nothing of the
stock modules is patched or copied.
"""

import re
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton

from . import _dsa_kernels as K

logger = init_logger(__name__)

MAX_TOKENS = 32
MAX_Q = 8
HEADS = 16  # local MLA heads (TP4)
LATENT = 512
NOPE = 256
VALUE = 256
HIDDEN = 4096
Q_LORA = 1536
INDEX_HEADS = 32
INDEX_DIM = 128
KPOOL = 4
TOPK = 2048
SELECT = TOPK // KPOOL
ROWS = TOPK + KPOOL - 1  # selected tokens per row at most
ROW_PITCH = 2064

# G1 output columns: q_a | kv_a | gate | wk | weights_proj.
GATE = Q_LORA + LATENT
KEY = GATE + INDEX_DIM
WPROJ = KEY + INDEX_DIM
G1_COLS = WPROJ + INDEX_HEADS

# (BN, BK, stages, split-K, warps) per GEMM, from the step-0 H200 sweep.
GEMM_TACTICS = {
    "g1": (64, 128, 6, 2, 4),
    "g2": (64, 128, 4, 1, 4),
    "g3": (64, 128, 4, 1, 4),
    "g4": (64, 128, 4, 1, 4),
    "g5": (64, 128, 6, 1, 4),
}
# Sparse MLA (rows per split, rows per tile, stages, warps) by token count, from
# the in-situ H200 sweep (`full/dsa` e1-2k): 17 splits of 128 rows up to 12
# tokens, 11 of 192 from 18 (-6 us per layer at 18 and 24 tokens).
ATTENTION_TACTICS = {
    m: (128, 64, 3, 4) if m <= 12 else (192, 64, 3, 4) for m in range(1, MAX_TOKENS + 1)
}
# Indexer select: pool tile, programs per row, stages, warps.
SELECT_TACTIC = (64, 16, 2, 4)
# Pools a row's select keeps in registers (contexts up to 4K tokens); longer
# rows run the pass-based select. 2048 is 1-2 us slower per layer at 2K-2.4K.
SELECT_REGISTER_POOLS = 1024
# The indexer query's Hadamard-128 as an MMA with the +-1 matrix instead of
# butterflies (the same values; see `select_kernel`): in situ -12 us per layer
# at 6 and 24 tokens (`full/dsa` e3-combo).
SELECT_HADAMARD_MMA = True
# Prefetch each GEMM's weights into L2 before its PDL wait (slower alone on
# H200: 13.7 -> 16.6 us for G1); release dependents right after the wait.
GEMM_PREFETCH = False
EARLY_TRIGGER = True
# Token counts whose GEMMs release their dependents only after the main loop
# (in situ: -2.7 us per layer at 12 tokens, +3.0 at 6, +0.5-1.0 at 18 and 24).
LATE_TRIGGER_TOKENS = frozenset(range(7, 13))
# The Triton absorb reads a K-major copy of W_UK (4.2 MB per layer): 3.8-4.0
# against 5.3-6.4 us on the column-major W_UK_T at 12-24 tokens.
ABSORB_KMAJOR = True
# Programmatic dependent launch for the Triton kernels (off only to measure each
# kernel alone).
PDL = True
# Token counts whose W_UK absorb runs as main's CuTe `head_gemv` (H200 alone:
# 2.2 / 3.2 us at 1 / 6 tokens against 4.3 for the Triton GEMM; even at 8).
ABSORB_GEMV_TOKENS = frozenset(range(1, 7))
# Token counts whose o_proj runs as main's CuTe skinny GEMM (`gemm.Projection`,
# weight loads issued before the PDL wait); H200 alone at 1 token: 10.7 us
# against 12.9 for the Triton GEMM (graph launch excluded).
O_PROJ_SKINNY_TOKENS = frozenset((1, 2, 4))
# Token counts whose o_proj runs as Kimi K3's warp split-K GEMM, with its
# (output tile, k tile, stages, B loader warps) per token tile.
O_PROJ_WSK_TOKENS: frozenset[int] = frozenset()
O_PROJ_WSK_TACTICS = {8: (16, 256, 8, 2), 16: (16, 256, 8, 2), 32: (16, 256, 8, 2)}


def _bm(tokens: int) -> int:
    return 16 if tokens <= 16 else 32


@dataclass
class DsaWeights:
    """The tensors of one DSA layer the kernels read (no copies)."""

    qkv_a: torch.Tensor  # [q_lora + latent, hidden]
    q_b: torch.Tensor  # [heads * nope, q_lora]
    o_proj: torch.Tensor  # [hidden, heads * value]
    wq_b: torch.Tensor  # [index heads * index dim, q_lora]
    wk_weights: torch.Tensor  # [index dim + index heads, hidden]
    gate: torch.Tensor  # [index dim, hidden]
    ape: torch.Tensor  # [kpool, index dim] FP32
    k_norm_weight: torch.Tensor  # [index dim] FP32
    k_norm_bias: torch.Tensor  # [index dim] FP32
    k_norm_eps: float
    q_norm: torch.Tensor  # [q_lora]
    kv_norm: torch.Tensor  # [latent]
    rms_eps: float
    w_uk_t: torch.Tensor  # [heads, nope, latent], any strides
    w_uv: torch.Tensor  # [heads, latent, value], any strides
    softmax_scale: float
    index_scale: float  # indexer softmax scale * index heads ** -0.5
    max_pools: int  # max_model_len // kpool


@dataclass
class DsaCaches:
    kv: torch.Tensor  # [blocks, block size, latent] BF16
    index: torch.Tensor  # [pages, pools per page, index dim + 4] uint8
    tail: torch.Tensor  # [blocks, 2, ring, index dim] BF16


@dataclass
class DsaInputs:
    """Per-step tensors (persistent metadata buffers under CUDA graphs)."""

    positions: torch.Tensor  # [T]
    kv_slots: torch.Tensor  # [T] MLA KV slots, -1 for padded rows
    req_ids: torch.Tensor  # [T] int32
    block_table: torch.Tensor  # [R, W] int32, MLA
    block_size: int
    index_slots: torch.Tensor  # [T] pool-granular indexer cache slots
    index_block_table: torch.Tensor  # [R, Wi] int32
    tail_slots: torch.Tensor  # [T]


class _Workspace:
    """Intermediates shared by every layer on a device (the layers of a step run
    one after another on one stream)."""

    def __init__(self, device: torch.device, max_pools: int):
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)

        m = MAX_TOKENS
        splits = max(triton.cdiv(ROWS, t[0]) for t in ATTENTION_TACTICS.values())
        self.max_pools = max_pools
        self.y = empty(m, G1_COLS, dtype=torch.float32)
        self.qc = empty(m, Q_LORA)
        self.q = empty(m, 2 * HEADS * NOPE)  # q_b | wq_b
        self.qabs = empty(m, HEADS, LATENT)
        self.ob = empty(m, HEADS, LATENT)
        self.ao = empty(m, HEADS * VALUE)
        self.logits = empty(m, max(max_pools, 1), dtype=torch.float32)
        self.pools = empty(m, SELECT, dtype=torch.int32)
        self.rows = empty(m, ROW_PITCH, dtype=torch.int32)
        self.counts = empty(m, dtype=torch.int32)
        self.partial = empty(m, splits, HEADS, LATENT, dtype=torch.float32)
        self.lse = empty(m, splits, HEADS, dtype=torch.float32)
        self.split_k = empty(1 << 20, dtype=torch.float32)
        h = torch.ones(1, 1)
        while h.shape[0] < INDEX_DIM:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        self.hadamard = h.to(device=device, dtype=torch.bfloat16).contiguous()
        # Counters are reset by the program that consumes them.
        self.counters = torch.zeros(4096, dtype=torch.int32, device=device)


_workspaces: dict[tuple[int, int], _Workspace] = {}


def workspace(device: torch.device, max_pools: int) -> _Workspace:
    key = (device.index, max_pools)
    if key not in _workspaces:
        _workspaces[key] = _Workspace(device, max_pools)
    return _workspaces[key]


def _missing(attention) -> list[str]:
    wrapper = attention.mla_attn
    mla = wrapper.mla_attn
    indexer = getattr(attention, "indexer", None)
    checks = [
        (wrapper.q_lora_rank == Q_LORA and wrapper.fuse_qkv_rmsnorm, "q-LoRA 1536"),
        (wrapper.rotary_emb is None and wrapper.qk_rope_head_dim == 0, "NoPE"),
        (wrapper.num_heads == HEADS, f"{HEADS} heads per rank"),
        (wrapper.kv_lora_rank == LATENT and wrapper.v_head_dim == VALUE, "latent 512"),
        (indexer is not None, "its own indexer"),
        (wrapper.g_proj is None and not wrapper.dcp_q_replicate, "no gate or DCP q"),
        (str(mla.kv_cache_dtype) in ("auto", "bfloat16"), "a BF16 KV cache"),
        (mla.q_pad_num_heads is None and not mla.use_pcp, "no q padding or PCP"),
        (getattr(mla, "hisparse_cache", None) is None, "no HiSparse"),
        (getattr(mla.impl, "dcp_world_size", 1) == 1, "no DCP"),
    ]
    if indexer is not None:
        checks += [
            (indexer.index_kpool == KPOOL, "kpool 4"),
            (
                indexer.n_head == INDEX_HEADS and indexer.head_dim == INDEX_DIM,
                "32 indexer heads of 128",
            ),  # fmt: skip
            (indexer.topk_tokens == TOPK and indexer.rope_dim == 0, "top-k 2048, NoPE"),
            (getattr(indexer.indexer_op, "dcp_world_size", 1) == 1, "no indexer DCP"),
            (not getattr(indexer.indexer_op, "use_fp4_cache", False), "FP8 index"),
        ]
    for name in ("fused_qkv_a_proj", "q_b_proj", "o_proj"):
        weight = getattr(getattr(wrapper, name, None), "weight", None)
        checks.append(
            (weight is not None and weight.dtype == torch.bfloat16, f"BF16 {name}")
        )
    return [what for ok, what in checks if not ok]


def _is_draft(attention) -> bool:
    found = re.search(r"layers\.(\d+)\.", attention.mla_attn.prefix + ".")
    layers = getattr(attention.indexer.config, "num_hidden_layers", None)
    return bool(found and layers is not None and int(found.group(1)) >= layers)


class HopperDsaLayer:
    """`Glm5NextMLAAttention.forward` of planned decode steps; see the module
    docstring. Built from the stock layer after weight loading; reads it,
    never modifies it."""

    def __init__(self, attention, tokens=(), q: int | None = None, draft=None):
        missing = _missing(attention)
        if missing:
            raise ValueError("Hopper DSA layer needs " + ", ".join(missing))
        wrapper = attention.mla_attn
        mla = wrapper.mla_attn
        indexer = attention.indexer
        self.attention, self.wrapper, self.mla, self.indexer = (
            attention, wrapper, mla, indexer,
        )  # fmt: skip
        weights = DsaWeights(
            qkv_a=wrapper.fused_qkv_a_proj.weight,
            q_b=wrapper.q_b_proj.weight,
            o_proj=wrapper.o_proj.weight,
            wq_b=indexer.wq_b.weight,
            wk_weights=indexer.wk_weights_proj.weight,
            gate=indexer.index_kpool_compress_gate.data,
            ape=indexer.index_kpool_compress_ape.data,
            k_norm_weight=indexer.k_norm.weight.data,
            k_norm_bias=indexer.k_norm.bias.data,
            k_norm_eps=indexer.k_norm.eps,
            q_norm=wrapper.q_a_layernorm.weight.data,
            kv_norm=wrapper.kv_a_layernorm.weight.data,
            rms_eps=wrapper.q_a_layernorm.variance_epsilon,
            w_uk_t=mla.W_UK_T,
            w_uv=mla.W_UV,
            softmax_scale=mla.impl.scale,
            index_scale=indexer.softmax_scale * indexer.n_head**-0.5,
            max_pools=indexer.max_pool_len,
        )
        self.draft = _is_draft(attention) if draft is None else draft
        self._setup(weights, wrapper.topk_indices_buffer if self.draft else None)
        self.prepare(tokens, q)

    @classmethod
    def from_weights(cls, weights: DsaWeights, tokens=(), q=None, topk_buffer=None):
        """A layer over explicit tensors (tests and benchmarks)."""
        self = cls.__new__(cls)
        self.attention = self.wrapper = self.mla = self.indexer = None
        self.draft = topk_buffer is not None
        self._setup(weights, topk_buffer)
        self.prepare(tokens, q)
        return self

    def _setup(self, w: DsaWeights, topk_buffer):
        for t in (w.qkv_a, w.q_b, w.o_proj, w.wq_b, w.wk_weights, w.gate):
            if not t.is_contiguous() or t.dtype != torch.bfloat16:
                raise ValueError("Hopper DSA layer needs contiguous BF16 projections")
        if w.qkv_a.shape != (Q_LORA + LATENT, HIDDEN) or w.wk_weights.shape != (
            INDEX_DIM + INDEX_HEADS, HIDDEN,
        ):  # fmt: skip
            raise ValueError("Hopper DSA layer: unexpected projection shapes")
        self.w = w
        self.topk_buffer = topk_buffer
        self.w_uk_kmajor = w.w_uk_t.transpose(
            1, 2
        ).contiguous()  # [heads, latent, nope]
        self.o_proj = None
        self.o_proj_wsk = None
        if O_PROJ_WSK_TOKENS:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.warp_splitk import (  # noqa: E501
                PreparedWarpSplitK,
                WarpSplitKTactic,
            )

            def tactic(m):
                tile = 8 if m <= 8 else 16 if m <= 16 else 32
                out, k, stages, loaders = O_PROJ_WSK_TACTICS[tile]
                return WarpSplitKTactic(out, tile, k, stages, loaders)

            self.o_proj_wsk = PreparedWarpSplitK(
                w.o_proj, {m: tactic(m) for m in sorted(O_PROJ_WSK_TOKENS)}
            )
        if O_PROJ_SKINNY_TOKENS:
            from ..gemm import Projection

            try:
                self.o_proj = Projection(w.o_proj)
            except Exception as error:  # the CuTe DSL build may refuse a shape
                logger.info_once("Hopper DSA layer: Triton o_proj (%s)", error)
        self.ws = workspace(w.qkv_a.device, w.max_pools)
        self.keys: set[tuple[int, int]] = set()

    def prepare(self, tokens, q: int | None = None) -> None:
        """Plans `tokens` (requests x q) at `q` tokens per request; with q None
        every divisor up to `MAX_Q`."""
        for m in tokens:
            if not 1 <= m <= MAX_TOKENS:
                continue
            for tq in [q] if q is not None else range(1, MAX_Q + 1):
                if tq is not None and 1 <= tq <= MAX_Q and m % tq == 0:
                    self.keys.add((m, tq))

    def supports(self, tokens: int, q: int) -> bool:
        return (tokens, q) in self.keys

    # ------------------------------------------------------------------ inputs

    def caches(self) -> DsaCaches:
        return DsaCaches(
            self.mla.kv_cache,
            self.indexer.k_cache.kv_cache,
            self.indexer.tail_cache.kv_cache,
        )

    def inputs(self, positions, metadata) -> DsaInputs:
        from vllm.forward_context import get_forward_context

        md = metadata[self.mla.layer_name]
        index = metadata[self.indexer.k_cache.prefix]
        tail = metadata[self.indexer.tail_cache.prefix]
        return DsaInputs(
            positions=positions,
            kv_slots=get_forward_context().slot_mapping[self.mla.layer_name],
            req_ids=md.req_id_per_token,
            block_table=md.block_table,
            block_size=md.block_size,
            index_slots=index.slot_mapping,
            index_block_table=index.block_table,
            tail_slots=tail.slot_mapping,
        )

    def prepare_caches(self, *_) -> None:
        """Compiles every planned kernel on the bound caches before capture:
        every slot is -1, so nothing is written."""
        caches = self.caches()
        dev = caches.kv.device
        for m, q in sorted(self.keys):
            neg = torch.full((m,), -1, dtype=torch.int64, device=dev)
            inputs = DsaInputs(
                positions=torch.zeros(m, dtype=torch.int64, device=dev),
                kv_slots=neg,
                req_ids=torch.zeros(m, dtype=torch.int32, device=dev),
                block_table=torch.zeros((m, 8), dtype=torch.int32, device=dev),
                block_size=caches.kv.shape[1],
                index_slots=neg,
                index_block_table=torch.zeros((m, 8), dtype=torch.int32, device=dev),
                tail_slots=neg,
            )
            x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            self.forward(x, inputs, caches, q)
            if self.draft:
                self.forward(x, inputs, caches, q, skip_topk=True)
        torch.cuda.synchronize(dev)

    # ------------------------------------------------------------------ run

    def run(self, x, positions, metadata, q: int) -> torch.Tensor:
        """The layer's TP-partial o_proj output `[T, hidden]` for a planned step."""
        return self.forward(
            x, self.inputs(positions, metadata), self.caches(), q,
            skip_topk=bool(self.wrapper.skip_topk),
        )  # fmt: skip

    __call__ = run

    def _gemm(self, name, x, segments, out, m, n, k, swn, swk, *, sxm, som,
              batch=1, sxb=0, swb=0, sob=0, counters=0):  # fmt: skip
        early = EARLY_TRIGGER and m not in LATE_TRIGGER_TOKENS
        bn, bk, stages, split, warps = GEMM_TACTICS[name]
        w0 = segments[0]
        w1 = segments[1] if len(segments) > 1 else w0
        w2 = segments[2] if len(segments) > 2 else w1
        bounds = [s.shape[-2] for s in segments]
        n0 = bounds[0] if len(segments) > 1 else n
        n1 = n0 + bounds[1] if len(segments) > 2 else n
        grid = (triton.cdiv(n, bn), split, batch)
        K.gemm_kernel[grid](
            x, w0, w1, w2, out, self.ws.split_k, self.ws.counters[counters:],
            m, sxm, sxb, swb, som, sob, n0, n1,
            N=n, K=k, SWN=swn, SWK=swk, BM=_bm(m), BN=bn, BK=bk, SPLIT=split,
            NSTAGE=stages, PREFETCH=GEMM_PREFETCH, EARLY=early,
            num_warps=warps, launch_pdl=PDL,
        )  # fmt: skip

    def forward(
        self,
        x: torch.Tensor,
        inputs: DsaInputs,
        caches: DsaCaches,
        q: int,
        skip_topk: bool = False,
    ) -> torch.Tensor:
        w, ws = self.w, self.ws
        m = x.shape[0]
        assert x.is_contiguous() and x.dtype == torch.bfloat16 and m <= MAX_TOKENS
        assert m % q == 0
        dev = x.device
        kv, index, tail = caches.kv, caches.index, caches.tail
        assert (
            kv.dtype == torch.bfloat16 and kv.stride(2) == 1 and kv.stride(1) == LATENT
        )
        rows_per_block = kv.stride(0) // LATENT
        assert kv.stride(0) % LATENT == 0
        y = ws.y[:m]
        # 1. a-projections (and, with the indexer, its key / gate / head weights)
        segments = [w.qkv_a] if skip_topk else [w.qkv_a, w.gate, w.wk_weights]
        self._gemm("g1", x, segments, y, m, GATE if skip_topk else G1_COLS, HIDDEN,
                   HIDDEN, 1, sxm=HIDDEN, som=G1_COLS, counters=0)  # fmt: skip
        # 2. norms, KV insert, indexer key and kpool
        ring = tail.shape[2]
        K.post_g1_kernel[(m * (2 if skip_topk else 3),)](
            y, w.q_norm, w.kv_norm, w.rms_eps, ws.qc, kv, kv.stride(0),
            inputs.kv_slots, w.k_norm_weight, w.k_norm_bias, w.k_norm_eps, w.ape,
            inputs.positions, inputs.index_slots, inputs.tail_slots, tail,
            index.view(torch.float8_e4m3fn), index.view(torch.float32), m,
            SY=G1_COLS, QA=Q_LORA, KVR=LATENT, GATE=GATE, KEY=KEY, BLOCK=2048,
            KV_BS=kv.shape[1], NEXT_N=q, HD=INDEX_DIM, KPOOL=KPOOL, RING=ring,
            TAIL_BLK=tail.stride(0), TAIL_HALF=tail.stride(1), PAGE=index.shape[1],
            PAGE_BYTES=index.stride(0), KPOOL_PROGRAMS=not skip_topk, num_warps=8,
            launch_pdl=PDL,
        )  # fmt: skip
        # 3. q_b (and the indexer query)
        qn = HEADS * NOPE
        segments = [w.q_b] if skip_topk else [w.q_b, w.wq_b]
        self._gemm("g2", ws.qc, segments, ws.q, m, qn * (1 if skip_topk else 2),
                   Q_LORA, Q_LORA, 1, sxm=Q_LORA, som=2 * qn, counters=512)  # fmt: skip
        # 4. W_UK absorb: qabs[t, h] = q[t, h] @ W_UK_T[h]
        uk = w.w_uk_t
        if m in ABSORB_GEMV_TOKENS and uk.stride(2) == 1:
            from ..dsa_attention import head_gemv

            head_gemv(ws.q[:m, :qn].view(m, HEADS, NOPE), uk, ws.qabs[:m])
        else:
            if ABSORB_KMAJOR:
                uk = self.w_uk_kmajor
                swn, swk = uk.stride(1), uk.stride(2)
            else:
                swn, swk = uk.stride(2), uk.stride(1)
            self._gemm("g3", ws.q, [uk], ws.qabs, m, LATENT, NOPE, swn, swk,
                       sxm=2 * qn, som=HEADS * LATENT, batch=HEADS, sxb=NOPE,
                       swb=uk.stride(0), sob=LATENT, counters=1024)  # fmt: skip
        # 5. top-k selection -> MLA KV rows
        mbt = inputs.block_table
        assert mbt.dtype == torch.int32 and mbt.stride(1) == 1
        topk = self.topk_buffer
        if skip_topk:
            assert topk is not None
            K.remap_kernel[(m,)](
                topk, topk.stride(0), topk.shape[1], inputs.req_ids, inputs.kv_slots,
                mbt, mbt.stride(0), mbt.shape[1], ws.rows, ws.counts,
                MBS=inputs.block_size, MSTRIDE=rows_per_block, SLW=ROW_PITCH,
                BLOCK=1024, num_warps=4, launch_pdl=PDL,
            )  # fmt: skip
        else:
            ibt = inputs.index_block_table
            assert ibt.dtype == torch.int32 and ibt.stride(1) == 1
            bp, nch, stages, warps = SELECT_TACTIC
            write = topk is not None
            if not write:
                topk = ws.rows
            K.select_kernel[(m, nch)](
                ws.q[:, qn:], 2 * qn, y[:, WPROJ:], G1_COLS, inputs.positions,
                inputs.req_ids, inputs.kv_slots, index.view(torch.float8_e4m3fn),
                index.view(torch.float32), ibt, ibt.stride(0), ws.logits,
                ws.logits.stride(0), ws.counters[2048:], ws.pools, mbt, mbt.stride(0),
                mbt.shape[1], ws.rows, ws.counts, topk, topk.stride(0), topk.shape[1],
                w.index_scale, NH=INDEX_HEADS, PAGE=index.shape[1],
                PAGE_BYTES=index.stride(0), MBS=inputs.block_size,
                MSTRIDE=rows_per_block, KSEL=SELECT, KPOOL=KPOOL, BP=bp, NCH=nch,
                NSTAGE=stages, BLOCK_SEL=SELECT_REGISTER_POOLS, SLW=ROW_PITCH,
                EXP=128, WRITE_LOGICAL=write, TOPW_BLOCK=512, had_ptr=ws.hadamard,
                HADAMARD_MMA=SELECT_HADAMARD_MMA, num_warps=warps, launch_pdl=PDL,
            )  # fmt: skip
        # 6-7. sparse MLA and the merge of its splits
        chunk, tile, stages, warps = ATTENTION_TACTICS[m]
        splits = triton.cdiv(ROWS, chunk)
        flat = kv.as_strided(((kv.shape[0] - 1) * rows_per_block + kv.shape[1], LATENT),
                             (LATENT, 1))  # fmt: skip
        K.attention_kernel[(m, splits)](
            ws.qabs, ws.rows, ws.counts, flat, ws.partial, ws.lse,
            w.softmax_scale * 1.4426950408889634, S=splits, CH=chunk, BT=tile,
            H=HEADS, D=LATENT, SLW=ROW_PITCH, NSTAGE=stages, num_warps=warps,
            launch_pdl=PDL,
        )  # fmt: skip
        K.combine_kernel[(m, HEADS)](
            ws.partial, ws.lse, ws.ob, S=splits, SP=triton.next_power_of_2(splits),
            H=HEADS, D=LATENT, num_warps=4, launch_pdl=PDL,
        )  # fmt: skip
        # 8. W_UV: ao[t, h] = ob[t, h] @ W_UV[h]
        uv = w.w_uv
        self._gemm("g4", ws.ob, [uv], ws.ao, m, VALUE, LATENT, uv.stride(2),
                   uv.stride(1), sxm=HEADS * LATENT, som=HEADS * VALUE, batch=HEADS,
                   sxb=LATENT, swb=uv.stride(0), sob=VALUE, counters=1536)  # fmt: skip
        # 9. o_proj (TP-partial)
        skinny = self.o_proj
        if m in O_PROJ_SKINNY_TOKENS and skinny is not None and m in skinny.configs:
            return skinny(ws.ao[:m])
        wsk = self.o_proj_wsk
        if m in O_PROJ_WSK_TOKENS and wsk is not None and m in wsk.codes:
            # Its stores cover whole token tiles: rows up to the tile are scratch.
            tile = 8 if m <= 8 else 16 if m <= 16 else 32
            out = torch.empty((tile, HIDDEN), dtype=torch.bfloat16, device=dev)
            return wsk.run(ws.ao[:m], out[:m])
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        self._gemm("g5", ws.ao, [w.o_proj], out, m, HIDDEN, HEADS * VALUE,
                   HEADS * VALUE, 1, sxm=HEADS * VALUE, som=HIDDEN,
                   counters=3072)  # fmt: skip
        return out
