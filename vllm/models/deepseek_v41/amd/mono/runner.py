# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash mono decode layer: host side.

One ``DSV41MonoLayer`` a TP rank runs every mono layer of a decode step (M <= 48
rows): two persistent launches a layer (``layer``) from the layer's
inputs at the attention seam to its outputs at the next one -- the MoE's
reduced output, the residual after the FFN seam and that seam's mixes -- with
both TP all-reduces inside the kernels (symmetric peer memory). A layer whose
attention vLLM runs (``ffn``) takes one launch from its unreduced attention
output on. Every argument is a device pointer: the launches can be captured in a
HIP graph. The kernels move their mailbox epoch on themselves; every rank runs
the same launches, so the ranks' epochs agree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention.device import BLOCKS
from .attention.front import FRONT_POINTS
from .attention.plan import HEAD_DIM, HIDDEN, KEYS, Dims
from .common.arch import GFX942
from .layer import (
    EPOCH_WORDS,
    K2_POINTS,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)

__all__ = ["MAX_TOKENS", "AttnWeights", "DSV41MonoLayer", "MonoLayerWeights"]

_LOG2E = 1.4426950408889634
HC = 4


def _ensure_writable_flydsl_cache() -> None:
    """Aiter points FlyDSL at its bundled, read-only cache; new kernels need a
    writable one."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


@dataclass
class AttnWeights:
    """One layer's attention tensors on this rank, in vLLM's loaded layout."""

    layer_id: int
    wqkv: torch.Tensor  # [1792, 5120] e4m3: [wq_a; wkv]
    wqkv_scale: torch.Tensor  # [56, 160] uint8 E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wq_b: torch.Tensor  # [H * 512, 1280] e4m3
    wq_b_scale: torch.Tensor  # [H * 16, 40] uint8
    wo_a: torch.Tensor  # [G * 1024, 4096] e4m3
    wo_a_scale: torch.Tensor  # [G * 32, 128] uint8
    wo_b: torch.Tensor  # [5120, G * 1024] e4m3
    wo_b_scale: torch.Tensor  # [160, G * 32] uint8
    attn_sink: torch.Tensor  # [>= H] f32
    cos_sin: torch.Tensor  # [positions, 64] f32 (the layer's rope table)
    ratio: int  # 0 / 1 / 2: the layer's compress ratio

    def check(self, d: Dims) -> None:
        h, g = d.heads, d.groups
        # gfx942 reads its own copies (weights942.linear_copy): FNUZ bytes in
        # the same shapes, as every attention K is a multiple of 128.
        fp8 = torch.uint8 if GFX942 else torch.float8_e4m3fn
        want = {
            "wqkv": ((1792, HIDDEN), fp8),
            "wqkv_scale": ((56, HIDDEN // 32), torch.uint8),
            "wq_b": ((h * HEAD_DIM, 1280), fp8),
            "wq_b_scale": ((h * HEAD_DIM // 32, 40), torch.uint8),
            "wo_a": ((g * 1024, d.group_k), fp8),
            "wo_a_scale": ((g * 32, d.group_k // 32), torch.uint8),
            "wo_b": ((HIDDEN, g * 1024), fp8),
            "wo_b_scale": ((HIDDEN // 32, g * 32), torch.uint8),
        }
        for name, (shape, dtype) in want.items():
            t = getattr(self, name)
            assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous(), (
                f"layer {self.layer_id} {name}: {tuple(t.shape)} {t.dtype}, "
                f"want {shape} {dtype}"
            )
        assert self.cos_sin.dtype == torch.float32 and self.cos_sin.shape[-1] == 64


@dataclass
class MonoLayerWeights:
    """One layer's tensors on this rank, in vLLM's loaded layout (``attn``:
    None for a layer whose attention vLLM runs)."""

    attn: AttnWeights | None
    hc_attn_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_attn_scale: torch.Tensor  # [3] f32
    hc_attn_base: torch.Tensor  # [24] f32
    attn_norm: torch.Tensor  # [5120] bf16
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    ffn_norm: torch.Tensor
    gate_w: torch.Tensor  # [384, 5120] bf16
    bias: torch.Tensor  # [384] f32 (e_score_correction_bias)
    w13: torch.Tensor  # [384, 2 inter, 2560] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [384, 5120, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, 5120] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, 160] E8M0
    sw2: torch.Tensor  # shared down [5120, inter] e4m3
    sw2_s: torch.Tensor


class DSV41MonoLayer:
    """The mono decode runner of one TP rank (``group``: its TP group, for the
    peer-memory handle exchange; a gloo / CPU group)."""

    def __init__(
        self,
        tp: int,
        rank: int,
        group,
        device: torch.device | str = "cuda",
        timeline: bool = False,
    ):
        _ensure_writable_flydsl_cache()
        from .common.peer_memory import PeerBuffer

        self.tp, self.rank = tp, rank
        # A timeline runner builds K1 and K2 with clock stamps. Each CTA
        # writes its stamps into its own record of these buffers (100 MHz
        # ticks), and a profiling script reads them after a step.
        self.timeline = timeline
        self.tl1 = torch.zeros(BLOCKS, FRONT_POINTS, dtype=torch.int64, device=device)
        self.tl2 = torch.zeros(BLOCKS, K2_POINTS, dtype=torch.int64, device=device)
        # The FFN launch uses K2's record layout from the seam point on, plus
        # its own start at point 0.
        self.tl3 = torch.zeros(BLOCKS, K2_POINTS, dtype=torch.int64, device=device)
        self.d = Dims(tp)
        dev = self.device = torch.device(device)
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.q = torch.zeros(
            MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        self._dummy = torch.zeros(4, dtype=torch.int32, device=dev)
        self._zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        # the attention seam's outputs, K1 -> K2
        self.res_mid = torch.zeros(
            MAX_TOKENS, HC, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        self._kernels: dict = {}

    def scratch(self, tokens: int) -> torch.Tensor:
        """Step width ``tokens``'s scratch: a width's layout has memory of its own
        (``layer.scratch_layout``). Allocated at the width's first step, which is
        eager: vLLM runs every graph's batch eagerly before capturing it."""
        buf = self._scratch.get(tokens)
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"DSv4.1 mono decode: step width {tokens} first reached inside "
                    "a CUDA graph capture"
                )
            buf = torch.zeros(
                scratch_bytes(tokens, self.tp), dtype=torch.uint8, device=self.device
            )
            self._scratch[tokens] = buf
        return buf

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def kernels(self, tokens: int, ratio: int, index: bool = False):
        """(K1, K2) for steps of ``tokens`` rows on the layers of compress ratio
        ``ratio``. ``index`` gives an index layer's K1 (MonoBuild.index). Both
        K1 builds run before the same K2, so K2 is built once."""
        k1_key = (tokens, ratio, "k1", index)
        k2_key = (tokens, ratio, "k2")
        if k1_key not in self._kernels:
            self._kernels[k1_key] = build_mono_k1(
                MonoBuild(tokens, self.tp, ratio, timeline=self.timeline, index=index)
            )
        if k2_key not in self._kernels:
            self._kernels[k2_key] = build_mono_k2(
                MonoBuild(tokens, self.tp, ratio, timeline=self.timeline)
            )
        return self._kernels[k1_key], self._kernels[k2_key]

    def ffn_kernel(self, tokens: int, experts: int, topk: int):
        key = (tokens, "ffn", experts, topk)
        if key not in self._kernels:
            self._kernels[key] = build_mono_ffn(
                MonoBuild(
                    tokens,
                    self.tp,
                    0,
                    timeline=self.timeline,
                    experts=experts,
                    topk=topk,
                )
            )
        return self._kernels[key]

    @staticmethod
    def _outs(M: int, residual: torch.Tensor) -> tuple:
        dev = residual.device
        return (
            torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.empty_like(residual),
            torch.empty(M, HC, 1, dtype=torch.float32, device=dev),
            torch.empty(M, HC, HC, dtype=torch.float32, device=dev),
            torch.empty(M, HC, dtype=torch.float32, device=dev),
        )

    def forward(
        self,
        w: MonoLayerWeights,
        x: torch.Tensor,  # [M, 5120] bf16: the previous FFN output (reduced)
        residual: torch.Tensor,  # [M, 4, 5120] bf16
        post_mix: torch.Tensor,  # [M, 4, 1] f32
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8
        swa_indices: torch.Tensor,  # [>= M, 1, 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        topk_indices: torch.Tensor | None = None,
        comp_cache: torch.Tensor | None = None,
        comp_block_table: torch.Tensor | None = None,
        outs: tuple | None = None,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix): the layer's outputs
        (``outs``: buffers to write them into)."""
        M = x.shape[0]
        assert self.supports(M), M
        a = w.attn
        assert a is not None
        k1, k2 = self.kernels(M, a.ratio)
        if outs is None:
            outs = self._outs(M, residual)
        self._k1(
            k1, w, x, residual, post_mix, res_mix, pre_mix, positions, slot_mapping,
            swa_cache, swa_indices, swa_lens, token_to_req, topk_indices, comp_cache,
            comp_block_table,
        )  # fmt: skip
        comp = self._comp(a.ratio, comp_cache)
        self._k2(
            k2,
            w,
            self.q,
            self.res_mid,
            self.post_a,
            self.comb_a,
            self.pre_a,
            positions,
            swa_cache,
            comp,
            outs,
        )
        return outs

    def index_front(
        self,
        w: MonoLayerWeights,
        x: torch.Tensor,  # [M, 5120] bf16: the previous FFN output (reduced)
        residual: torch.Tensor,  # [M, 4, 5120] bf16
        post_mix: torch.Tensor,  # [M, 4, 1] f32
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8
        swa_indices: torch.Tensor,  # [>= M, 1, 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        x_out: torch.Tensor,  # [M, 5120] bf16: the attention's normed input
        qr_out: torch.Tensor,  # [M, q_lora_rank] bf16: the normed q latent
    ) -> None:
        """An index layer's K1: the attention seam and front as in ``forward``,
        without the keys, which need the layer's top-k from vLLM's indexer.
        It also writes the indexer's and compressor's inputs, x_out and qr_out.
        ``index_back`` then runs the rest of the layer."""
        M = x.shape[0]
        assert self.supports(M), M
        a = w.attn
        assert a is not None and a.ratio in (1, 2)
        assert x_out.dtype == qr_out.dtype == torch.bfloat16
        assert x_out.shape == (M, HIDDEN) and x_out.is_contiguous()
        assert qr_out.shape[0] == M and qr_out.is_contiguous()
        k1, _ = self.kernels(M, a.ratio, index=True)
        self._k1(
            k1, w, x, residual, post_mix, res_mix, pre_mix, positions, slot_mapping,
            swa_cache, swa_indices, swa_lens, token_to_req, None, None, None,
            x_out=x_out, qr_out=qr_out,
        )  # fmt: skip

    def index_back(
        self,
        w: MonoLayerWeights,
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8
        swa_indices: torch.Tensor,  # [>= M, 1, 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        topk_indices: torch.Tensor,  # [>= M, 512] int32: the indexer's
        comp_cache: torch.Tensor,  # [blocks, block, 584] uint8
        comp_block_table: torch.Tensor,  # [reqs, blocks] int32
        outs: tuple | None = None,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix) of an index layer
        after ``index_front`` and vLLM's indexer: the keys by
        ``keylist.launch``, then K2 on K1's q and attention seam outputs."""
        M = positions.shape[0]
        return self.mixed(
            w,
            self.q[:M],
            self.res_mid[:M],
            self.post_a[:M],
            self.comb_a[:M],
            self.pre_a[:M],
            positions,
            slot_mapping,
            swa_cache,
            swa_indices,
            swa_lens,
            token_to_req,
            topk_indices,
            comp_cache,
            comp_block_table,
            outs,
        )

    def _comp(self, ratio: int, comp_cache: torch.Tensor | None) -> tuple:
        """(the compressed cache or a dummy, its block stride, its rows a
        block), as K2 takes them."""
        if ratio:
            assert comp_cache is not None
            return comp_cache, comp_cache.stride(0), comp_cache.shape[1]
        return self._dummy, 0, 1

    def _k1(
        self,
        k1,
        w: MonoLayerWeights,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        token_to_req: torch.Tensor,
        topk_indices: torch.Tensor | None,
        comp_cache: torch.Tensor | None,
        comp_block_table: torch.Tensor | None,
        x_out: torch.Tensor | None = None,
        qr_out: torch.Tensor | None = None,
    ) -> None:
        """K1 on this stream: the attention seam into res_mid, post_a, comb_a
        and pre_a, and the attention front into q, the SWA cache and the keys
        (KT, KLEN). An index layer's K1 writes x_out and qr_out instead of the
        keys, and gets no top-k (None for the three compressed arguments)."""
        a = w.attn
        assert a is not None, "K1 runs only on layers with attention weights"
        ratio = a.ratio
        st = torch.cuda.current_stream()
        for t_ in (x, residual, post_mix, res_mix, pre_mix, positions, slot_mapping):
            assert t_.is_contiguous()
        assert positions.dtype == slot_mapping.dtype == torch.int64
        assert swa_indices.dtype == swa_lens.dtype == token_to_req.dtype == torch.int32
        assert swa_indices.stride(0) == 128 and swa_cache.dtype == torch.uint8
        assert swa_cache.shape[-1] == 584 and swa_cache.stride(1) == 584
        swa_block = swa_cache.shape[1]
        index = x_out is not None
        assert index == (qr_out is not None)
        if ratio and not index:
            assert topk_indices is not None and comp_cache is not None
            assert comp_block_table is not None
            assert topk_indices.dtype == torch.int32 and topk_indices.stride(0) == 512
            assert comp_cache.dtype == torch.uint8 and comp_cache.stride(1) == 584
            assert comp_block_table.dtype == torch.int32
            comp_block = comp_cache.shape[1]
            bt, bt_stride, topk = (
                comp_block_table,
                comp_block_table.stride(0),
                topk_indices,
            )
        else:
            comp_block = 1
            bt, bt_stride, topk = self._dummy, 0, self._dummy
        k1(
            residual.data_ptr(),
            x.data_ptr(),
            post_mix.data_ptr(),
            res_mix.data_ptr(),
            pre_mix.data_ptr(),
            w.hc_attn_fn.data_ptr(),
            w.hc_attn_scale.data_ptr(),
            w.hc_attn_base.data_ptr(),
            w.attn_norm.data_ptr(),
            self.res_mid.data_ptr(),
            self.post_a.data_ptr(),
            self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            a.wqkv.data_ptr(),
            a.wqkv_scale.data_ptr(),
            a.q_norm.data_ptr(),
            a.kv_norm.data_ptr(),
            a.wq_b.data_ptr(),
            a.wq_b_scale.data_ptr(),
            a.cos_sin.data_ptr(),
            positions.data_ptr(),
            slot_mapping.data_ptr(),
            swa_cache.data_ptr(),
            swa_cache.stride(0),
            swa_block,
            self.q.data_ptr(),
            swa_indices.data_ptr(),
            swa_lens.data_ptr(),
            comp_block,
            bt.data_ptr(),
            bt_stride,
            topk.data_ptr(),
            token_to_req.data_ptr(),
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            x_out.data_ptr() if x_out is not None else 0,
            qr_out.data_ptr() if qr_out is not None else 0,
            self.scratch(x.shape[0]).data_ptr(),
            self.epoch.data_ptr(),
            self.tl1.data_ptr() if self.timeline else 0,
            stream=st,
        )

    def mixed(
        self,
        w: MonoLayerWeights,
        q: torch.Tensor,  # [M, heads, 512] bf16: vLLM's q, roped, contiguous
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        positions: torch.Tensor,  # [M] int64
        slot_mapping: torch.Tensor,  # [M] int64
        swa_cache: torch.Tensor,  # [blocks, block, 584] uint8
        swa_indices: torch.Tensor,  # [>= M, 1, 128] int32
        swa_lens: torch.Tensor,  # [>= M] int32
        token_to_req: torch.Tensor,  # [>= M] int32
        topk_indices: torch.Tensor,  # [>= M, 512] int32
        comp_cache: torch.Tensor,  # [blocks, block, 584] uint8
        comp_block_table: torch.Tensor,  # [reqs, blocks] int32
        outs: tuple | None = None,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix) of a layer whose
        attention front vLLM ran: its projections, KV insert, compressor and
        indexer. K2 takes vLLM's q and the attention seam's outputs where it
        takes K1's in ``forward``, after ``keylist.launch`` wrote the keys
        K1's ``stage_kt`` would have written."""
        from .keylist import launch as keylist_launch

        M = q.shape[0]
        assert self.supports(M), M
        a = w.attn
        assert a is not None and a.ratio in (1, 2)
        assert q.dtype == torch.bfloat16 and q.is_contiguous()
        assert q.shape[1:] == (self.d.heads, HEAD_DIM), q.shape
        for t_ in (residual, post_mix, res_mix, pre_mix, positions, slot_mapping):
            assert t_.is_contiguous()
        assert positions.dtype == slot_mapping.dtype == torch.int64
        assert swa_indices.dtype == swa_lens.dtype == token_to_req.dtype == torch.int32
        assert topk_indices.dtype == comp_block_table.dtype == torch.int32
        assert swa_cache.dtype == torch.uint8 and swa_cache.stride(1) == 584
        assert comp_cache.dtype == torch.uint8 and comp_cache.stride(1) == 584
        if outs is None:
            outs = self._outs(M, residual)
        keylist_launch(
            self.kt,
            self.klen,
            slot_mapping,
            positions,
            swa_indices,
            swa_lens,
            token_to_req,
            a.ratio,
            topk=topk_indices,
            comp_block_table=comp_block_table,
            comp_block=comp_cache.shape[1],
        )
        _, k2 = self.kernels(M, a.ratio)
        self._k2(
            k2,
            w,
            q,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            positions,
            swa_cache,
            (comp_cache, comp_cache.stride(0), comp_cache.shape[1]),
            outs,
        )
        return outs

    def _k2(
        self,
        k2,
        w: MonoLayerWeights,
        q: torch.Tensor,
        res_mid: torch.Tensor,
        post_a: torch.Tensor,
        comb_a: torch.Tensor,
        pre_a: torch.Tensor,
        positions: torch.Tensor,
        swa_cache: torch.Tensor,
        comp: tuple,
        outs: tuple,
    ) -> None:
        """K2 on this stream: the attention back on q and the keys in KT and
        KLEN, then the FFN seam from the attention seam's outputs (res_mid,
        post_a, comb_a, pre_a) and the MoE. ``comp`` is (the compressed
        cache or a dummy, its block stride, its rows a block)."""
        a = w.attn
        assert a is not None, "K2 runs only on layers with attention weights"
        comp_cache, comp_stride, comp_block = comp
        out, res_out, post_out, comb_out, pre_out = outs
        k2(
            q.data_ptr(),
            swa_cache.data_ptr(),
            swa_cache.stride(0),
            swa_cache.shape[1],
            comp_cache.data_ptr(),
            comp_stride,
            comp_block,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            positions.data_ptr(),
            a.attn_sink.data_ptr(),
            self._qk_scale,
            a.cos_sin.data_ptr(),
            a.wo_a.data_ptr(),
            a.wo_a_scale.data_ptr(),
            a.wo_b.data_ptr(),
            a.wo_b_scale.data_ptr(),
            self._zero_rec.data_ptr(),
            res_mid.data_ptr(),
            post_a.data_ptr(),
            comb_a.data_ptr(),
            pre_a.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            # The step's rows are the positions' count. q may be the whole
            # MAX_TOKENS buffer.
            self.scratch(positions.shape[0]).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            self.tl2.data_ptr() if self.timeline else 0,
            stream=torch.cuda.current_stream(),
        )

    def ffn(
        self,
        w: MonoLayerWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
        topk: int = 6,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix) of a layer whose
        attention vLLM ran: the attention's TP reduce, the FFN seam, the MoE and
        its all-reduce in one launch. The routed expert count is the gate
        weight's row count, and ``topk`` is the layer's picks a token: 384 / 6
        for the target, 128 / 3 for a DSpark draft layer."""
        M = part.shape[0]
        assert self.supports(M), M
        for t_ in (part, residual, post_mix, res_mix, pre_mix):
            assert t_.is_contiguous()
        assert part.dtype == residual.dtype == torch.bfloat16
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        self.ffn_kernel(M, w.gate_w.shape[0], topk)(
            part.data_ptr(),
            residual.data_ptr(),
            post_mix.data_ptr(),
            res_mix.data_ptr(),
            pre_mix.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            self.tl3.data_ptr() if self.timeline else 0,
            stream=torch.cuda.current_stream(),
        )
        return outs
