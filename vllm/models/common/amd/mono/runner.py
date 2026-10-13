# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode layer's host side.

One ``MonoRunner`` a TP rank runs every mono layer of a decode step (at most
``spec.max_tokens`` rows): two persistent launches a layer (``layer``) from the layer's
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .common.plan import BLOCKS
from .layer import (
    EPOCH_WORDS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)
from .spec import HC, HEAD_DIM, Dims, MonoSpec

__all__ = ["BLOCKS", "AttnWeights", "MonoLayerWeights", "MonoRunner"]

_LOG2E = 1.4426950408889634


def _ensure_writable_flydsl_cache() -> None:
    """Aiter points FlyDSL at its bundled, read-only cache; new kernels need a
    writable one."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _check_tensors(owner, tag: str, want: dict, lds: dict | None = None) -> None:
    """Each named tensor of ``owner`` has the (shape, dtype) the kernels read,
    contiguous -- or a 2-D one's rows ``lds[name]`` apart: their buffer loads
    are unbounded, so a mismatch would read garbage rather than fault."""
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        ld = (lds or {}).get(name)
        laid = t.is_contiguous() if ld is None else t.stride() == (ld, 1)
        assert tuple(t.shape) == shape and t.dtype == dtype and laid, (
            f"{tag} {name}: {tuple(t.shape)} {t.dtype} {t.stride()}, want {shape} "
            f"{dtype}" + ("" if ld is None else f" rows {ld} apart")
        )


@dataclass
class AttnWeights:
    """One layer's attention tensors on this rank, in vLLM's loaded layout."""

    layer_id: int
    wqkv: torch.Tensor  # [q_rank + 512, hidden] e4m3: [wq_a; wkv]
    wqkv_scale: torch.Tensor  # [rows / 32, hidden / 32] uint8 E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [q_rank] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wq_b: torch.Tensor  # [H * 512, q_rank] e4m3
    wq_b_scale: torch.Tensor  # [H * 16, q_rank / 32] uint8
    wo_a: torch.Tensor  # [G * 1024, group_k] e4m3
    wo_a_scale: torch.Tensor  # [G * 32, group_k / 32] uint8
    wo_b: torch.Tensor  # [hidden, G * 1024] e4m3
    wo_b_scale: torch.Tensor  # [hidden / 32, G * 32] uint8
    attn_sink: torch.Tensor  # [>= H] f32
    cos_sin: torch.Tensor  # [positions, 64] f32 (the layer's rope table)
    ratio: int  # the layer's compress ratio

    def check(self, d: Dims) -> None:
        h, g, k, q = d.heads, d.groups, d.hidden, d.q_rank
        want = {
            "wqkv": ((d.qkv_rows, k), torch.float8_e4m3fn),
            "wqkv_scale": ((d.qkv_rows // 32, k // 32), torch.uint8),
            "wq_b": ((h * HEAD_DIM, q), torch.float8_e4m3fn),
            "wq_b_scale": ((h * HEAD_DIM // 32, q // 32), torch.uint8),
            "wo_a": ((g * 1024, d.group_k), torch.float8_e4m3fn),
            "wo_a_scale": ((g * 32, d.group_k // 32), torch.uint8),
            "wo_b": ((k, g * 1024), torch.float8_e4m3fn),
            "wo_b_scale": ((k // 32, g * 32), torch.uint8),
        }
        lds = {"wq_b": d.ld(q), "wo_a": d.ld(d.group_k)}
        _check_tensors(self, f"layer {self.layer_id}", want, lds)
        assert self.cos_sin.dtype == torch.float32 and self.cos_sin.shape[-1] == 64


@dataclass
class MonoLayerWeights:
    """One layer's tensors on this rank, in vLLM's loaded layout (``attn``:
    None for a layer whose attention vLLM runs)."""

    attn: AttnWeights | None
    hc_attn_fn: torch.Tensor  # [24, 4 * hidden] f32
    hc_attn_scale: torch.Tensor  # [3] f32
    hc_attn_base: torch.Tensor  # [24] f32
    attn_norm: torch.Tensor  # [hidden] bf16
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    ffn_norm: torch.Tensor
    gate_w: torch.Tensor  # [experts, hidden] bf16
    # [experts] f32 (e_score_correction_bias); None for a hash layer, whose
    # tokens' experts each step gives (``route_ids``)
    bias: torch.Tensor | None
    w13: torch.Tensor  # [experts, 2 inter, hidden / 2] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [experts, hidden, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, hidden] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, hidden / 32] E8M0
    sw2: torch.Tensor  # shared down [hidden, inter] e4m3
    sw2_s: torch.Tensor

    def check(self, d: Dims) -> None:
        """The MoE's tensors: every routed expert, TP-sharded intermediates (the
        routed one padded as the loader pads it), AITER's A8W4 layout, 32 x 32
        E8M0 blocks for the shared expert."""
        fp4, e4m3, u8 = torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.uint8
        e, k = d.spec.experts, d.hidden
        want = {
            "gate_w": ((e, k), torch.bfloat16),
            "w13": ((e, 2 * d.inter, k // 2), fp4),
            "w13_s": ((e * 2 * d.inter, k // 32), u8),
            "w2": ((e, k, d.inter // 2), fp4),
            "w2_s": ((e * k, d.down_scale_cols), u8),
            "sgu": ((2 * d.sh_inter, k), e4m3),
            "sgu_s": ((2 * d.sh_inter // 32, k // 32), u8),
            "sw2": ((k, d.sh_inter), e4m3),
            "sw2_s": ((k // 32, d.sh_inter // 32), u8),
        }
        if self.bias is not None:
            want["bias"] = ((e,), torch.float32)
        _check_tensors(self, "MoE", want)


class MonoRunner:
    """The mono decode runner of one TP rank for model ``spec`` (``group``: its
    TP group, for the peer-memory handle exchange; a gloo / CPU group)."""

    def __init__(
        self,
        spec: MonoSpec,
        tp: int,
        rank: int,
        group,
        device: torch.device | str = "cuda",
        max_topk: int = 0,
        timeline: bool = False,
    ):
        """``max_topk``: the widest key table a layer's build takes (``Dims.topk``),
        past the model's top-k. ``timeline``: the kernels' stage stamps into
        ``tl`` (K1's, K2's: [BLOCKS, points] of the device clock), a profile."""
        _ensure_writable_flydsl_cache()
        from .common.peer_memory import PeerBuffer

        self.spec, self.tp, self.rank = spec, tp, rank
        self.d = d = Dims(spec, tp)
        keys = Dims(spec, tp, max(max_topk, spec.index_topk)).keys
        rows = self.max_tokens = spec.max_tokens
        dev = self.device = torch.device(device)
        self._scratch: dict[tuple[int, int], torch.Tensor] = {}
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.q = torch.zeros(rows, d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.kt = torch.zeros(rows * keys, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * rows, dtype=torch.int32, device=dev)
        self._dummy = torch.zeros(4, dtype=torch.int32, device=dev)
        self._zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        # the first layer's incoming pre-mix: stream 0 of the broadcast embedding
        self._first_pre = torch.zeros(rows, HC, dtype=torch.float32, device=dev)
        self._first_pre[:, 0] = 1.0
        # an index source's K1 outputs, vLLM's indexer's inputs
        self.normed = torch.zeros(rows, d.hidden, dtype=torch.bfloat16, device=dev)
        self.qr = torch.zeros(rows, d.q_rank, dtype=torch.bfloat16, device=dev)
        # the attention seam's outputs, K1 -> K2
        self.res_mid = torch.zeros(rows, HC, d.hidden, dtype=torch.bfloat16, device=dev)
        self.post_a = torch.zeros(rows, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(rows, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(rows, HC, dtype=torch.float32, device=dev)
        self.peer = PeerBuffer(2 * peer_half_bytes(spec, tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        self._kernels: dict = {}
        self.timeline = timeline
        self.tl = (
            torch.zeros(2, BLOCKS, 64, dtype=torch.int64, device=dev)
            if timeline
            else None
        )

    def scratch(self, tokens: int, topk: int = 0) -> torch.Tensor:
        """Step width ``tokens``'s scratch (of the builds of key table width
        ``topk``): a layout has memory of its own (``layer.scratch_layout``).
        Allocated at the width's first step, which is eager: vLLM runs every
        graph's batch eagerly before capturing it."""
        buf = self._scratch.get((tokens, topk))
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"Mono decode: step width {tokens} first reached inside "
                    "a CUDA graph capture"
                )
            buf = torch.zeros(
                scratch_bytes(self.spec, tokens, self.tp, topk),
                dtype=torch.uint8,
                device=self.device,
            )
            self._scratch[(tokens, topk)] = buf
        return buf

    def first_pre_mix(self, tokens: int) -> torch.Tensor:
        return self._first_pre[:tokens]

    def supports(self, tokens: int) -> bool:
        return 1 <= tokens <= self.max_tokens

    def two_hop(self, tokens: int) -> bool:
        """A step width's all-reduces in two hops (``stages.allreduce``)."""
        rows = self.spec.two_hop_rows
        return bool(rows) and tokens >= rows

    def kernels(
        self,
        tokens: int,
        ratio: int,
        fold: bool = True,
        index: bool = False,
        hash_route: bool = False,
        topk: int = 0,
        global_keys: bool = False,
    ):
        key = (tokens, ratio, fold, index, hash_route, topk, global_keys)
        if key not in self._kernels:
            b = MonoBuild(
                self.spec,
                tokens,
                self.tp,
                ratio,
                fold=fold,
                index=index,
                hash_route=hash_route,
                topk=topk,
                global_keys=global_keys,
                timeline=self.timeline,
                two_hop=self.two_hop(tokens),
            )
            self._kernels[key] = (build_mono_k1(b), build_mono_k2(b))
        return self._kernels[key]

    def ffn_kernel(self, tokens: int, hash_route: bool = False):
        key = (tokens, "ffn", hash_route)
        if key not in self._kernels:
            b = MonoBuild(
                self.spec,
                tokens,
                self.tp,
                0,
                hash_route=hash_route,
                two_hop=self.two_hop(tokens),
            )
            self._kernels[key] = build_mono_ffn(b)
        return self._kernels[key]

    @staticmethod
    def _route(w: MonoLayerWeights, route_ids: torch.Tensor | None) -> torch.Tensor:
        """The MoE's bias argument: the layer's bias, or a hash layer's step
        experts ([M, topk] int32)."""
        if route_ids is None:
            assert w.bias is not None
            return w.bias
        assert route_ids.dtype == torch.int32 and route_ids.is_contiguous()
        return route_ids

    def _outs(self, M: int, residual: torch.Tensor) -> tuple:
        dev = residual.device
        return (
            torch.empty(M, self.d.hidden, dtype=torch.bfloat16, device=dev),
            torch.empty_like(residual),
            torch.empty(M, HC, 1, dtype=torch.float32, device=dev),
            torch.empty(M, HC, HC, dtype=torch.float32, device=dev),
            torch.empty(M, HC, dtype=torch.float32, device=dev),
        )

    def forward(
        self,
        w: MonoLayerWeights,
        x: torch.Tensor,  # [M, hidden] bf16: the previous FFN output (reduced)
        residual: torch.Tensor,  # [M, 4, hidden] bf16
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
        fold: bool = True,
        select: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
        route_ids: torch.Tensor | None = None,
        topk: int = 0,
        global_keys: bool = False,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix): the layer's outputs
        (``outs``: buffers to write them into). ``select(normed, qr)``: an index
        source's top-k selection into ``topk_indices``, run between the
        kernels on K1's normed input and qr. ``route_ids``: a hash layer's
        experts of the step's tokens. ``topk``: the build's key table width
        (``Dims.topk``); ``global_keys``: ``topk_indices`` holds compressed
        slots (a row of its own width), not indices through
        ``comp_block_table``."""
        M = x.shape[0]
        assert self.supports(M), M
        a = w.attn
        assert a is not None
        ratio = a.ratio
        index = select is not None
        assert not index or ratio
        k1, k2 = self.kernels(
            M, ratio, fold, index, route_ids is not None, topk, global_keys
        )
        scratch = self.scratch(M, topk)
        st = torch.cuda.current_stream()
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        for t_ in (x, residual, post_mix, res_mix, pre_mix, positions, slot_mapping):
            assert t_.is_contiguous()
        assert positions.dtype == slot_mapping.dtype == torch.int64
        assert swa_indices.dtype == swa_lens.dtype == token_to_req.dtype == torch.int32
        assert swa_indices.stride(0) == 128 and swa_cache.dtype == torch.uint8
        assert swa_cache.shape[-1] == 584 and swa_cache.stride(1) == 584
        swa_block = swa_cache.shape[1]
        if ratio:
            assert topk_indices is not None and comp_cache is not None
            assert global_keys or comp_block_table is not None
            assert topk_indices.dtype == torch.int32
            assert comp_cache.dtype == torch.uint8 and comp_cache.stride(1) == 584
            comp, comp_stride, comp_block = (
                comp_cache,
                comp_cache.stride(0),
                comp_cache.shape[1],
            )
            if global_keys:
                # the table's own width where the block table's stride was
                width = topk_indices.stride(0)
                assert width <= Dims(self.spec, self.tp, topk).topk_width
                bt, bt_stride, topk_t = self._dummy, width, topk_indices
            else:
                assert comp_block_table is not None
                assert topk_indices.stride(0) == self.d.topk_width
                assert comp_block_table.dtype == torch.int32
                bt, bt_stride, topk_t = (
                    comp_block_table,
                    comp_block_table.stride(0),
                    topk_indices,
                )
        else:
            comp, comp_stride, comp_block = self._dummy, 0, 1
            bt, bt_stride, topk_t = self._dummy, 0, self._dummy
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
            topk_t.data_ptr(),
            token_to_req.data_ptr(),
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            self.normed.data_ptr(),
            self.qr.data_ptr(),
            scratch.data_ptr(),
            self.epoch.data_ptr(),
            self.tl[0].data_ptr() if self.tl is not None else 0,
            stream=st,
        )
        if select is not None:
            select(self.normed[:M], self.qr[:M])
        k2(
            self.q.data_ptr(),
            swa_cache.data_ptr(),
            swa_cache.stride(0),
            swa_block,
            comp.data_ptr(),
            comp_stride,
            comp_block,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            slot_mapping.data_ptr(),
            swa_indices.data_ptr(),
            swa_lens.data_ptr(),
            token_to_req.data_ptr(),
            topk_t.data_ptr(),
            bt.data_ptr(),
            bt_stride,
            positions.data_ptr(),
            a.attn_sink.data_ptr(),
            self._qk_scale,
            a.cos_sin.data_ptr(),
            a.wo_a.data_ptr(),
            a.wo_a_scale.data_ptr(),
            a.wo_b.data_ptr(),
            a.wo_b_scale.data_ptr(),
            self._zero_rec.data_ptr(),
            self.res_mid.data_ptr(),
            self.post_a.data_ptr(),
            self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            self._route(w, route_ids).data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            scratch.data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            self.tl[1].data_ptr() if self.tl is not None else 0,
            stream=st,
        )
        return outs

    def ffn(
        self,
        w: MonoLayerWeights,
        part: torch.Tensor,  # [M, hidden] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, hidden] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
        route_ids: torch.Tensor | None = None,
    ):
        """-> (out, residual, post_mix, res_mix, pre_mix) of a layer whose
        attention vLLM ran: the attention's TP reduce, the FFN seam, the MoE and
        its all-reduce in one launch."""
        M = part.shape[0]
        assert self.supports(M), M
        for t_ in (part, residual, post_mix, res_mix, pre_mix):
            assert t_.is_contiguous()
        assert part.dtype == residual.dtype == torch.bfloat16
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        self.ffn_kernel(M, route_ids is not None)(
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
            self._route(w, route_ids).data_ptr(),
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
            stream=torch.cuda.current_stream(),
        )
        return outs
