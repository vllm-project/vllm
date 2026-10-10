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
from typing import TYPE_CHECKING

import torch

from vllm.models.common.mono import check_tensors

from .attention.plan import HEAD_DIM, HIDDEN, KEYS, Dims
from .common.plan import BLOCKS
from .layer import (
    EPOCH_WORDS,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS

if TYPE_CHECKING:
    from vllm.models.common.mono import MonoRuntime

    from .common.peer_memory import PeerBuffer

__all__ = [
    "BLOCKS",
    "EPOCH_WORDS",
    "MAX_TOKENS",
    "PEER",
    "AttnWeights",
    "DSV41MonoLayer",
    "MonoLayerWeights",
    "peer_factory",
]

_LOG2E = 1.4426950408889634
HC = 4
PEER = "peer"


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
        want = {
            "wqkv": ((1792, HIDDEN), torch.float8_e4m3fn),
            "wqkv_scale": ((56, HIDDEN // 32), torch.uint8),
            "wq_b": ((h * HEAD_DIM, 1280), torch.float8_e4m3fn),
            "wq_b_scale": ((h * HEAD_DIM // 32, 40), torch.uint8),
            "wo_a": ((g * 1024, d.group_k), torch.float8_e4m3fn),
            "wo_a_scale": ((g * 32, d.group_k // 32), torch.uint8),
            "wo_b": ((HIDDEN, g * 1024), torch.float8_e4m3fn),
            "wo_b_scale": ((HIDDEN // 32, g * 32), torch.uint8),
        }
        check_tensors(self, f"layer {self.layer_id}", want)
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

    def check(self, tp: int) -> None:
        """The MoE's tensors: 384 experts, TP-sharded intermediates (the routed
        one padded as the loader pads it), AITER's A8W4 layout, 32 x 32 E8M0
        blocks for the shared expert."""
        d, fp4, e4m3, u8 = (
            MoeDims(tp),
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.uint8,
        )
        want = {
            "gate_w": ((EXPERTS, HIDDEN), torch.bfloat16),
            "bias": ((EXPERTS,), torch.float32),
            "w13": ((EXPERTS, 2 * d.inter, HIDDEN // 2), fp4),
            "w13_s": ((EXPERTS * 2 * d.inter, HIDDEN // 32), u8),
            "w2": ((EXPERTS, HIDDEN, d.inter // 2), fp4),
            "w2_s": ((EXPERTS * HIDDEN, d.down_scale_cols), u8),
            "sgu": ((2 * d.sh_inter, HIDDEN), e4m3),
            "sgu_s": ((2 * d.sh_inter // 32, HIDDEN // 32), u8),
            "sw2": ((HIDDEN, d.sh_inter), e4m3),
            "sw2_s": ((HIDDEN // 32, d.sh_inter // 32), u8),
        }
        check_tensors(self, "MoE", want)


def peer_factory(nbytes: int, rank: int, world: int, group, device) -> PeerBuffer:
    """Zeroed symmetric peer memory for the in-kernel TP all-reduces."""
    from .common.peer_memory import PeerBuffer

    peer = PeerBuffer(nbytes, group, rank, world, device)
    peer.bytes.zero_()
    return peer


class DSV41MonoLayer:
    """The mono decode kernels of one TP rank, over a shared ``MonoRuntime``.

    The runtime holds what the persistent launches wait on (a width's scratch,
    the symmetric peer buffer, the mailbox epoch); this object holds what the
    kernels read between steps (the staging buffers and the built kernels).
    """

    def __init__(self, rt: MonoRuntime):
        _ensure_writable_flydsl_cache()
        self.rt = rt
        self.tp, self.rank = rt.world, rt.rank
        self.d = Dims(self.tp)
        dev = self.device = rt.device
        assert rt.epoch is not None, "the mono runtime needs EPOCH_WORDS"
        self.epoch = rt.epoch
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
        self.peer = rt.reserve(PEER, peer_bytes=2 * peer_half_bytes(self.tp)).peer
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        self._kernels: dict = {}

    def scratch(self, tokens: int) -> torch.Tensor:
        """Step width ``tokens``'s scratch: a width's layout has memory of its own
        (``layer.scratch_layout``)."""
        res = self.rt.reserve(
            (tokens, "layer"), scratch_bytes=scratch_bytes(tokens, self.tp)
        )
        assert res.scratch is not None
        return res.scratch

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def kernels(self, tokens: int, ratio: int):
        key = (tokens, ratio)
        if key not in self._kernels:
            b = MonoBuild(tokens, self.tp, ratio)
            self._kernels[key] = (build_mono_k1(b), build_mono_k2(b))
        return self._kernels[key]

    def ffn_kernel(self, tokens: int):
        key = (tokens, "ffn")
        if key not in self._kernels:
            self._kernels[key] = build_mono_ffn(MonoBuild(tokens, self.tp, 0))
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
        ratio = a.ratio
        k1, k2 = self.kernels(M, ratio)
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
            assert comp_block_table is not None
            assert topk_indices.dtype == torch.int32 and topk_indices.stride(0) == 512
            assert comp_cache.dtype == torch.uint8 and comp_cache.stride(1) == 584
            assert comp_block_table.dtype == torch.int32
            comp, comp_stride, comp_block = (
                comp_cache,
                comp_cache.stride(0),
                comp_cache.shape[1],
            )
            bt, bt_stride, topk = (
                comp_block_table,
                comp_block_table.stride(0),
                topk_indices,
            )
        else:
            comp, comp_stride, comp_block = self._dummy, 0, 1
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
            self.scratch(M).data_ptr(),
            self.epoch.data_ptr(),
            0,
            stream=st,
        )
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
            0,
            stream=st,
        )
        return outs

    def ffn(
        self,
        w: MonoLayerWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
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
        self.ffn_kernel(M)(
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
            stream=torch.cuda.current_stream(),
        )
        return outs
