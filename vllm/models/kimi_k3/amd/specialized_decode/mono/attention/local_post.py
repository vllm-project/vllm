# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's KDA after in_proj (``kda_post``): the spec conv, gates, recurrence and
gated RMSNorm of in_proj's output, one independent CTA a (request, head).

For the widest steps, where vLLM's in_proj GEMM beats K1's: no CTA waits on
another, so the launch needs no mailbox and no residency. K1's KDA stages run
as they are, reading the projected rows and their own q / k in LDS."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import kda
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    butterfly,
    hw_rsq,
    rsrc,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    THREADS,
    KernelAbi,
    key_tuple,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.sync import sreg
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.tiles import (
    MAX_TOKENS,
    TILE_TOKENS,
)

_BUILDS: dict = {}


@dataclass(frozen=True)
class KdaPostBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    state_len: int  # conv state entries: CONV_W - 1 + num_spec
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0


ABI = KernelAbi(
    (
        "projected",
        "w_fb",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "on_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "st_stride",
        "num_acc",
        "core_out",
    )
)


class LocalInputs:
    """K1's ``poll`` / ``put`` over the projected rows (``proj``) and this CTA's
    q / k in LDS (``qk``); the partial norms (``onrm``) are dropped, the CTA
    norms its head whole."""

    def __init__(self, projected, qkl, base):
        self.projected, self.qkl, self.base = projected, qkl, base

    def poll(self, specs):
        result = []
        for addr, index, count in specs:
            assert count == 1
            if fx.const_expr(addr.region == "proj"):
                value = kda.ld_bf(self.projected, index)
            else:
                assert addr.region == "qk"
                value = fx.ptr_load(self.qkl + (index - self.base))
            result.append([value.bitcast(fx.Int32)])
        return result

    def put(self, addr, index, value):
        if fx.const_expr(addr.region == "qk"):
            fx.ptr_store(value, self.qkl + (index - self.base))
        else:
            assert addr.region == "onrm"


def build(key: KdaPostBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    assert 1 <= key.tokens <= MAX_TOKENS
    assert 1 <= key.qlen <= TILE_TOKENS and key.tokens % key.qlen == 0
    L, V = key.qlen, kda.HD
    keyed = key_tuple(key, kda._SOURCES)

    @fx.struct
    class Smem:
        qkl: fx.Array[fx.Float32, L * 2 * kda.HD, 16]
        decay: fx.Array[fx.Float32, L * kda.HD, 16]
        vloc: fx.Array[fx.Float32, L * V, 16]
        gate_or_output: fx.Array[fx.Float32, L * V, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @flyc.kernel(
        name=f"k3_mono_kda_post_s{key.tokens}_l{L}", known_block_size=[THREADS, 1, 1]
    )
    def kda_post(
        projected: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
    ):
        _ = keyed
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        req, head = bid // kda.NH, bid % kda.NH
        alloc = fx.SharedAllocator()
        shared = alloc.allocate(Smem).peek()
        # the gates read f_a before the recurrence writes its output there
        fa = ol = shared.gate_or_output.ptr
        inputs = LocalInputs(projected, shared.qkl.ptr, bid * L * 2 * kda.HD)
        c = {
            "S": key.tokens,
            "L": L,
            "VCH": V,
            "NVCH": 1,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "lb": key.lower_bound,
            "fa": fa,
            "qkl": shared.qkl.ptr,
            "decay": shared.decay.ptr,
            "vloc": shared.vloc.ptr,
            "ol": ol,
            "beta": shared.beta.ptr,
            "put": inputs.put,
            "poll": inputs.poll,
            "proj": sreg(projected, 0, "proj"),
            "qk": sreg(projected, 0, "qk"),
            "onrm": sreg(projected, 0, "onrm"),
            "args": {
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
            },
        }
        staged = kda.kda_stage(c, req, head, 0)
        kda.kda_conv(c, staged, req, head, 0)
        kda.kda_gates(c, staged, req, head)
        kda.kda_recur(c, staged, req, head, 0)
        # the gated RMSNorm: wave t token t's 128 outputs, two a lane
        lane, token = tid % 64, tid // 64
        if token < L:
            x0 = fx.ptr_load(ol + (token * V + lane))
            x1 = fx.ptr_load(ol + (token * V + lane + 64))
            ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
            scale = hw_rsq(ss * fx.Float32(1.0 / V) + fx.Float32(key.onorm_eps))
            for offset, value in ((0, x0), (64, x1)):
                col = lane + offset
                gate = kda.ld_bf(
                    projected, (req * L + token) * kda.NPROJ + kda.G2_0 + head * V + col
                )
                result = value * scale * kda.ld_bf(on_w, col) * kda.sigmoid(gate)
                result = (staged["slot0"] > 0).select(result, fx.Float32(0.0))
                bo.buffer_store(
                    result.to(fx.BFloat16),
                    rsrc(core_out),
                    (req * L + token) * kda.PROJ + head * V + col,
                )
        gpu.barrier()

    @flyc.jit
    def launch(
        projected: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        stream: fx.Stream = kda._STREAM,
    ):
        _ = keyed
        kda_post(
            projected,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
        ).launch(grid=(key.tokens // L * kda.NH,), block=(THREADS,), stream=stream)

    ABI.check(kda_post, launch)
    _BUILDS[key] = launch
    return launch


def kda_post(
    key: KdaPostBuild,
    *,
    projected,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    core_out,
):
    """Host launcher: ``projected`` [S, NPROJ] bf16 (in_proj's output),
    ``conv_state`` / ``rstate`` / ``st_idx`` as ``kda.kda_pre``'s."""
    build(key)(
        projected.data_ptr(),
        w_fb.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        on_w.data_ptr(),
        rstate.data_ptr(),
        rstate.stride(0),
        st_idx.data_ptr(),
        st_idx.stride(0),
        num_acc.data_ptr(),
        core_out.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
