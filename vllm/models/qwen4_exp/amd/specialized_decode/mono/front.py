# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HC front: a ``GatedResidual``'s ``mix`` of the materialized multi-stream
state as one persistent launch, ``BLOCKS`` x ``THREADS``, S <= ``MAX_TOKENS``:

    msq   (``self_msq``, CTAs 0 .. 19): the state's sums of squares (the
          model's first layer and the one after a PLE; past them the previous
          block's combine hands them over as ``msq_in``)
    down  (``hc.DOWN_TASKS``, CTAs 88 ..): xn of the state, [down; inject]
          -> LORA and the injection logits -> ``inj_out`` (bf16)
    up    (``hc.UP_TASKS``, CTAs 0 .. 159): the gate-mix -> ``bin_out``, the
          block's input
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.mono import hc
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.debug import region_ids
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ops import (
    bf16_pair,
    rsrc,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
    source_digest,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    sreg,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.layout import need
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import step_tag

H, HH, HC = L.HIDDEN, L.HH, L.HC
D0 = BLOCKS - hc.DOWN_TASKS  # 88: the down CTAs, to the last

assert hc.UP_TASKS <= BLOCKS and D0 >= 0 and hc.NRED <= hc.UP_TASKS

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class FrontBuild:
    tokens: int
    tp: int = 2
    self_msq: bool = False  # the state's sums of squares computed here
    eps: float = 1e-6  # the HC norm


ABI = KernelAbi(
    (
        "state",
        "msq_in",
        "hc_w",
        "w_down",
        "w_up",
        "inj_out",
        "bin_out",
        "scratch",
        "epoch",
        "health",
        "alarm",
        "layer",
    )
)


def scratch_layout(key: FrontBuild) -> dict:
    return pair_layout(hc.regions(key.tokens).items())


def scratch_bytes(key: FrontBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def _put_inj(c, t, r, vs):
    """The injection rows r, r + 1 of token t -> ``inj_out`` [S, HC] bf16."""
    bo.buffer_store(
        bf16_pair(vs[0], vs[1]).bitcast(fx.Int32),
        rsrc(c["args"]["inj_out"]),
        (t * HC + r) // 2,
    )


_BUILDS: dict = {}


def build(key: FrontBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    need(1 <= s <= L.MAX_TOKENS, f"{s} rows, the kernels take 1..{L.MAX_TOKENS}")
    scr = scratch_layout(key)

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rs: fx.Array[fx.Float32, s * HC, 16]
        ll: fx.Array[fx.Int32, s * hc.LROW // 2, 16]
        xl: fx.Array[fx.Int32, s * hc.DROW // 2, 16]

    keyed = key_tuple(key, source_digest())
    name = f"flashnext_hc_front_s{s}_t{key.tp}_m{int(key.self_msq)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kfront(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(scr))
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rs": lds.rs.ptr,
            "ll": lds.ll.ptr,
            "xl": lds.xl.ptr,
            "args": {"inj_out": inj_out},
        }
        c["hc"] = {
            "state": state,
            "state_cm": 0,
            "w_norm": hc_w,
            "w_down": w_down,
            "w_up": w_up,
            "msq": sreg(scratch, scr["msq"][0], "msq"),
            "msq_in": None if key.self_msq else msq_in,
            "dpart": sreg(scratch, scr["dpart"][0], "dpart"),
            "lora": sreg(scratch, scr["lora"][0], "lora"),
            "bin": bin_out,
            "inj_out": lambda t, r, vs: _put_inj(c, t, r, vs),
        }
        # a CTA's whole sequence in one branch, its weights in flight first
        if const_expr(key.self_msq):  # noqa: SIM102 (traced: `and` does not combine traced values)
            if bid < hc.NRED:
                hc.stage_msq(c, bid)
        if bid < D0:
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_up(c, bid, uops)
        if (bid >= D0) & (bid < hc.UP_TASKS):
            dops = hc.down_loads(c, bid - D0)
            uops = hc.up_loads(c, bid)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)
            hc.stage_up(c, bid, uops)
        if bid >= hc.UP_TASKS:
            dops = hc.down_loads(c, bid - D0)
            hc.load_rrms(c)
            hc.stage_down(c, bid - D0, dops)

    @flyc.jit
    def launch(
        state: Int64,
        msq_in: Int64,
        hc_w: Int64,
        w_down: Int64,
        w_up: Int64,
        inj_out: Int64,
        bin_out: Int64,
        scratch: Int64,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kfront(
            state,
            msq_in,
            hc_w,
            w_down,
            w_up,
            inj_out,
            bin_out,
            scratch,
            epoch,
            health,
            alarm,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kfront, launch)
    _BUILDS[key] = launch
    return launch


def hc_front(
    key: FrontBuild,
    *,
    state,
    msq_in,
    hc_w,
    w_down,
    w_up,
    inj_out,
    bin_out,
    scratch,
    epoch,
    health,
    alarm,
    layer,
):
    """``GatedResidual.mix`` of ``state`` [S, 4 H] bf16 -> ``bin_out`` [S, H]
    bf16 (the block's input) and ``inj_out`` [S, 4] bf16 (its injection
    logits). ``msq_in``: ``state``'s sums of squares [NRED, S, 4] fp32 as the
    previous block's combine wrote them, None with ``key.self_msq``."""
    s = key.tokens
    bf = torch.bfloat16
    need(state.shape == (s, HH) and state.dtype == bf, "state: bf16 [S, 4 H]")
    need(
        (msq_in is None) == key.self_msq
        and (
            msq_in is None
            or (msq_in.numel() == hc.NRED * s * HC and msq_in.dtype == torch.float32)
        ),
        "msq_in: fp32 [20, S, 4], absent with self_msq only",
    )
    need(hc_w.shape == (HH,) and hc_w.dtype == bf, "HC norm weight: [4 H]")
    need(
        w_down.shape == (L.DOWN_ROWS, HH) and w_down.dtype == bf, "HC down: [336, 4 H]"
    )
    need(w_up.shape == (HH, L.LORA) and w_up.dtype == bf, "HC up: [4 H, 320]")
    need(inj_out.shape == (s, HC) and inj_out.dtype == bf, "inj_out: bf16 [S, 4]")
    need(bin_out.shape == (s, H) and bin_out.dtype == bf, "bin_out: bf16 [S, H]")
    tensors = [state, hc_w, w_down, w_up, inj_out, bin_out]
    tensors += [msq_in] if msq_in is not None else []
    need(all(t.is_contiguous() for t in tensors), "a non-contiguous HC input")
    vals = {
        "state": state.data_ptr(),
        "msq_in": 0 if msq_in is None else msq_in.data_ptr(),
        "hc_w": hc_w.data_ptr(),
        "w_down": w_down.data_ptr(),
        "w_up": w_up.data_ptr(),
        "inj_out": inj_out.data_ptr(),
        "bin_out": bin_out.data_ptr(),
        "scratch": scratch.data_ptr(),
        "epoch": epoch.data_ptr(),
        "health": health.data_ptr(),
        "alarm": alarm,
        "layer": layer,
    }
    build(key)(*ABI.pack(vals), stream=torch.cuda.current_stream())
