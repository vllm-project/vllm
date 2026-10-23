# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1: one Muse Spark kernel layer's attention front (Phase B1, DESIGN_BC 5),
from the previous layer's deferred FFN boundary to the qkv rows and this
step's K / V in the cache, in one launch a layer and rank. It replaces vLLM's
``_norm_residual_norm`` boundary, the qkv GEMM and the fused QK-norm / RoPE /
KV insert; the stock attention core and K2 follow it.

    boundary      CTAs 0 .. 63, no weight: the previous layer's post_ffn_norm
                  (weight first), the gated blend and this layer's
                  pre_qkv_proj_norm on slice j (``front/boundary.py``) ->
                  ``stream_out``, the ``arow`` rows, ``ardy[j]``
    epilogue      CTAs 0 .. 5 S - 1, after their slice: unit u = (token u // 5,
                  row group u % 5) (``front/epilogue.py``): the two GEMV
                  partials -> QK-norm, RoPE (``rope`` builds) -> ``qkvo``, the
                  k / v heads into the cache
    qkv           CTAs 64 .. 255: 12 (tile, K part) tasks each, every weight in
                  flight from entry, the ``arow`` rows staged behind ``ardy``
                  (``front/qkv.py``) -> ``qpart``

The roles are split by CTA (5.3) so that no CTA polls or publishes with a
load in flight that is not its own task's (rule 8, DESIGN_BC 10.4): the
boundary CTAs issue no weight load (their polls hold the boundary operands
and the epilogue's three step inputs), the epilogue polls ``qpart`` holding
only its cos / sin row, a GEMV CTA polls ``ardy`` holding only its own
weights. The boundary CTAs wait only on each other, the GEMV CTAs only on the
boundary, the epilogue only on the GEMV CTAs; every CTA is resident (one a
CU, below), so the launch cannot deadlock. A wait that gives up records
itself and fails the rank (``sync._spin``); every address stays clamped and
a FAILED rank writes no K / V, so the launch completes.

K1 is rank-local (E2): the qkv rows are this rank's column shard and the
boundary's inputs are replicated, so it takes no ``peers`` and pushes nothing
to another rank (only ``alarm``, on a give-up). Its tags carry the layer's K1
slot (``spec.k1_slot``), so a step's K1 and K2 pairs never alias. The step
epoch is the one K2 reads: the step's first kernel layer bumps it before its
K1 (``runtime.layers``).

LDS: the fixed ``red`` (the GEMV's accumulators), no ``RouteLds``, then a
union of the GEMV's staged rows (``qkv``) and the residency pin (``pin``).
The pin makes every build hold more than half a CU's LDS, so the dispatcher
places one CTA a CU (5.9): at S <= 2 the code alone (~120 VGPRs, 20 KiB)
would let two CTAs share a CU and leave CUs without a GEMV CTA. The
``amdgpu-waves-per-eu`` attribute 5.9 names is a register-allocation bound,
not an occupancy one (the hardware places by the descriptor's VGPR granules
and LDS), so it cannot do this. The epilogue and the boundary use no LDS.

Adapted from K2 (``mono/layer.py``: the kernel / launcher pair, ``build``,
``k2_launch``), with the context built rank-local
(``make_context(peers=False, route_lds=False)``) and the arguments packed by
name.
"""

from collections.abc import Mapping

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import Int32, Int64

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.common.context import (
    RED_WORDS,
    lds_total_bytes,
    make_context,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import traced
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    LDS_BYTES,
    THREADS,
    KernelAbi,
    key_tuple,
    source_digest,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.front import (
    boundary,
    epilogue,
    qkv,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.layer import _bind_cache
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    BLOCKS,
    E_UNITS,
    K1_GEMV0,
    NA,
    K1Build,
)

# The kernel's parameters: spec.K1_ARGS, the residual stream renamed
# ``stream_in`` as in K2 (the launcher's own ``stream`` is the HIP stream);
# ``k1_launch`` and ``c["args"]`` key them by K1_ARGS.
KERNEL_NAMES = {"stream": "stream_in"}
ABI = KernelAbi(tuple(KERNEL_NAMES.get(n, n) for n in spec.K1_ARGS))
# Every .py under mono/: joins every build key (K2's digest; the front stages
# and the helpers K1 shares with K2 are all in it)
SOURCES = source_digest(".")
REGIONS = spec.REGIONS  # debug.region_ids order of the wait records
# ``program``'s two role branches partition the grid: every CTA runs exactly
# one of them
assert K1_GEMV0 == NA and E_UNITS * spec.MAX_ROWS <= NA

# One CTA a CU (DESIGN_BC 5.9): every build's LDS past half a CU's
RESIDENT_LDS = LDS_BYTES // 2 + 16
PIN_WORDS = (RESIDENT_LDS - RED_WORDS * 4) // 4


@fx.struct
class PinLds:
    pin: fx.Array[fx.Int32, PIN_WORDS, 16]


_STREAM = fx.Stream(None)


def lds_structs(s: int) -> dict:
    """The stage union's members at ``s`` rows: the GEMV's staged ``arow``
    rows and the residency pin (no stage touches it)."""
    return {qkv.STAGE: qkv.lds_struct(s), "pin": PinLds}


def lds_bytes(s: int) -> int:
    """K1's static LDS at ``s`` rows: ``red``, then the union (no
    ``RouteLds``). Past ``LDS_BYTES // 2`` at every width."""
    return lds_total_bytes(lds_structs(s), route_lds=False)


@traced
def program(c) -> None:
    """K1's roles on CTA ``c["bid"]`` (DESIGN_BC 5.3). Every branch is
    CTA-uniform, so every thread reaches every barrier."""
    s = c["S"]
    b = c["bid"]
    b32 = fx.Int32(b)
    # -- the weight-free CTAs: every operand of the boundary and the
    # epilogue's step inputs issued before the boundary's first poll; the
    # epilogue after the slice's publish
    if b < NA:
        pre = boundary.boundary_loads(c, b)
        epi = boundary.epilogue_inputs(c, b)
        boundary.stage_boundary(c, b, pre)
        if b32 < E_UNITS * s:
            epilogue.stage_epilogue(c, b32, epi)
    # -- the GEMV CTAs: their weights the first loads, then the ardy polls
    if b >= K1_GEMV0:
        g = b32 - K1_GEMV0
        wops = qkv.qkv_loads(c, g)
        qkv.stage_qkv(c, g, wops)


_BUILDS: dict = {}


def build(key: K1Build):
    """The ``@flyc.jit`` launcher of build ``key`` (``ABI`` order, then
    ``stream``), cached. Every field of ``key`` and the mono sources are in
    the JIT key (``key_tuple``), referenced by both the kernel and the
    launcher."""
    if key in _BUILDS:
        return _BUILDS[key]
    if not isinstance(key, K1Build):
        raise TypeError(f"a K1 build key is a K1Build, not {type(key).__name__}")
    s = key.tokens
    lds = lds_structs(s)
    nbytes = lds_bytes(s)
    if nbytes > LDS_BYTES:
        raise ValueError(f"K1 at {s} rows: {nbytes} B of LDS > {LDS_BYTES}")
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=key.kernel_name(), known_block_size=[THREADS, 1, 1])
    def k1(
        branch: Int64,
        stream_in: Int64,
        pad: Int64,
        positions: Int64,
        slots: Int64,
        w_postffn: Int64,
        ffn_alpha: Int64,
        ffn_beta: Int64,
        w_preqkv: Int64,
        w_qkv: Int64,
        cos_sin: Int64,
        kv: Int64,
        kv_sb: Int64,
        kv_sh: Int64,
        kv_st: Int64,
        kv_blocks: Int32,
        max_pos: Int32,
        stream_out: Int64,
        qkvo: Int64,
        scratch: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        args = {
            "branch": branch,
            "stream": stream_in,
            "pad": pad,
            "positions": positions,
            "slots": slots,
            "w_postffn": w_postffn,
            "ffn_alpha": ffn_alpha,
            "ffn_beta": ffn_beta,
            "w_preqkv": w_preqkv,
            "w_qkv": w_qkv,
            "cos_sin": cos_sin,
            "kv": kv,
            "kv_sb": kv_sb,
            "kv_sh": kv_sh,
            "kv_st": kv_st,
            "kv_blocks": kv_blocks,
            "max_pos": max_pos,
            "stream_out": stream_out,
            "qkvo": qkvo,
            "scratch": scratch,
            "rank": rank,
            "epoch": epoch,
            "slot": slot,
            "health": health,
            "alarm": alarm,
        }
        # rank-local: no peers, no route table
        c = make_context(s, args, lds, key=key, peers=False, route_lds=False)
        program(c)

    @flyc.jit
    def launch(
        branch: Int64,
        stream_in: Int64,
        pad: Int64,
        positions: Int64,
        slots: Int64,
        w_postffn: Int64,
        ffn_alpha: Int64,
        ffn_beta: Int64,
        w_preqkv: Int64,
        w_qkv: Int64,
        cos_sin: Int64,
        kv: Int64,
        kv_sb: Int64,
        kv_sh: Int64,
        kv_st: Int64,
        kv_blocks: Int32,
        max_pos: Int32,
        stream_out: Int64,
        qkvo: Int64,
        scratch: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            branch,
            stream_in,
            pad,
            positions,
            slots,
            w_postffn,
            ffn_alpha,
            ffn_beta,
            w_preqkv,
            w_qkv,
            cos_sin,
            kv,
            kv_sb,
            kv_sh,
            kv_st,
            kv_blocks,
            max_pos,
            stream_out,
            qkvo,
            scratch,
            rank,
            epoch,
            slot,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(k1, launch)
    # K2's FlyDSL cache root (``layer.k2_cache_dir``), not aiter's
    _bind_cache(launch)
    _BUILDS[key] = launch
    return launch


def k1_launch(key: K1Build, args: Mapping[str, torch.Tensor | int]) -> None:
    """One K1 launch of build ``key`` on the current stream. ``args``: every
    ``spec.K1_ARGS`` name, nothing else (``ABI.pack``); a tensor passes its
    address, an int (``kv_sb``, ``kv_sh``, ``kv_st``, ``kv_blocks``,
    ``max_pos``, ``rank``, ``slot``, ``alarm``; ``cos_sin`` 0 on a NoPE
    layer) itself."""
    f = build(key)
    vals = {
        KERNEL_NAMES.get(n, n): v.data_ptr() if isinstance(v, torch.Tensor) else int(v)
        for n, v in args.items()
    }
    f(*ABI.pack(vals), stream=torch.cuda.current_stream())
