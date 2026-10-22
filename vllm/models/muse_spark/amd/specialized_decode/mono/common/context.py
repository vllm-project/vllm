# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stage context every K2 and K1 stage takes (``c``), built once at the
top of a kernel by ``make_context``.

The context (built in the kernel's trace; every value below is traced unless
noted):

    S                     launch rows (a Python int, a build constant)
    tid, bid, lane, wave  thread, CTA, tid % 64, tid // 64
    rank, slot, e, par    ABI scalars; the step epoch (22 bits) and its parity
    put, put_bf,          the launch's ``Mailbox`` methods (bound: the AST
    put_words, poll       rewriter must see no method call on a local)
    red                   LDS f32[WAVES 64 4]: MFMA accumulators (``row_sum``)
    rt                    ``RouteLds`` field -> LDS pointer ({} without it)
    lds                   stage -> field -> LDS pointer (the stage union)
    args                  ABI name -> kernel argument (Int64 address / Int32)
    scr                   scratch mailbox region -> ``Addr`` (Phases A, B and
                          C: ``spec.SCRATCH_REGIONS_ALL``); the plain regions
                          (``xrow``, ``spec.SCRATCH_PLAIN_BC``): Int64 bases
    peer                  peer region -> [an ``Addr`` a rank], and
                          region + "_own" -> this rank's, parity-shifted ({}
                          without peers)

A rank-local kernel (K1) builds its context with ``peers=False`` (no
``peers`` argument, no peer regions) and ``route_lds=False`` (no route table
in its LDS).

LDS: the fixed ``red`` and ``RouteLds`` (if any), then one union of the
stages' structs (``lds_structs``: stage -> ``fx.struct``, or stage ->
{member: ``fx.struct``} for a sequential composite whose members are all live
at once, e.g. K2's ``xq_ug``). A composite's fields are reachable both under
its own name and under each member's. Union members share bytes: a stage that
touches its struct or ``red`` ends with ``gpu.barrier()`` (the hand-over rule).
"""

from collections.abc import Mapping

import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_align_of, dsl_size_of
from flydsl.expr.struct import is_struct_type

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
    region_ids,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import load_ptr64
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    LDS_BYTES,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import peer_bases
from vllm.models.muse_spark.amd.specialized_decode.mono.common.route_lds import (
    RouteLds,
    route_ptrs,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    Mailbox,
    load_budget,
    load_epoch,
    preg,
    shift,
    sreg,
    stall,
    step_tag,
)

RED_WORDS = WAVES * 64 * 4  # a wave's 16x16 accumulator, 4 f32 a lane


@fx.struct
class FixedLds:
    red: fx.Array[fx.Float32, RED_WORDS, 16]


# The arguments make_context reads; a kernel's ``args`` carries them and any
# other K2 argument its stages use. Without peers (K1), all but ``peers``.
CONTEXT_ARGS = ("scratch", "peers", "rank", "epoch", "slot", "health", "alarm")
LOCAL_CONTEXT_ARGS = tuple(n for n in CONTEXT_ARGS if n != "peers")


# ---------------------------------------------------------------- LDS


def _member(stage: str, st):
    """A union member: a stage's struct, or its members laid out in order."""
    if isinstance(st, Mapping):
        if not st:
            raise ValueError(f"LDS composite {stage!r} has no member")
        for name, t in st.items():
            if not is_struct_type(t):
                raise TypeError(f"LDS {stage}.{name} is not an fx.struct: {t!r}")
        return fx.struct[tuple(slice(name, t) for name, t in st.items())]
    if not is_struct_type(st):
        raise TypeError(f"LDS {stage!r} is not an fx.struct: {st!r}")
    return st


def stage_union(lds_structs: Mapping | None):
    """The union of the stages' LDS structs (None when there is none)."""
    if not lds_structs:
        return None
    return fx.union[
        tuple(slice(stage, _member(stage, st)) for stage, st in lds_structs.items())
    ]


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def lds_total_bytes(lds_structs: Mapping | None = None, route_lds: bool = True) -> int:
    """The LDS a context with ``lds_structs`` allocates: ``red``, ``RouteLds``
    (with ``route_lds``), then the stage union, each at its alignment
    (``SharedAllocator``'s bump)."""
    off = 0
    parts = [FixedLds, RouteLds] if route_lds else [FixedLds]
    union = stage_union(lds_structs)
    if union is not None:
        parts.append(union)
    for t in parts:
        off = _align_up(off, dsl_align_of(t)) + dsl_size_of(t)
    return off


def _field_ptrs(inst, t, out: dict, where: str) -> dict:
    """Field -> LDS pointer of a peeked struct ``inst`` of type ``t``, nested
    structs flattened (field names must be unique)."""
    for f in t.__dsl_field_defs__:
        v = getattr(inst, f.name)
        if is_struct_type(f.type_spec):
            _field_ptrs(v, f.type_spec, out, f"{where}.{f.name}")
            continue
        if f.name in out:
            raise ValueError(f"LDS field {f.name!r} appears twice in {where}")
        out[f.name] = v.ptr
    return out


def _stage_ptrs(alloc, lds_structs: Mapping | None) -> dict:
    """Allocate the stage union; stage (and composite member) -> field ->
    pointer."""
    lds: dict = {}
    union = stage_union(lds_structs)
    if union is None or lds_structs is None:
        return lds
    st = alloc.allocate(union)
    for stage, member in lds_structs.items():
        inst = getattr(st, stage).peek()
        if isinstance(member, Mapping):
            flat: dict = {}
            for name, t in member.items():
                if name in lds or name in lds_structs:
                    raise ValueError(f"LDS name {name!r} names two stages")
                lds[name] = _field_ptrs(getattr(inst, name), t, {}, f"{stage}.{name}")
                for k, p in lds[name].items():
                    if k in flat:
                        raise ValueError(f"LDS field {k!r} appears twice in {stage}")
                    flat[k] = p
            lds[stage] = flat
        else:
            if stage in lds:
                raise ValueError(f"LDS name {stage!r} names two stages")
            lds[stage] = _field_ptrs(inst, member, {}, stage)
    return lds


# ---------------------------------------------------------------- the context


def _peer_regions(peers, rank, par) -> dict:
    """Peer region -> [an ``Addr`` a rank] and region + "_own" -> this rank's,
    in the epoch parity ``par``'s half."""
    bases = peer_bases(peers, spec.TP)
    own = load_ptr64(peers, rank)
    delta = fx.Int64(par) * spec.peer_half_bytes()
    peer: dict = {}
    for region, (off, _) in spec.peer_layout().items():
        peer[region] = [shift(preg(b, off, region), delta) for b in bases]
        peer[region + "_own"] = shift(preg(own, off, region), delta)
    return peer


def make_context(
    S: int,
    args: Mapping,
    lds_structs: Mapping | None = None,
    *,
    key: spec.K2Build | spec.K1Build | None = None,
    peers: bool = True,
    route_lds: bool = True,
) -> dict:
    """The stage context ``c`` (module docstring), traced in a kernel body.

    ``args``: ABI name -> kernel argument, at least ``CONTEXT_ARGS``
    (``LOCAL_CONTEXT_ARGS`` without ``peers``).
    ``lds_structs``: stage -> struct (or {member: struct}), the stage union.
    ``key``: a K2 or K1 build; a K2 fault build compiles ``sync.stall`` in first
    (its prologue).
    ``peers``: build the peer regions from the ``peers`` argument (False: a
    rank-local kernel, no ``peers`` argument read and ``peer`` empty).
    ``route_lds``: allocate ``RouteLds`` (False: ``rt`` empty).
    """
    if not 1 <= S <= spec.MAX_ROWS:
        raise ValueError(f"launch rows {S} (1 .. {spec.MAX_ROWS})")
    needs = CONTEXT_ARGS if peers else LOCAL_CONTEXT_ARGS
    missing = [n for n in needs if n not in args]
    if missing:
        raise KeyError(f"make_context needs the arguments {missing}")
    lds_bytes = lds_total_bytes(lds_structs, route_lds)
    if lds_bytes > LDS_BYTES:
        raise ValueError(f"{lds_bytes} B of LDS > {LDS_BYTES} (a CU's)")
    scratch, rank = args["scratch"], args["rank"]
    epoch, slot, health, alarm = (
        args["epoch"],
        args["slot"],
        args["health"],
        args["alarm"],
    )
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    alloc = fx.SharedAllocator()
    fixed = alloc.allocate(FixedLds).peek()
    rl = alloc.allocate(RouteLds).peek() if route_lds else None
    lds = _stage_ptrs(alloc, lds_structs)
    if isinstance(key, spec.K2Build) and key.faulty:
        stall(
            key.fault_rank,
            key.fault_epoch,
            key.fault_slot,
            key.fault_ticks,
            rank,
            epoch,
            slot,
        )
    e = load_epoch(epoch)
    mb = Mailbox(
        step_tag(e, slot) - 1,
        health,
        alarm,
        region_ids(spec.REGIONS),
        load_budget(health),
    )
    par = e & 1
    peer = _peer_regions(args["peers"], rank, par) if peers else {}
    lay = spec.scratch_layout()
    scr: dict = {r: sreg(scratch, lay[r][0], r) for r in spec.SCRATCH_REGIONS}
    scr[spec.XROW] = scratch + fx.Int64(lay[spec.XROW][0])
    # Phases B and C, laid out past xrow (Phase A's offsets stay)
    for r in spec.SCRATCH_REGIONS_BC:
        scr[r] = sreg(scratch, lay[r][0], r)
    for name, _, _ in spec.SCRATCH_PLAIN_BC:
        if name is not None:  # None: a reserved gap
            scr[name] = scratch + fx.Int64(lay[name][0])
    return {
        "S": S,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rank": rank,
        "slot": slot,
        "e": e,
        "par": par,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "put_words": mb.put_words,
        "poll": mb.poll,
        "red": fixed.red.ptr,
        "rt": route_ptrs(rl) if rl is not None else {},
        "lds": lds,
        "args": dict(args),
        "scr": scr,
        "peer": peer,
        "key": key,
    }
