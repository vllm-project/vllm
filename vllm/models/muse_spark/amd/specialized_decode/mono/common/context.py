# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The stage context every K2 and K1 stage takes (``c``), built once at the
top of a kernel by ``make_context``.

The context (built in the kernel's trace; every value below is traced unless
noted):

    S                     launch rows (a Python int, a build constant)
    tid, bid, lane, wave  thread, CTA, tid % 64, tid // 64
    rank, slot            ABI scalars
    has_peers             ``make_context``'s ``peers``
    red                   LDS f32[WAVES 64 4]: MFMA accumulators (``row_sum``)
    rt                    ``RouteLds`` field -> LDS pointer ({} without it)
    lds                   stage -> field -> LDS pointer (the stage union)
    args                  ABI name -> kernel argument (Int64 address / Int32)
    scr                   scratch mailbox region -> ``Addr`` (Phases A, B and
                          C: ``spec.SCRATCH_REGIONS_ALL``); the plain regions
                          (``xrow``, ``spec.SCRATCH_PLAIN_BC``): Int64 bases

and, from ``settle`` on (the launch's scalar words: a kernel loads them once
its first loads are issued, so their wait never lands ahead of those):

    e, par                the step epoch (22 bits) and its parity
    put, put_bf,          the launch's ``Mailbox`` methods (bound: the AST
    put_words, poll       rewriter must see no method call on a local)
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
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    ld_s32,
    pin_s,
    ptr64,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    LDS_BYTES,
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import (
    peer_bases,
    peer_words,
)
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
    step_epoch,
    step_tag,
    wait_budget,
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


def _peer_regions(words, par) -> dict:
    """Peer region -> [an ``Addr`` a rank] and region + "_own" -> this rank's,
    in the epoch parity ``par``'s half. ``words``: the peer table's, then this
    rank's entry (``settle``)."""
    bases = peer_bases(words[:-2])
    own = ptr64(words[-2], words[-1])
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
    ``peers``: ``settle`` builds the peer regions from the ``peers`` argument
    (False: a rank-local kernel, no ``peers`` argument read and ``peer``
    empty).
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
    epoch, slot = args["epoch"], args["slot"]
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
        "has_peers": peers,
        "red": fixed.red.ptr,
        "rt": route_ptrs(rl) if rl is not None else {},
        "lds": lds,
        "args": dict(args),
        "scr": scr,
        "key": key,
    }


def settle(c) -> None:
    """Loads the launch's scalar words (``ld_s32``: the epoch, the
    ``WAIT_BUDGET`` word, the peer table and this rank's entry), pins them
    (``pin_s``) and adds to ``c`` what is built from them: ``e``, ``par``, the
    ``Mailbox`` methods and ``peer``. A kernel calls it right after its first
    loads are issued, at top level or once in each CTA-uniform branch that
    publishes or polls: the words' round trip (an lgkmcnt wait, never a vmcnt
    one) then overlaps those loads (unpinned, a scalar load sinks to its first
    use, a poll's, and exposes its round trip there). A stage that publishes
    or polls before it fails at trace (no ``put`` / ``poll``)."""
    a = c["args"]
    # this rank's entry (the one load at a traced offset) issued first: last,
    # LLVM put it behind a first lgkmcnt wait, a second round trip
    own = []
    if c["has_peers"]:
        own = [ld_s32(a["peers"], 2 * c["rank"] + h) for h in range(2)]
    words = own + [load_epoch(a["epoch"]), load_budget(a["health"])]
    if c["has_peers"]:
        words += peer_words(a["peers"], spec.TP)
    w = pin_s(words)
    w = w[len(own) :] + w[: len(own)]  # epoch, budget, table, this rank's
    e = step_epoch(w[0])
    mb = Mailbox(
        step_tag(e, c["slot"]) - 1,
        a["health"],
        a["alarm"],
        region_ids(spec.REGIONS),
        wait_budget(w[1]),
    )
    par = e & 1
    c["e"] = e
    c["par"] = par
    c["put"] = mb.put
    c["put_bf"] = mb.put_bf
    c["put_words"] = mb.put_words
    c["poll"] = mb.poll
    c["peer"] = _peer_regions(w[2:], par) if len(w) > 2 else {}
