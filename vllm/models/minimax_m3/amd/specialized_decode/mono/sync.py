# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded tagged hand-offs and the sticky error word.

The contract, in the LLVM "AMDGPU Memory Model" (llvm/docs/AMDGPUMemoryModel.md,
AMM) and AMDGPUUsage "Execution Barriers":

- A pair is one naturally aligned 64-bit atomic, (value | tag << 32),
  ``monotonic`` at its space's scope (``agent`` for this GPU's scratch, system
  for the peer heap). An atomic access is never split, so a load that sees a
  pair's tag sees that store's value.
- Data a pair announces is stored *store-available* and loaded *load-visible*
  at agent scope (AMM: agent atomics of 16 to 64 bits, ``llvm.amdgcn.av.load.
  b128``; ``store_dev`` / ``load_dev``).
- Happens-before from producer to consumer: the producer's release fence
  before the pair (``publish``), the consumer's acquire fence after the poll
  that saw it (``acquire``), and within a CTA release, ``s_barrier``, acquire
  (``gpu.barrier``; "Execution Barriers": they synchronize). These fences are
  tagged ``!{"amdgcn-av", "none"}``: they synchronize and keep the compiler
  from moving accesses across them, but make nothing available or visible.
- AMM visibility: a store-available write that happens-before a load-visible
  read of the same agent is visible to it, so the read returns that write (the
  previous launch's writes to the location are ordered before it by the
  kernel boundary).

The av operations are lowered with the agent-scope cache bits (``sc1``) and
the av-none fences to waits only, so no L2 write-back or invalidate is
emitted. The GFX942 workgroup release waits ``lgkmcnt(0)`` only outside
tgsplit mode, so a publishing CTA also has every wave wait for its own stores
before the barrier (``cta_sync(drain=True)``). The step-end fault handshake
uses plain system release / acquire once a step.

A poll gives up after ``ticks`` of the 100 MHz clock or once any rank has set
the error word; a timeout sets ``[1, rank, epoch, site]`` on every rank.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from vllm.models.deepseek_v41.amd.mono.common.ops import memrealtime, traced
from vllm.models.deepseek_v41.amd.mono.common.sync import Space

WAIT_TICKS = 1_000_000_000  # 10 s
ERROR_WORDS = 4
POLL_MAX = 12

_SCOPE = {Space.SCRATCH: "agent", Space.PEER: None}


def _ptr(addr, i, bits=64):
    """The global pointer to element i (of ``bits``) of address ``addr``."""
    at = addr + fx.Int64(i) * (bits // 8)
    return fx.inttoptr(
        fx.PointerType.get(
            T.i64 if bits == 64 else T.i32, fx.AddressSpace.Global, bits // 8
        ),
        at,
    )


def store64(addr, i, word, scope):
    fx.generic_store(
        _ptr(addr, i), word, memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope
    )


def load64(addr, i, scope):
    return fx.generic_load(
        _ptr(addr, i),
        dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def store32(addr, i, word, scope):
    fx.generic_store(
        _ptr(addr, i, 32),
        fx.Int32(word),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def load32(addr, i, scope):
    return fx.generic_load(
        _ptr(addr, i, 32),
        dtype=fx.Int32,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def store16(addr, i, v, scope):
    at = addr + fx.Int64(i) * 2
    fx.generic_store(
        fx.inttoptr(fx.PointerType.get(T.bf16, fx.AddressSpace.Global, 2), at),
        v,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=scope,
    )


def load128(addr, i, scope):
    """Dwords i .. i + 3 (i a multiple of 4): one load-visible at ``scope``."""
    ptr = _llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr + fx.Int64(i) * 4)
    ).result
    md = _llvm.MetadataAsValueOp(
        ir.Attribute.parse(f'#llvm.md_node<#llvm.md_string<"{scope}">>'),
        results=[ir.Type.parse("!llvm.metadata")],
    ).result
    return fx.Vector(
        _llvm.CallIntrinsicOp(
            T.vec(4, T.i32), "llvm.amdgcn.av.load.b128.p1", [ptr, md], [], []
        ).results[0]
    )


def store_dev(addr, i, words):
    """Dwords i .. of this GPU's scratch := ``words`` (1 or 2, i aligned),
    store-available at agent scope."""
    if const_expr(len(words) == 1):
        store32(addr, i, words[0], "agent")
    else:
        store64(addr + fx.Int64(i) * 4, 0, pack(words[0], words[1]), "agent")


def load_dev(addr, i, n=1):
    """Dwords i .. i + n of this GPU's scratch (n in {1, 2, 4}, i aligned),
    load-visible at agent scope."""
    if const_expr(n == 4):
        return load128(addr, i, "agent")
    if const_expr(n == 2):
        w = load64(addr + fx.Int64(i) * 4, 0, "agent")
        return fx.Vector.from_elements([fx.Int32(w), fx.Int32(w >> 32)], fx.Int32)
    return load32(addr, i, "agent")


def _sync_fence(ordering, scope):
    """A fence that synchronizes without making anything available or
    visible."""
    fence = _llvm.FenceOp(ordering, syncscope=scope)
    fence.attributes["llvm.mmra"] = ir.Attribute.parse(
        '#llvm.mmra_tag<"amdgcn-av":"none">'
    )


def sync_release(scope):
    _sync_fence(_llvm.AtomicOrdering.release, scope)


def sync_acquire(scope):
    _sync_fence(_llvm.AtomicOrdering.acquire, scope)


def release(scope):
    fx.memory_fence(syncscope=scope, ordering=fx.AtomicOrdering.Release)


def acquire_fence(scope):
    fx.memory_fence(syncscope=scope, ordering=fx.AtomicOrdering.Acquire)


def cta_sync(drain=False):
    """``gpu.barrier``: workgroup release fence, ``s_barrier``, acquire fence
    (HIP's ``__syncthreads``). ``drain``: every wave's stores complete first."""
    if const_expr(drain):
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


def error_word(base):
    return load32(base, 0, None)


def pack(word, tag):
    return (fx.Int64(tag) << 32) | fx.Int64(fx.Uint32(fx.Int32(word)))


@traced
def _spin(load_all, pending, err_base, ticks):
    t0 = memrealtime()
    v = load_all()
    err = error_word(err_base)
    late = fx.Int64(0)
    while pending(v) & (err == 0) & (late < ticks):
        rocdl.s_sleep(1)
        v = load_all()
        err = error_word(err_base)
        late = memrealtime() - t0
    return v, pending(v) & (err == 0)


@traced
def _fail_if(timed_out, fail, site):
    if timed_out:
        fail(site)


class Mailbox:
    """Tagged pairs of one launch (``tag``), every address an ``Addr``."""

    def __init__(self, tag, err_base, peer_bases, rank, epoch, ticks=WAIT_TICKS):
        self.tag = tag
        self.err_base = err_base
        self.peer_bases = peer_bases
        self.rank = rank
        self.epoch = epoch
        self.ticks = ticks

    def put(self, a, i, v):
        """Pair i of ``a`` := (v, tag), monotonic: a value no other data hangs on."""
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        store64(a.value, i, pack(bits, self.tag), _SCOPE[a.space])

    def put_words(self, a, i, words):
        for k in range_constexpr(len(words)):
            store64(a.value, i + k, pack(words[k], self.tag), _SCOPE[a.space])

    def publish(self, a, i, v):
        """Pair i of ``a`` := (v, tag) after a synchronizing release: what
        happened before it (the CTA's stores past a ``cta_sync(drain=True)``,
        pairs it acquired) happens before an acquire of the pair."""
        sync_release(_SCOPE[a.space])
        self.put(a, i, v)

    def acquire(self, space):
        """After a poll of published pairs, before what they announce."""
        sync_acquire(_SCOPE[space])

    def put_release_all(self, addrs, i, v):
        """Pair i of every ``addrs`` := (v, tag) after one release fence."""
        release(_SCOPE[addrs[0].space])
        for a in addrs:
            self.put(a, i, v)

    def acquire_system(self):
        acquire_fence(None)

    def fail(self, site):
        lo = (fx.Int64(fx.Int32(self.rank)) << 32) | fx.Int64(1)
        hi = (fx.Int64(fx.Int32(site)) << 32) | fx.Int64(
            fx.Uint32(fx.Int32(self.epoch))
        )
        for base in self.peer_bases:
            store64(base, 1, hi, None)
            store64(base, 0, lo, None)

    def poll(self, specs, batch=POLL_MAX, site=0):
        """Pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})], all of one
        space, re-loaded together until every tag is this launch's. Returns the
        value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch, site) + self.poll(
                specs[batch:], batch, site
            )
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        scope = _SCOPE[spaces.pop()]
        tag = self.tag

        def load_all():
            words = []
            for a, i, n in specs:
                for q in range(n):
                    w = load64(a.value, fx.Int32(i) + q, scope)
                    words += [fx.Int32(w), fx.Int32(w >> 32)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v, timed_out = _spin(load_all, pending, self.err_base, self.ticks)
        _fail_if(timed_out, self.fail, site)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs
