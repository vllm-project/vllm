# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded tagged hand-offs and the sticky error word.

Contract, on the LLVM AMDGPU memory model (AMDGPUUsage, "Memory Model GFX942",
whose code sequences GFX950 uses):

- A pair is one naturally aligned 64-bit LLVM atomic, (value | tag << 32),
  stored and loaded ``monotonic`` at its space's scope: ``agent`` for this
  GPU's scratch, system for the fine-grained uncached (MTYPE UC) peer heap.
  An atomic access may not be split (LLVM Atomics, "Unordered": "a load cannot
  see a value which was never stored"), so a load that sees a pair's tag sees
  that store's value. Lowered: one ``global_store/load_dwordx2`` with ``sc1``
  (agent) or ``sc0 sc1`` (system).
- Plain data a pair announces on this GPU is stored and loaded with ``sc1``,
  the GFX942 encoding of agent-scope atomic accesses, which are coherent
  across the agent's L2s; so no ``buffer_wbl2`` / ``buffer_inv`` is needed
  for it. Its order against the pair is the vector-memory counter: the
  producer's ``s_waitcnt vmcnt(0)`` before the pair store (``publish``; a CTA
  first drains every wave and barriers, ``cta_sync``), and the consumer's
  data loads issued after its poll completed. GFX942: "A s_waitcnt vmcnt(0)
  ... ensures a previous vector memory operation has completed before
  executing a subsequent vector memory ... operation and so can be used to
  meet the requirements of acquire and release."
- The step-end fault handshake orders the error words across GPUs with
  system-scope release / acquire fences (``publish_release``,
  ``acquire_system``).

A poll gives up after ``ticks`` of the 100 MHz clock or once any rank has set
the error word; a timeout sets ``[1, rank, epoch, site]`` on every rank.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

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


def release(scope):
    fx.memory_fence(syncscope=scope, ordering=fx.AtomicOrdering.Release)


def acquire_fence(scope):
    fx.memory_fence(syncscope=scope, ordering=fx.AtomicOrdering.Acquire)


def cta_sync(drain=False):
    """A workgroup barrier that orders memory both ways (HIP's
    ``__syncthreads``). ``drain``: every wave's stores complete first, so one
    thread's release afterwards covers the whole CTA's."""
    if const_expr(drain):
        rocdl.s_waitcnt(vmcnt=0)
    release("workgroup")
    gpu.barrier()
    acquire_fence("workgroup")


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
        """Pair i of ``a`` (scratch) := (v, tag) once the calling wave's ``sc1``
        stores (past a ``cta_sync(drain=True)``, its CTA's) have completed."""
        assert a.space is Space.SCRATCH
        rocdl.s_waitcnt(vmcnt=0)
        self.put(a, i, v)

    def publish_release(self, a, i, v):
        """Pair i of ``a`` := (v, tag) after a release fence at its scope."""
        release(_SCOPE[a.space])
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
