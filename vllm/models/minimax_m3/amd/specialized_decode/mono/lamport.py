# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport (sentinel) inboxes: a cross-GPU hand-off whose payload is its own
flag.

A sender stores plain data into an inbox in a peer's buffer (`PeerBuffer`,
uncached). Every 16-bit half-word of an inbox holds the sentinel 0xFFFF until a
push lands. Finite bf16 data never holds it (bf16 0xFFFF is a NaN), nor does an
E8M0 scale (at most 254) or a finite e4m3 code; a sentinel half-word in pushed
data is rewritten to 0x7FFF first (`canon`: still a NaN). The reader re-loads
its slots until none of the half-words it uses is the sentinel: one round trip
after the last byte lands, no flag hop and no tag bytes on the wire.

The protocol the users keep (see glue/kernel.py, the MiniMax-M3.1 decode
glue's reduce-scatter and all-gather):

- A slot is polled and re-armed (the sentinel written back) by exactly
  one lane, after that lane consumed it: a slot re-armed by a fast reader before
  a slow one saw it would lose the hand-off.
- Inboxes are double-buffered by launch parity (the caller's epoch or colour):
  a slot of parity p is written again two collective launches later, which a
  peer reaches only after this rank finished the next launch (it needs this
  rank's next pushes), i.e. after the re-arm stores completed at a kernel
  boundary.
- The host arms every inbox once at setup and again after a fault reset
  (`arm`), with no launch in flight.

Every wait is bounded: `ticks` of the 100 MHz s_memrealtime clock, and it gives
up once any rank set the error record (word 0 at `err_base`, see
`Mailbox.fail` in sync.py); a lane that timed out calls `fail(site)`. The error word
and the clock are read only after a first miss, so a hand-off that already
landed costs one load round trip."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    buf_load,
    lane_gather,
    memrealtime,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import sel

SENTINEL_BYTE = 0xFF
# Bit 15 of both half-words (0x80008000), as a signed Int32.
SENT_SIGN = -0x7FFF8000


# ---------------------------------------------------------------- host
def arm(buf, offset: int, nbytes: int) -> None:
    """Every half-word of the inbox bytes [offset, offset + nbytes) of the uint8
    tensor `buf` := the sentinel. Host side, no launch in flight."""
    buf[offset : offset + nbytes].fill_(SENTINEL_BYTE)


# ---------------------------------------------------------------- loads / stores
def gptr(addr, ty, align):
    """The global pointer of type `ty` at the Int64 address `addr`."""
    return fx.inttoptr(
        fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align),
        fx.Int64(addr),
    )


def ld32(addr):
    """32-bit system-scope monotonic load: uncached, never hoisted or merged
    (a poll)."""
    return fx.generic_load(
        gptr(addr, T.i32, 4), dtype=fx.Int32, memory_order=fx.AtomicOrdering.Monotonic
    )


def ld64(addr):
    """64-bit system-scope monotonic load (a poll of two words)."""
    return fx.generic_load(
        gptr(addr, T.i64, 8), dtype=fx.Int64, memory_order=fx.AtomicOrdering.Monotonic
    )


def st16(addr, ws):
    """Four Int32 words at a 16-byte aligned address: one plain 16-byte store
    (a push into an inbox)."""
    fx.generic_store(gptr(addr, T.i32, 16), fx.Vector.from_elements(ws, fx.Int32))


def load16(base, dw):
    """Four dwords at dword `dw` of a (cached) device buffer, as a Vector."""
    return fx.Vector(buf_load(base, dw, vec_width=4, dtype=T.i32))


def load_dwords(base, dw, n):
    """`n` (1, 2 or 4) dwords at dword `dw` of a (cached) device buffer."""
    return fx.Vector(buf_load(base, dw, vec_width=n, dtype=T.i32))


def words(vec, n=4):
    """The first `n` elements of a Vector as Int32 words."""
    return [fx.Int32(vec[e]) for e in range(n)]


# ---------------------------------------------------------------- sentinel
def sent_mask(w):
    """Bit 15 of every half-word of `w` that is the sentinel 0xFFFF (AITER
    one_shot_allreduce's test)."""
    x = w ^ fx.Int32(-1)
    nz = ((x & fx.Int32(0x7FFF7FFF)) + fx.Int32(0x7FFF7FFF)) | x
    return (nz & fx.Int32(SENT_SIGN)) ^ fx.Int32(SENT_SIGN)


def canon(ws):
    """Sentinel half-words 0xFFFF -> 0x7FFF (a NaN stays a NaN): what a sender
    pushes."""
    return [w ^ sent_mask(w) for w in ws]


def wave_last(a):
    """Lane 63's value of the per-lane Int64 `a` (both halves through
    ds_bpermute)."""
    lo = lane_gather(fx.Int32(a), 63)
    hi = lane_gather(fx.Int32(a >> 32), 63)
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def triple_off(tl):
    """Byte offset, in a run of packed bf16, of the two dwords that hold triple
    tl (values 3 tl .. 3 tl + 2)."""
    return 6 * tl - 2 * (tl & 1)


# ---------------------------------------------------------------- waits
@traced
def spin(load_all, pending, err_word, ticks, fail, site):
    """Re-load (`load_all`) until `pending` clears; bounded by `ticks`, gives up
    once `err_word()` is set, calls `fail(site)` on a timeout. The error word and
    the clock are read only after a first miss."""
    v = load_all()
    late = fx.Int64(0)
    err = fx.Int32(0)
    if pending(v):
        t0 = memrealtime()
        err = err_word()
        while pending(v) & (err == 0) & (late < ticks):
            rocdl.s_sleep(1)
            v = load_all()
            err = err_word()
            late = memrealtime() - t0
    if pending(v) & (err == 0):
        fail(site)
    return v


@traced
def probe(addr, wave_live, err_word, ticks):
    """Every lane of the wave spins on lane 63's word at `addr` (one request a
    wave) until it is sentinel-free: a hint before the full poll, bounded, with
    no fault of its own. `wave_live`: false for a wave with no live lane."""
    pa = wave_last(addr)
    w = ld32(pa)
    late = fx.Int64(0)
    err = fx.Int32(0)
    if (sent_mask(w) != 0) & wave_live:
        t0 = memrealtime()
        while (sent_mask(w) != 0) & (err == 0) & (late < ticks):
            rocdl.s_sleep(1)
            w = ld32(pa)
            err = err_word()
            late = memrealtime() - t0


@traced
def poll_triples(addrs, tl, lives, err_word, ticks, fail, site):
    """Triple tl's two dwords at each address of `addrs` (+ `triple_off(tl)`)
    -> [(lo, hi)] with the 3 bf16 in bits 0 .. 47, once none of the lane's 3
    half-words is the sentinel (the 4th half-word loaded belongs to a neighbour
    triple and is not checked). `lives`: an address the lane does not need is
    loaded once and never checked."""
    odd = (tl & 1) != 0
    # odd: only the high half of word 0 is ours; even: only the low half of
    # word 1
    m0 = odd.select(fx.Int32(-0x80000000), fx.Int32(SENT_SIGN))
    m1 = odd.select(fx.Int32(SENT_SIGN), fx.Int32(0x8000))
    off = fx.Int64(triple_off(tl))
    pa = [a + off for a in addrs]
    masks = []
    for live in lives:
        masks += [sel(live, m0, fx.Int32(0)), sel(live, m1, fx.Int32(0))]

    def load_all():
        ws = []
        for a in pa:
            ws += [ld32(a), ld32(a + fx.Int64(4))]
        return fx.Vector.from_elements(ws, fx.Int32)

    def pending(v):
        acc = sent_mask(fx.Int32(v[0])) & masks[0]
        for e in range_constexpr(1, 2 * len(pa)):
            acc = acc | (sent_mask(fx.Int32(v[e])) & masks[e])
        return acc != fx.Int32(0)

    v = spin(load_all, pending, err_word, ticks, fail, site)
    out = []
    for i in range_constexpr(len(pa)):
        w = (fx.Int64(fx.Uint32(fx.Int32(v[2 * i + 1]))) << 32) | fx.Int64(
            fx.Uint32(fx.Int32(v[2 * i]))
        )
        w = w >> fx.Int64(16 * (tl & 1))
        out.append((fx.Int32(w), fx.Int32(w >> 32)))
    return out


@traced
def poll_vecs(addrs, live, err_word, ticks, fail, site):
    """16-byte slots at the per-lane addresses `addrs` -> 4 Int32 words each,
    once no half-word is the sentinel (two 64-bit loads a slot). `live` false:
    loaded once, never checked."""
    mask = sel(live, fx.Int32(SENT_SIGN), fx.Int32(0))

    def load_all():
        ws = []
        for a in addrs:
            for h in range_constexpr(2):
                w = ld64(a + fx.Int64(8 * h))
                ws += [fx.Int32(w), fx.Int32(w >> 32)]
        return fx.Vector.from_elements(ws, fx.Int32)

    def pending(v):
        acc = sent_mask(fx.Int32(v[0]))
        for e in range_constexpr(1, 4 * len(addrs)):
            acc = acc | sent_mask(fx.Int32(v[e]))
        return (acc & mask) != fx.Int32(0)

    v = spin(load_all, pending, err_word, ticks, fail, site)
    return [[fx.Int32(v[4 * i + e]) for e in range(4)] for i in range(len(addrs))]
