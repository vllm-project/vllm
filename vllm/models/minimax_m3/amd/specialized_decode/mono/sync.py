# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded tagged hand-offs and the sticky error word.

Pairs are DS V4.1's ``Mailbox`` (value + epoch tag in one 8-byte store). A poll
here gives up after ``WAIT_TICKS`` of the 100 MHz clock or as soon as any rank
has set the error word; a timeout sets ``[1, rank, epoch, site]`` on every
rank. A failed launch still runs to completion; the host checks the word
before the step's hidden states are used.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_SYS,
    memrealtime,
    rsrc,
    traced,
)
from vllm.models.deepseek_v41.amd.mono.common.sync import (
    _SCOPE_CM,
    POLL_MAX,
    Mailbox,
)

WAIT_TICKS = 1_000_000_000  # 10 s
ERROR_WORDS = 4


def error_word(base):
    return fx.Int32(
        bo.buffer_load(rsrc(base), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS)
    )


@traced
def _spin(load_all, pending, err_base):
    t0 = memrealtime()
    v = load_all()
    err = error_word(err_base)
    late = fx.Int64(0)
    while pending(v) & (err == 0) & (late < WAIT_TICKS):
        rocdl.s_sleep(1)
        v = load_all()
        err = error_word(err_base)
        late = memrealtime() - t0
    return v, pending(v) & (err == 0)


@traced
def _fail_if(timed_out, fail, site):
    if timed_out:
        fail(site)


class BoundedMailbox(Mailbox):
    """``Mailbox`` whose polls are bounded; ``fail(site)`` raises the error."""

    def __init__(self, tag, err_base, peer_bases, rank, epoch):
        super().__init__(tag)
        self.err_base = err_base
        self.peer_bases = peer_bases
        self.rank = rank
        self.epoch = epoch

    def fail(self, site):
        words = fx.Vector.from_elements(
            [fx.Int32(1), fx.Int32(self.rank), fx.Int32(self.epoch), fx.Int32(site)],
            fx.Int32,
        )
        for base in self.peer_bases:
            bo.buffer_store(words, rsrc(base), 0, cache_modifier=CM_SYS)

    def poll(self, specs, batch=POLL_MAX, site=0):
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch, site) + self.poll(
                specs[batch:], batch, site
            )
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        bases = [a.value for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v, timed_out = _spin(load_all, pending, self.err_base)
        _fail_if(timed_out, self.fail, site)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs
