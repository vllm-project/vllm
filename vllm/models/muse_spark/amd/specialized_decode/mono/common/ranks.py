# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ranks.py
"""A TP group seen from inside a kernel: every rank's peer buffer, a traced
rank's address (``peer_at``), and an all-reduce's sum of the ranks' partials.
``tp`` is a build-time constant."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf2_f32,
    load_ptr64,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import Addr


def peer_bases(peers, tp):
    """Every rank's peer buffer base from the ``peers`` table (an i64 a rank),
    loaded once: a put's address must not wait on a load of its own."""
    return [load_ptr64(peers, p) for p in range(tp)]


def peer_at(addrs, p):
    """``addrs[p]`` (an ``Addr`` a rank, e.g. a region's ``peer`` list) for a
    traced rank ``p``: a select chain over the ranks' addresses, the region and
    space of rank 0's (they are all the same region)."""
    if isinstance(p, int):
        return addrs[p]
    v = addrs[0].value
    for q in range(1, len(addrs)):
        v = fx.Int64((p == q).select(addrs[q].value, v))
    return Addr(addrs[0].region, addrs[0].space, v)


def sum_partials(poll, own, pair_of, tp):
    """The ``tp`` ranks' bf16 pair partials at ``pair_of(src)`` of the mailbox
    region ``own``, summed in fp32 in rank order 0 .. tp - 1: one poll batch ->
    (lo sum, hi sum). Every rank sums in the same order, so they agree."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1
