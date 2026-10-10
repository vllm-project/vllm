# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ranks.py
"""A TP group seen from inside a kernel: every rank's peer buffer, and an
all-reduce's sum of the ranks' partials. ``tp`` is a build-time constant."""

from flydsl.expr import range_constexpr

from vllm.models.kimi_k3.amd.mono.common.ops import bf2_f32, load_ptr64


def peer_bases(peers, tp):
    """Every rank's peer buffer base from the ``peers`` table (an i64 a rank),
    loaded once: a put's address must not wait on a load of its own."""
    return [load_ptr64(peers, p) for p in range(tp)]


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
