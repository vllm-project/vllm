# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/layout.py
"""A kernel's mailbox regions laid out in its scratch or peer buffer."""

PAIR_BYTES = 8  # a (value, tag) pair
# a debug build's wait record, one a CTA: flag, region id, pair index, tag,
# seen tag, CTA, clock, pad (``vllm.models.kimi_k3.amd.mono.common.sync._spin_bounded``)
DIAG_WORDS = 8


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out
