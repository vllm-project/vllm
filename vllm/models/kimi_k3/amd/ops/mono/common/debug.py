# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/debug.py
"""Wait records: which mailbox region a CTA's given-up wait was on."""

import zlib

import torch

from vllm.models.common.mono.abi import RECORD_WORDS
from vllm.models.kimi_k3.amd.ops.mono.common.plan import BLOCKS


def region_id(name: str) -> int:
    """A mailbox region's id in a wait record: its name's CRC-32 (31 bits), the
    same in every kernel and table that names it."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def region_ids(names) -> dict[str, int]:
    """Name -> ``region_id``: a ``Mailbox``'s table."""
    return {n: region_id(n) for n in names}


def given_up_waits(health: torch.Tensor, names) -> list[str]:
    """Each CTA's first wait that gave up, the earliest first, from a rank's
    ``health`` words; ``names`` the regions a record may name."""
    by_id = {region_id(n): n for n in names}
    records = health.view(BLOCKS + 1, RECORD_WORDS)[1:]
    records = records[records[:, 0] != 0].tolist()
    t0 = min((r[6] for r in records), default=0)
    return [
        f"+{(at - t0) / 100:.0f} us CTA {cta}: {by_id.get(rid, rid)} pair {idx}"
        f" waited for tag {tag}, saw {seen}"
        for _, rid, idx, tag, seen, cta, at, _ in sorted(records, key=lambda r: r[6])
    ]
