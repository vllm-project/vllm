# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/debug.py
"""An ``ATOM_MONO_DEBUG`` step's report: the mailbox waits that gave up
(``sync._spin_bounded`` records, one a CTA) and the ranks the
step fence gave up on (``StepMailboxes.missing_ranks``). Synchronizes: eager only."""

import zlib

import torch

from vllm.models.kimi_k3.amd.mono.common.execution import BLOCKS
from vllm.models.kimi_k3.amd.mono.common.layout import DIAG_WORDS

DIAG_BYTES = BLOCKS * DIAG_WORDS * 4


def region_id(name: str) -> int:
    """A mailbox region's id in a wait record: its name's CRC-32 (31 bits), the
    same in every kernel and table that names it."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def region_ids(names) -> dict[str, int]:
    """Name -> ``region_id``: a debug build's ``Mailbox`` table."""
    return {n: region_id(n) for n in names}


def region_namer(names):
    """``region_id`` -> name for ``given_up_waits`` (an unknown id as itself)."""
    by_id = {region_id(n): n for n in names}
    return lambda rid: by_id.get(rid, rid)


def given_up_waits(diag: torch.Tensor, name_of) -> list[str]:
    """Each CTA's first wait of the step that gave up, the earliest first:
    ``diag`` the records' ``DIAG_BYTES`` bytes, ``name_of(region_id)`` the name a
    record's region goes by."""
    torch.accelerator.synchronize()
    records = diag.view(torch.int32).view(BLOCKS, DIAG_WORDS)
    records = records[records[:, 0] != 0].tolist()
    t0 = min((r[6] for r in records), default=0)
    return [
        f"+{(at - t0) / 100:.0f} us CTA {cta}: {name_of(rid)} pair {idx} waited for"
        f" tag {tag}, saw {seen}"
        for _, rid, idx, tag, seen, cta, at, _ in sorted(records, key=lambda r: r[6])
    ]


def raise_if_given_up(what: str, rank: int, rows: int, waits, mailboxes) -> None:
    """Raise with ``waits`` and the fence's missing ranks, if any."""
    lines = list(waits) + [
        f"step fence: rank {r} never arrived" for r in mailboxes.missing_ranks()
    ]
    if lines:
        raise RuntimeError(
            f"{what} rank {rank}: {len(lines)} waits gave up ({rows}-row step):\n"
            + "\n".join(lines)
        )
