# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/consensus.py
"""Decisions every TP rank must take alike.

A mono kernel waits in-kernel on every peer rank, so a rank that turned mono off
alone -- or failed to build what the others launch -- leaves the others spinning
on the GPU for ever. Each such decision is a local verdict agreed on over the TP
group's CPU group before anything collective (peer handshake, launch) follows.
"""

import torch
import torch.distributed as dist


class MonoUnsupported(Exception):
    """The loaded model or runtime configuration is outside what mono serves."""


def tp_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true. Every rank must call
    it at the same point; ``group`` is a CPU (gloo) process group, or None for a
    single rank."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def bind_agreed(bind, group) -> None:
    """``bind()`` on this rank (raising ``MonoUnsupported`` to refuse), then every
    rank's verdict agreed: a rank that raised re-raises, the others raise too."""
    try:
        bind()
    except Exception:
        tp_agree(False, group)
        raise
    if not tp_agree(True, group):
        raise MonoUnsupported("another TP rank refused mono")
