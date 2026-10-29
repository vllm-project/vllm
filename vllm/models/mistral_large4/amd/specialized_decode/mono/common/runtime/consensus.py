# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
    except MonoUnsupported:
        tp_agree(False, group)
        raise
    except Exception as error:
        tp_agree(False, group)
        # Peers see only `another TP rank refused mono`, a MonoUnsupported that
        # LazyRunner catches and falls back on. A rank that failed for some
        # other reason -- an OOM dequantizing latent-MoE weights, say -- has to
        # refuse the same way, or it alone propagates an exception nobody
        # catches and dies while every peer keeps serving. The original is
        # chained, and its type and message are in the refusal text.
        raise MonoUnsupported(f"{type(error).__name__}: {error}") from error
    if not tp_agree(True, group):
        raise MonoUnsupported("another TP rank refused mono")


def tp_uniform_local_validation(
    error: Exception | None,
    *,
    group,
    world_size: int,
    context: str,
) -> None:
    """``bind_agreed``'s verbose form: every rank's refusal reason gathered, so
    the raised message names which rank refused and why. ``error`` is this
    rank's local verdict (None when it passed). Every rank must call it at the
    same point; ``group`` is a CPU (gloo) process group."""
    local = None if error is None else f"{type(error).__name__}: {error}"
    if world_size == 1:
        reports = [local]
    else:
        import torch.distributed as dist

        reports = [None] * world_size
        dist.all_gather_object(reports, local, group=group)
    failed = [
        (rank, report) for rank, report in enumerate(reports) if report is not None
    ]
    if failed:
        detail = "; ".join(f"rank {rank}: {report}" for rank, report in failed)
        raise MonoUnsupported(f"{context}: {detail}")
