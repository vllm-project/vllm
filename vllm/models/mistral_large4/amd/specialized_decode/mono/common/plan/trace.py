# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A kernel's mailbox accesses, recorded while FlyDSL traces it.

Tracing runs the kernel body's Python once, so every mailbox put / poll executes
once as Python: noting it here costs the device nothing (no IR is emitted). The
stage currently being traced is set by ``enter_stage``, called where the program
starts each stage, so the record is ``(stage, region, put | poll)`` in program
order.

Everything here is plain functions on a module-level recorder: FlyDSL's AST
rewriter carries every local whose method is called inside a dynamic if / for as
loop state, so a kernel body must not call methods on these objects.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Access:
    stage: str
    region: str
    space: Space
    kind: str  # "put" | "poll"


@dataclass
class Record:
    stages: list[str] = field(default_factory=list)  # in the order entered
    accesses: list[Access] = field(default_factory=list)
    unnamed: list[tuple[str, str]] = field(default_factory=list)  # (stage, kind)


_record: Record | None = None


@contextmanager
def recording():
    """Record the accesses of kernels traced inside this block."""
    global _record
    outer, _record = _record, Record()
    try:
        yield _record
    finally:
        _record = outer


def enter_stage(name: str) -> None:
    if _record is not None:
        _record.stages.append(name)


def _stage() -> str:
    return _record.stages[-1] if _record.stages else "<before any stage>"


def note(region: str, space: Space, kind: str) -> None:
    if _record is not None:
        _record.accesses.append(Access(_stage(), region, space, kind))


def note_unnamed(kind: str) -> None:
    """A put / poll on a raw address: the check reports it (every mailbox access of
    a checked kernel names its region)."""
    if _record is not None:
        _record.unnamed.append((_stage(), kind))
