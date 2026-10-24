# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/mono/plan/check.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hand-off contract of a traced kernel, checked at build time.

A mono kernel's CTAs are all co-resident and each walks the program's stages in
order, a stage's tasks spread over the grid. A poll then always makes progress if
the stage it waits on comes earlier in the program, or is its own stage and the
region an exchange (an all-reduce: a task pushes before it waits on its peers'
pushes). ``check`` holds a ``trace.Record`` to that and to the regions'
declarations: the stages that put each and the declared space (scope).
"""

from dataclasses import dataclass

from .trace import (
    Record,
    Space,
)


@dataclass(frozen=True)
class RegionDecl:
    name: str
    space: Space
    # the stage that puts it; several when the step's input picks which one puts a
    # cell (a token's last index tree level): every one runs before a poller
    writer: str | tuple[str, ...]
    exchange: bool = False  # polled by its writer stage too (an all-reduce)

    @property
    def writers(self) -> tuple[str, ...]:
        return (self.writer,) if isinstance(self.writer, str) else self.writer


class ContractError(Exception):
    """A traced kernel breaks its hand-off contract."""


def check(record: Record, decls: list[RegionDecl]) -> None:
    """Raise ``ContractError`` listing every breach of ``record`` against
    ``decls``."""
    by_name = {d.name: d for d in decls}
    first: dict[str, int] = {}
    for i, stage in enumerate(record.stages):
        first.setdefault(stage, i)
    problems: list[str] = []
    used = set()
    polled_before_put: set[tuple[str, str]] = set()
    put_seen: set[tuple[str, str]] = set()
    for a in record.accesses:
        d = by_name.get(a.region)
        if d is None:
            problems.append(f"{a.kind} of undeclared region {a.region} in {a.stage}")
            continue
        used.add(a.region)
        if a.space is not d.space:
            problems.append(
                f"{a.kind} of {a.region} in {a.stage} as {a.space.value}, "
                f"declared {d.space.value}"
            )
        if a.kind == "put":
            put_seen.add((a.stage, a.region))
            if a.stage not in d.writers:
                problems.append(
                    f"{a.region} put by {a.stage}, its writer is {d.writer}"
                )
            continue
        if a.stage in d.writers:
            if not d.exchange:
                problems.append(
                    f"{a.region} polled by its own writer {a.stage} (not an exchange)"
                )
            elif (a.stage, a.region) not in put_seen:
                polled_before_put.add((a.stage, a.region))
        else:
            for w in d.writers:
                if w not in first:
                    problems.append(
                        f"{a.region} polled by {a.stage}, its writer {w} never runs"
                    )
                elif first[w] > first[a.stage]:
                    problems.append(
                        f"{a.region} polled by {a.stage}, "
                        f"which runs before its writer {w}"
                    )
    for stage, region in sorted(polled_before_put):
        problems.append(f"exchange {region} polled in {stage} before any put of it")
    for name in sorted(set(by_name) - used):
        problems.append(f"declared region {name} is never accessed")
    if problems:
        raise ContractError("\n".join(problems))
