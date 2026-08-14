# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare prefetch weight-offload schedules against one recorded run.

Picking `--offload-group-size` / `--offload-num-in-group` /
`--offload-prefetch-step` is mostly guesswork: the combinations multiply, and
the ones that do not fit only say so by failing to start.

Two quantities decide whether a schedule is viable, and both are exact
arithmetic once the per-position weight and buffer sizes are known:

* how many bytes stay resident on the GPU, and
* how many bytes are re-copied host-to-device on every forward.

Those sizes depend on the model, its quantization, and the tensor/expert
parallel split, so only a running engine knows them. Serve once with
`VLLM_PREFETCH_LOG_SCHEDULE=1`; the offloader emits a manifest after
post-init, when buffer-fallback decisions are final. This module reads that
manifest and evaluates other schedules from it, so choosing between N of them
costs one launch rather than N.

Everything else about a deployment -- KV cache, non-offloadable weights,
allocator slack, CUDA graph reserve -- is identical across schedules, so it
cancels. Results are therefore reported as a delta against the recorded run,
which needs no further input from the operator.

This deliberately models memory, not latency. Whether a transfer hides behind
compute depends on measured bandwidth and per-layer compute time, which this
cannot obtain and does not guess at.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import regex as re

SUPPORTED_SCHEMA_VERSION = 1

_MANIFEST_LINE = re.compile(r"manifest_json=(\{.*\})\s*$")


@dataclass(frozen=True)
class Schedule:
    group_size: int
    num_in_group: int
    prefetch_step: int

    def __str__(self) -> str:
        return f"{self.group_size}/{self.num_in_group}/{self.prefetch_step}"


@dataclass(frozen=True)
class Position:
    """One ordered module position the offload planner walks."""

    module_index: int
    selector_bytes: int
    offloadable: bool
    # Known only for positions the recorded run actually offloaded: buffer
    # layouts are finalized per selected unit.
    pooled_bytes_per_slot: int | None
    direct_bytes: int | None


@dataclass(frozen=True)
class RunProfile:
    rank: int | None
    schedule: Schedule
    positions: tuple[Position, ...]

    @property
    def num_positions(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class Candidate:
    schedule: Schedule
    selected: tuple[int, ...]
    offloaded_bytes: int
    runtime_buffer_bytes: int
    resident_delta_bytes: int
    h2d_bytes_per_forward: int
    # Schedules that differ only in which positions they pick, while moving
    # and holding exactly the same bytes, are one choice as far as this tool
    # can tell. The simplest is kept and the rest counted here.
    equivalent_schedules: int = 1


def iter_manifests(text: str) -> list[dict[str, Any]]:
    """Read every manifest in `text`.

    Accepts a bare manifest document, a JSON list of them, or server log
    output carrying one `manifest_json=` line per rank.
    """

    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        document = json.loads(stripped)
        return document if isinstance(document, list) else [document]

    manifests = [
        json.loads(match.group(1))
        for match in (_MANIFEST_LINE.search(line) for line in text.splitlines())
        if match
    ]
    if not manifests:
        raise ValueError(
            "no manifest found: expected a manifest document, or a server log "
            "containing a 'manifest_json=' line (serve with "
            "VLLM_PREFETCH_LOG_SCHEDULE=1)"
        )
    return manifests


def select_manifest(
    manifests: Sequence[dict[str, Any]], rank: int | None
) -> dict[str, Any]:
    """Pick one rank's manifest.

    Offload schedules are rank-local, so merging ranks would be meaningless.
    Ask rather than guess when they differ.
    """

    if rank is not None:
        for manifest in manifests:
            if manifest.get("rank") == rank:
                return manifest
        found = ", ".join(sorted(str(item.get("rank")) for item in manifests))
        raise ValueError(f"no manifest for rank {rank}; found rank(s) {found}")
    if len(manifests) == 1:
        return manifests[0]
    shapes = {json.dumps(item.get("positions"), sort_keys=True) for item in manifests}
    if len(shapes) > 1:
        raise ValueError(
            f"{len(manifests)} manifests describe different rank-local models; "
            "pass --rank to choose one"
        )
    return manifests[0]


def profile_from_manifest(manifest: dict[str, Any]) -> RunProfile:
    version = manifest.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version {version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )

    raw_positions = sorted(manifest["positions"], key=lambda item: item["module_index"])
    if len(raw_positions) != int(manifest["module_count"]):
        raise ValueError(
            f"manifest lists {len(raw_positions)} positions but module_count is "
            f"{manifest['module_count']}"
        )

    units = manifest.get("units", ())
    # A manifest indexes buffers by runtime unit; a schedule selects module
    # positions. Everything below works in module positions.
    pooled_of_module: dict[int, int] = {}
    direct_of_module: dict[int, int] = {}
    layout_of_unit: dict[int, int] = {}
    for layout in manifest.get("pooled_buffer_layouts", ()):
        for unit_idx in layout.get("unit_indices", ()):
            layout_of_unit[int(unit_idx)] = int(layout["bytes_per_slot"])
    for unit in units:
        module_index = int(unit["module_index"])
        direct_of_module[module_index] = int(unit.get("direct_runtime_buffer_bytes", 0))
        unit_idx = int(unit["unit_idx"])
        if unit_idx in layout_of_unit:
            pooled_of_module[module_index] = layout_of_unit[unit_idx]

    positions = tuple(
        Position(
            module_index=int(item["module_index"]),
            selector_bytes=int(item["logical_parameter_bytes"]),
            offloadable=bool(item["offloadable"]),
            pooled_bytes_per_slot=pooled_of_module.get(int(item["module_index"])),
            direct_bytes=direct_of_module.get(int(item["module_index"])),
        )
        for item in raw_positions
    )

    config = manifest["config"]
    return RunProfile(
        rank=manifest.get("rank"),
        schedule=Schedule(
            group_size=int(config["group_size"]),
            num_in_group=int(config["num_in_group"]),
            prefetch_step=int(config["prefetch_step"]),
        ),
        positions=positions,
    )


def selected_positions(profile: RunProfile, schedule: Schedule) -> tuple[int, ...]:
    """Offload the last `num_in_group` of each `group_size`, minus skips.

    Mirrors the runtime planner: a position whose selector matches nothing on
    this rank never becomes a unit.
    """

    if schedule.group_size <= 0:
        raise ValueError("group_size must be greater than zero")
    if not 0 <= schedule.num_in_group <= schedule.group_size:
        raise ValueError("num_in_group must be between zero and group_size")
    keep_from = schedule.group_size - schedule.num_in_group
    return tuple(
        position.module_index
        for index, position in enumerate(profile.positions)
        if index % schedule.group_size >= keep_from and position.offloadable
    )


def prefetch_after_units(unit_count: int, prefetch_step: int) -> tuple[int | None, ...]:
    """Match `PrefetchRuntimeController` slot reuse.

    Units live in slot `unit_idx % prefetch_step` and advance only through
    their own slot, which differs from a plain `unit + step` rule when the
    unit count is not divisible by the step. A slot holding a single unit
    keeps it resident, so that unit costs no steady-state transfer.
    """

    if unit_count <= 0:
        return ()
    if not 1 <= prefetch_step <= unit_count:
        raise ValueError("prefetch_step must be in [1, unit count]")

    slots: list[list[int]] = [[] for _ in range(prefetch_step)]
    for unit_index in range(unit_count):
        slots[unit_index % prefetch_step].append(unit_index)

    after: list[int | None] = [None] * unit_count
    for units in slots:
        if len(units) > 1:
            for index, unit_index in enumerate(units):
                after[unit_index] = units[(index + 1) % len(units)]
    return tuple(after)


def _buffer_bytes(
    profile: RunProfile, selected: Sequence[int], prefetch_step: int
) -> int:
    """Runtime buffers: `prefetch_step` slots per distinct pooled layout."""

    by_module = {position.module_index: position for position in profile.positions}

    known_pooled = {
        position.pooled_bytes_per_slot
        for position in profile.positions
        if position.pooled_bytes_per_slot is not None
    }
    unknown = [
        index for index in selected if by_module[index].pooled_bytes_per_slot is None
    ]
    if unknown and len(known_pooled) != 1:
        raise ValueError(
            f"positions {unknown} were not offloaded in the recorded run, and "
            "the run used more than one pooled layout, so their buffer size is "
            "unknown. Record a run whose schedule covers them."
        )
    fallback = next(iter(known_pooled)) if known_pooled else 0

    layouts: set[int] = set()
    for index in selected:
        slot_bytes = by_module[index].pooled_bytes_per_slot
        layouts.add(fallback if slot_bytes is None else slot_bytes)
    pooled = prefetch_step * sum(layouts)

    direct_known = {
        position.direct_bytes
        for position in profile.positions
        if position.direct_bytes is not None
    }
    direct_fallback = next(iter(direct_known)) if len(direct_known) == 1 else 0
    direct = 0
    for index in selected:
        position_direct = by_module[index].direct_bytes
        direct += direct_fallback if position_direct is None else position_direct
    return pooled + direct


def schedule_terms(
    profile: RunProfile, schedule: Schedule
) -> tuple[tuple[int, ...], int, int]:
    """The only three quantities a schedule changes.

    Returns the selected positions, the bytes they move off the GPU, and the
    runtime buffer bytes they hold on it.
    """

    selected = selected_positions(profile, schedule)
    if not selected:
        return (), 0, 0
    if not 1 <= schedule.prefetch_step <= len(selected):
        raise ValueError("prefetch_step must be in [1, selected position count]")
    by_module = {position.module_index: position for position in profile.positions}
    offloaded = sum(by_module[index].selector_bytes for index in selected)
    buffers = _buffer_bytes(profile, selected, schedule.prefetch_step)
    return selected, offloaded, buffers


def resident_bytes(profile: RunProfile, schedule: Schedule) -> int:
    """Schedule-dependent resident bytes: buffers held minus weights evicted."""

    _, offloaded, buffers = schedule_terms(profile, schedule)
    return buffers - offloaded


def evaluate(
    profile: RunProfile, schedule: Schedule, *, baseline: int | None = None
) -> Candidate:
    """Score one schedule against the recorded run.

    Everything that does not depend on the schedule -- KV cache, resident
    non-offloadable weights, allocator slack, graph reserve -- is identical
    across candidates and cancels, so the recorded run is the reference point
    and no absolute memory model is needed.
    """

    selected, offloaded, buffers = schedule_terms(profile, schedule)
    by_module = {position.module_index: position for position in profile.positions}
    after = prefetch_after_units(len(selected), schedule.prefetch_step)
    h2d = sum(
        by_module[selected[target]].selector_bytes
        for target in after
        if target is not None
    )
    if baseline is None:
        baseline = resident_bytes(profile, profile.schedule)
    return Candidate(
        schedule=schedule,
        selected=selected,
        offloaded_bytes=offloaded,
        runtime_buffer_bytes=buffers,
        resident_delta_bytes=(buffers - offloaded) - baseline,
        h2d_bytes_per_forward=h2d,
    )


def enumerate_candidates(profile: RunProfile, *, max_prefetch: int) -> list[Candidate]:
    """Score every schedule that selects at least one position."""

    baseline = resident_bytes(profile, profile.schedule)
    results: list[Candidate] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for group_size in range(1, profile.num_positions + 1):
        for num_in_group in range(1, group_size + 1):
            selected = selected_positions(
                profile, Schedule(group_size, num_in_group, 1)
            )
            if not selected:
                continue
            for prefetch_step in range(1, min(max_prefetch, len(selected)) + 1):
                # Different (G, O) that select the same positions with the same
                # window are the same deployment.
                key = (selected, prefetch_step)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    evaluate(
                        profile,
                        Schedule(group_size, num_in_group, prefetch_step),
                        baseline=baseline,
                    )
                )
    return _collapse_equivalent(results)


def _collapse_equivalent(candidates: list[Candidate]) -> list[Candidate]:
    """Keep one schedule per distinct outcome.

    The report is a memory delta and a transfer volume, so two schedules that
    produce both identically are the same choice here. Listing each one is
    noise; dropping them silently would hide alternatives, so the count is
    carried instead.
    """

    by_outcome: dict[tuple[int, int], list[Candidate]] = {}
    for candidate in candidates:
        key = (candidate.resident_delta_bytes, candidate.h2d_bytes_per_forward)
        by_outcome.setdefault(key, []).append(candidate)

    collapsed = []
    for group in by_outcome.values():
        simplest = min(
            group,
            key=lambda item: (
                item.schedule.group_size,
                item.schedule.num_in_group,
                item.schedule.prefetch_step,
            ),
        )
        collapsed.append(
            replace(simplest, equivalent_schedules=len(group))
            if len(group) > 1
            else simplest
        )
    return collapsed


def _gib(value: int) -> str:
    return f"{value / 2**30:+.3f} GiB"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prefetch weight-offload schedules against one recorded "
            "run. Serve with VLLM_PREFETCH_LOG_SCHEDULE=1 to produce the "
            "manifest this reads."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="manifest JSON, or a server log containing a manifest_json= line",
    )
    parser.add_argument(
        "--rank", type=int, help="rank to read when the log holds several"
    )
    parser.add_argument(
        "--max-prefetch",
        type=int,
        default=6,
        help="largest prefetch_step to consider (default: 6)",
    )
    parser.add_argument(
        "--headroom-bytes",
        type=int,
        help=(
            "free GPU bytes observed during the recorded run; when given, "
            "schedules needing more than this are marked infeasible"
        ),
    )
    parser.add_argument(
        "--top", type=int, default=20, help="rows to print (default: 20)"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead")
    args = parser.parse_args(argv)

    manifests = iter_manifests(args.source.read_text())
    profile = profile_from_manifest(select_manifest(manifests, args.rank))
    candidates = enumerate_candidates(profile, max_prefetch=args.max_prefetch)
    candidates.sort(
        key=lambda item: (item.resident_delta_bytes, item.h2d_bytes_per_forward)
    )

    if args.headroom_bytes is not None:
        candidates = [
            item
            for item in candidates
            if item.resident_delta_bytes <= args.headroom_bytes
        ]

    if args.json:
        print(
            json.dumps(
                {
                    "rank": profile.rank,
                    "recorded_schedule": str(profile.schedule),
                    "candidates": [
                        {
                            "schedule": str(item.schedule),
                            "selected_positions": len(item.selected),
                            "offloaded_bytes": item.offloaded_bytes,
                            "runtime_buffer_bytes": item.runtime_buffer_bytes,
                            "resident_delta_bytes": item.resident_delta_bytes,
                            "h2d_bytes_per_forward": item.h2d_bytes_per_forward,
                            "equivalent_schedules": item.equivalent_schedules,
                        }
                        for item in candidates[: args.top]
                    ],
                },
                indent=2,
            )
        )
        return 0

    recorded = evaluate(profile, profile.schedule)
    print(
        f"recorded run: rank={profile.rank} schedule={profile.schedule} "
        f"units={len(recorded.selected)} "
        f"H2D/forward={recorded.h2d_bytes_per_forward / 2**30:.3f} GiB"
    )
    print("(every row is relative to that run; H2D is a volume, not a time)")
    print(
        f"{'G/O/P':>10}  {'units':>5}  {'resident vs run':>16}  "
        f"{'H2D/forward':>13}  {'alt':>4}"
    )
    for item in candidates[: args.top]:
        alt = (
            str(item.equivalent_schedules - 1) if item.equivalent_schedules > 1 else ""
        )
        print(
            f"{str(item.schedule):>10}  {len(item.selected):>5}  "
            f"{_gib(item.resident_delta_bytes):>16}  "
            f"{item.h2d_bytes_per_forward / 2**30:>9.3f} GiB  {alt:>4}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
