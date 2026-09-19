# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regenerate tests/watermarking/watermarking_goldens.json; see README.md."""

import argparse
import json
from pathlib import Path

from tests.watermarking.golden_candidates import (
    GOLDEN_SCHEMA_VERSION,
    REGENERATE_COMMAND,
    WATERMARKING_CANDIDATES,
    GoldenCandidatePayload,
    GoldenFormatError,
    GoldenGuardError,
    GoldenPayload,
    compare_entries,
    compare_golden,
    environment_block,
    frozen_entry,
    golden_payload,
    read_goldens,
    validate_golden_guards,
)

GOLDENS_PATH = Path(__file__).with_name("watermarking_goldens.json")

CANDIDATES_BY_ID = {candidate.id: candidate for candidate in WATERMARKING_CANDIDATES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=REGENERATE_COMMAND,
        description=(
            "Regenerate the watermarking goldens. With no flags every candidate "
            "is regenerated and the file is overwritten; the ids whose entry "
            "changed are printed. Never hand-edit the file."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Regenerate in memory and report every field that differs from the "
            "stored goldens, then exit 1. Score and p_value are compared with "
            "the same tolerance the test uses and the environment block is "
            "ignored. Nothing is written."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        metavar="ID",
        default=[],
        choices=sorted(CANDIDATES_BY_ID),
        help=(
            "Regenerate only this candidate, keeping the stored entry of every "
            "other one. Repeatable. If the stored file cannot be read or has a "
            "different schema version, every candidate is regenerated instead "
            "and the reason is printed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check and args.candidate:
        parser.error("--check and --candidate cannot be combined")
    if args.check:
        return _check()
    if args.candidate:
        return _write_candidates(args.candidate)
    return _write(golden_payload())


def _check() -> int:
    try:
        stored = read_goldens(GOLDENS_PATH)
        validate_golden_guards(stored)
    except (OSError, GoldenFormatError, GoldenGuardError) as error:
        print(error)
        return 1

    differences: list[str] = []
    for candidate in WATERMARKING_CANDIDATES:
        golden = stored.get(candidate.id)
        if golden is None:
            differences.append(f"{candidate.id}: missing from {GOLDENS_PATH}")
            continue
        differences.extend(
            f"{candidate.id}: {line}" for line in compare_golden(candidate, golden)
        )
    for extra in sorted(set(stored) - set(CANDIDATES_BY_ID)):
        differences.append(f"{extra}: stored but no longer a candidate")

    if differences:
        print("\n".join(differences))
        return 1
    print(f"{len(stored)} candidates reproduce {GOLDENS_PATH}")
    return 0


def _write_candidates(ids: list[str]) -> int:
    try:
        stored = read_goldens(GOLDENS_PATH)
    except (OSError, GoldenFormatError) as error:
        print(f"regenerating every candidate: {error}")
        return _write(golden_payload())

    candidates: dict[str, GoldenCandidatePayload] = {
        candidate.id: stored[candidate.id]
        for candidate in WATERMARKING_CANDIDATES
        if candidate.id in stored
    }
    missing = [
        candidate.id
        for candidate in WATERMARKING_CANDIDATES
        if candidate.id not in stored and candidate.id not in ids
    ]
    if missing:
        print(f"also regenerating candidates absent from the file: {missing}")
    for candidate_id in ids + missing:
        candidates[candidate_id] = frozen_entry(CANDIDATES_BY_ID[candidate_id])
    return _write(
        {
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "environment": environment_block(),
            "candidates": candidates,
        },
        previous=stored,
    )


def _write(
    payload: GoldenPayload,
    previous: dict[str, GoldenCandidatePayload] | None = None,
) -> int:
    validate_golden_guards(payload["candidates"])
    if previous is None:
        try:
            previous = read_goldens(GOLDENS_PATH)
        except (OSError, GoldenFormatError):
            previous = None
    GOLDENS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(payload['candidates'])} candidates to {GOLDENS_PATH}")
    print(f"changed: {_changed_ids(payload['candidates'], previous)}")
    return 0


def _changed_ids(
    candidates: dict[str, GoldenCandidatePayload],
    previous: dict[str, GoldenCandidatePayload] | None,
) -> str:
    if previous is None:
        return "all (the previous file was unreadable or a different schema)"
    changed = [
        candidate_id
        for candidate_id, entry in candidates.items()
        if candidate_id not in previous
        or compare_entries(entry, previous[candidate_id])
    ]
    dropped = sorted(set(previous) - set(candidates))
    return ", ".join(changed + [f"{name} (dropped)" for name in dropped]) or "none"


if __name__ == "__main__":
    raise SystemExit(main())
