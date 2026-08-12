# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advisory uninvoked-test report: test files no live CI step invokes.

Directory targets count as invoking every test file under them, and narrowing
(-m/-k/--ignore) is deliberately NOT applied: the report must only name
confident orphans. Files invoked only by the orphaned legacy test-amd.yaml are
reported as their own "legacy-only" class, not as tethered.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..jobs.buildkite import load_pipeline_configs, load_steps
from ..jobs.invoked import invoked_files, legacy_amd_invoked
from ..jobs.model import LoadReport, Step
from ..jobs.scripts import scan_script
from ..jobs.testmap import StepTargets, map_step
from ..repo import test_file_catalog


@dataclass
class UninvokedReport:
    orphans: list[str] = field(default_factory=list)
    legacy_only: list[str] = field(default_factory=list)
    invoked_count: int = 0
    catalog_count: int = 0


def collect_step_targets(repo: Path, steps: list[Step]) -> list[StepTargets]:
    return [map_step(repo, step, script_scanner=scan_script) for step in steps]


def uninvoked_report(
    repo: Path, all_steps: list[Step]
) -> tuple[UninvokedReport, list[StepTargets]]:
    catalog = test_file_catalog(repo)
    targets = collect_step_targets(repo, all_steps)
    invoked = invoked_files(catalog, targets)
    legacy = legacy_amd_invoked(repo, catalog)
    report = UninvokedReport(invoked_count=len(invoked), catalog_count=len(catalog))
    for f in catalog:
        if f in invoked:
            continue
        if f in legacy:
            report.legacy_only.append(f)
        else:
            report.orphans.append(f)
    return report, targets


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)


def run(args) -> int:
    repo = args.repo.resolve()
    load_report = LoadReport()
    steps = []
    for config in load_pipeline_configs(repo):
        steps.extend(load_steps(repo, config, load_report))
    ur, _ = uninvoked_report(repo, steps)
    print(f"catalog: {ur.catalog_count} test files; invoked: {ur.invoked_count}")
    print(f"orphans ({len(ur.orphans)}): no live CI step invokes these")
    for o in ur.orphans:
        print(f"  {o}")
    print(
        f"legacy-only ({len(ur.legacy_only)}): reached only via the "
        "orphaned test-amd.yaml"
    )
    for o in ur.legacy_only:
        print(f"  {o}")
    # Detection floor: an empty catalog or zero invoked files means the
    # derivation broke, and "no orphans" then reads as a clean bill of health.
    if not ur.catalog_count or not ur.invoked_count:
        print(
            f"  COLLAPSE: catalog={ur.catalog_count} invoked={ur.invoked_count}; "
            "the orphan derivation ran against nothing"
        )
        return 1
    return 0
