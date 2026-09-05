# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which test files a live CI step actually invokes.

A directory target counts as invoking every test file under it, and narrowing
flags are ignored: a file skipped by one shard still belongs to the step.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ...handwritten import DEFAULT_WORKING_DIR, LEGACY_CI_FILES
from .scripts import scan_script
from .step import Step
from .targets import StepTargets, map_step


def invoked_files(catalog: list[str], targets: list[StepTargets]) -> set[str]:
    invoked: set[str] = set()
    dir_prefixes: set[str] = set()
    for st in targets:
        for t in st.targets:
            if t.path.endswith(".py"):
                invoked.add(t.path)
            else:
                dir_prefixes.add(t.path.rstrip("/") + "/")
    for f in catalog:
        if any(f.startswith(p) for p in dir_prefixes):
            invoked.add(f)
    return invoked


def legacy_amd_invoked(repo: Path, catalog: list[str]) -> set[str]:
    """Targets of the legacy test-amd.yaml, read with the same parser."""
    legacy = LEGACY_CI_FILES[0]
    path = repo / legacy
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text())
    steps = []
    for raw in data.get("steps", []):
        if not isinstance(raw, dict) or not (raw.get("commands") or raw.get("command")):
            continue
        commands = raw.get("commands") or raw.get("command")
        if isinstance(commands, str):
            commands = [commands]
        steps.append(
            Step(
                pipeline="legacy_amd",
                source_file=legacy,
                label=str(raw.get("label", "")),
                key=None,
                group=None,
                commands=list(commands),
                source_file_dependencies=raw.get("source_file_dependencies"),
                working_dir=raw.get("working_dir") or DEFAULT_WORKING_DIR,
            )
        )
    targets = [map_step(repo, s, script_scanner=scan_script) for s in steps]
    return invoked_files(catalog, targets)
