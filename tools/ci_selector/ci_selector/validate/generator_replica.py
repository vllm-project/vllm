# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What today's CI rules would pick, re-implemented here.

The generator lives in a repo a vLLM checkout cannot read, so this mirrors its
decision: the run predicate, the dependency match, run-all, docs-only. Assumes a
plain PR: not on main, no labels, no RUN_ALL / NIGHTLY / NOAUTO.

Borrowing the parsed pipeline and two predicates from `codemap.claim` is
deliberate. The pipeline is the shared INPUT, and the predicates are copies of
the same upstream functions our selector copies, so a second copy here could
only diverge through a maintenance slip, and the comparison would then report a
difference that is not a selection difference. That kind of drift is caught by
neither side, but by what CI actually ran.

Never borrow our selection logic. Two tests in `tests/test_crosscheck_units.py`
fail if anything from `classify.py` or `selection.py` reaches this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

from ..codemap.claim import deps_match, docs_only
from ..codemap.pipeline.step import PipelineConfig, Step
from ..handwritten import AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES


def should_run_all(config: PipelineConfig, paths: list[str]) -> bool:
    for path in paths:
        for pattern in config.run_all_patterns:
            if re.match(pattern, path) and not any(
                re.match(e, path) for e in config.run_all_exclude_patterns
            ):
                return True
    return False


def step_should_run(step: Step, paths: list[str], run_all: bool) -> bool:
    from ..codemap.hardware import family_of_device

    if step.always_runs:
        return True
    if step.optional:
        return False
    if (
        family_of_device(step.device) == "amd"
        and not step.dind
        and deps_match(list(AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES), paths)
    ):
        return True
    if run_all:
        return True
    return deps_match(step.source_file_dependencies, paths)


@dataclass
class TodaySelection:
    selected: dict[str, set[str]] = field(default_factory=dict)  # pipeline->ids
    run_all: dict[str, bool] = field(default_factory=dict)
    docs_only: bool = False


def today_select(
    pipelines: list[tuple[PipelineConfig, list[Step]]], paths: list[str]
) -> TodaySelection:
    sel = TodaySelection()
    if docs_only(paths):
        sel.docs_only = True
        return sel
    for config, steps in pipelines:
        run_all = should_run_all(config, paths)
        sel.run_all[config.name] = run_all
        sel.selected[config.name] = {
            s.step_id for s in steps if step_should_run(s, paths, run_all)
        }
    return sel
