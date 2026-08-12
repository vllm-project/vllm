# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Faithful replica of the ci-infra v2 generator's per-PR step selection.

Mirrors the generator's _step_should_run, source-dependency match, run-all, and
docs-only predicates, plus its AMD always-run constants. Assumes a static PR
context: branch != main, no labels, and none of RUN_ALL/NIGHTLY/NOAUTO/
only-step-keys set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

from ..curated import AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES
from ..jobs.model import PipelineConfig, Step
from ..policy import deps_match, docs_only, is_catch_all_dep

__all__ = ["is_catch_all_dep"]


def should_run_all(config: PipelineConfig, paths: list[str]) -> bool:
    for path in paths:
        for pattern in config.run_all_patterns:
            if re.match(pattern, path) and not any(
                re.match(e, path) for e in config.run_all_exclude_patterns
            ):
                return True
    return False


def step_should_run(step: Step, paths: list[str], run_all: bool) -> bool:
    from ..hardware import family_of_device

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
