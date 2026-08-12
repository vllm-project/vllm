# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data model for Buildkite pipeline configs and steps.

Semantics mirror the ci-infra v2 generator: a step auto-runs on a PR only via
the image-build/AMD always-run shortcut, run_all, or a source_file_dependencies
match. optional: true blocks a step even under run_all (manual unblock, nightly
only). autorun_on_main is a legacy-Jinja field the v2 generator drops; we parse
it only to report as dead config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The generator's working-dir default (step.py read_steps_from_job_dir). It
# applies ONLY to steps in a file with a truthy group-level depends_on;
# depends_on: [] disables it along with the yaml self-dep append.
from ..curated import (
    AMD_ALWAYS_RUN_STEP_KEYS,
    DEFAULT_WORKING_DIR,
)


@dataclass
class PipelineConfig:
    name: str
    config_file: str
    job_dirs: list[str]
    run_all_patterns: list[str]
    run_all_exclude_patterns: list[str]


@dataclass
class Step:
    pipeline: str
    source_file: str  # repo-relative path of the defining yaml
    label: str
    key: str | None
    group: str | None
    commands: list[str]
    source_file_dependencies: list[str] | None  # None = field absent
    device: str | None = None
    num_devices: int | None = None
    num_nodes: int | None = None
    working_dir: str = DEFAULT_WORKING_DIR
    timeout_in_minutes: int | None = None
    optional: bool = False
    soft_fail: bool = False
    autorun_on_main: bool = False
    no_plugin: bool = False
    dind: bool = True
    parallelism: int | None = None
    depends_on: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    mirror_hw: str | None = None  # set on expanded mirror variants, e.g. "amd"
    mirror_of: str | None = None  # parent step id for mirror variants
    extra: dict = field(default_factory=dict)

    @property
    def step_id(self) -> str:
        base = self.key or self.label
        return f"{self.pipeline}:{base}" + (
            f":{self.mirror_hw}" if self.mirror_hw else ""
        )

    @property
    def always_runs(self) -> bool:
        """The generator's key shortcut, checked BEFORE optional."""
        return bool(self.key) and (
            self.key.startswith("image-build") or self.key in AMD_ALWAYS_RUN_STEP_KEYS
        )

    @property
    def manual_only(self) -> bool:
        return self.optional and not self.always_runs


@dataclass
class LoadReport:
    """Parser observations that are findings, not fatal errors."""

    duplicate_keys: list[str] = field(default_factory=list)
    # step_ids shared by two steps (incl. keyless same-label): their target
    # maps collide, so selection cannot tell them apart
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_fields: dict[str, list[str]] = field(
        default_factory=dict
    )  # field -> step ids

    def record_unknown(self, fields: dict, step_id: str) -> None:
        for name in fields:
            self.unknown_fields.setdefault(name, []).append(step_id)


def repo_rel(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()
