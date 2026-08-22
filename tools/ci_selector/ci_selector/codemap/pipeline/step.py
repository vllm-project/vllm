# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data model for Buildkite pipeline configs and steps.

Follows the ci-infra v2 generator: a step auto-runs on a PR through the
always-run key shortcut, run_all, or a source_file_dependencies match, and
`optional` blocks it even under run_all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import regex as re

from ...handwritten import (
    AMD_ALWAYS_RUN_STEP_KEYS,
    DEFAULT_WORKING_DIR,
    IMAGE_BUILD_KEY_PREFIX,
)

# Fitted against the keys real builds published, not copied from the generator,
# so there is no source to re-read when a label shape changes.
_KEY_STRIP = re.compile(r"[()%]")
_KEY_DASH = re.compile(r"[ ,+:./]+")
_KEY_RUNS = re.compile(r"-+")


def derive_step_key(label: str) -> str:
    """Reproduce the key ci-infra mints when the yaml omits one. The whole
    label is used, `%N` included: a sharded step keeps the literal `-n`, and
    stripping a trailing number would eat the real digit in other labels."""
    slug = _KEY_STRIP.sub("", label.lower())
    return _KEY_RUNS.sub("-", _KEY_DASH.sub("-", slug)).strip("-")


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
    # Never read, but parsed: without the field the yaml key lands in `extra`
    # and the unknown-field guard selects every step using it.
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
    def buildkite_key(self) -> str:
        """The key the generator publishes, which is not always our internal one.

        A mirror is published as `<hw>-<key>` while we mint `<key>-<hw>`, and a
        step whose yaml omits a key gets one derived from its label. Recordings
        and status contexts carry the generator's spelling, so anything matched
        against real CI needs this and not `key`.
        """
        if not self.key:
            return derive_step_key(self.label)
        if not self.mirror_hw:
            return self.key
        return f"{self.mirror_hw}-{self.key.removesuffix(f'-{self.mirror_hw}')}"

    @property
    def always_runs(self) -> bool:
        """The generator's key shortcut, checked BEFORE optional."""
        return bool(self.key) and (
            self.key.startswith(IMAGE_BUILD_KEY_PREFIX)
            or self.key in AMD_ALWAYS_RUN_STEP_KEYS
        )

    @property
    def manual_only(self) -> bool:
        return self.optional and not self.always_runs


@dataclass
class LoadReport:
    """Parser observations that are findings, not fatal errors."""

    duplicate_keys: list[str] = field(default_factory=list)
    # shared by two steps, whose target maps then collide
    duplicate_ids: list[str] = field(default_factory=list)
    unknown_fields: dict[str, list[str]] = field(
        default_factory=dict
    )  # field -> step ids

    def record_unknown(self, fields: dict, step_id: str) -> None:
        for name in fields:
            self.unknown_fields.setdefault(name, []).append(step_id)


def repo_rel(path: Path, repo: Path) -> str:
    return path.relative_to(repo).as_posix()
