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

from ...handwritten import (
    AMD_ALWAYS_RUN_STEP_KEYS,
    DEFAULT_WORKING_DIR,
    IMAGE_BUILD_KEY_PREFIX,
)

_KEY_DELETE = str.maketrans("", "", "()%")
_KEY_DASH = str.maketrans(",+:./", "-----")


def derive_step_key(label: str) -> str:
    """The key ci-infra mints when the yaml omits one.

    Transcribed from `_generate_step_key` in ci-infra's
    `buildkite/pipeline_generator/buildkite_step.py`, and executed against the
    real function by the snapshot suite, so drift shows up as a red test.
    """
    return label.replace(" ", "-").lower().translate(_KEY_DELETE).translate(_KEY_DASH)


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
    # The mirror's own yaml `label`, kept beside the derived one rather than
    # replacing it: `step_id` falls back to the label, so overwriting it would
    # rekey a keyless mirror against every stored table.
    mirror_label: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def identity(self) -> tuple:
        """What the step IS, for comparing one checkout's steps against
        another's.

        `step_id` cannot do it: it falls back to the label, so a reword makes
        one step look like two and a comparison by id deletes the survivor's
        twin. Every field here survives a reword: where the step is defined,
        what it runs, which hardware, which mirror.

        `device` is in because without it the same commands from the same file
        on a100/h100/b200 collide, 11 steps at HEAD. All 11 are explicitly
        keyed today so nothing breaks, but the next collision may not be.
        """
        return (
            self.pipeline,
            self.source_file,
            tuple(self.commands),
            self.mirror_hw,
            self.device,
        )

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
