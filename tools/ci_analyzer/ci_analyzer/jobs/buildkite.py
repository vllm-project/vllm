# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse .buildkite ci_config*.yaml pipelines into Step records.

Only the v2 generator surface is parsed (ci_config.yaml job_dirs). The legacy
test-amd.yaml is in no job_dirs (external Jinja pipeline) and is deliberately
excluded; test-pipeline.yaml is a deprecated stub.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..curated import KNOWN_STEP_FIELDS, MIRROR_OVERRIDABLE
from .model import (
    DEFAULT_WORKING_DIR,
    LoadReport,
    PipelineConfig,
    Step,
    repo_rel,
)


def load_pipeline_configs(repo: Path) -> list[PipelineConfig]:
    configs = []
    for path in sorted((repo / ".buildkite").glob("ci_config*.yaml")):
        data = yaml.safe_load(path.read_text())
        configs.append(
            PipelineConfig(
                name=data["name"],
                config_file=repo_rel(path, repo),
                job_dirs=data.get("job_dirs", []),
                run_all_patterns=data.get("run_all_patterns", []),
                run_all_exclude_patterns=data.get("run_all_exclude_patterns", []),
            )
        )
    if not configs:
        raise FileNotFoundError(f"no .buildkite/ci_config*.yaml under {repo}")
    return configs


def load_steps(
    repo: Path, config: PipelineConfig, report: LoadReport | None = None
) -> list[Step]:
    report = report if report is not None else LoadReport()
    steps: list[Step] = []
    seen_keys: set[str] = set()
    for job_dir in config.job_dirs:
        base = repo / job_dir
        if not base.is_dir():
            raise FileNotFoundError(f"{config.name}: job_dir {job_dir} missing")
        for path in sorted(base.rglob("*.yaml")):
            steps.extend(_parse_job_file(path, repo, config.name, report, seen_keys))
    return steps


def _parse_job_file(
    path: Path,
    repo: Path,
    pipeline: str,
    report: LoadReport,
    seen_keys: set[str],
) -> list[Step]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "steps" not in data:
        return []
    group = data.get("group")
    # Generator quirk (step.py read_steps_from_job_dir): depends_on inheritance,
    # the working-dir default, and the self-dep append fire only when file-level
    # depends_on is TRUTHY; `depends_on: []` (hardware_tests/cpu.yaml) gets none.
    group_depends_raw = data.get("depends_on")
    group_active = bool(group_depends_raw)
    group_depends = _as_list(group_depends_raw)
    steps: list[Step] = []
    for raw in data["steps"]:
        step = _parse_step(
            raw, path, repo, pipeline, group, group_depends, group_active, report
        )
        if step.key:
            if step.key in seen_keys:
                report.duplicate_keys.append(f"{step.pipeline}:{step.key}")
            seen_keys.add(step.key)
        steps.append(step)
        mirror = raw.get("mirror") or {}
        for hw, overrides in mirror.items():
            steps.append(_expand_mirror(step, hw, overrides or {}, report))
    return steps


def _parse_step(
    raw: dict,
    path: Path,
    repo: Path,
    pipeline: str,
    group: str | None,
    group_depends: list[str],
    group_active: bool,
    report: LoadReport,
) -> Step:
    commands = raw.get("commands")
    if commands is None and "command" in raw:  # legacy singular
        commands = raw["command"]
    if isinstance(commands, str):
        commands = [commands]
    depends = raw.get("depends_on", ...)
    source_file = repo_rel(path, repo)
    deps = (
        list(raw["source_file_dependencies"])
        if raw.get("source_file_dependencies") is not None
        else None
    )
    working_dir = raw.get("working_dir")
    if group_active:
        deps = (deps or []) + [source_file]  # generator's self-dep append
        working_dir = working_dir or DEFAULT_WORKING_DIR
    step = Step(
        pipeline=pipeline,
        source_file=source_file,
        label=str(raw.get("label", "")),
        key=raw.get("key"),
        group=group,
        commands=list(commands or []),
        source_file_dependencies=deps,
        device=raw.get("device") or raw.get("gpu"),  # gpu = legacy alias
        num_devices=raw.get("num_devices") or raw.get("num_gpus"),
        num_nodes=raw.get("num_nodes"),
        working_dir=working_dir or "",
        timeout_in_minutes=raw.get("timeout_in_minutes"),
        optional=bool(raw.get("optional", False)),
        soft_fail=bool(raw.get("soft_fail", False)),
        autorun_on_main=bool(raw.get("autorun_on_main", False)),
        no_plugin=bool(raw.get("no_plugin", False)),
        dind=bool(raw.get("dind", True)),
        parallelism=raw.get("parallelism"),
        # In a group-active file the generator overwrites any FALSY own
        # depends_on ([]/~/absent) with the group's.
        depends_on=(
            group_depends
            if (depends is ... or (group_active and not depends))
            else _as_list(depends)
        ),
        env=dict(raw.get("env") or {}),
        extra={k: v for k, v in raw.items() if k not in KNOWN_STEP_FIELDS},
    )
    if step.extra:
        report.record_unknown(step.extra, step.step_id)
    return step


def _expand_mirror(parent: Step, hw: str, overrides: dict, report: LoadReport) -> Step:
    """Mirrors ci-infra's AMD mirror-step derivation: deps are a UNION (mirror's
    first, then the parent's not already present), commands/working_dir override
    only when the mirror provides commands, optional/soft_fail default to the
    parent's."""
    deps = list(overrides.get("source_file_dependencies") or [])
    for dep in parent.source_file_dependencies or []:
        if dep not in deps:
            deps.append(dep)
    custom_commands = overrides.get("commands")
    env = dict(parent.env)
    env.update(overrides.get("env") or {})
    variant = Step(
        pipeline=parent.pipeline,
        source_file=parent.source_file,
        label=f"{parent.label} ({hw})",
        key=f"{parent.key}-{hw}" if parent.key else None,
        group=parent.group,
        commands=list(custom_commands or parent.commands),
        source_file_dependencies=deps or None,
        device=overrides.get("device", parent.device),
        num_devices=(
            overrides.get("num_devices")
            or overrides.get("num_gpus")
            or parent.num_devices
        ),
        num_nodes=overrides.get("num_nodes", parent.num_nodes),
        working_dir=(
            overrides.get("working_dir", parent.working_dir)
            if custom_commands
            else parent.working_dir
        ),
        timeout_in_minutes=overrides.get(
            "timeout_in_minutes", parent.timeout_in_minutes
        ),
        optional=bool(overrides.get("optional", parent.optional)),
        soft_fail=bool(overrides.get("soft_fail", parent.soft_fail)),
        autorun_on_main=parent.autorun_on_main,
        no_plugin=bool(overrides.get("no_plugin", parent.no_plugin)),
        dind=bool(overrides.get("dind", True)),
        parallelism=parent.parallelism,
        depends_on=_as_list(overrides.get("depends_on")) or ["image-build-amd"],
        env=env,
        mirror_hw=hw,
        mirror_of=parent.step_id,
    )
    unknown = {k: v for k, v in overrides.items() if k not in MIRROR_OVERRIDABLE}
    if unknown:
        report.record_unknown(unknown, variant.step_id)
    return variant


def _as_list(value) -> list[str]:
    if value is None or value is ...:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
