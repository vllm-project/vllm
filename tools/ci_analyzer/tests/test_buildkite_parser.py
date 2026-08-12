# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buildkite parser vs. the real checkout.

Every count asserted against the live repo is re-derived here independently
(regex over the yaml sources), never hardcoded, so these tests survive repo
drift and fail only on real parser/repo skew.
"""

from pathlib import Path

import pytest
import regex as re
from ci_analyzer.curated import AMD_ALWAYS_RUN_STEP_KEYS
from ci_analyzer.jobs.buildkite import load_pipeline_configs, load_steps
from ci_analyzer.jobs.model import DEFAULT_WORKING_DIR, LoadReport

LABEL_RE = re.compile(r"^\s*- label:", re.MULTILINE)
MIRROR_RE = re.compile(r"^\s+mirror:\s*$", re.MULTILINE)


def job_files(repo: Path, config) -> list[Path]:
    files = []
    for job_dir in config.job_dirs:
        files.extend(sorted((repo / job_dir).rglob("*.yaml")))
    return files


def _uncommented(path: Path) -> str:
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def loaded(repo):
    configs = load_pipeline_configs(repo)
    report = LoadReport()
    steps = {c.name: load_steps(repo, c, report) for c in configs}
    return configs, steps, report


def test_pipeline_configs_found(loaded, repo):
    configs, _, _ = loaded
    names = {c.name for c in configs}
    assert "vllm_ci" in names
    main = next(c for c in configs if c.name == "vllm_ci")
    assert ".buildkite/test_areas" in main.job_dirs
    assert main.run_all_patterns and main.run_all_exclude_patterns


def test_step_count_matches_label_oracle(loaded, repo):
    configs, steps, _ = loaded
    for config in configs:
        oracle = sum(
            len(LABEL_RE.findall(_uncommented(f))) for f in job_files(repo, config)
        )
        # Both sides walk config.job_dirs, so an emptied job_dirs collapses them
        # together and 0 == 0 passes. vllm_rocm_ci shares its only job_dir with
        # vllm_ci, so no orphan-yaml check would notice either.
        assert oracle, f"{config.name} parsed no job labels at all"
        parsed = [s for s in steps[config.name] if s.mirror_hw is None]
        assert len(parsed) == oracle, config.name


def test_mirror_expansion_matches_mirror_oracle(loaded, repo):
    configs, steps, _ = loaded
    for config in configs:
        oracle = sum(
            len(MIRROR_RE.findall(_uncommented(f))) for f in job_files(repo, config)
        )
        mirrors = [s for s in steps[config.name] if s.mirror_hw is not None]
        assert len(mirrors) == oracle, config.name


def test_lora_step_shapes(loaded):
    _, steps, _ = loaded
    lora = [
        s for s in steps["vllm_ci"] if s.source_file.endswith("test_areas/lora.yaml")
    ]
    parent = next(s for s in lora if s.key == "lora")
    assert parent.group == "LoRA"
    assert parent.parallelism == 4
    assert parent.working_dir == DEFAULT_WORKING_DIR
    assert parent.depends_on == ["image-build"]  # group-level default
    assert "vllm/lora" in parent.source_file_dependencies

    # Generator quirk: group-active files append the defining yaml to deps.
    assert ".buildkite/test_areas/lora.yaml" in parent.source_file_dependencies

    amd = next(s for s in lora if s.mirror_hw == "amd")
    assert amd.mirror_of == parent.step_id
    assert amd.device == "mi300_1"
    assert "vllm/platforms/rocm.py" in amd.source_file_dependencies
    # Mirror deps are a UNION with the parent's, not an override.
    assert ".buildkite/test_areas/lora.yaml" in amd.source_file_dependencies
    assert amd.depends_on == ["image-build-amd"]
    assert amd.commands == parent.commands  # not overridden in the mirror


def test_auto_run_semantics(loaded):
    """always_runs is the generator's key shortcut (image-build*/AMD list),
    NOT "has no deps": a no-deps step without the shortcut never auto-runs."""
    _, steps, _ = loaded
    by_key = {s.key: s for s in steps["vllm_ci"] if s.key}
    assert by_key["image-build"].always_runs
    assert by_key["image-build"].source_file_dependencies is None
    # Prefix shortcut, not substring: this key contains "image-build" but
    # doesn't start with it, so no-deps alone never grants always_runs.
    assert by_key["cpu-arm64-image-build"].source_file_dependencies is None
    assert not by_key["cpu-arm64-image-build"].always_runs
    assert not by_key["arm64-image-build"].always_runs
    # Membership asserted: these steps declare no deps, so always_runs
    # is the only thing that ever selects them, and this is the
    # only stale-entry guard on the curated set. Under `if key in rocm` a
    # rename upstream skipped the check and killed the entry in one move.
    rocm = {s.key: s for s in steps["vllm_rocm_ci"] if s.key}
    for key in AMD_ALWAYS_RUN_STEP_KEYS:
        assert key in rocm, f"curated AMD always-run key is gone: {key}"
        assert rocm[key].always_runs, key
    no_deps_no_shortcut = [
        s
        for s in steps["vllm_ci"]
        if s.source_file_dependencies is None
        and not s.always_runs
        and not s.optional
        and s.mirror_hw is None
    ]
    # Must be non-empty: an empty set would mean the parser wrongly granted
    # always_runs to no-deps steps, running them on every PR.
    assert no_deps_no_shortcut


def test_yaml_anchor_resolution(loaded):
    _, steps, _ = loaded
    cpu = [
        s
        for s in steps["vllm_ci"]
        if s.source_file.endswith("hardware_tests/cpu.yaml")
        and s.source_file_dependencies is not None
    ]
    dep_lists = [tuple(s.source_file_dependencies) for s in cpu]
    # &cpu_distributed_deps / *cpu_distributed_deps: two steps share one list
    assert len(dep_lists) != len(set(dep_lists)), (
        "expected at least two cpu.yaml steps sharing an anchored dep list"
    )


def test_legacy_field_aliases(loaded):
    _, steps, _ = loaded
    misc = next(
        s for s in steps["vllm_ci"] if s.key == "acceptance-length-test-large-models"
    )
    # This step carries BOTH `device: h200_35gb` and a stale legacy `gpu: h100`;
    # the v2 field must win, the legacy one is only a fallback.
    assert misc.device == "h200_35gb"
    assert misc.num_devices == 1  # from legacy `num_gpus:` (no num_devices)


def test_no_unknown_step_fields_at_head(loaded):
    *_, report = loaded
    assert report.unknown_fields == {}, (
        "new/unknown step fields appeared; teach the model or update "
        f"KNOWN_STEP_FIELDS: {report.unknown_fields}"
    )


def test_no_duplicate_step_ids_at_head(loaded):
    """Colliding step_ids silently overwrite each other in selection's target
    map; a new one means dedup the yaml or fix the id derivation.

    detect_duplicate_ids is the only writer of duplicate_ids and load_steps
    does not call it, so asserting on the fixture's report alone passed for
    any yaml at all."""
    from ci_analyzer.select import detect_duplicate_ids

    _, steps, _ = loaded
    report = LoadReport()
    for pipeline_steps in steps.values():
        detect_duplicate_ids(pipeline_steps, report)
    assert report.duplicate_ids == [], report.duplicate_ids


def test_autorun_on_main_steps_exist(loaded, repo):
    _, steps, _ = loaded
    oracle = sum(
        _uncommented(f).count("autorun_on_main")
        for c in load_pipeline_configs(repo)
        for f in job_files(repo, c)
    )
    flagged = [
        s
        for pipeline in steps.values()
        for s in pipeline
        if s.autorun_on_main and s.mirror_hw is None
    ]
    assert len(flagged) == oracle


def test_no_orphan_step_shaped_yaml(repo, loaded):
    """A step-shaped yaml outside every known consumer means a new pipeline
    convention landed: extend load_pipeline_configs or classify the file.
    Allowlisted files are consumed by Buildkite directly, not the PR generator."""
    import yaml as _yaml
    from ci_analyzer.curated import INERT_CI_PREFIXES, LEGACY_CI_FILES

    configs, _steps, _report = loaded
    known_dirs = tuple(
        f".buildkite/{d.rstrip('/')}/"
        if not d.startswith(".buildkite")
        else d.rstrip("/") + "/"
        for c in configs
        for d in c.job_dirs
    )
    allow = {".buildkite/release-pipeline.yaml"}
    orphans = []
    for path in sorted((repo / ".buildkite").rglob("*.yaml")):
        rel = path.relative_to(repo).as_posix()
        if (
            rel.startswith(known_dirs)
            or rel in LEGACY_CI_FILES
            or rel.startswith(INERT_CI_PREFIXES)
            or rel in allow
            or any(rel == c.config_file for c in configs)
        ):
            continue
        try:
            data = _yaml.safe_load(path.read_text())
        except _yaml.YAMLError:
            continue
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            continue
        if any(
            isinstance(s, dict) and "label" in s and ("commands" in s or "command" in s)
            for s in steps
        ):
            orphans.append(rel)
    assert not orphans, orphans
