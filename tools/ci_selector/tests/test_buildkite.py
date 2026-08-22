# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buildkite parser vs. the real checkout.

Every count asserted against the live checkout is re-derived here independently
(regex over the yaml sources), never hardcoded, so these tests survive checkout
drift and fail only on real parser/checkout skew.
"""

from pathlib import Path

import pytest
import regex as re
from ci_selector.codemap.pipeline.buildkite import load_pipeline_configs, load_steps
from ci_selector.codemap.pipeline.step import DEFAULT_WORKING_DIR, LoadReport
from ci_selector.handwritten import AMD_ALWAYS_RUN_STEP_KEYS
from helpers import HW, drift_message

LABEL_RE = re.compile(r"^\s*- label:", re.MULTILINE)
MIRROR_RE = re.compile(r"^\s+mirror:\s*$", re.MULTILINE)


def job_files(vllm_repo: Path, config) -> list[Path]:
    files = []
    for job_dir in config.job_dirs:
        files.extend(sorted((vllm_repo / job_dir).rglob("*.yaml")))
    return files


def _uncommented(path: Path) -> str:
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


@pytest.fixture(scope="module")
def loaded(vllm_repo):
    configs = load_pipeline_configs(vllm_repo)
    report = LoadReport()
    steps = {c.name: load_steps(vllm_repo, c, report) for c in configs}
    return configs, steps, report


def test_pipeline_configs_found(loaded, vllm_repo):
    configs, _, _ = loaded
    names = {c.name for c in configs}
    assert "vllm_ci" in names
    main = next(c for c in configs if c.name == "vllm_ci")
    assert ".buildkite/test_areas" in main.job_dirs
    assert main.run_all_patterns and main.run_all_exclude_patterns


def test_step_count_matches_label_oracle(loaded, vllm_repo):
    configs, steps, _ = loaded
    for config in configs:
        oracle = sum(
            len(LABEL_RE.findall(_uncommented(f))) for f in job_files(vllm_repo, config)
        )
        # Both sides walk config.job_dirs, so an emptied job_dirs collapses them
        # together and 0 == 0 passes. vllm_rocm_ci shares its only job_dir with
        # vllm_ci, so no orphan-yaml check would notice either.
        assert oracle, f"{config.name} parsed no job labels at all"
        parsed = [s for s in steps[config.name] if s.mirror_hw is None]
        assert len(parsed) == oracle, config.name


def test_mirror_expansion_matches_mirror_oracle(loaded, vllm_repo):
    configs, steps, _ = loaded
    for config in configs:
        oracle = sum(
            len(MIRROR_RE.findall(_uncommented(f)))
            for f in job_files(vllm_repo, config)
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


SWEEPS = Path(__file__).resolve().parents[3] / "zzzzz/auto-test/covspike/sweeps"


@pytest.mark.skipif(not SWEEPS.is_dir(), reason="raw sweeps not present")
def test_derived_step_keys_match_what_buildkite_published(loaded):
    """The key rule was fitted against real builds, so pin it against them.

    Most steps carry an author-written key. When the yaml omits one the
    generator derives it from the label, and we have to reproduce that or the
    step cannot be named -- which under an exhaustive-whitelist emission means
    it silently does not run.
    """
    import json

    from ci_selector.codemap.pipeline.step import derive_step_key

    published: dict[str, str] = {}
    for build in sorted(SWEEPS.glob("vllm-ci-*")):
        index = build / "index.json"
        if not index.is_file():
            continue
        for job in json.loads(index.read_text())["jobs"]:
            if job.get("step_key") and job.get("label"):
                published.setdefault(job["label"], job["step_key"])

    _, steps, _ = loaded
    checked = []
    for step in (s for group in steps.values() for s in group):
        if step.key or step.mirror_hw:
            continue
        # Runtime labels carry the shard number the yaml writes as %N.
        stem = re.sub(r"\s+\d+$", "", step.label)
        for label, key in published.items():
            if re.sub(r"\s+\d+$", "", label) in (stem, step.label):
                checked.append((step.label, key, derive_step_key(step.label)))
                break

    assert len(checked) > 30, f"oracle collapsed: only {len(checked)} pairs matched"
    wrong = [(lbl, real, got) for lbl, real, got in checked if real != got]
    assert not wrong, "\n".join(
        f"{lbl!r}: published {real!r}, derived {got!r}" for lbl, real, got in wrong
    )


def test_a_mirror_publishes_the_hardware_as_a_prefix(loaded):
    """Our step id suffixes the hardware; real CI prefixes it. Recordings and
    status contexts carry the generator's spelling, so the two must not be
    confused for each other."""
    _, steps, _ = loaded
    amd = next(
        s for s in steps["vllm_ci"] if s.mirror_hw == "amd" and s.key == "lora-amd"
    )
    assert amd.buildkite_key == "amd-lora"
    assert amd.buildkite_key != amd.key

    plain = next(s for s in steps["vllm_ci"] if s.key == "lora")
    assert plain.buildkite_key == "lora"


@pytest.mark.drift
def test_auto_run_semantics(loaded):
    """always_runs is the generator's key shortcut (image-build*/AMD list),
    NOT "has no deps": a no-deps step without the shortcut never auto-runs."""
    _, steps, _ = loaded
    by_key = {s.key: s for s in steps["vllm_ci"] if s.key}
    specimens = ("image-build", "cpu-arm64-image-build", "arm64-image-build")
    for key in specimens:
        assert key in by_key, drift_message(
            f"{key} is gone from vllm_ci, and this test reads always-run "
            "semantics off it.",
            "always_runs decides which steps run on every PR regardless of the "
            "diff. Without a specimen the rule is unverified, and it is the "
            "rule that keeps the image builds everything else depends on.",
            "the step was renamed upstream: pick the new key as the specimen",
            f"the prefix itself moved: update IMAGE_BUILD_KEY_PREFIX in {HW}",
        )
    prefix = drift_message(
        "always_runs no longer reads as a key-prefix shortcut.",
        "The generator grants it by key prefix, not by 'declares no deps'. If "
        "we grant it more widely we run steps CI does not; more narrowly and we "
        "skip the image builds the rest of the pipeline waits on.",
        "check Step.always_runs against ci-infra's _step_should_run",
        f"the prefix changed: update IMAGE_BUILD_KEY_PREFIX in {HW}",
    )
    assert by_key["image-build"].always_runs, prefix
    assert by_key["image-build"].source_file_dependencies is None, prefix
    # Prefix shortcut, not substring: this key contains "image-build" but
    # doesn't start with it, so no-deps alone never grants always_runs.
    assert by_key["cpu-arm64-image-build"].source_file_dependencies is None, prefix
    assert not by_key["cpu-arm64-image-build"].always_runs, prefix
    assert not by_key["arm64-image-build"].always_runs, prefix
    # Membership asserted: these steps declare no deps, so always_runs
    # is the only thing that ever selects them, and this is the
    # only stale-entry guard on the curated set. Under `if key in rocm` a
    # rename upstream skipped the check and killed the entry in one move.
    rocm = {s.key: s for s in steps["vllm_rocm_ci"] if s.key}
    for key in AMD_ALWAYS_RUN_STEP_KEYS:
        assert key in rocm, drift_message(
            f"AMD_ALWAYS_RUN_STEP_KEYS names a step that no longer exists: {key}",
            "These steps declare no dependencies, so always_runs is the only "
            "thing that ever selects them. A renamed key means we stop naming "
            "the step and CI stops running it.",
            "the step was renamed upstream in ci-infra's amd.py: update "
            f"AMD_ALWAYS_RUN_STEP_KEYS in {HW}",
        )
        assert rocm[key].always_runs, drift_message(
            f"{key} is in AMD_ALWAYS_RUN_STEP_KEYS but does not read as always-run.",
            "It would only be selected if some rule happened to pick it, and "
            "these steps have no dependencies for a rule to match on.",
            "check Step.always_runs against ci-infra's _step_should_run",
        )
    no_deps_no_shortcut = [
        s
        for s in steps["vllm_ci"]
        if s.source_file_dependencies is None
        and not s.always_runs
        and not s.optional
        and s.mirror_hw is None
    ]
    assert no_deps_no_shortcut, drift_message(
        "Every no-deps step now reads as always-run, so the distinction this "
        "test exists to check has collapsed.",
        "Granting always_runs to any step without dependencies would run a "
        "large slice of CI on every PR.",
        "the parser started inferring always_runs from missing deps: fix "
        "Step.always_runs in ci_selector/codemap/pipeline/step.py",
    )


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


@pytest.mark.drift
def test_no_unknown_step_fields_at_head(loaded):
    *_, report = loaded
    assert report.unknown_fields == {}, drift_message(
        "Step fields we do not model appeared in the job yaml:\n"
        + "\n".join(
            f"    {name}  ({len(ids)} steps, e.g. {ids[0]})"
            for name, ids in sorted(report.unknown_fields.items())
        ),
        "Preflight force-selects every step carrying one, because an unknown "
        "field may change what the step runs. So this costs CI time until it "
        "is modelled, and it hides whatever the field actually does.",
        f"a plain step field: add it to KNOWN_STEP_FIELDS in {HW}",
        f"a key under a mirror's `amd:` block: add it to MIRROR_OVERRIDABLE in {HW}",
        "it changes how the step runs: also teach Step in "
        "ci_selector/codemap/pipeline/step.py to read it",
    )


@pytest.mark.drift
def test_no_duplicate_step_ids_at_head(loaded):
    """Colliding step_ids silently overwrite each other in selection's target
    map; a new one means dedup the yaml or fix the id derivation.

    detect_duplicate_ids is the only writer of duplicate_ids and load_steps
    does not call it, so asserting on the fixture's report alone passed for
    any yaml at all."""
    from ci_selector.codemap.state import detect_duplicate_ids

    _, steps, _ = loaded
    report = LoadReport()
    for pipeline_steps in steps.values():
        detect_duplicate_ids(pipeline_steps, report)
    assert report.duplicate_ids == [], drift_message(
        f"Two steps resolve to the same id: {report.duplicate_ids}",
        "Step ids key the target map, so one silently overwrites the other and "
        "whatever the loser tested stops being selectable.",
        "the yaml really has two steps with one key: give one a distinct key",
        "they differ but our id derivation collapses them: fix derive_step_key "
        "in ci_selector/codemap/pipeline/step.py",
    )


def test_autorun_on_main_steps_exist(loaded, vllm_repo):
    _, steps, _ = loaded
    oracle = sum(
        _uncommented(f).count("autorun_on_main")
        for c in load_pipeline_configs(vllm_repo)
        for f in job_files(vllm_repo, c)
    )
    flagged = [
        s
        for pipeline in steps.values()
        for s in pipeline
        if s.autorun_on_main and s.mirror_hw is None
    ]
    assert len(flagged) == oracle


@pytest.mark.drift
def test_no_orphan_step_shaped_yaml(vllm_repo, loaded):
    """A step-shaped yaml outside every known consumer means a new pipeline
    convention landed: extend load_pipeline_configs or classify the file.
    Allowlisted files are consumed by Buildkite directly, not the PR generator."""
    import yaml as _yaml
    from ci_selector.handwritten import INERT_CI_PREFIXES, LEGACY_CI_FILES

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
    for path in sorted((vllm_repo / ".buildkite").rglob("*.yaml")):
        rel = path.relative_to(vllm_repo).as_posix()
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
    assert not orphans, drift_message(
        f"These look like Buildkite step definitions but no pipeline config "
        f"consumes them: {orphans}",
        "Steps we never load cannot be selected, so anything they test is "
        "invisible to the selector.",
        "a new pipeline landed: add its config so load_pipeline_configs finds it",
        f"nothing runs it: add it to LEGACY_CI_FILES or INERT_CI_PREFIXES in {HW}",
    )
