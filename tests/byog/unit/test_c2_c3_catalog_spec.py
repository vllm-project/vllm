# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C2 auto-discovered catalog and C3 single-job spec resolver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools.byog.catalog import Catalog, CatalogError, instantiate_plugin
from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.catalog import WorkloadKind
from tools.byog.schemas.spec import RunSpec
from tools.byog.spec import (
    SpecResolutionError,
    load_spec_file,
    resolve_single_job,
)

VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


class DummyBenchPlugin(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint(point_id="p0", params=params)]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        return PointResult(metrics={"ok": 1})


def _make_catalog(tmp_path: Path) -> Catalog:
    wdir = tmp_path / "workloads"
    sdir = tmp_path / "suites"
    wdir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    (wdir / "bench_a.yaml").write_text(
        "schema_version: 1\n"
        "name: bench_a\n"
        "kind: benchmark\n"
        "execution_mode: server\n"
        "lifecycle: reset_cache\n"
        "platforms: [nvidia]\n"
        f"entrypoint: {__name__}:DummyBenchPlugin\n"
        "defaults:\n"
        "  num_requests: 4\n"
        "  concurrency: [1]\n"
        "allowed_params: [warmup_requests]\n",
        encoding="utf-8",
    )
    (wdir / "rocm_only.yml").write_text(
        "schema_version: 1\n"
        "name: rocm_only\n"
        "kind: benchmark\n"
        "execution_mode: direct\n"
        "lifecycle: none\n"
        "platforms: [rocm]\n"
        f"entrypoint: {__name__}:DummyBenchPlugin\n"
        "defaults:\n"
        "  iterations: 1\n",
        encoding="utf-8",
    )
    (wdir / "eval_a.yaml").write_text(
        "schema_version: 1\n"
        "name: eval_a\n"
        "kind: eval\n"
        "execution_mode: server\n"
        "lifecycle: reset_cache\n"
        "platforms: [nvidia, rocm]\n"
        f"entrypoint: {__name__}:DummyBenchPlugin\n"
        "defaults:\n"
        "  num_questions: 5\n"
        "  repeats: 1\n"
        "  concurrency: [1]\n",
        encoding="utf-8",
    )
    (sdir / "mini.yaml").write_text(
        "schema_version: 1\n"
        "name: mini\n"
        "benchmarks:\n"
        "  - type: bench_a\n"
        "evals:\n"
        "  - name: eval_a\n",
        encoding="utf-8",
    )
    return Catalog.discover(tmp_path)


def test_catalog_discovers_yaml_and_yml_and_errors(tmp_path: Path) -> None:
    cat = _make_catalog(tmp_path)
    assert [w.name for w in cat.list_workloads()] == [
        "bench_a",
        "rocm_only",
        "eval_a",
    ]
    assert [s.name for s in cat.list_suites()] == ["mini"]

    # Instantiate valid benchmark plugin; reject kind mismatch for eval_a.
    inst = instantiate_plugin(
        cat.get_workload("bench_a").entrypoint, WorkloadKind.BENCHMARK
    )
    assert isinstance(inst, DummyBenchPlugin)
    with pytest.raises(CatalogError, match="EvalRunner"):
        instantiate_plugin(cat.get_workload("eval_a").entrypoint, WorkloadKind.EVAL)

    # Filename mismatch raises CatalogError.
    (tmp_path / "workloads" / "wrong_stem.yaml").write_text(
        "schema_version: 1\nname: other_name\nkind: benchmark\n"
        "execution_mode: direct\nlifecycle: none\n"
        f"entrypoint: {__name__}:DummyBenchPlugin\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="wrong_stem.yaml"):
        Catalog.discover(tmp_path)
    (tmp_path / "workloads" / "wrong_stem.yaml").unlink()

    # Malformed YAML raises CatalogError with path.
    (tmp_path / "workloads" / "broken.yaml").write_text(
        ":\n  - [unclosed", encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="broken.yaml"):
        Catalog.discover(tmp_path)
    (tmp_path / "workloads" / "broken.yaml").unlink()

    # Invalid suite entry (repeats: 0) rejected on discovery.
    (tmp_path / "suites" / "bad_suite.yaml").write_text(
        "schema_version: 1\nname: bad_suite\nevals:\n"
        "  - name: eval_a\n    repeats: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="bad_suite.yaml"):
        Catalog.discover(tmp_path)


def test_resolve_single_job_and_validation_guards(tmp_path: Path) -> None:
    cat = _make_catalog(tmp_path)
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "name": "s1",
            "serve": {
                "config": {"model": "org/model", "revision": VALID_SHA},
                "env": {"VLLM_USE_V1": "1"},
            },
            "hardware": {"gpu": ["h200"], "count": 1},
            "benchmarks": [{"type": "bench_a", "warmup_requests": 2}],
            "evals": [{"name": "eval_a", "min_accuracy": 0.6}],
        }
    )
    job = resolve_single_job(spec, cat, attempt_id="att-fixed")
    assert job.job_id == "default"
    assert job.attempt_id == "att-fixed"
    assert len(job.workloads) == 2
    assert job.workloads[0].params["warmup_requests"] == 2
    assert job.workloads[1].quality_thresholds[0].min_value == 0.6

    # Dash vs underscore serve.config options hash identically.
    s_dash = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {
                "config": {
                    "model": "org/model",
                    "revision": VALID_SHA,
                    "max-model-len": 4096,
                }
            },
            "benchmarks": [{"type": "bench_a"}],
        }
    )
    s_under = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {
                "config": {
                    "model": "org/model",
                    "revision": VALID_SHA,
                    "max_model_len": 4096,
                }
            },
            "benchmarks": [{"type": "bench_a"}],
        }
    )
    assert (
        resolve_single_job(s_dash, cat).resolved_config_hash
        == resolve_single_job(s_under, cat).resolved_config_hash
    )


def test_resolver_rejects_bad_inputs(tmp_path: Path) -> None:
    cat = _make_catalog(tmp_path)

    # Unknown workload parameter rejected.
    typo_spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": VALID_SHA}},
            "benchmarks": [{"type": "bench_a", "unknown_param_xyz": [1]}],
        }
    )
    with pytest.raises(SpecResolutionError, match="unknown_param_xyz"):
        resolve_single_job(typo_spec, cat)

    # Duplicate workloads in a single job.
    dup_spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": VALID_SHA}},
            "benchmarks": [{"type": "bench_a"}, {"type": "bench_a"}],
        }
    )
    with pytest.raises(SpecResolutionError, match="Duplicate workload"):
        resolve_single_job(dup_spec, cat)

    # Platform mismatch (rocm_only on h200).
    plat_spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": VALID_SHA}},
            "hardware": {"gpu": ["h200"]},
            "benchmarks": [{"type": "rocm_only"}],
        }
    )
    with pytest.raises(SpecResolutionError, match="does not support platform"):
        resolve_single_job(plat_spec, cat)

    # Non-SHA revision (branch name or octal int).
    for bad_rev in ("main", "v1.0", 123456, "A" * 40):
        rev_spec = RunSpec.model_validate(
            {
                "schema_version": 1,
                "serve": {"config": {"model": "m", "revision": bad_rev}},
                "benchmarks": [{"type": "bench_a"}],
            }
        )
        with pytest.raises(SpecResolutionError, match="40-character"):
            resolve_single_job(rev_spec, cat)

    # Blocked env vars (HF_ENDPOINT, SSL_CERT_DIR, HF_TOKEN, API_KEY).
    for bad_env in ("HF_ENDPOINT", "SSL_CERT_DIR", "HF_TOKEN", "OPENAI_API_KEY"):
        env_spec = RunSpec.model_validate(
            {
                "schema_version": 1,
                "serve": {
                    "config": {"model": "m", "revision": VALID_SHA},
                    "env": {bad_env: "x"},
                },
                "benchmarks": [{"type": "bench_a"}],
            }
        )
        with pytest.raises(SpecResolutionError, match=bad_env):
            resolve_single_job(env_spec, cat)

    # Phase 1 features rejected cleanly: compare_baseline, nodes > 1, matrix, recipe.
    for extra in (
        {"compare_baseline": True},
        {"hardware": {"nodes": 2}},
        {"matrix": {"tp": [1, 2]}},
        {"recipe": {"model": "meta-llama/Llama-3.1-8B"}},
    ):
        payload: dict[str, Any] = {
            "schema_version": 1,
            "serve": {"config": {"model": "m", "revision": VALID_SHA}},
            "benchmarks": [{"type": "bench_a"}],
        }
        payload.update(extra)
        with pytest.raises(SpecResolutionError):
            resolve_single_job(RunSpec.model_validate(payload), cat)

    # Malformed YAML spec file wraps with file path.
    bad_file = tmp_path / "bad_spec.yaml"
    bad_file.write_text("schema_version: [1", encoding="utf-8")
    with pytest.raises(SpecResolutionError, match="bad_spec.yaml"):
        load_spec_file(bad_file)
