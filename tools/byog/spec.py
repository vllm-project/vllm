# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec resolver (C3): parse `spec.yaml`, validate rules, and resolve jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tools.byog.catalog import Catalog, CatalogError
from tools.byog.schemas.execution import (
    BenchmarkSelection,
    EvalSelection,
    ResolvedJob,
    ResolvedWorkload,
    RunSpec,
    WorkloadDefinition,
    WorkloadKind,
)

DENIED_ENV_VARS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "PATH",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    }
)
_DENIED_ENV_PREFIXES = ("LD_",)
_SECRET_SUBSTRS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "APIKEY",
)
_SECRET_TOKENS = frozenset({"KEY", "PAT"})
_HEX_LOWER = frozenset("0123456789abcdef")


class SpecResolutionError(ValueError):
    """Raised when `spec.yaml` cannot be validated or resolved."""


def is_valid_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(ch in _HEX_LOWER for ch in value)
    )


def validate_serve_env(env: dict[str, str]) -> dict[str, str]:
    blocked = []
    for k in env:
        up = k.upper()
        if (
            up in DENIED_ENV_VARS
            or up.startswith(_DENIED_ENV_PREFIXES)
            or any(s in up for s in _SECRET_SUBSTRS)
            or bool(set(up.split("_")) & _SECRET_TOKENS)
        ):
            blocked.append(k)
    if blocked:
        raise SpecResolutionError(
            "Disallowed or secret env var(s) in serve.env: "
            f"{', '.join(sorted(blocked))}."
        )
    return dict(env)


def canonicalize_serve_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(config.items()):
        canon = k.replace("_", "-")
        if canon in out:
            raise SpecResolutionError(f"Duplicate serve.config option '{k}'.")
        out[canon] = v
    return out


def load_spec_file(spec_path: Path) -> RunSpec:
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SpecResolutionError(f"Failed to read spec '{spec_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecResolutionError(f"Spec '{spec_path}' must be a YAML mapping.")
    try:
        return RunSpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecResolutionError(f"Invalid spec '{spec_path}': {exc}") from exc


def _resolve_workload(
    defn: WorkloadDefinition,
    overrides: dict[str, Any],
) -> ResolvedWorkload:
    # Only `allowed_params` (plus eval `repeats`/`concurrency`) may be set by a spec;
    # pinned catalog `defaults` (datasets, hashes, shots, etc.) are immutable.
    valid_keys = defn.valid_param_keys() | (
        {"repeats", "concurrency"} if defn.kind == WorkloadKind.EVAL else set()
    )
    unknown = sorted(k for k in overrides if k not in valid_keys)
    if unknown:
        raise SpecResolutionError(
            f"Disallowed or unknown param(s) for '{defn.name}': {', '.join(unknown)}."
        )
    params = {**defn.defaults, **{k: v for k, v in overrides.items() if v is not None}}
    return ResolvedWorkload(definition=defn, params=params)


def resolve_spec(
    spec: RunSpec,
    catalog: Catalog,
    *,
    allow_unpinned_revision: bool = False,
) -> list[ResolvedJob]:
    """Resolve `spec.yaml` into executable jobs (Phase 0 emits 1 job: `default`)."""
    serve_config = dict(spec.serve.config)
    model = serve_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SpecResolutionError("`serve.config.model` must be a non-empty string.")
    if not allow_unpinned_revision and not is_valid_git_sha(
        serve_config.get("revision")
    ):
        raise SpecResolutionError(
            "`serve.config.revision` must be a 40-char lowercase hex commit SHA "
            "(or pass --allow-unpinned-revision)."
        )

    canonicalize_serve_config(serve_config)
    serve_env = validate_serve_env(spec.serve.env)

    bench_entries: list[BenchmarkSelection] = []
    eval_entries: list[EvalSelection] = []
    try:
        # Explicit list replaces suite; `[]` explicitly skips that category.
        if spec.benchmarks is not None:
            bench_entries = list(spec.benchmarks)
        elif spec.benchmark_suite:
            bench_entries = list(catalog.get_suite(spec.benchmark_suite).benchmarks)

        if spec.evals is not None:
            eval_entries = list(spec.evals)
        elif spec.eval_suite:
            eval_entries = list(catalog.get_suite(spec.eval_suite).evals)
    except CatalogError as exc:
        raise SpecResolutionError(str(exc)) from exc

    if not bench_entries and not eval_entries:
        raise SpecResolutionError("Spec must select at least one benchmark or eval.")

    workloads: list[ResolvedWorkload] = []
    seen: set[str] = set()
    for b in bench_entries:
        try:
            defn = catalog.get_workload(b.type)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.BENCHMARK or defn.name in seen:
            raise SpecResolutionError(f"Invalid or duplicate benchmark '{defn.name}'.")
        seen.add(defn.name)
        workloads.append(_resolve_workload(defn, b.model_dump(exclude={"type"})))

    for e in eval_entries:
        try:
            defn = catalog.get_workload(e.name)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.EVAL or defn.name in seen:
            raise SpecResolutionError(f"Invalid or duplicate eval '{defn.name}'.")
        seen.add(defn.name)
        overrides = e.model_dump(exclude={"name"}, exclude_unset=True)
        workloads.append(_resolve_workload(defn, overrides))

    return [
        ResolvedJob(
            job_id="default",
            spec_name=spec.name,
            serve_config=serve_config,
            serve_env=serve_env,
            hardware=spec.hardware,
            workloads=workloads,
            timeout_s=spec.timeout_s,
        )
    ]
