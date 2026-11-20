# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C3: Parse spec.yaml, enforce Phase 0 validation rules, and emit ResolvedJob."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tools.byog.catalog import Catalog, CatalogError
from tools.byog.schemas.catalog import (
    PlatformKind,
    QualityThreshold,
    WorkloadDefinition,
    WorkloadKind,
)
from tools.byog.schemas.spec import (
    BenchmarkSelection,
    EvalSelection,
    ResolvedJob,
    ResolvedWorkload,
    RunSpec,
)

DENIED_ENV_VARS: frozenset[str] = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
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
    }
)

_SECRET_SUBSTRINGS: tuple[str, ...] = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
)
_SECRET_TOKENS: frozenset[str] = frozenset({"KEY", "PAT"})

GPU_PLATFORM_MAP: dict[str, PlatformKind] = {
    "h100": PlatformKind.NVIDIA,
    "h200": PlatformKind.NVIDIA,
    "b200": PlatformKind.NVIDIA,
    "b300": PlatformKind.NVIDIA,
    "gb200": PlatformKind.NVIDIA,
    "gb300": PlatformKind.NVIDIA,
    "a100": PlatformKind.NVIDIA,
    "l40s": PlatformKind.NVIDIA,
    "l4": PlatformKind.NVIDIA,
    "rtx_5090": PlatformKind.NVIDIA,
    "rtx_pro_6000": PlatformKind.NVIDIA,
    "mi300x": PlatformKind.ROCM,
    "mi325x": PlatformKind.ROCM,
    "mi350x": PlatformKind.ROCM,
    "mi355x": PlatformKind.ROCM,
}

_HEX_LOWER = frozenset("0123456789abcdef")


class SpecResolutionError(ValueError):
    """Raised when a spec cannot be validated or resolved in Phase 0."""


def is_valid_git_sha(value: object) -> bool:
    """Return True iff `value` is a 40-character lowercase hex commit SHA."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(ch in _HEX_LOWER for ch in value)
    )


def _is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    if any(sub in upper for sub in _SECRET_SUBSTRINGS):
        return True
    tokens = {tok for tok in upper.split("_") if tok}
    return bool(tokens & _SECRET_TOKENS)


def validate_serve_env(env: dict[str, str]) -> dict[str, str]:
    """Enforce Phase 0 `serve.env` safety denylist and secret-name guard."""
    blocked: list[str] = []
    for key in env:
        upper = key.upper()
        if upper in DENIED_ENV_VARS or _is_secret_env_name(upper):
            blocked.append(key)
    if blocked:
        raise SpecResolutionError(
            "Disallowed or secret-bearing environment variable(s) in serve.env: "
            f"{', '.join(sorted(blocked))}."
        )
    return dict(env)


def _check_json_compatible(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (date, datetime)):
        raise SpecResolutionError(
            f"Non-JSON date/datetime value at '{path}': quote dates as strings in YAML."
        )
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _check_json_compatible(item, f"{path}[{idx}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise SpecResolutionError(
                    f"Non-string mapping key {k!r} at '{path}' is not allowed."
                )
            _check_json_compatible(v, f"{path}.{k}")
        return
    raise SpecResolutionError(
        f"Unsupported value type {type(value).__name__} at '{path}'."
    )


def canonicalize_serve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize top-level `-` vs `_` keys for deterministic config hashing."""
    normalized: dict[str, Any] = {}
    for key, val in sorted(config.items()):
        canon_key = key.replace("_", "-")
        if canon_key in normalized:
            raise SpecResolutionError(
                f"Duplicate serve.config option '{key}' conflicts with '{canon_key}'."
            )
        normalized[canon_key] = val
    return normalized


def _deep_merge_params(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, val in overrides.items():
        if val is None:
            merged.pop(key, None)
        elif isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_params(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_spec_file(spec_path: Path) -> RunSpec:
    """Load and schema-validate a YAML `RunSpec` with clear path context."""
    try:
        text = spec_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecResolutionError(
            f"Failed to read spec file '{spec_path}': {exc}"
        ) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecResolutionError(
            f"Malformed YAML in spec file '{spec_path}': {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SpecResolutionError(
            f"Spec file '{spec_path}' must contain a YAML mapping, got "
            f"{type(raw).__name__}."
        )
    try:
        return RunSpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecResolutionError(
            f"Invalid spec schema in '{spec_path}': {exc}"
        ) from exc


def _resolve_target_platform(gpus: list[str]) -> PlatformKind | None:
    if not gpus:
        return None
    platforms: set[PlatformKind] = set()
    unknown: list[str] = []
    for gpu in gpus:
        plat = GPU_PLATFORM_MAP.get(gpu)
        if plat is None:
            unknown.append(gpu)
        else:
            platforms.add(plat)
    if unknown:
        valid = ", ".join(sorted(GPU_PLATFORM_MAP))
        raise SpecResolutionError(
            f"Unknown hardware.gpu ID(s): {', '.join(unknown)}. Valid canonical IDs: "
            f"{valid}"
        )
    if len(platforms) > 1:
        raise SpecResolutionError(
            f"hardware.gpu cannot mix platforms in a single job: {gpus}"
        )
    return next(iter(platforms))


def _apply_thresholds(
    defn: WorkloadDefinition, min_accuracy: float | None
) -> list[QualityThreshold]:
    thresholds = list(defn.quality_thresholds)
    if min_accuracy is not None:
        thresholds = [t for t in thresholds if t.metric != "accuracy"]
        thresholds.append(QualityThreshold(metric="accuracy", min_value=min_accuracy))
    return thresholds


def _build_resolved_workload(
    defn: WorkloadDefinition,
    overrides: dict[str, Any],
    *,
    target_platform: PlatformKind | None,
    min_accuracy: float | None = None,
) -> ResolvedWorkload:
    if target_platform is not None and target_platform not in defn.platforms:
        supported = ", ".join(p.value for p in defn.platforms)
        raise SpecResolutionError(
            f"Workload '{defn.name}' does not support platform "
            f"'{target_platform.value}' (supported: {supported})."
        )
    valid_keys = defn.valid_param_keys()
    if defn.kind == WorkloadKind.EVAL:
        valid_keys = valid_keys | {"repeats", "concurrency"}
    unknown_keys = sorted(k for k in overrides if k not in valid_keys)
    if unknown_keys:
        allowed = ", ".join(sorted(valid_keys)) or "<none>"
        raise SpecResolutionError(
            f"Unknown parameter(s) for workload '{defn.name}': "
            f"{', '.join(unknown_keys)}. Allowed: {allowed}"
        )

    params = _deep_merge_params(defn.defaults, overrides)
    _check_json_compatible(params, f"workloads.{defn.name}.params")
    return ResolvedWorkload(
        name=defn.name,
        kind=defn.kind.value,
        execution_mode=defn.execution_mode.value,
        lifecycle=defn.lifecycle.value,
        allow_lifecycle_fallback=defn.allow_lifecycle_fallback,
        evidence_source=defn.evidence_source,
        platforms=list(defn.platforms),
        entrypoint=defn.entrypoint,
        protocol_version=defn.protocol_version,
        assets=dict(defn.assets),
        publication_policy=defn.publication_policy,
        quality_thresholds=_apply_thresholds(defn, min_accuracy),
        required_for_quality=defn.required_for_quality,
        params=params,
    )


def resolve_single_job(
    spec: RunSpec,
    catalog: Catalog,
    *,
    allow_unpinned_revision: bool = False,
    attempt_id: str | None = None,
    source_tier: str = "community",
) -> ResolvedJob:
    """Resolve a Phase 0 single-job `RunSpec` into a concrete `ResolvedJob`."""
    if spec.recipe is not None:
        raise SpecResolutionError(
            "Recipe resolution (`recipe:`) is not supported in Phase 0 (planned in "
            "P1.3). Provide an explicit `serve:` block instead."
        )
    if spec.matrix or spec.matrix_include or spec.matrix_exclude:
        raise SpecResolutionError(
            "Matrix expansion (`matrix`, `matrix_include`, `matrix_exclude`) is not "
            "supported in Phase 0 (planned in P1.4)."
        )
    if spec.compare_baseline:
        raise SpecResolutionError(
            "`compare_baseline: true` is not supported in Phase 0 (requires Results "
            "Service in Phase 1)."
        )
    if spec.hardware.nodes > 1:
        raise SpecResolutionError(
            "Multi-node jobs (`hardware.nodes > 1`) are not supported in Phase 0."
        )
    if spec.serve is None:
        raise SpecResolutionError("Spec must define a `serve:` block.")

    serve_config = dict(spec.serve.config)
    _check_json_compatible(serve_config, "serve.config")
    if "model" not in serve_config or not isinstance(serve_config["model"], str):
        raise SpecResolutionError("`serve.config.model` must be a non-empty string.")
    if not serve_config["model"].strip():
        raise SpecResolutionError("`serve.config.model` must be a non-empty string.")

    revision = serve_config.get("revision")
    if not allow_unpinned_revision and not is_valid_git_sha(revision):
        raise SpecResolutionError(
            "`serve.config.revision` must be a 40-character lowercase hex commit SHA "
            "(got "
            f"{revision!r}). Pass --allow-unpinned-revision for local experiments."
        )

    canonical_cfg = canonicalize_serve_config(serve_config)
    serve_env = validate_serve_env(spec.serve.env)
    target_platform = _resolve_target_platform(spec.hardware.gpu)

    # Expand benchmark and eval selections from suite + explicit lists.
    bench_entries: list[BenchmarkSelection] = []
    eval_entries: list[EvalSelection] = []
    try:
        if spec.benchmark_suite:
            suite = catalog.get_suite(spec.benchmark_suite)
            bench_entries.extend(
                BenchmarkSelection.model_validate(item) for item in suite.benchmarks
            )
        if spec.benchmarks:
            bench_entries.extend(spec.benchmarks)

        if spec.eval_suite:
            suite = catalog.get_suite(spec.eval_suite)
            eval_entries.extend(
                EvalSelection.model_validate(item) for item in suite.evals
            )
        if spec.evals:
            eval_entries.extend(spec.evals)
    except (CatalogError, ValidationError) as exc:
        raise SpecResolutionError(str(exc)) from exc

    if not bench_entries and not eval_entries:
        raise SpecResolutionError(
            "Spec must select at least one benchmark or eval (via suite or list)."
        )

    workloads: list[ResolvedWorkload] = []
    seen_names: set[str] = set()

    for b_sel in bench_entries:
        try:
            defn = catalog.get_workload(b_sel.type)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.BENCHMARK:
            raise SpecResolutionError(
                f"Workload '{defn.name}' is an eval, not a benchmark."
            )
        if defn.name in seen_names:
            raise SpecResolutionError(
                f"Duplicate workload '{defn.name}' in single job spec."
            )
        seen_names.add(defn.name)
        overrides = b_sel.model_dump(exclude={"type"})
        workloads.append(
            _build_resolved_workload(defn, overrides, target_platform=target_platform)
        )

    for e_sel in eval_entries:
        try:
            defn = catalog.get_workload(e_sel.name)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.EVAL:
            raise SpecResolutionError(
                f"Workload '{defn.name}' is a benchmark, not an eval."
            )
        if defn.name in seen_names:
            raise SpecResolutionError(
                f"Duplicate workload '{defn.name}' in single job spec."
            )
        seen_names.add(defn.name)
        raw_dump = e_sel.model_dump(
            exclude={"name", "min_accuracy"}, exclude_unset=True
        )
        workloads.append(
            _build_resolved_workload(
                defn,
                raw_dump,
                target_platform=target_platform,
                min_accuracy=e_sel.min_accuracy,
            )
        )

    config_payload = {
        "serve_config": canonical_cfg,
        "serve_env": dict(sorted(serve_env.items())),
        "hardware": spec.hardware.model_dump(),
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    return ResolvedJob(
        job_id="default",
        attempt_id=attempt_id or f"att-{uuid.uuid4().hex[:16]}",
        spec_name=spec.name,
        serve_config=serve_config,
        serve_env=serve_env,
        hardware=spec.hardware,
        workloads=workloads,
        timeout_s=spec.timeout_s,
        source_tier=source_tier,  # type: ignore[arg-type]
        resolved_config_hash=config_hash,
    )
