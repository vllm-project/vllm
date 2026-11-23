# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4: Coordinator owning lifecycle, deadlines, quality checks, and bundle sealing."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.byog.bundle.eval_records import DeduplicatedEvalRecordWriter
from tools.byog.bundle.writer import (
    redact_env,
    redact_serve_config,
    seal_bundle,
    write_bundle_metadata,
)
from tools.byog.catalog import instantiate_plugin
from tools.byog.executor.interfaces import (
    AssetProvider,
    BenchmarkRunner,
    CacheResetUnsupportedError,
    DirectCommandResult,
    DirectRunContext,
    EvalRunner,
    PlatformAdapter,
    PointResult,
    ServerHandle,
    ServerRunContext,
    ServerTransportError,
    WorkloadPoint,
)
from tools.byog.schemas.catalog import (
    LifecyclePolicy,
    QualityThreshold,
    WorkloadKind,
)
from tools.byog.schemas.result import (
    ExecutionStatus,
    MeasurementRecord,
    ProtocolVariant,
    QualityState,
    RunResult,
    ThresholdCheckResult,
    compute_compatibility_key,
)
from tools.byog.schemas.spec import ResolvedJob, ResolvedWorkload
from tools.byog.spec.resolver import canonicalize_serve_config

HARNESS_VERSION = "0.1.0-phase0"
COMPATIBILITY_RULE_VERSION = "1"

_NON_FUNCTIONAL_ENV_KEYS = frozenset(
    {
        "VLLM_LOGGING_LEVEL",
        "VLLM_LOG_STATS_INTERVAL",
        "VLLM_CONFIGURE_LOGGING",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_USAGE_STATS_SERVER",
        "VLLM_NO_USAGE_STATS",
        "DO_NOT_TRACK",
    }
)


class JobCancelledError(BaseException):
    """Raised internally when SIGTERM/SIGHUP cancels an active job."""


class JobTimeoutError(TimeoutError):
    """Raised internally when a job exceeds its configured `timeout_s` deadline."""


def _functional_env(serve_env: dict[str, str]) -> dict[str, str]:
    redacted = redact_env(serve_env, allowlist_only=False)
    return {
        k: v
        for k, v in sorted(redacted.items())
        if k.upper() not in _NON_FUNCTIONAL_ENV_KEYS
    }


def _compute_compatibility(
    *,
    job: ResolvedJob,
    workload: ResolvedWorkload,
    point: WorkloadPoint,
    protocol_variant: ProtocolVariant,
    observed_hardware: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    components: dict[str, Any] = {
        "workload": workload.name,
        "kind": workload.kind,
        "point_id": point.point_id,
        "point_params": point.params,
        "execution_mode": workload.execution_mode,
        "protocol_version": workload.protocol_version,
        "protocol_variant": protocol_variant.value,
        "evidence_source": workload.evidence_source.value,
        "serve_config": canonicalize_serve_config(
            redact_serve_config(job.serve_config)
        ),
        "serve_env": _functional_env(job.serve_env),
        "hardware": {
            "platform": observed_hardware.get("platform"),
            "gpu_id": observed_hardware.get("gpu_id"),
            "gpu_count": observed_hardware.get("gpu_count"),
            "driver_version": observed_hardware.get("driver_version"),
        },
        "assets": {
            aid: {"revision": a.revision, "sha256": a.sha256}
            for aid, a in sorted(workload.assets.items())
        },
    }
    if workload.kind == "eval":
        components["eval_settings"] = {
            k: point.params[k]
            for k in sorted(point.params)
            if k
            in (
                "temperature",
                "max_tokens",
                "num_shots",
                "num_questions",
                "repeats",
                "concurrency",
            )
        }
    digest = compute_compatibility_key(components, COMPATIBILITY_RULE_VERSION)
    return components, digest


def _evaluate_thresholds(
    thresholds: list[QualityThreshold],
    metrics: dict[str, float | int | str | None],
) -> tuple[QualityState, list[ThresholdCheckResult]]:
    if not thresholds:
        return QualityState.INCONCLUSIVE, []
    checks: list[ThresholdCheckResult] = []
    any_missing = False
    for th in thresholds:
        raw = metrics.get(th.metric)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            any_missing = True
            checks.append(
                ThresholdCheckResult(
                    metric=th.metric,
                    observed=None,
                    min_value=th.min_value,
                    max_value=th.max_value,
                    source=th.source,
                    passed=False,
                )
            )
            continue
        val = float(raw)
        ok = True
        if th.min_value is not None and val < th.min_value:
            ok = False
        if th.max_value is not None and val > th.max_value:
            ok = False
        checks.append(
            ThresholdCheckResult(
                metric=th.metric,
                observed=val,
                min_value=th.min_value,
                max_value=th.max_value,
                source=th.source,
                passed=ok,
            )
        )
    if any_missing:
        return QualityState.INCONCLUSIVE, checks
    if not all(c.passed for c in checks):
        return QualityState.FAILED, checks
    if not any(c.source == "catalog" for c in checks):
        return QualityState.INCONCLUSIVE, checks
    return QualityState.PASSED, checks


def _validate_observed_hardware(
    job: ResolvedJob, observed: dict[str, Any]
) -> str | None:
    obs_gpu = observed.get("gpu_id")
    if (
        job.hardware.gpu
        and isinstance(obs_gpu, str)
        and obs_gpu != "unknown"
        and obs_gpu not in job.hardware.gpu
    ):
        return (
            f"Observed GPU '{obs_gpu}' does not match requested hardware.gpu "
            f"{job.hardware.gpu}."
        )
    obs_count = observed.get("gpu_count")
    if isinstance(obs_count, int) and obs_count > 0 and obs_count < job.hardware.count:
        return (
            f"Observed GPU count {obs_count} is less than requested "
            f"hardware.count={job.hardware.count}."
        )
    return None


def _run_with_deadline(fn: Callable[[], PointResult], timeout_s: float) -> PointResult:
    """Run a plugin point with a hard wall-clock deadline."""
    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome["exc"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=max(0.05, timeout_s))
    if worker.is_alive():
        raise JobTimeoutError(
            f"Workload point exceeded remaining job timeout ({timeout_s:.1f}s)."
        )
    if "exc" in outcome:
        raise outcome["exc"]
    return outcome["result"]


def _aggregate_quality_state(
    execution_status: ExecutionStatus, measurements: list[MeasurementRecord]
) -> QualityState:
    required = [m for m in measurements if m.required_for_quality]
    if not required:
        eval_ms = [m for m in measurements if m.kind == "eval"]
        if not eval_ms:
            return QualityState.NOT_EVALUATED
        if any(m.quality_state == QualityState.FAILED for m in eval_ms):
            return QualityState.FAILED
        return QualityState.INCONCLUSIVE

    if any(m.quality_state == QualityState.FAILED for m in required):
        return QualityState.FAILED
    if execution_status != ExecutionStatus.COMPLETED or any(
        m.status != ExecutionStatus.COMPLETED for m in measurements
    ):
        return QualityState.INCONCLUSIVE
    if any(m.quality_state != QualityState.PASSED for m in required):
        return QualityState.INCONCLUSIVE
    return QualityState.PASSED


class Coordinator:
    """Executes a `ResolvedJob`, owns server lifecycle, and seals `bundle.tar.gz`."""

    def __init__(
        self,
        *,
        launcher: PlatformAdapter,
        assets: AssetProvider,
        image_ref: str | None = None,
        vllm_commit: str | None = None,
        image_digest: str | None = None,
    ):
        self._launcher = launcher
        self._assets = assets
        self._image_ref = image_ref
        self._vllm_commit = vllm_commit
        self._image_digest = image_digest

    def run(
        self,
        job: ResolvedJob,
        output_root: Path,
        *,
        source_spec_path: Path | None = None,
    ) -> tuple[RunResult, Path]:
        bundle_dir = output_root / job.job_id / job.attempt_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = bundle_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(UTC)
        start_mono = time.monotonic()
        deadline = start_mono + job.timeout_s

        def _remaining_s() -> float:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise JobTimeoutError(
                    f"Job exceeded overall timeout_s={job.timeout_s}s."
                )
            return rem

        observed_hw = self._launcher.detect_hardware()
        baseline_mem = observed_hw.get("device_memory_used_mib")
        baseline_mem_list: list[int] | None = (
            list(baseline_mem) if isinstance(baseline_mem, list) else None
        )

        measurements: list[MeasurementRecord] = []
        withheld_artifacts: list[dict[str, str]] = []
        errors: list[str] = []
        execution_status = ExecutionStatus.COMPLETED
        cleanup_records: list[dict[str, Any]] = []
        active_handle: ServerHandle | None = None
        active_dev_mode: bool = False

        def _stop_active_server() -> bool:
            nonlocal active_handle, active_dev_mode
            if active_handle is None:
                return True
            handle = active_handle
            active_handle = None
            active_dev_mode = False
            report = self._launcher.stop(handle, baseline_memory_mib=baseline_mem_list)
            cleanup_records.append(report)
            ok = True
            if report.get("terminated_pgid") is False:
                errors.append("Server cleanup failed: process group did not terminate.")
                ok = False
            if report.get("vram_reclaimed") is False:
                errors.append(
                    "Server cleanup failed: GPU VRAM was not reclaimed to baseline."
                )
                ok = False
            return ok

        def _ensure_server(enable_dev_mode: bool) -> ServerHandle:
            nonlocal active_handle, active_dev_mode
            if active_handle is not None and active_dev_mode == enable_dev_mode:
                return active_handle
            if active_handle is not None and not _stop_active_server():
                raise RuntimeError("Aborting launch due to failed server cleanup.")
            rem = int(max(1.0, _remaining_s()))
            active_handle = self._launcher.start(
                job.serve_config,
                job.serve_env,
                logs_dir,
                timeout_s=rem,
                enable_dev_mode=enable_dev_mode,
            )
            active_dev_mode = enable_dev_mode
            return active_handle

        # Install SIGTERM/SIGHUP handler on main thread to guarantee graceful
        # server teardown and CANCELLED bundle sealing.
        prev_handlers: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():

            def _on_signal(signum: int, _frame: Any) -> None:
                raise JobCancelledError(f"Received signal {signum}")

            for sig in (signal.SIGTERM, signal.SIGHUP):
                try:
                    prev_handlers[int(sig)] = signal.getsignal(sig)
                    signal.signal(sig, _on_signal)
                except (ValueError, OSError):
                    pass

        try:
            hw_err = _validate_observed_hardware(job, observed_hw)
            if hw_err is not None:
                errors.append(hw_err)
                execution_status = ExecutionStatus.FAILED
            else:
                self._execute_workloads(
                    job=job,
                    bundle_dir=bundle_dir,
                    observed_hw=observed_hw,
                    remaining_s=_remaining_s,
                    ensure_server=_ensure_server,
                    stop_active_server=_stop_active_server,
                    measurements=measurements,
                    withheld_artifacts=withheld_artifacts,
                    errors=errors,
                )
                if errors or any(
                    m.status != ExecutionStatus.COMPLETED for m in measurements
                ):
                    execution_status = ExecutionStatus.FAILED
        except JobCancelledError as exc:
            errors.append(str(exc))
            execution_status = ExecutionStatus.CANCELLED
        except (JobTimeoutError, TimeoutError) as exc:
            errors.append(str(exc))
            execution_status = ExecutionStatus.TIMED_OUT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            execution_status = ExecutionStatus.FAILED
        finally:
            if (
                not _stop_active_server()
                and execution_status == ExecutionStatus.COMPLETED
            ):
                execution_status = ExecutionStatus.FAILED
            for sig_num, prev in prev_handlers.items():
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig_num, prev)

        finished_at = datetime.now(UTC)
        duration_s = max(0.0, time.monotonic() - start_mono)

        if cleanup_records:
            (logs_dir / "cleanup.json").write_text(
                json.dumps(cleanup_records, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        combo_key = {
            "platform": observed_hw.get("platform"),
            "gpu": observed_hw.get("gpu_id"),
            "gpu_count": observed_hw.get("gpu_count"),
            "model": job.serve_config.get("model"),
        }
        quality_state = _aggregate_quality_state(execution_status, measurements)

        result = RunResult(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            spec_name=job.spec_name,
            claimed_source_tier=job.claimed_source_tier,
            execution_status=execution_status,
            quality_state=quality_state,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            harness_version=HARNESS_VERSION,
            vllm_commit=self._vllm_commit,
            image_digest=self._image_digest,
            requested_hardware=job.hardware.model_dump(),
            observed_hardware=observed_hw,
            combo_key=combo_key,
            resolved_config_hash=job.resolved_config_hash,
            measurements=measurements,
            withheld_artifacts=withheld_artifacts,
            errors=errors,
        )

        write_bundle_metadata(
            bundle_dir,
            job=job,
            result=result,
            source_spec_path=source_spec_path,
            image_ref=self._image_ref,
        )
        archive_path = seal_bundle(bundle_dir, job.job_id, job.attempt_id)
        return result, archive_path

    def _execute_workloads(
        self,
        *,
        job: ResolvedJob,
        bundle_dir: Path,
        observed_hw: dict[str, Any],
        remaining_s: Callable[[], float],
        ensure_server: Callable[[bool], ServerHandle],
        stop_active_server: Callable[[], bool],
        measurements: list[MeasurementRecord],
        withheld_artifacts: list[dict[str, str]],
        errors: list[str],
    ) -> None:
        served_model = str(
            job.serve_config.get("served-model-name")
            or job.serve_config.get("served_model_name")
            or job.serve_config.get("model", "")
        )

        for workload in job.workloads:
            plugin = instantiate_plugin(
                workload.entrypoint, expected_kind=WorkloadKind(workload.kind)
            )
            points = plugin.points(workload.params)
            if not points:
                errors.append(f"Workload '{workload.name}' yielded zero points.")
                continue

            subdir_root = "evals" if workload.kind == "eval" else "benchmarks"
            workload_out = bundle_dir / subdir_root / workload.name
            workload_out.mkdir(parents=True, exist_ok=True)

            if workload.execution_mode == "direct":
                if not stop_active_server():
                    return
                assert isinstance(plugin, BenchmarkRunner)
                self._run_direct_workload(
                    job=job,
                    workload=workload,
                    plugin=plugin,
                    points=points,
                    workload_out=workload_out,
                    bundle_dir=bundle_dir,
                    observed_hw=observed_hw,
                    remaining_s=remaining_s,
                    measurements=measurements,
                )
            else:
                self._run_server_workload(
                    job=job,
                    workload=workload,
                    plugin=plugin,
                    points=points,
                    served_model=served_model,
                    workload_out=workload_out,
                    bundle_dir=bundle_dir,
                    observed_hw=observed_hw,
                    remaining_s=remaining_s,
                    ensure_server=ensure_server,
                    stop_active_server=stop_active_server,
                    measurements=measurements,
                    withheld_artifacts=withheld_artifacts,
                    errors=errors,
                )

    def _run_direct_workload(
        self,
        *,
        job: ResolvedJob,
        workload: ResolvedWorkload,
        plugin: BenchmarkRunner,
        points: list[WorkloadPoint],
        workload_out: Path,
        bundle_dir: Path,
        observed_hw: dict[str, Any],
        remaining_s: Callable[[], float],
        measurements: list[MeasurementRecord],
    ) -> None:
        for point in points:
            rem = remaining_s()

            def _run_cmd(
                cmd: list[str], cmd_timeout_s: int | None = None
            ) -> DirectCommandResult:
                budget = int(max(1.0, remaining_s()))
                eff = min(budget, cmd_timeout_s) if cmd_timeout_s else budget
                return self._launcher.run_direct(cmd, job.serve_env, eff)

            ctx = DirectRunContext(
                serve_config=dict(job.serve_config),
                serve_env=dict(job.serve_env),
                assets=self._assets,
                workload_assets=dict(workload.assets),
                output_dir=workload_out,
                timeout_s=int(max(1.0, rem)),
                run_command=_run_cmd,
            )
            try:
                pr = _run_with_deadline(
                    lambda p=point, c=ctx: plugin.run_point(p, c), rem
                )
            except (JobCancelledError, JobTimeoutError):
                raise
            except Exception as exc:  # noqa: BLE001
                pr = PointResult(
                    status=ExecutionStatus.FAILED,
                    error_message=f"{type(exc).__name__}: {exc}",
                )

            comps, compat_key = _compute_compatibility(
                job=job,
                workload=workload,
                point=point,
                protocol_variant=ProtocolVariant.DIRECT,
                observed_hardware=observed_hw,
            )
            measurements.append(
                MeasurementRecord(
                    workload=workload.name,
                    kind="benchmark",
                    point_id=point.point_id,
                    execution_mode="direct",
                    protocol_version=workload.protocol_version,
                    requested_lifecycle=LifecyclePolicy.NONE,
                    protocol_variant=ProtocolVariant.DIRECT,
                    evidence_source=workload.evidence_source,
                    required_for_quality=workload.required_for_quality,
                    compatibility_rule_version=COMPATIBILITY_RULE_VERSION,
                    compatibility_components=comps,
                    compatibility_key=compat_key,
                    status=pr.status,
                    quality_state=QualityState.NOT_EVALUATED,
                    metrics=pr.metrics,
                    attempted=pr.attempted,
                    completed=pr.completed,
                    scoring_denominator=pr.scoring_denominator,
                    model_errors=pr.model_errors,
                    infra_errors=pr.infra_errors,
                    error_message=pr.error_message,
                    artifacts=self._rel_artifacts(pr.artifacts, bundle_dir),
                )
            )

    def _run_server_workload(
        self,
        *,
        job: ResolvedJob,
        workload: ResolvedWorkload,
        plugin: BenchmarkRunner | EvalRunner,
        points: list[WorkloadPoint],
        served_model: str,
        workload_out: Path,
        bundle_dir: Path,
        observed_hw: dict[str, Any],
        remaining_s: Callable[[], float],
        ensure_server: Callable[[bool], ServerHandle],
        stop_active_server: Callable[[], bool],
        measurements: list[MeasurementRecord],
        withheld_artifacts: list[dict[str, str]],
        errors: list[str],
    ) -> None:
        requested_lc = LifecyclePolicy(workload.lifecycle)
        needs_dev_mode = requested_lc == LifecyclePolicy.RESET_CACHE

        record_writer: DeduplicatedEvalRecordWriter | None = None
        if workload.kind == "eval":
            record_writer = DeduplicatedEvalRecordWriter(
                workload=workload.name,
                output_dir=workload_out,
                policy=workload.publication_policy,
            )

        try:
            for idx, point in enumerate(points):
                rem = remaining_s()
                protocol_variant = ProtocolVariant(requested_lc.value)
                fallback_reason: str | None = None

                if idx == 0:
                    handle = ensure_server(needs_dev_mode)
                elif requested_lc == LifecyclePolicy.RESTART_SERVER:
                    if not stop_active_server():
                        return
                    handle = ensure_server(needs_dev_mode)
                elif requested_lc == LifecyclePolicy.RESET_CACHE:
                    handle = ensure_server(needs_dev_mode)
                    try:
                        protocol_variant = self._launcher.reset_cache(handle)
                    except CacheResetUnsupportedError as exc:
                        if not workload.allow_lifecycle_fallback:
                            msg = (
                                f"Lifecycle reset_cache unsupported for "
                                f"'{workload.name}' and allow_lifecycle_fallback is "
                                f"False: {exc}"
                            )
                            errors.append(msg)
                            comps, compat_key = _compute_compatibility(
                                job=job,
                                workload=workload,
                                point=point,
                                protocol_variant=ProtocolVariant.RESET_CACHE,
                                observed_hardware=observed_hw,
                            )
                            measurements.append(
                                MeasurementRecord(
                                    workload=workload.name,
                                    kind=workload.kind,
                                    point_id=point.point_id,
                                    execution_mode="server",
                                    protocol_version=workload.protocol_version,
                                    requested_lifecycle=requested_lc,
                                    protocol_variant=ProtocolVariant.RESET_CACHE,
                                    evidence_source=workload.evidence_source,
                                    required_for_quality=workload.required_for_quality,
                                    compatibility_rule_version=COMPATIBILITY_RULE_VERSION,
                                    compatibility_components=comps,
                                    compatibility_key=compat_key,
                                    status=ExecutionStatus.FAILED,
                                    quality_state=(
                                        QualityState.INCONCLUSIVE
                                        if workload.kind == "eval"
                                        else QualityState.NOT_EVALUATED
                                    ),
                                    error_message=msg,
                                )
                            )
                            return
                        fallback_reason = str(exc)
                        if not stop_active_server():
                            return
                        handle = ensure_server(False)
                        protocol_variant = ProtocolVariant.RESTART_SERVER
                    except ServerTransportError as exc:
                        msg = (
                            f"Server transport error during reset_cache for "
                            f"'{workload.name}': {exc}"
                        )
                        errors.append(msg)
                        comps, compat_key = _compute_compatibility(
                            job=job,
                            workload=workload,
                            point=point,
                            protocol_variant=ProtocolVariant.RESET_CACHE,
                            observed_hardware=observed_hw,
                        )
                        measurements.append(
                            MeasurementRecord(
                                workload=workload.name,
                                kind=workload.kind,
                                point_id=point.point_id,
                                execution_mode="server",
                                protocol_version=workload.protocol_version,
                                requested_lifecycle=requested_lc,
                                protocol_variant=ProtocolVariant.RESET_CACHE,
                                evidence_source=workload.evidence_source,
                                required_for_quality=workload.required_for_quality,
                                compatibility_rule_version=COMPATIBILITY_RULE_VERSION,
                                compatibility_components=comps,
                                compatibility_key=compat_key,
                                status=ExecutionStatus.FAILED,
                                quality_state=(
                                    QualityState.INCONCLUSIVE
                                    if workload.kind == "eval"
                                    else QualityState.NOT_EVALUATED
                                ),
                                error_message=msg,
                            )
                        )
                        return
                else:
                    handle = ensure_server(needs_dev_mode)

                ctx = ServerRunContext(
                    base_url=handle.base_url,
                    model=served_model,
                    assets=self._assets,
                    workload_assets=dict(workload.assets),
                    output_dir=workload_out,
                    timeout_s=int(max(1.0, remaining_s())),
                )
                before_written = (
                    record_writer.written_count if record_writer is not None else 0
                )
                try:
                    if workload.kind == "eval":
                        assert isinstance(plugin, EvalRunner)
                        assert record_writer is not None
                        pr = _run_with_deadline(
                            lambda p=point, c=ctx, w=record_writer: plugin.run_point(
                                p, c, w
                            ),
                            rem,
                        )
                    else:
                        assert isinstance(plugin, BenchmarkRunner)
                        pr = _run_with_deadline(
                            lambda p=point, c=ctx: plugin.run_point(p, c),
                            rem,
                        )
                except (JobCancelledError, JobTimeoutError):
                    raise
                except Exception as exc:  # noqa: BLE001
                    pr = PointResult(
                        status=ExecutionStatus.FAILED,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )

                point_written = (
                    record_writer.written_count - before_written
                    if record_writer is not None
                    else 0
                )
                q_state = QualityState.NOT_EVALUATED
                th_checks: list[ThresholdCheckResult] = []
                err_msg = pr.error_message
                if workload.kind == "eval":
                    if (
                        pr.attempted is not None
                        and pr.status == ExecutionStatus.COMPLETED
                        and point_written != pr.attempted
                    ):
                        mismatch = (
                            f"Eval '{workload.name}' point '{point.point_id}' "
                            f"reported attempted={pr.attempted} but wrote "
                            f"{point_written} response records."
                        )
                        errors.append(mismatch)
                        err_msg = f"{err_msg}; {mismatch}" if err_msg else mismatch
                        q_state = QualityState.INCONCLUSIVE
                    elif (
                        pr.status != ExecutionStatus.COMPLETED
                        or pr.infra_errors > 0
                        or not pr.scoring_denominator
                    ):
                        q_state = QualityState.INCONCLUSIVE
                    else:
                        q_state, th_checks = _evaluate_thresholds(
                            workload.quality_thresholds, pr.metrics
                        )

                comps, compat_key = _compute_compatibility(
                    job=job,
                    workload=workload,
                    point=point,
                    protocol_variant=protocol_variant,
                    observed_hardware=observed_hw,
                )
                measurements.append(
                    MeasurementRecord(
                        workload=workload.name,
                        kind=workload.kind,
                        point_id=point.point_id,
                        execution_mode="server",
                        protocol_version=workload.protocol_version,
                        requested_lifecycle=requested_lc,
                        protocol_variant=protocol_variant,
                        lifecycle_fallback_reason=fallback_reason,
                        evidence_source=workload.evidence_source,
                        required_for_quality=workload.required_for_quality,
                        compatibility_rule_version=COMPATIBILITY_RULE_VERSION,
                        compatibility_components=comps,
                        compatibility_key=compat_key,
                        status=pr.status,
                        quality_state=q_state,
                        threshold_checks=th_checks,
                        metrics=pr.metrics,
                        attempted=pr.attempted,
                        completed=pr.completed,
                        scoring_denominator=pr.scoring_denominator,
                        model_errors=pr.model_errors,
                        infra_errors=pr.infra_errors,
                        error_message=err_msg,
                        artifacts=self._rel_artifacts(pr.artifacts, bundle_dir),
                    )
                )
        finally:
            if record_writer is not None:
                record_paths = record_writer.close()
                withheld_artifacts.extend(record_writer.withheld_summary())
                rel_records = self._rel_artifacts(record_paths, bundle_dir)
                for m in measurements:
                    if m.workload == workload.name:
                        for rp in rel_records:
                            if rp not in m.artifacts:
                                m.artifacts.append(rp)

    @staticmethod
    def _rel_artifacts(paths: list[Path], bundle_dir: Path) -> list[str]:
        rels: list[str] = []
        root = bundle_dir.resolve()
        for p in paths:
            try:
                rels.append(p.resolve().relative_to(root).as_posix())
            except ValueError:
                rels.append(p.name)
        return rels
