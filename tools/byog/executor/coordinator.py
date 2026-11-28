# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor / Coordinator (C4): owns server lifecycle, cleanup, and result bundling."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.byog.bundle import (
    DeduplicatedEvalRecordWriter,
    seal_bundle,
    write_bundle_metadata,
)
from tools.byog.catalog import instantiate_plugin
from tools.byog.interfaces import (
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
from tools.byog.schemas.execution import (
    ExecutionMode,
    ExecutionStatus,
    LifecyclePolicy,
    ResolvedJob,
    ResolvedWorkload,
    WorkloadKind,
)
from tools.byog.schemas.results import (
    MeasurementRecord,
    ObservedHardware,
    RunResult,
)

HARNESS_VERSION = "0.1.0-phase0"
_HANDLED_SIGNALS = tuple(
    s
    for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", None))
    if s is not None
)


class _JobCancelled(BaseException):
    pass


def _run_with_timeout(fn: Callable[[], PointResult], timeout_s: float) -> PointResult:
    """Run a plugin point with a hard wall-clock deadline."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=max(0.05, timeout_s))
    except concurrent.futures.TimeoutError as exc:
        fut.cancel()
        raise TimeoutError(
            f"Point exceeded remaining timeout ({timeout_s:.1f}s)."
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class Coordinator:
    """Runs one `ResolvedJob`, owns server lifecycle, and seals `bundle.tar.gz`."""

    def __init__(
        self,
        *,
        launcher: PlatformAdapter,
        image_ref: str | None = None,
        vllm_commit: str | None = None,
        image_digest: str | None = None,
    ):
        self._launcher = launcher
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
        attempt_id = f"att-{uuid.uuid4().hex[:16]}"
        bundle_dir = output_root / job.job_id / attempt_id
        logs_dir = bundle_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        deadline = t0 + job.timeout_s
        hw = self._launcher.detect_hardware()
        base_mem_list = (
            list(hw.device_memory_used_mib)
            if hw.device_memory_used_mib is not None
            else None
        )

        measurements: list[MeasurementRecord] = []
        errors: list[str] = []
        cleanups: list[dict[str, Any]] = []
        status = ExecutionStatus.COMPLETED
        active: ServerHandle | None = None
        active_dev = False

        def _rem_f() -> float:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"Job exceeded timeout_s={job.timeout_s}s.")
            return rem

        def _rem_s() -> int:
            return max(1, int(_rem_f()))

        def _stop_server() -> bool:
            nonlocal active, active_dev
            if active is None:
                return True
            h, active, active_dev = active, None, False
            rep = self._launcher.stop(h, baseline_memory_mib=base_mem_list)
            cleanups.append(asdict(rep))
            ok = True
            if not rep.terminated_pgid:
                errors.append("Server process group failed to terminate.")
                ok = False
            if rep.vram_reclaimed is False:
                errors.append("GPU VRAM was not reclaimed to baseline.")
                ok = False
            return ok

        def _ensure_server(dev_mode: bool) -> tuple[ServerHandle, bool]:
            """Return `(handle, reused)` where `reused` is True if already running."""
            nonlocal active, active_dev
            if active is not None and active_dev == dev_mode:
                return active, True
            if active is not None and not _stop_server():
                raise RuntimeError("Aborting launch after failed server cleanup.")
            active = self._launcher.start(
                job.serve_config,
                job.serve_env,
                logs_dir,
                _rem_s(),
                enable_dev_mode=dev_mode,
            )
            active_dev = dev_mode
            return active, False

        def _record(
            w: ResolvedWorkload,
            pt: WorkloadPoint,
            actual_lc: LifecyclePolicy,
            pr: PointResult,
        ) -> None:
            defn = w.definition
            rels = []
            root = bundle_dir.resolve()
            for p in pr.artifacts:
                try:
                    rels.append(p.resolve().relative_to(root).as_posix())
                except ValueError:
                    rels.append(p.name)
            measurements.append(
                MeasurementRecord(
                    workload=defn.name,
                    kind=defn.kind,
                    point_id=pt.point_id,
                    execution_mode=defn.execution_mode,
                    protocol_version=defn.protocol_version,
                    requested_lifecycle=defn.lifecycle,
                    actual_lifecycle=actual_lc,
                    status=pr.status,
                    metrics=pr.metrics,
                    attempted=pr.attempted,
                    completed=pr.completed,
                    scoring_denominator=pr.scoring_denominator,
                    model_errors=pr.model_errors,
                    infra_errors=pr.infra_errors,
                    error_message=pr.error_message,
                    artifacts=rels,
                )
            )

        prev_sigs: dict[int, Any] = {}
        in_main = threading.current_thread() is threading.main_thread()
        if in_main:
            for sig in _HANDLED_SIGNALS:
                prev_sigs[sig] = signal.getsignal(sig)
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(
                        sig,
                        lambda s, _f: (_ for _ in ()).throw(
                            _JobCancelled(f"Signal {s}")
                        ),
                    )

        try:
            if (
                job.hardware.gpu
                and hw.gpu_id != "unknown"
                and hw.gpu_id not in job.hardware.gpu
            ):
                raise RuntimeError(
                    f"Observed GPU '{hw.gpu_id}' not in {job.hardware.gpu}."
                )

            served_model = str(
                job.serve_config.get("served-model-name")
                or job.serve_config.get("served_model_name")
                or job.serve_config.get("model", "")
            )

            for w in job.workloads:
                defn = w.definition
                plugin = instantiate_plugin(defn.entrypoint, defn.kind)
                pts = plugin.points(w.params)
                w_out = (
                    bundle_dir
                    / ("evals" if defn.kind == WorkloadKind.EVAL else "bench")
                    / defn.name
                )
                w_out.mkdir(parents=True, exist_ok=True)

                if defn.execution_mode == ExecutionMode.DIRECT:
                    if not _stop_server():
                        break
                    assert isinstance(plugin, BenchmarkRunner)
                    for pt in pts:
                        rem_f = _rem_f()

                        def _run_cmd(
                            cmd: list[str], t: int | None = None
                        ) -> DirectCommandResult:
                            b = _rem_s()
                            return self._launcher.run_direct(
                                cmd, job.serve_env, min(b, t) if t else b
                            )

                        ctx = DirectRunContext(
                            serve_config=dict(job.serve_config),
                            serve_env=dict(job.serve_env),
                            output_dir=w_out,
                            timeout_s=max(1, int(rem_f)),
                            run_command=_run_cmd,
                        )
                        try:
                            pr = _run_with_timeout(
                                lambda p=plugin, pt=pt, c=ctx: p.run_point(pt, c),
                                rem_f,
                            )
                        except TimeoutError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        _record(w, pt, LifecyclePolicy.NONE, pr)
                    continue

                req_lc = defn.lifecycle
                dev_mode = req_lc == LifecyclePolicy.RESET_CACHE
                writer = (
                    DeduplicatedEvalRecordWriter(workload=defn.name, output_dir=w_out)
                    if defn.kind == WorkloadKind.EVAL
                    else None
                )
                try:
                    for idx, pt in enumerate(pts):
                        actual_lc = LifecyclePolicy.NONE
                        if idx > 0 and req_lc == LifecyclePolicy.RESTART_SERVER:
                            if not _stop_server():
                                break
                            handle, _ = _ensure_server(dev_mode)
                            actual_lc = LifecyclePolicy.RESTART_SERVER
                        else:
                            handle, reused = _ensure_server(dev_mode)
                            if req_lc == LifecyclePolicy.RESET_CACHE and (
                                idx > 0 or reused
                            ):
                                try:
                                    self._launcher.reset_cache(handle)
                                    actual_lc = LifecyclePolicy.RESET_CACHE
                                except (
                                    CacheResetUnsupportedError,
                                    ServerTransportError,
                                ) as exc:
                                    msg = f"reset_cache failed: {exc}"
                                    errors.append(msg)
                                    _record(
                                        w,
                                        pt,
                                        LifecyclePolicy.NONE,
                                        PointResult(
                                            status=ExecutionStatus.FAILED,
                                            error_message=msg,
                                        ),
                                    )
                                    break

                        rem_f = _rem_f()
                        s_ctx = ServerRunContext(
                            base_url=handle.base_url,
                            model=served_model,
                            output_dir=w_out,
                            timeout_s=max(1, int(rem_f)),
                        )
                        try:
                            if defn.kind == WorkloadKind.EVAL:
                                assert isinstance(plugin, EvalRunner)
                                assert writer is not None
                                pr = _run_with_timeout(
                                    lambda p=plugin, pt=pt, c=s_ctx, wr=writer: (
                                        p.run_point(pt, c, wr)
                                    ),
                                    rem_f,
                                )
                            else:
                                assert isinstance(plugin, BenchmarkRunner)
                                pr = _run_with_timeout(
                                    lambda p=plugin, pt=pt, c=s_ctx: p.run_point(pt, c),
                                    rem_f,
                                )
                        except TimeoutError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        _record(w, pt, actual_lc, pr)
                finally:
                    if writer is not None:
                        rec_paths = writer.close()
                        root = bundle_dir.resolve()
                        rels = [
                            p.resolve().relative_to(root).as_posix() for p in rec_paths
                        ]
                        for m in measurements:
                            if m.workload == defn.name:
                                m.artifacts.extend(
                                    r for r in rels if r not in m.artifacts
                                )

            if errors or any(
                m.status != ExecutionStatus.COMPLETED for m in measurements
            ):
                status = ExecutionStatus.FAILED
        except _JobCancelled as exc:
            errors.append(str(exc))
            status = ExecutionStatus.CANCELLED
        except TimeoutError as exc:
            errors.append(str(exc))
            status = ExecutionStatus.TIMED_OUT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            status = ExecutionStatus.FAILED
        finally:
            # Block repeat signals during cleanup so the server PGID is never orphaned.
            if in_main:
                for sig in prev_sigs:
                    with contextlib.suppress(ValueError, OSError):
                        signal.signal(sig, signal.SIG_IGN)
            try:
                if not _stop_server() and status == ExecutionStatus.COMPLETED:
                    status = ExecutionStatus.FAILED
            finally:
                if in_main:
                    for sig, prev in prev_sigs.items():
                        with contextlib.suppress(ValueError, OSError):
                            signal.signal(sig, prev)

        if cleanups:
            (logs_dir / "cleanup.json").write_text(
                json.dumps(cleanups, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        result = RunResult(
            job_id=job.job_id,
            attempt_id=attempt_id,
            spec_name=job.spec_name,
            execution_status=status,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_s=max(0.0, time.monotonic() - t0),
            harness_version=HARNESS_VERSION,
            vllm_commit=self._vllm_commit,
            image_digest=self._image_digest,
            requested_hardware=job.hardware,
            observed_hardware=ObservedHardware(**asdict(hw)),
            measurements=measurements,
            errors=errors,
        )
        write_bundle_metadata(
            bundle_dir,
            job=job,
            result=result,
            source_spec_path=source_spec_path,
            image_ref=self._image_ref,
        )
        return result, seal_bundle(bundle_dir, job.job_id, attempt_id)
