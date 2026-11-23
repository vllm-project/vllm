# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for C4 coordinator, lifecycle isolation, timeouts, and NVIDIA adapter."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.byog.fake_launcher import FakeLauncher
from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.bundle import validate_bundle
from tools.byog.executor.coordinator import Coordinator
from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    CacheResetUnsupportedError,
    DirectRunContext,
    EvalRecordWriter,
    EvalRunner,
    FixtureAssetProvider,
    PointResult,
    ServerHandle,
    ServerRunContext,
    ServerTransportError,
    WorkloadPoint,
)
from tools.byog.executor.platform.nvidia import (
    NvidiaServerLauncher,
    _build_subprocess_env,
    build_vllm_serve_command,
)
from tools.byog.schemas.catalog import (
    PublicationPolicy,
    QualityThreshold,
)
from tools.byog.schemas.result import (
    ExecutionStatus,
    ProtocolVariant,
    QualityState,
)
from tools.byog.schemas.spec import HardwareSpec, ResolvedJob, ResolvedWorkload

VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


class TwoPointBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("p1"), WorkloadPoint("p2")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        return PointResult(metrics={"val": 1.0}, attempted=1, completed=1)


class HangingBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("hang")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        time.sleep(5.0)
        return PointResult(metrics={"val": 1.0})


class SigtermBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("sig")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
        return PointResult(metrics={"val": 1.0})


class DirectEnvCheckBench(BenchmarkRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("env_check")]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, DirectRunContext)
        res = ctx.run_command(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('HF_TOKEN', 'CLEAN'))",
            ],
            5,
        )
        return PointResult(
            metrics={"hf_token_seen": res.stdout.strip()},
            attempted=1,
            completed=1,
        )


class SimpleEval(EvalRunner):
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("c1", {"accuracy": params.get("accuracy", 0.8)})]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        acc = float(point.params["accuracy"])
        record_writer.write_response(
            dataset_id="mini",
            dataset_revision="r1",
            sample_id="s-0",
            question="1+1?",
            reference_answer="2",
            rendered_input="1+1?",
            point_id=point.point_id,
            repeat=0,
            concurrency=1,
            model_response="2",
            extracted_answer="2",
            score=acc,
            correct=acc >= 0.5,
        )
        return PointResult(
            metrics={"accuracy": acc},
            attempted=1,
            completed=1,
            scoring_denominator=1,
        )


def _job_for(
    workloads: list[ResolvedWorkload],
    *,
    timeout_s: int = 30,
    serve_env: dict[str, str] | None = None,
    gpu: list[str] | None = None,
) -> ResolvedJob:
    return ResolvedJob(
        job_id="default",
        attempt_id="att-exec-1",
        serve_config={"model": "org/m", "revision": VALID_SHA},
        serve_env=serve_env or {},
        hardware=HardwareSpec(gpu=gpu or ["h200"], count=1),
        workloads=workloads,
        timeout_s=timeout_s,
    )


def test_lifecycle_no_silent_fallback_vs_opt_in_and_transport_error(
    tmp_path: Path,
) -> None:
    w_strict = ResolvedWorkload(
        name="b_strict",
        kind="benchmark",
        execution_mode="server",
        lifecycle="reset_cache",
        allow_lifecycle_fallback=False,
        entrypoint=f"{__name__}:TwoPointBench",
        protocol_version="1.0",
    )
    # 1. Unsupported reset_cache WITHOUT allow_lifecycle_fallback -> FAILED.
    launcher1 = FakeLauncher(allow_reset_cache=False)
    res1, tar1 = Coordinator(
        launcher=launcher1, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_strict]), tmp_path / "out1")
    assert res1.execution_status == ExecutionStatus.FAILED
    assert launcher1.last_enable_dev_mode is True
    assert launcher1.stops == launcher1.starts
    validate_bundle(tar1)

    # 2. Unsupported reset_cache WITH allow_lifecycle_fallback -> RESTART_SERVER.
    w_fallback = w_strict.model_copy(update={"allow_lifecycle_fallback": True})
    launcher2 = FakeLauncher(allow_reset_cache=False)
    res2, tar2 = Coordinator(
        launcher=launcher2, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_fallback]), tmp_path / "out2")
    assert res2.execution_status == ExecutionStatus.COMPLETED
    assert res2.measurements[1].protocol_variant == ProtocolVariant.RESTART_SERVER
    assert res2.measurements[1].lifecycle_fallback_reason is not None
    validate_bundle(tar2)

    # 3. Transport error during reset_cache ALWAYS fails, even with fallback!
    launcher3 = FakeLauncher(fail_reset_transport=True)
    res3, _ = Coordinator(
        launcher=launcher3, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_fallback]), tmp_path / "out3")
    assert res3.execution_status == ExecutionStatus.FAILED


def test_deadline_enforcement_and_sigterm_cancellation(tmp_path: Path) -> None:
    # Hanging point exceeds job timeout_s=1 -> TIMED_OUT and server cleaned up.
    w_hang = ResolvedWorkload(
        name="hang",
        kind="benchmark",
        execution_mode="server",
        lifecycle="none",
        entrypoint=f"{__name__}:HangingBench",
        protocol_version="1.0",
    )
    launcher = FakeLauncher()
    res, tar = Coordinator(
        launcher=launcher, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_hang], timeout_s=1), tmp_path / "out_to")
    assert res.execution_status == ExecutionStatus.TIMED_OUT
    assert launcher.stops == launcher.starts
    validate_bundle(tar)

    # Startup timeout -> TIMED_OUT.
    launcher_to = FakeLauncher(fail_start_timeout=True)
    res_st, _ = Coordinator(
        launcher=launcher_to, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_hang], timeout_s=5), tmp_path / "out_st")
    assert res_st.execution_status == ExecutionStatus.TIMED_OUT

    # SIGTERM cancels job cleanly and seals CANCELLED bundle.
    w_sig = ResolvedWorkload(
        name="sig",
        kind="benchmark",
        execution_mode="server",
        lifecycle="none",
        entrypoint=f"{__name__}:SigtermBench",
        protocol_version="1.0",
    )
    launcher_sig = FakeLauncher()
    res_sig, tar_sig = Coordinator(
        launcher=launcher_sig, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_sig], timeout_s=10), tmp_path / "out_sig")
    assert res_sig.execution_status == ExecutionStatus.CANCELLED
    assert launcher_sig.stops == launcher_sig.starts
    validate_bundle(tar_sig)


def test_dirty_cleanup_aborts_next_workload_and_hw_mismatch(tmp_path: Path) -> None:
    w1 = ResolvedWorkload(
        name="b1",
        kind="benchmark",
        execution_mode="server",
        lifecycle="none",
        entrypoint=f"{__name__}:TwoPointBench",
        protocol_version="1.0",
    )
    w2 = ResolvedWorkload(
        name="d1",
        kind="benchmark",
        execution_mode="direct",
        lifecycle="none",
        entrypoint=f"{__name__}:DirectEnvCheckBench",
        protocol_version="1.0",
    )
    # Failed VRAM reclaim aborts direct workload d1.
    launcher_dirty = FakeLauncher(vram_reclaimed=False)
    res, _ = Coordinator(
        launcher=launcher_dirty, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w1, w2]), tmp_path / "out_dirty")
    assert res.execution_status == ExecutionStatus.FAILED
    assert launcher_dirty.direct_calls == 0

    # Hardware mismatch rejected before launching server.
    launcher_hw = FakeLauncher()
    res_hw, _ = Coordinator(
        launcher=launcher_hw, assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w1], gpu=["b200"]), tmp_path / "out_hw")
    assert res_hw.execution_status == ExecutionStatus.FAILED
    assert launcher_hw.starts == 0


def test_quality_states_and_compatibility_key_scoping(tmp_path: Path) -> None:
    pub = PublicationPolicy(
        reviewed=True,
        allow_question_text=True,
        allow_reference_answer=True,
        allow_model_response=True,
    )
    # 1. No thresholds -> INCONCLUSIVE (never PASSED!).
    w_no_th = ResolvedWorkload(
        name="ev",
        kind="eval",
        execution_mode="server",
        lifecycle="none",
        entrypoint=f"{__name__}:SimpleEval",
        protocol_version="1.0",
        publication_policy=pub,
        params={"accuracy": 1.0},
    )
    res_inc, _ = Coordinator(
        launcher=FakeLauncher(), assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_no_th]), tmp_path / "q1")
    assert res_inc.quality_state == QualityState.INCONCLUSIVE

    # 2. Spec-only threshold cannot authorize PASSED (stays INCONCLUSIVE).
    w_spec_only = w_no_th.model_copy(
        update={
            "quality_thresholds": [
                QualityThreshold(metric="accuracy", min_value=0.0, source="spec")
            ],
            "params": {"accuracy": 0.9},
        }
    )
    res_spec_only, _ = Coordinator(
        launcher=FakeLauncher(), assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_spec_only]), tmp_path / "q1b")
    assert res_spec_only.quality_state == QualityState.INCONCLUSIVE

    # 3. Passing required catalog threshold -> PASSED.
    w_pass = w_no_th.model_copy(
        update={
            "required_for_quality": True,
            "quality_thresholds": [
                QualityThreshold(metric="accuracy", min_value=0.7, source="catalog")
            ],
            "params": {"accuracy": 0.9},
        }
    )
    res_pass, _ = Coordinator(
        launcher=FakeLauncher(),
        assets=FixtureAssetProvider(tmp_path),
        vllm_commit=VALID_SHA,
    ).run(
        _job_for([w_pass], serve_env={"VLLM_LOGGING_LEVEL": "DEBUG"}),
        tmp_path / "q2",
    )
    assert res_pass.quality_state == QualityState.PASSED
    assert res_pass.vllm_commit == VALID_SHA

    # Non-functional env var VLLM_LOGGING_LEVEL must NOT alter compatibility_key.
    res_pass2, _ = Coordinator(
        launcher=FakeLauncher(), assets=FixtureAssetProvider(tmp_path)
    ).run(
        _job_for([w_pass], serve_env={"VLLM_LOGGING_LEVEL": "WARNING"}),
        tmp_path / "q2b",
    )
    assert (
        res_pass.measurements[0].compatibility_key
        == res_pass2.measurements[0].compatibility_key
    )

    # 3. Failing threshold -> FAILED.
    w_fail = w_pass.model_copy(update={"params": {"accuracy": 0.3}})
    res_fail, _ = Coordinator(
        launcher=FakeLauncher(), assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_fail]), tmp_path / "q3")
    assert res_fail.quality_state == QualityState.FAILED


def test_direct_env_isolation_and_nvidia_adapter_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "super-secret-hf-token")
    w_dir = ResolvedWorkload(
        name="d_env",
        kind="benchmark",
        execution_mode="direct",
        lifecycle="none",
        entrypoint=f"{__name__}:DirectEnvCheckBench",
        protocol_version="1.0",
    )
    res, _ = Coordinator(
        launcher=FakeLauncher(), assets=FixtureAssetProvider(tmp_path)
    ).run(_job_for([w_dir]), tmp_path / "d_out")
    assert res.measurements[0].metrics["hf_token_seen"] == "CLEAN"

    # NvidiaServerLauncher env allowlist & dev-mode toggle.
    env_off = _build_subprocess_env({}, enable_dev_mode=False, allow_hf_token=False)
    assert "HF_TOKEN" not in env_off
    assert "VLLM_SERVER_DEV_MODE" not in env_off
    env_on = _build_subprocess_env({}, enable_dev_mode=True, allow_hf_token=False)
    assert env_on["VLLM_SERVER_DEV_MODE"] == "1"

    # build_vllm_serve_command formatting.
    cmd = build_vllm_serve_command(
        {"model": "org/m", "max_model_len": 1024, "enforce_eager": True},
        port=9000,
    )
    assert cmd[:6] == ["vllm", "serve", "org/m", "--host", "127.0.0.1", "--port"]
    assert "--max-model-len" in cmd and "--enforce-eager" in cmd

    # reset_cache distinguishes HTTP 404 (unsupported) from HTTP 500 (transport error).
    adapter = NvidiaServerLauncher()
    with FakeVLLMServer(
        FakeServerState(allow_reset_cache=False, reset_error_code=404)
    ) as srv404:
        h404 = ServerHandle(base_url=srv404.base_url, command=[], log_path=tmp_path)
        with pytest.raises(CacheResetUnsupportedError):
            adapter.reset_cache(h404)
    with FakeVLLMServer(
        FakeServerState(allow_reset_cache=False, reset_error_code=500)
    ) as srv500:
        h500 = ServerHandle(base_url=srv500.base_url, command=[], log_path=tmp_path)
        with pytest.raises(ServerTransportError):
            adapter.reset_cache(h500)
