# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FakeLauncher, test-only dummy plugins, and `assert_plugin_contract()` helper."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from tests.byog.fake_server import FakeServerState, FakeVLLMServer
from tools.byog.bundle import validate_bundle
from tools.byog.catalog import Catalog
from tools.byog.executor.coordinator import Coordinator
from tools.byog.interfaces import (
    BenchmarkRunner,
    CacheResetUnsupportedError,
    CleanupResult,
    DetectedHardware,
    DirectCommandResult,
    DirectRunContext,
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
    RunSpec,
    WorkloadKind,
)
from tools.byog.schemas.results import RunResult
from tools.byog.spec import resolve_spec

_DIRECT_ENV = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH")
_TEST_SHA = "0123456789abcdef0123456789abcdef01234567"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TEST_CATALOG_DIR = FIXTURES_DIR / "catalog"


class DummyServerBenchmark(BenchmarkRunner):
    """Test-only HTTP benchmark plugin exercising `ServerRunContext`."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concs = params.get("concurrency", [1])
        return [
            WorkloadPoint(f"c{int(c)}", {**params, "concurrency": int(c)})
            for c in concs
        ]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, ServerRunContext)
        sleep_s = float(point.params.get("sleep_s", 0.0))
        if sleep_s > 0:
            time.sleep(sleep_s)
        body = json.dumps(
            {"model": ctx.model, "prompt": "hi", "max_tokens": 4, "temperature": 0.0}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{ctx.base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            json.loads(resp.read().decode("utf-8"))
        return PointResult(
            status=ExecutionStatus.COMPLETED,
            metrics={"throughput_req_s": 10.0},
            attempted=1,
            completed=1,
        )


class DummyDirectBenchmark(BenchmarkRunner):
    """Test-only direct benchmark plugin exercising `DirectRunContext`."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        return [WorkloadPoint("prefill_128", dict(params))]

    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        assert isinstance(ctx, DirectRunContext)
        code = (
            "import json, os; "
            "print(json.dumps({'hf_token_leaked': int('HF_TOKEN' in os.environ)}))"
        )
        res = ctx.run_command([sys.executable, "-c", code], ctx.timeout_s)
        parsed = json.loads(res.stdout.strip())
        return PointResult(
            status=ExecutionStatus.COMPLETED,
            metrics={"hf_token_leaked": int(parsed["hf_token_leaked"])},
            attempted=1,
            completed=1,
        )


def make_offline_gsm8k_catalog(tmp_path: Path) -> Path:
    """Write a temp catalog dir with `gsm8k.yaml` pinned to local mini fixtures."""
    cat_dir = tmp_path / "offline_catalog"
    (cat_dir / "workloads").mkdir(parents=True, exist_ok=True)
    train_p = FIXTURES_DIR / "gsm8k_mini_train.jsonl"
    test_p = FIXTURES_DIR / "gsm8k_mini_test.jsonl"
    prod_gsm8k = Catalog.discover().get_workload("gsm8k").model_dump(mode="json")
    prod_gsm8k["defaults"]["train_source"] = str(train_p)
    prod_gsm8k["defaults"]["train_sha256"] = hashlib.sha256(
        train_p.read_bytes()
    ).hexdigest()
    prod_gsm8k["defaults"]["test_source"] = str(test_p)
    prod_gsm8k["defaults"]["test_sha256"] = hashlib.sha256(
        test_p.read_bytes()
    ).hexdigest()
    (cat_dir / "workloads" / "gsm8k.yaml").write_text(
        yaml.safe_dump(prod_gsm8k, sort_keys=False), encoding="utf-8"
    )
    return cat_dir


class FakeLauncher(PlatformAdapter):
    """In-process PlatformAdapter backed by `FakeVLLMServer`."""

    def __init__(
        self,
        *,
        allow_reset_cache: bool = True,
        fail_reset_transport: bool = False,
        fail_on_calls: set[int] | None = None,
        wrong_answer_mode: bool = False,
        vram_reclaimed: bool | None = True,
    ):
        self.allow_reset_cache = allow_reset_cache
        self.fail_reset_transport = fail_reset_transport
        self.fail_on_calls = set(fail_on_calls or set())
        self.wrong_answer_mode = wrong_answer_mode
        self.vram_reclaimed = vram_reclaimed
        self.starts = self.stops = self.resets = self.direct_calls = 0
        self.states: list[FakeServerState] = []
        self._active: dict[str, FakeVLLMServer] = {}

    @property
    def total_requests(self) -> int:
        return sum(s.calls for s in self.states)

    def detect_hardware(self) -> DetectedHardware:
        return DetectedHardware(
            platform="nvidia",
            gpu_id="h200",
            gpu_count=1,
            gpu_name="NVIDIA H200 (fake)",
            driver_version="570.00-fake",
            device_memory_used_mib=[0],
        )

    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        self.starts += 1
        model = str(serve_config.get("model", "fake-model"))
        st = FakeServerState(
            model_name=model,
            allow_reset_cache=self.allow_reset_cache and enable_dev_mode,
            fail_on_calls=set(self.fail_on_calls),
            wrong_answer_mode=self.wrong_answer_mode,
        )
        self.states.append(st)
        srv = FakeVLLMServer(st)
        url = srv.start()
        self._active[url] = srv
        log_path = log_dir / f"fake_{self.starts}.log"
        log_path.write_text("fake server\n", encoding="utf-8")
        return ServerHandle(base_url=url, command=["fake"], log_path=log_path)

    def reset_cache(self, handle: ServerHandle) -> None:
        self.resets += 1
        if self.fail_reset_transport:
            raise ServerTransportError("simulated transport error")
        srv = self._active.get(handle.base_url)
        if srv is None or not srv.state.allow_reset_cache:
            raise CacheResetUnsupportedError("HTTP 404")
        srv.state.reset_calls += 1

    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> CleanupResult:
        self.stops += 1
        if srv := self._active.pop(handle.base_url, None):
            srv.stop()
        return CleanupResult(
            terminated_pgid=True,
            vram_reclaimed=self.vram_reclaimed,
            baseline_memory_mib=baseline_memory_mib,
        )

    def run_direct(
        self, cmd: list[str], serve_env: dict[str, str], timeout_s: int
    ) -> DirectCommandResult:
        self.direct_calls += 1
        env = {k: v for k, v in os.environ.items() if k in _DIRECT_ENV}
        env.update(serve_env)
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return DirectCommandResult(
            proc.returncode, proc.stdout, proc.stderr, time.monotonic() - t0
        )


def assert_plugin_contract(
    *,
    workload_name: str,
    tmp_path: Path,
    catalog: Catalog | None = None,
    launcher: FakeLauncher | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[RunResult, Path]:
    """Run one plugin against `FakeLauncher` and verify end-to-end contract."""
    cat = catalog or Catalog.discover()
    defn = cat.get_workload(workload_name)
    key = "benchmarks" if defn.kind == WorkloadKind.BENCHMARK else "evals"
    id_field = "type" if defn.kind == WorkloadKind.BENCHMARK else "name"
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "fake/m", "revision": _TEST_SHA}},
            "hardware": {"gpu": ["h200"]},
            key: [{id_field: workload_name, **(overrides or {})}],
        }
    )
    job = resolve_spec(spec, cat)[0]
    active_launcher = launcher or FakeLauncher()
    res, archive = Coordinator(launcher=active_launcher).run(job, tmp_path / "out")

    assert res.execution_status == ExecutionStatus.COMPLETED, res.errors
    assert res.measurements and active_launcher.stops == active_launcher.starts
    for m in res.measurements:
        assert m.metrics and (m.attempted or 0) >= 1 and (m.completed or 0) >= 1
    if defn.execution_mode == ExecutionMode.SERVER:
        assert active_launcher.total_requests >= 1
    else:
        assert active_launcher.direct_calls >= 1

    validate_bundle(archive)
    if defn.kind == WorkloadKind.EVAL:
        with tarfile.open(archive, "r:gz") as tar:
            fh = tar.extractfile(f"evals/{workload_name}/responses.jsonl")
            assert fh is not None
            resp_txt = fh.read().decode()
            assert len(resp_txt.splitlines()) == sum(
                m.attempted or 0 for m in res.measurements
            )
    return res, archive
