# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BYOG runtime boundaries (design doc C0): plugins, platform launcher, eval writer.

Reading map (top-down by boundary):
  1. Workload plugins  -- `BenchmarkRunner`, `EvalRunner`, and the point/context
                          types passed between the coordinator (C4) and plugins.
  2. Platform launcher -- `PlatformAdapter` (NVIDIA/ROCm) plus its typed hardware,
                          cleanup, direct-command records, and transport errors.
  3. Eval recording    -- `EvalRecordWriter` (per-question JSONL writer in C5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from tools.byog.schemas.execution import ErrorKind, ExecutionStatus, is_safe_id

# ---------------------------------------------------------------------------
# 1. Workload plugins (C4 Coordinator -> Benchmark / Eval plugins)
# ---------------------------------------------------------------------------


class BenchmarkRunner(ABC):
    """Plugin interface for server-mode (HTTP) or direct-mode benchmarks."""

    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        """Expand workload params into ordered execution points (e.g. `c1`, `c4`)."""

    @abstractmethod
    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult:
        """Run one benchmark point and return normalized metrics and artifacts."""


class EvalRunner(ABC):
    """Plugin interface for accuracy evals running against a vLLM HTTP server."""

    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        """Expand eval params into ordered points (e.g. per concurrency level)."""

    @abstractmethod
    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        """Run one eval point, recording every question via `record_writer`."""


@dataclass(frozen=True)
class WorkloadPoint:
    """One execution point (e.g. `c1`, `c4`, `prefill_128`) scheduled by C4."""

    point_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_safe_id(self.point_id):
            raise ValueError(f"Invalid point_id {self.point_id!r}.")


@dataclass
class ServerRunContext:
    """Context for server-mode plugins; exposes `base_url`, never `ServerHandle`.

    The coordinator (C4) owns server startup, cache reset, and teardown so plugins
    cannot bypass lifecycle isolation or leak GPU processes.
    """

    base_url: str
    model: str
    output_dir: Path
    timeout_s: int


@dataclass
class DirectRunContext:
    """Context for direct-mode plugins (e.g. `vllm bench forward`, no HTTP server).

    Plugins invoke candidate commands through `run_command` rather than spawning
    subprocesses directly so env allowlisting and PGID timeout cleanup apply.
    """

    serve_config: dict[str, Any]
    serve_env: dict[str, str]
    output_dir: Path
    timeout_s: int
    run_command: Callable[[list[str], int | None], DirectCommandResult]


@dataclass
class PointResult:
    """Normalized metrics, request/sample counts, and artifacts for one point."""

    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    attempted: int | None = None
    completed: int | None = None
    scoring_denominator: int | None = None
    model_errors: int = 0
    infra_errors: int = 0
    error_message: str | None = None
    artifacts: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Platform launcher (C4 Coordinator -> NVIDIA / ROCm adapters)
# ---------------------------------------------------------------------------


class PlatformAdapter(ABC):
    """Hardware-specific server launch, cache reset, cleanup, and direct exec.

    Phase 0 ships a local-subprocess NVIDIA adapter. The interface only exchanges
    configs, a `ServerHandle` with a `base_url`, and typed results, so a later
    adapter can run the candidate in a separate container (for PR isolation)
    without changing the coordinator or plugins.
    """

    @abstractmethod
    def detect_hardware(self) -> DetectedHardware:
        """Inspect visible GPUs and return canonical hardware telemetry."""

    @abstractmethod
    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        # Enables `/reset_prefix_cache` when a workload requests `reset_cache`.
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        """Launch `vllm serve` and wait until ready, or clean up and raise."""

    @abstractmethod
    def reset_cache(self, handle: ServerHandle) -> None:
        """Reset prefix cache or raise; coordinator owns any fallback policy."""

    @abstractmethod
    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> CleanupResult:
        """Terminate the server process group (`PGID`) and verify VRAM release."""

    @abstractmethod
    def run_direct(
        self, cmd: list[str], serve_env: dict[str, str], timeout_s: int
    ) -> DirectCommandResult:
        """Run a direct-mode command with an allowlisted env and PGID timeout."""


@dataclass
class ServerHandle:
    """Opaque handle to a running `vllm serve` process owned strictly by C4."""

    base_url: str
    command: list[str]
    log_path: Path
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedHardware:
    """Typed GPU telemetry used for hardware checks and result provenance."""

    platform: Literal["nvidia", "rocm"]
    gpu_id: str
    gpu_count: int
    gpu_name: str | None = None
    driver_version: str | None = None
    device_memory_used_mib: list[int] | None = None

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError(f"gpu_count must be >= 0, got {self.gpu_count}.")


@dataclass(frozen=True)
class CleanupResult:
    """Typed outcome of stopping a server and checking GPU VRAM release.

    `vram_reclaimed` is True when memory returned to baseline, False when it did
    not, and None when it could not be checked (e.g. no `nvidia-smi` or MIG
    reporting N/A). Callers that gate on cleanup, such as CI, must treat None as
    a failure rather than a pass.
    """

    terminated_pgid: bool
    vram_reclaimed: bool | None
    baseline_memory_mib: list[int] | None = None
    after_memory_mib: list[int] | None = None


@dataclass(frozen=True)
class DirectCommandResult:
    """Subprocess exit status, output, and timing from `PlatformAdapter.run_direct`."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


class CacheResetUnsupportedError(RuntimeError):
    """Raised ONLY when `/reset_prefix_cache` returns HTTP 404/405/501."""


class ServerTransportError(RuntimeError):
    """Raised on connection, 5xx, or timeout errors talking to the vLLM server."""


# ---------------------------------------------------------------------------
# 3. Eval record writer (implemented by Result packaging, C5)
# ---------------------------------------------------------------------------


class EvalRecordWriter(Protocol):
    """Writes deduplicated eval JSONL (`samples`, `inputs`, `responses`)."""

    @property
    def written_count(self) -> int:
        """Number of response records written so far."""

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str | None,
        reference_answer: str | None,
        rendered_input: Any,
        point_id: str,
        repeat: int,
        concurrency: int,
        model_response: str | None,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
    ) -> None:
        """Record one scored model response; C5 handles prompt/sample deduplication."""

    def close(self) -> list[Path]:
        """Flush and close all JSONL files, returning their paths."""
