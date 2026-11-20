# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C0: Plugin, asset-provider, record-writer, and platform launcher interfaces."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tools.byog.schemas.catalog import PinnedAsset, is_safe_id
from tools.byog.schemas.result import (
    ErrorKind,
    ExecutionStatus,
    ProtocolVariant,
)


class CacheResetUnsupportedError(RuntimeError):
    """Raised ONLY when the server explicitly rejects cache reset (404/405/501)."""


class ServerTransportError(RuntimeError):
    """Raised when a server endpoint fails due to transport/5xx/timeout errors."""


@dataclass(frozen=True)
class ResolvedAsset:
    """Resolved local path and verified provenance for a pinned catalog asset."""

    asset_id: str
    path: Path
    resolved_revision: str
    sha256: str
    is_override: bool = False


class AssetProvider(ABC):
    """Resolves pinned catalog assets without leaking fetch/mirror/auth details."""

    @abstractmethod
    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        """Return local path and verified revision/hash for the pinned asset."""


class FixtureAssetProvider(AssetProvider):
    """Local fixture AssetProvider used in CPU unit/contract tests."""

    def __init__(self, base_dir: Path, overrides: dict[str, Path] | None = None):
        self._base_dir = base_dir
        self._overrides = overrides or {}

    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        is_override = asset.id in self._overrides
        if is_override:
            path = self._overrides[asset.id]
        else:
            candidate = Path(asset.source)
            path = candidate if candidate.is_absolute() else self._base_dir / candidate
        if not path.exists():
            raise FileNotFoundError(
                f"Asset '{asset.id}' not found at '{path}' (source='{asset.source}')."
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not is_override and digest != asset.sha256:
            raise ValueError(
                f"SHA256 mismatch for asset '{asset.id}': "
                f"expected {asset.sha256}, got {digest}"
            )
        return ResolvedAsset(
            asset_id=asset.id,
            path=path,
            resolved_revision="fixture-override" if is_override else asset.revision,
            sha256=digest,
            is_override=is_override,
        )


@dataclass(frozen=True)
class WorkloadPoint:
    """One execution point (e.g., concurrency or case) scheduled by the coordinator."""

    point_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_safe_id(self.point_id):
            raise ValueError(
                f"Invalid WorkloadPoint.point_id {self.point_id!r}: must match "
                "SAFE_ID_PATTERN."
            )


@dataclass(frozen=True)
class DirectCommandResult:
    """Outcome of executing a direct-mode subprocess via the platform launcher."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


@dataclass
class ServerRunContext:
    """Context passed to HTTP server-mode plugins.

    Plugins receive only an endpoint URL and asset resolver; server lifecycle
    (start, reset, restart, cleanup) is owned exclusively by the coordinator.
    """

    base_url: str
    model: str
    assets: AssetProvider
    workload_assets: dict[str, PinnedAsset]
    output_dir: Path
    timeout_s: int


@dataclass
class DirectRunContext:
    """Context passed to direct-mode plugins (no HTTP server is running).

    `run_command` executes commands through the platform launcher using an
    allowlisted environment (credentials stripped) and process-group timeouts.
    """

    serve_config: dict[str, Any]
    serve_env: dict[str, str]
    assets: AssetProvider
    workload_assets: dict[str, PinnedAsset]
    output_dir: Path
    timeout_s: int
    run_command: Callable[[list[str], int | None], DirectCommandResult]


@dataclass
class PointResult:
    """Result returned by a BenchmarkRunner or EvalRunner for one WorkloadPoint.

    Note: `evidence_source` is declared in the trusted catalog, not here.
    """

    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    attempted: int | None = None
    completed: int | None = None
    scoring_denominator: int | None = None
    model_errors: int = 0
    infra_errors: int = 0
    error_message: str | None = None
    artifacts: list[Path] = field(default_factory=list)


class EvalRecordWriter(Protocol):
    """Shared deduplicated writer for eval samples, rendered inputs, and responses."""

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str,
        reference_answer: str,
        rendered_input: Any,
        point_id: str,
        repeat: int,
        concurrency: int,
        model_response: str,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
        output_tokens: int | None = None,
        media_refs: list[str] | None = None,
    ) -> None: ...

    @property
    def written_count(self) -> int: ...

    def close(self) -> list[Path]: ...


class BenchmarkRunner(ABC):
    """Base interface for all BYOG benchmark plugins (HTTP or direct)."""

    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        """Return ordered workload points for the coordinator to schedule."""

    @abstractmethod
    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext | DirectRunContext,
    ) -> PointResult:
        """Execute a single workload point and return normalized metrics/artifacts."""


class EvalRunner(ABC):
    """Base interface for all BYOG eval plugins."""

    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        """Return ordered eval points (e.g., per concurrency level)."""

    @abstractmethod
    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        """Run one eval point, writing per-question records via record_writer."""


@dataclass
class ServerHandle:
    """Opaque handle owned by the coordinator (subprocess or isolated container)."""

    base_url: str
    command: list[str]
    log_path: Path
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlatformAdapter(ABC):
    """Platform-specific accelerator discovery, env, server, and direct execution."""

    @abstractmethod
    def detect_hardware(self) -> dict[str, Any]:
        """Detect visible accelerators and return canonical taxonomy metadata."""

    @abstractmethod
    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle:
        """Start vLLM and block until healthy or raise on failure/timeout."""

    @abstractmethod
    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant:
        """Reset prefix cache; raise CacheResetUnsupportedError only on 404/405/501."""

    @abstractmethod
    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> dict[str, Any]:
        """Terminate server + owned child processes and verify device cleanup."""

    @abstractmethod
    def run_direct(
        self,
        cmd: list[str],
        serve_env: dict[str, str],
        timeout_s: int,
    ) -> DirectCommandResult:
        """Execute a direct-mode command with allowlisted env and PGID timeout."""


# Alias preserved for readability across coordinator/tests.
ServerLauncher = PlatformAdapter
