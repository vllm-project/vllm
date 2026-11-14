# SPDX-License-Identifier: Apache-2.0
"""Workload and sweep configuration for the Inco inference benchmark.

The workload is defined once here and in `scripts/workload.env` so that the
server side (vLLM flags) and the client side (aiperf flags) cannot drift apart.
Every field is overridable by an environment variable of the same name,
upper-cased and prefixed with ``INCO_``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Powers of two to 64. Measured KV capacity at ISL+OSL=1280 with the cache
# pinned to 12GiB is ~102 resident requests, so 64 leaves headroom; every value
# is also a captured CUDA graph size, so no point pays padding. Extend to 96 if
# tokens/s/gpu is still climbing at 64 -- a later run merges into the same
# curve, since results are collected from disk by label.
DEFAULT_CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

INCO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_ROOT = str(INCO_ROOT / "results")


def _env(name: str) -> str | None:
    value = os.environ.get(f"INCO_{name.upper()}")
    return value if value not in (None, "") else None


def _as_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"cannot parse {value!r} as a boolean")


def _as_int_tuple(value: str) -> tuple[int, ...]:
    parts = [p for p in value.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("expected at least one integer")
    return tuple(int(p) for p in parts)


def _coerce(raw: str, annotation: Any) -> Any:
    """Coerce an env-var string to the type named by a dataclass annotation.

    Annotations arrive as strings because of ``from __future__ import
    annotations``, so this matches on the textual type name.
    """
    text = str(annotation).replace(" ", "")
    optional = "|None" in text
    if optional:
        if raw.strip().lower() == "none":
            return None
        text = text.replace("|None", "")
    if text == "bool":
        return _as_bool(raw)
    if text == "int":
        return int(raw)
    if text == "float":
        return float(raw)
    if text in ("tuple[int,...]", "tuple[int, ...]"):
        return _as_int_tuple(raw)
    if text == "str":
        return raw
    raise TypeError(f"unsupported config annotation: {annotation!r}")


def _from_env(cls):
    """Build a dataclass, overriding defaults from ``INCO_*`` env vars."""
    kwargs = {}
    for f in fields(cls):
        raw = _env(f.name)
        if raw is not None:
            kwargs[f.name] = _coerce(raw, f.type)
    return cls(**kwargs)


@dataclass(frozen=True)
class Workload:
    """The precise workload under test.

    Defaults describe a mid-length single-turn chat workload: 1024 input
    tokens, 256 output tokens, streaming, against Qwen3-30B-A3B-Instruct
    (a 30.5B-total / 3.3B-active MoE) in bf16 on a single 80GB H100.
    """

    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer: str | None = None
    isl: int = 1024
    osl: int = 256
    isl_stddev: int = 0
    osl_stddev: int = 0
    endpoint_type: str = "chat"
    streaming: bool = True
    ignore_eos: bool = True
    random_seed: int = 100
    num_gpus: int = 1

    def __post_init__(self) -> None:
        for name in ("isl", "osl", "num_gpus"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("isl_stddev", "osl_stddev"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def tokenizer_id(self) -> str:
        return self.tokenizer or self.model

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for this workload."""
        model = self.model.replace("/", "_")
        return f"{model}-isl{self.isl}-osl{self.osl}-tp{self.num_gpus}"

    @classmethod
    def from_env(cls) -> Workload:
        return _from_env(cls)


@dataclass(frozen=True)
class SweepConfig:
    """How the client drives the (already running) server."""

    url: str = "http://localhost:8000"
    label: str = "baseline"
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES
    requests_per_concurrency: int = 8
    min_requests: int = 24
    max_requests: int = 100_000
    warmup_requests: int = 16
    benchmark_duration: float | None = None
    reset_prefix_cache: bool = True
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    aiperf_bin: str = "aiperf"
    ui: str = "simple"
    extra_aiperf_args: str = ""

    def __post_init__(self) -> None:
        if not self.concurrencies:
            raise ValueError("concurrencies must not be empty")
        if any(c < 1 for c in self.concurrencies):
            raise ValueError("concurrencies must all be >= 1")
        if self.min_requests > self.max_requests:
            raise ValueError("min_requests must be <= max_requests")
        if self.requests_per_concurrency < 1:
            raise ValueError("requests_per_concurrency must be >= 1")

    def warmup_count(self, concurrency: int) -> int:
        """Warmup requests to discard before measuring at ``concurrency``.

        At least one full wave at the target batch width. Warmup runs *at* the
        concurrency it is warming, so a fixed count below that concurrency
        warms a narrower CUDA graph and leaves the target width's first-touch
        cost inside the measurement -- a bias that reached 40% of the sample
        at concurrency 400. The configured value is a floor for low
        concurrency, where one wave is too few requests.
        """
        if self.warmup_requests <= 0:
            return 0
        return max(self.warmup_requests, concurrency)

    def request_count(self, concurrency: int) -> int:
        """Requests to send at ``concurrency``.

        Scales with concurrency so every point sees a comparable number of
        steady-state decode iterations, then clamps so that low concurrency
        does not take forever and high concurrency does not burn GPU time.
        """
        scaled = concurrency * self.requests_per_concurrency
        return max(self.min_requests, min(self.max_requests, scaled))

    def artifact_dir(self, workload: Workload, concurrency: int) -> Path:
        return (
            Path(self.artifact_root)
            / self.label
            / workload.slug
            / f"concurrency{concurrency:04d}"
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.artifact_root) / self.label

    @classmethod
    def from_env(cls) -> SweepConfig:
        return _from_env(cls)


@dataclass
class RunManifest:
    """Everything needed to reproduce a sweep, written next to the results."""

    workload: dict[str, Any]
    sweep: dict[str, Any]
    commands: list[list[str]] = field(default_factory=list)
    server_info: dict[str, Any] | None = None
    notes: str = ""

    @classmethod
    def create(cls, workload: Workload, sweep: SweepConfig) -> RunManifest:
        return cls(workload=asdict(workload), sweep=asdict(sweep))

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["sweep"]["concurrencies"] = list(payload["sweep"]["concurrencies"])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path
