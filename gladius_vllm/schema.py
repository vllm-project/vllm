"""Shared constants and id-resolution helpers for the GLADIUS control protocol.

Kept in one module so `GladiusScheduler`, `PolicyLoader`, and `TelemetryWriter`
derive `engine_id`/`model_id` identically instead of three independent
implementations that could disagree.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_MAJOR = SCHEMA_VERSION.split(".")[0]

DEFAULT_POLICY_POLL_INTERVAL_MS = 50
DEFAULT_TELEMETRY_SAMPLE_N = 1

POLICY_STATUSES = (
    "active",
    "no_policy",
    "expired",
    "corrupt",
    "rejected_regression",
    "rejected_engine_mismatch",
)

POLICY_SOURCES = ("file", "default")


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 UTC string, tolerating a trailing 'Z' (py3.10 compat)."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_iso8601(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def resolve_engine_id(vllm_config: "VllmConfig") -> str:
    """Stable id for this engine process.

    Prefers an explicit `GLADIUS_ENGINE_ID` env var (recommended for
    deployments that want a stable id across restarts, e.g. matching a pod
    name), else falls back to `kv_transfer_config.engine_id` if disaggregated
    KV transfer is configured, else mints a fresh uuid4 per process.
    """
    env_id = os.environ.get("GLADIUS_ENGINE_ID")
    if env_id:
        return env_id
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_engine_id = getattr(kv_transfer_config, "engine_id", None)
    if kv_engine_id:
        return str(kv_engine_id)
    return f"engine-{uuid.uuid4()}"


def resolve_model_id(vllm_config: "VllmConfig") -> str:
    model_config = vllm_config.model_config
    served_name = getattr(model_config, "served_model_name", None)
    return served_name or model_config.model
