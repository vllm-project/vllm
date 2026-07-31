"""Atomic policy-snapshot hot-reload for GladiusScheduler.

PolicyLoader.poll() is the single entry point: it never raises, and always
returns a PolicyDecision usable directly by the scheduler. See the
failure-mode table in the design plan for the exact mapping from file state
to (decision, status) pairs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from gladius_vllm.errors import (
    PolicyCorruptError,
    PolicyEngineMismatchError,
    PolicyStaleError,
)
from gladius_vllm.schema import (
    DEFAULT_POLICY_POLL_INTERVAL_MS,
    SUPPORTED_SCHEMA_MAJOR,
    parse_iso8601,
)

PolicyStatus = Literal[
    "active",
    "no_policy",
    "expired",
    "corrupt",
    "rejected_regression",
    "rejected_engine_mismatch",
]

PolicySource = Literal["file", "default"]

_REQUIRED_FIELDS = (
    "schema_version",
    "generation",
    "policy_id",
    "model_id",
    "engine_id",
    "created_at",
    "expires_at",
)
_ADMISSION_ALLOWED_KEYS = {"max_num_seqs", "max_num_batched_tokens"}


@dataclass(frozen=True)
class PolicyDecision:
    """The scheduler's per-step admission ceilings and their provenance."""

    max_num_seqs: int
    max_num_batched_tokens: int
    policy_id: str | None
    generation: int | None
    status: PolicyStatus
    source: PolicySource


def _default_decision(
    startup_max_num_seqs: int,
    startup_max_num_batched_tokens: int,
    status: PolicyStatus,
) -> PolicyDecision:
    return PolicyDecision(
        max_num_seqs=startup_max_num_seqs,
        max_num_batched_tokens=startup_max_num_batched_tokens,
        policy_id=None,
        generation=None,
        status=status,
        source="default",
    )


def _parse_snapshot(
    raw: dict, *, engine_id: str, model_id: str, last_accepted_generation: int | None
) -> dict:
    """Validate and return the parsed fields.

    Raises PolicyCorruptError, PolicyEngineMismatchError, or PolicyStaleError.
    """
    for field in _REQUIRED_FIELDS:
        if field not in raw:
            raise PolicyCorruptError(f"missing required field: {field}")

    schema_version = raw["schema_version"]
    if not isinstance(schema_version, str):
        raise PolicyCorruptError("schema_version must be a string")
    if schema_version.split(".")[0] != SUPPORTED_SCHEMA_MAJOR:
        raise PolicyCorruptError(
            f"unsupported schema_version major: {schema_version!r} "
            f"(expected major {SUPPORTED_SCHEMA_MAJOR!r})"
        )

    generation = raw["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise PolicyCorruptError(f"generation must be a non-negative int, got {generation!r}")

    policy_id = raw["policy_id"]
    if not isinstance(policy_id, str) or not policy_id:
        raise PolicyCorruptError("policy_id must be a non-empty string")

    if raw["model_id"] != model_id:
        raise PolicyEngineMismatchError(
            f"model_id mismatch: snapshot={raw['model_id']!r} engine={model_id!r}"
        )
    if raw["engine_id"] != engine_id:
        raise PolicyEngineMismatchError(
            f"engine_id mismatch: snapshot={raw['engine_id']!r} engine={engine_id!r}"
        )

    try:
        created_at = parse_iso8601(raw["created_at"])
        expires_at = parse_iso8601(raw["expires_at"])
    except (TypeError, ValueError) as exc:
        raise PolicyCorruptError(f"invalid timestamp: {exc}") from exc
    if expires_at <= created_at:
        raise PolicyCorruptError("expires_at must be strictly after created_at")

    admission = raw.get("admission") or {}
    if not isinstance(admission, dict):
        raise PolicyCorruptError("admission must be an object")
    unknown_keys = set(admission) - _ADMISSION_ALLOWED_KEYS
    if unknown_keys:
        raise PolicyCorruptError(f"unrecognized admission keys: {sorted(unknown_keys)}")

    max_num_seqs = admission.get("max_num_seqs")
    if max_num_seqs is not None:
        if not isinstance(max_num_seqs, int) or isinstance(max_num_seqs, bool) or max_num_seqs < 1:
            raise PolicyCorruptError(
                f"admission.max_num_seqs must be a positive int, got {max_num_seqs!r}"
            )

    max_num_batched_tokens = admission.get("max_num_batched_tokens")
    if max_num_batched_tokens is not None:
        if (
            not isinstance(max_num_batched_tokens, int)
            or isinstance(max_num_batched_tokens, bool)
            or max_num_batched_tokens < 1
        ):
            raise PolicyCorruptError(
                "admission.max_num_batched_tokens must be a positive int, "
                f"got {max_num_batched_tokens!r}"
            )

    if last_accepted_generation is not None and generation <= last_accepted_generation:
        raise PolicyStaleError(
            f"generation {generation} <= last accepted {last_accepted_generation}"
        )

    return {
        "generation": generation,
        "policy_id": policy_id,
        "expires_at": expires_at,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }


class PolicyLoader:
    """Polls `policy_snapshot.json` and returns hot-reloadable admission ceilings.

    Startup ceilings act as both the "no policy" default and the hard upper
    clamp — callers pass them in and get them back verbatim whenever no
    policy is active, expired, or rejected.
    """

    def __init__(
        self,
        snapshot_path: Path | None,
        engine_id: str,
        model_id: str,
        startup_max_num_seqs: int,
        startup_max_num_batched_tokens: int,
        poll_interval_ms: int = DEFAULT_POLICY_POLL_INTERVAL_MS,
    ) -> None:
        self._snapshot_path = snapshot_path
        self._engine_id = engine_id
        self._model_id = model_id
        self._startup_max_num_seqs = startup_max_num_seqs
        self._startup_max_num_batched_tokens = startup_max_num_batched_tokens
        self._poll_interval_s = poll_interval_ms / 1000.0

        self._last_stat: tuple[int, int] | None = None  # (mtime_ns, size)
        self._last_poll_monotonic: float | None = None
        self._last_accepted_generation: int | None = None
        self._last_accepted_policy_id: str | None = None
        self._last_accepted_expires_at: datetime | None = None
        self._last_accepted_max_num_seqs: int | None = None
        self._last_accepted_max_num_batched_tokens: int | None = None
        self._last_decision: PolicyDecision = _default_decision(
            startup_max_num_seqs, startup_max_num_batched_tokens, "no_policy"
        )

    def poll(self) -> PolicyDecision:
        if self._snapshot_path is None:
            return self._default("no_policy")

        now_monotonic = time.monotonic()
        if (
            self._last_poll_monotonic is not None
            and now_monotonic - self._last_poll_monotonic < self._poll_interval_s
        ):
            return self._check_expiry_only()
        self._last_poll_monotonic = now_monotonic

        try:
            stat = self._snapshot_path.stat()
        except FileNotFoundError:
            self._last_stat = None
            return self._default("no_policy")

        current_stat = (stat.st_mtime_ns, stat.st_size)
        if current_stat == self._last_stat:
            return self._check_expiry_only()

        self._last_stat = current_stat
        try:
            raw_text = self._snapshot_path.read_text()
            raw = json.loads(raw_text)
            parsed = _parse_snapshot(
                raw,
                engine_id=self._engine_id,
                model_id=self._model_id,
                last_accepted_generation=self._last_accepted_generation,
            )
        except (OSError, json.JSONDecodeError, PolicyCorruptError):
            return self._reject_keep_last_or_default("corrupt")
        except PolicyEngineMismatchError:
            return self._reject_keep_last_or_default("rejected_engine_mismatch")
        except PolicyStaleError:
            return self._reject_keep_last_or_default("rejected_regression")

        self._last_accepted_generation = parsed["generation"]
        self._last_accepted_policy_id = parsed["policy_id"]
        self._last_accepted_expires_at = parsed["expires_at"]
        self._last_accepted_max_num_seqs = (
            parsed["max_num_seqs"]
            if parsed["max_num_seqs"] is not None
            else self._startup_max_num_seqs
        )
        self._last_accepted_max_num_batched_tokens = (
            parsed["max_num_batched_tokens"]
            if parsed["max_num_batched_tokens"] is not None
            else self._startup_max_num_batched_tokens
        )
        self._last_decision = PolicyDecision(
            max_num_seqs=self._last_accepted_max_num_seqs,
            max_num_batched_tokens=self._last_accepted_max_num_batched_tokens,
            policy_id=self._last_accepted_policy_id,
            generation=self._last_accepted_generation,
            status="active",
            source="file",
        )
        return self._check_expiry_only()

    def _check_expiry_only(self) -> PolicyDecision:
        """Re-check expiry every call, independent of whether the file changed."""
        if self._last_accepted_expires_at is None:
            return self._last_decision
        if self._is_expired():
            return self._default("expired")
        return self._last_decision

    def _is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self._last_accepted_expires_at

    def _reject_keep_last_or_default(self, status: PolicyStatus) -> PolicyDecision:
        if self._last_accepted_generation is None:
            decision = self._default(status)
        else:
            decision = PolicyDecision(
                max_num_seqs=self._last_accepted_max_num_seqs,
                max_num_batched_tokens=self._last_accepted_max_num_batched_tokens,
                policy_id=self._last_accepted_policy_id,
                generation=self._last_accepted_generation,
                status=status,
                source="file",
            )
        self._last_decision = decision
        return decision

    def _default(self, status: PolicyStatus) -> PolicyDecision:
        decision = _default_decision(
            self._startup_max_num_seqs, self._startup_max_num_batched_tokens, status
        )
        self._last_decision = decision
        return decision
