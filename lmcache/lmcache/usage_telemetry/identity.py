# SPDX-License-Identifier: Apache-2.0
"""Opt-out gate and anonymous identity for usage telemetry."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
import os
import threading
import uuid

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def usage_config_dir() -> Path:
    """Directory holding LMCache usage-telemetry state files."""
    return Path(os.path.expanduser("~/.config/lmcache"))


def is_usage_tracking_enabled() -> bool:
    """Whether anonymous usage telemetry may be collected and sent.

    Tracking is enabled by default and disabled when either opt-out is
    present:

    - Environment variable ``LMCACHE_TRACK_USAGE=false``.
    - Environment variable ``DO_NOT_TRACK`` set to ``1``/``true``/``yes``
      (the cross-tool "do not track" console convention).

    Returns:
        True when usage telemetry may be sent, False otherwise.
    """
    if os.getenv("LMCACHE_TRACK_USAGE") == "false":
        return False
    if os.getenv("DO_NOT_TRACK", "").lower() in ("1", "true", "yes"):
        return False
    return True


@dataclass(frozen=True)
class UsageIdentity:
    """Anonymous identifiers stamped on every usage message.

    Both identifiers are random UUIDs, never derived from hardware
    (MAC address, hostname, or ``/etc/machine-id``), so they carry no
    identifying information.
    """

    session_id: str
    """Random UUID minted once per process. Joins the one-shot context
    messages with the continuous messages of the same run."""

    machine_id: str
    """Random UUID persisted at ``~/.config/lmcache/machine_id``. Groups
    sessions from the same machine across restarts. Empty string when the
    file can be neither read nor created."""


_usage_identity: UsageIdentity | None = None
_usage_identity_lock = threading.Lock()


def get_usage_identity() -> UsageIdentity:
    """Return the process-wide usage identity, minting it on first call.

    The first call creates the persistent machine id file if needed; callers
    must therefore consult :func:`is_usage_tracking_enabled` first so that
    opted-out users get no state files written.

    Returns:
        The identity shared by all usage messages of this process.
    """
    global _usage_identity
    with _usage_identity_lock:
        if _usage_identity is None:
            _usage_identity = UsageIdentity(
                session_id=str(uuid.uuid4()),
                machine_id=_read_or_create_machine_id(),
            )
        return _usage_identity


def _read_or_create_machine_id() -> str:
    """Read the persistent machine id, creating it on first use.

    Returns:
        The machine id, or an empty string when the state file can be
        neither read nor created (e.g. read-only filesystem).
    """
    path = usage_config_dir() / "machine_id"
    try:
        if path.is_file():
            machine_id = path.read_text().strip()
            if machine_id:
                return machine_id
        machine_id = str(uuid.uuid4())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(machine_id)
        return machine_id
    except OSError:
        logger.debug("Cannot persist usage machine id", exc_info=True)
        return ""
