# SPDX-License-Identifier: Apache-2.0
"""Payload construction and HTTP transport for usage telemetry."""

# Future
from __future__ import annotations

# Standard
from urllib.parse import urljoin
import json
import os

# Third Party
import torch

# First Party
from lmcache.connections import global_http_connection
from lmcache.logging import init_logger
from lmcache.usage_telemetry.identity import UsageIdentity
from lmcache.usage_telemetry.messages import (
    USAGE_SCHEMA_VERSION,
    DeploymentMode,
    UsageMessage,
)

logger = init_logger(__name__)


def usage_server_url(endpoint: str) -> str:
    """Build the stats-server URL for *endpoint*.

    Args:
        endpoint: Path component on the stats server, e.g. ``"context"``.

    Returns:
        The full URL, honoring the ``LMCACHE_USAGE_TRACK_URL`` override.
        Any path on the override is preserved (e.g. base
        ``http://host/api/v1`` yields ``http://host/api/v1/context``).
    """
    base = os.getenv("LMCACHE_USAGE_TRACK_URL", "http://stats.lmcache.ai:8080")
    # urljoin drops the last path segment of a base without a trailing
    # slash (http://host/api/v1 + context -> http://host/api/context).
    if not base.endswith("/"):
        base += "/"
    return urljoin(base, endpoint)


class UsageMessageSender:
    """Default HTTP transport for usage messages.

    Posts JSON payloads with a short timeout and swallows every failure
    (telemetry must never disturb serving). Inject a stub instance into the
    usage contexts to capture payloads in tests.
    """

    def send(self, url: str, payload: dict[str, object]) -> None:
        """POST one JSON payload to *url*; never raises.

        Args:
            url: Full endpoint URL on the stats server.
            payload: Flat JSON-serializable payload.
        """
        try:
            client = global_http_connection.get_sync_client()
            client.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
        except Exception as e:
            logger.debug("Unable to send lmcache usage message: %s", e)


DEFAULT_SENDER = UsageMessageSender()
"""Shared default sender; stateless, safe to reuse across contexts."""


def build_usage_payload(
    message: UsageMessage, identity: UsageIdentity, mode: DeploymentMode
) -> dict[str, object]:
    """Flatten *message* into a payload stamped with identity and schema info.

    The wire ``message_type`` is the message's class name, per the schema
    contract in :mod:`lmcache.usage_telemetry.messages`.

    Args:
        message: The message to serialize; its fields become payload keys.
        identity: Identifiers stamped on the payload.
        mode: Deployment mode stamped as the ``deployment_mode`` key.

    Returns:
        A flat JSON-serializable dict (``torch.dtype`` values stringified).
    """
    payload: dict[str, object] = {
        "message_type": type(message).__name__,
        "schema_version": USAGE_SCHEMA_VERSION,
        "session_id": identity.session_id,
        "machine_id": identity.machine_id,
        "deployment_mode": mode.value,
    }
    for key, value in vars(message).items():
        payload[key] = str(value) if isinstance(value, torch.dtype) else value
    return payload
