# SPDX-License-Identifier: Apache-2.0
"""Context reporting: snapshot messages describing a deployment.

Context reports are sent once each (at startup, or at KV-cache registration
for the MP path in :mod:`.mp`), in contrast to the periodic interval
counters in :mod:`.continuous`.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import (
    DeploymentMode,
    EngineMessage,
    MetadataMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class UsageContextBase(ABC):
    """Shared plumbing for one-shot usage reporting.

    Subclasses define which messages the report contains via
    :meth:`_collect_messages`; this base owns the identity, transport, and
    optional local logging. Each message is POSTed to the endpoint it
    declares in the schema (:mod:`lmcache.usage_telemetry.messages`).
    """

    def __init__(
        self,
        mode: DeploymentMode,
        local_log: str | None,
        sender: UsageMessageSender | None,
    ) -> None:
        """Initialize shared reporting state.

        Args:
            mode: Deployment mode stamped on every payload this reporter
                sends.
            local_log: Path of a human-readable local log of every sent
                payload; ``None`` disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._mode = mode
        self._local_log = local_log
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._identity = get_usage_identity()
        self._start_time = datetime.now()

    @abstractmethod
    def _collect_messages(self) -> list[UsageMessage]:
        """Return the messages of the one-shot report."""
        raise NotImplementedError

    @swallow_telemetry_errors
    def report_once(self) -> None:
        """Collect and send every one-shot message on the calling thread.

        Failures are swallowed; never raises.
        """
        for message in self._collect_messages():
            payload = build_usage_payload(message, self._identity, self._mode)
            self._sender.send(usage_server_url(message.ENDPOINT), payload)
            self._write_local(payload)

    def _write_local(self, payload: dict[str, object]) -> None:
        """Append *payload* to the local log file, if one is configured."""
        if self._local_log is None:
            return
        text = "".join(f"{key}: {value}\n" for key, value in payload.items()) + "\n"
        try:
            with open(self._local_log, "a") as f:
                f.write(text)
        except OSError:
            logger.debug("Unable to write usage log to %s", self._local_log)


class UsageContext(UsageContextBase):
    """One-shot usage reporter for the single-process LMCacheEngine path.

    Sends an ``EnvMessage``, an ``EngineMessage``, and a
    ``MetadataMessage`` to the stats server.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(DeploymentMode.SINGLE_PROCESS, local_log, sender)
        self._config = config
        self._metadata = metadata

    def _collect_messages(self) -> list[UsageMessage]:
        metadata_message = MetadataMessage(
            start_time=self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=(datetime.now() - self._start_time).total_seconds(),
        )
        return [
            collect_env_message(),
            EngineMessage.from_config(self._config, self._metadata),
            metadata_message,
        ]


@swallow_telemetry_errors
def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> UsageContext | None:
    """Start one-shot usage reporting for a single-process engine.

    Returns immediately; the report is sent in the background. Never
    blocks or raises.

    Args:
        config: The engine configuration to snapshot.
        metadata: The engine metadata (model, world size, kv layout).
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing usage context.")
    context = UsageContext(config, metadata, local_log, sender)
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    return context
