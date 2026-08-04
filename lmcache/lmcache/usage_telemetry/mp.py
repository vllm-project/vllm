# SPDX-License-Identifier: Apache-2.0
"""Usage reporting for the multiprocess (MP) cache server."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.context import UsageContextBase
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import is_usage_tracking_enabled
from lmcache.usage_telemetry.messages import (
    DeploymentMode,
    MPServerMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import UsageMessageSender

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.config import StorageManagerConfig
    from lmcache.v1.multiprocess.config import MPServerConfig

logger = init_logger(__name__)


class MPUsageContext(UsageContextBase):
    """Usage reporter for the multiprocess cache server.

    Sends an ``EnvMessage`` and an ``MPServerMessage`` at startup.
    """

    def __init__(
        self,
        mp_config: MPServerConfig,
        storage_manager_config: StorageManagerConfig,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            mp_config: The MP server configuration to snapshot.
            storage_manager_config: The storage manager configuration to
                snapshot.
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(DeploymentMode.MP_SERVER, local_log, sender)
        self._mp_config = mp_config
        self._storage_manager_config = storage_manager_config

    def _collect_messages(self) -> list[UsageMessage]:
        return [
            collect_env_message(),
            MPServerMessage.from_configs(self._mp_config, self._storage_manager_config),
        ]


@swallow_telemetry_errors
def InitializeMPUsageContext(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
) -> MPUsageContext | None:
    """Start usage reporting for a multiprocess cache server.

    Returns immediately; the startup report is sent in the background.
    Never blocks or raises.

    Args:
        mp_config: The MP server configuration to snapshot.
        storage_manager_config: The storage manager configuration to
            snapshot.
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing MP usage context.")
    context = MPUsageContext(mp_config, storage_manager_config, local_log, sender)
    threading.Thread(
        target=context.report_once, daemon=True, name="lmcache-usage-report"
    ).start()
    return context
