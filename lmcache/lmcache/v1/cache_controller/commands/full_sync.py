# SPDX-License-Identifier: Apache-2.0
"""Full sync command implementation"""

# Standard
from typing import TYPE_CHECKING
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.commands.base import HeartbeatCommand

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class FullSyncCommand(HeartbeatCommand, tag="full_sync"):
    """Command to request full state synchronization

    Sent when the controller needs the worker to perform a full sync,
    e.g., after controller restart or worker re-registration.
    """

    # FullSync-specific fields can be added here, e.g.:
    # sync_scope: Optional[str] = None  # "all", "metadata", "data"
    # priority: int = 0  # Sync priority level

    def describe(self) -> str:
        return f"FullSyncCommand(reason={self.reason}, args={self.args})"

    def execute(self, worker: "LMCacheWorker") -> None:
        """Trigger full sync on the worker"""
        logger.info(
            "Received full sync command with reason: %s, args: %s",
            self.reason,
            self.args,
        )

        sender = worker._get_full_sync_sender()

        # Check if full sync is already in progress
        if sender.is_syncing:
            logger.warning("Full sync already in progress, skipping")
            return

        # Trigger full sync in background
        asyncio.create_task(sender.start_full_sync(self.reason))
