# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from vllm.engine.snapshot.base import BaseSnapshotProvider
from vllm.plugins import load_plugins_by_group

logger = logging.getLogger(__name__)


class SnapshotManager:
    def __init__(self, provider_name: str | None):
        self.provider = None
        if provider_name:
            self.provider = self._find_provider(provider_name)

    def _find_provider(self, name: str) -> BaseSnapshotProvider | None:
        # Look for plugins registered under 'vllm.snapshot_providers'
        logger.info("Loading snapshot providers...")
        providers = load_plugins_by_group("vllm.snapshot_providers")
        logger.info("Discovered snapshot providers: %s", list(providers.keys()))

        provider_factory = providers.get(name)
        if not provider_factory:
            logger.error("Snapshot provider '%s' not found.", name)
            return None

        try:
            provider = provider_factory()
            logger.info("Successfully instantiated snapshot provider: %s", name)
            return provider
        except Exception as e:
            logger.exception(
                "Failed to instantiate snapshot provider '%s': %s", name, e
            )
            return None

    def run_snapshot(self):
        if self.provider:
            logger.info("Triggering snapshot...")
            try:
                self.provider.trigger()
                logger.info("Snapshot completed successfully.")
            except Exception as e:
                logger.exception("Failed to run snapshot: %s", e)
        else:
            logger.warning("No snapshot provider configured or available.")
