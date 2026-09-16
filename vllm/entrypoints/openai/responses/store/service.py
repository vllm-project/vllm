# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

from . import ResponsesStoreConfig
from .cleanup import PeriodicSessionStoreCleanup
from .disk import SQLiteSessionStore
from .key_provider import FileKeyProvider
from .memory import MemorySessionStore
from .metrics import ResponseStoreMetrics
from .rotation import PeriodicDatabaseKeyRotation
from .tiered import TieredSessionStore


class ResponsesStoreService:
    """Own the Responses store and its background tasks."""

    def __init__(self, config: ResponsesStoreConfig) -> None:
        self._config = config
        metrics = ResponseStoreMetrics()
        memory_store = MemorySessionStore(
            max_capacity_bytes=config.memory_capacity_bytes,
            mem_idle_ttl_seconds=config.memory_ttl_seconds,
        )
        key_provider = (
            FileKeyProvider(config.key_file) if config.key_file is not None else None
        )
        disk_store = (
            SQLiteSessionStore(
                db_path=config.disk_path,
                disk_idle_ttl_seconds=config.disk_ttl_seconds,
                write_interval_seconds=config.disk_write_interval_seconds,
                key_provider=key_provider,
                metrics=metrics,
                persistent_key=config.key_file is not None,
            )
            if config.disk_enabled
            else None
        )
        self._store = TieredSessionStore(
            memory_store=memory_store,
            disk_store=disk_store,
            num_shards=config.num_shards,
        )
        self._cleanup = PeriodicSessionStoreCleanup(
            store=self._store,
            config=config.build_cleanup_config(),
        )
        self._key_rotation = (
            PeriodicDatabaseKeyRotation(disk_store)
            if disk_store is not None and disk_store.automatic_key_rotation_enabled
            else None
        )
        self._started = False
        self._closed = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ResponsesStoreService:
        """Build a service from the OpenAI server CLI namespace."""
        return cls(ResponsesStoreConfig.from_cli_args(args))

    def start(self) -> None:
        """Start periodic cleanup in the current event loop."""
        if self._closed:
            raise RuntimeError("ResponsesStoreService is already closed")
        if self._started:
            return

        self._cleanup.start()
        if self._key_rotation is not None:
            self._key_rotation.start()
        self._started = True

    async def close(self) -> None:
        """Stop background work and close the storage tiers."""
        if self._closed:
            return

        if self._key_rotation is not None:
            await self._key_rotation.stop()
        await self._cleanup.stop()
        await self._store.close()
        self._started = False
        self._closed = True

    @property
    def config(self) -> ResponsesStoreConfig:
        return self._config

    @property
    def store(self) -> TieredSessionStore:
        return self._store

    @property
    def cleanup(self) -> PeriodicSessionStoreCleanup:
        return self._cleanup

    @property
    def key_rotation(self) -> PeriodicDatabaseKeyRotation | None:
        return self._key_rotation

    @property
    def is_running(self) -> bool:
        return self._started and self._cleanup.is_running


__all__ = ["ResponsesStoreService"]
