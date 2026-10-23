# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asynchronous CPU-primary and secondary-tier transfer orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import (
    ECReadResult,
    ECWriteResult,
)
from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    build_ec_cache_key,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class _PrimaryEntry(Protocol):
    block_ids: tuple[int, ...]
    ready: bool


class _PrimaryTier(Protocol):
    block_size_bytes: int

    def get(self, key: str) -> _PrimaryEntry | None: ...

    def pin(self, key: str) -> None: ...

    def unpin(self, key: str) -> None: ...

    def alloc(
        self,
        key: str,
        n_blocks: int,
        shape: tuple[int, int] | None = None,
    ) -> _PrimaryEntry | None: ...

    def mark_ready(self, key: str) -> None: ...

    def discard(self, key: str) -> None: ...

    def block_views(self, block_ids: Sequence[int]) -> tuple[memoryview, ...]: ...


class _SecondaryTier(Protocol):
    def contains(self, cache_key: str) -> bool: ...

    def submit_write(
        self,
        cache_key: str,
        sources: Sequence[memoryview],
        file_size_bytes: int,
    ) -> bool: ...

    def submit_read(
        self,
        cache_key: str,
        destinations: Sequence[memoryview],
    ) -> bool: ...

    def poll_completed_reads(self) -> list[ECReadResult]: ...

    def poll_completed_writes(self) -> list[ECWriteResult]: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class _StoreDescriptor:
    cache_key: str
    file_size_bytes: int


class ECTieringManager:
    """Orchestrate asynchronous transfers between CPU and a secondary tier."""

    def __init__(
        self,
        primary: _PrimaryTier,
        secondary: _SecondaryTier,
        namespace_digest: str,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._namespace_digest = namespace_digest
        self._descriptors: dict[str, _StoreDescriptor] = {}
        self._pending_writes: dict[str, str] = {}
        self._pending_reads: dict[str, str] = {}

    def register_cpu_save(
        self,
        identifier: str,
        modality: str,
        num_tokens: int,
    ) -> None:
        """Record identity for a newly allocated CPU save."""
        entry = self._primary.get(identifier)
        if entry is None:
            return
        file_size_bytes = len(entry.block_ids) * self._primary.block_size_bytes
        cache_key = build_ec_cache_key(
            self._namespace_digest,
            identifier,
            modality,
            num_tokens,
            file_size_bytes,
        )
        self._descriptors[identifier] = _StoreDescriptor(
            cache_key=cache_key,
            file_size_bytes=file_size_bytes,
        )

    def on_cpu_save_completed(self, identifier: str) -> None:
        """Submit one write-through store after the CPU entry becomes ready."""
        descriptor = self._descriptors.pop(identifier, None)
        if descriptor is None:
            return
        if self._secondary.contains(descriptor.cache_key):
            return
        if descriptor.cache_key in self._pending_writes:
            return

        entry = self._primary.get(identifier)
        if entry is None or not entry.ready:
            return
        self._primary.pin(identifier)
        try:
            sources = self._primary.block_views(entry.block_ids)
            accepted = self._secondary.submit_write(
                descriptor.cache_key,
                sources,
                descriptor.file_size_bytes,
            )
        except Exception:
            self._primary.unpin(identifier)
            logger.warning(
                "Failed to submit encoder-cache filesystem store for %s",
                identifier,
                exc_info=True,
            )
            return
        if not accepted:
            self._primary.unpin(identifier)
            return
        self._pending_writes[descriptor.cache_key] = identifier

    def ensure_primary_entry(
        self,
        identifier: str,
        modality: str,
        num_tokens: int,
        shape: tuple[int, int],
        element_size: int,
    ) -> bool:
        """Ensure an SSD hit is promoted, returning false while it is pending."""
        self.poll()
        entry = self._primary.get(identifier)
        if entry is not None:
            if entry.ready:
                return True
            return identifier not in self._pending_reads.values()

        logical_size_bytes = shape[0] * shape[1] * element_size
        block_size_bytes = self._primary.block_size_bytes
        num_blocks = (logical_size_bytes + block_size_bytes - 1) // block_size_bytes
        file_size_bytes = num_blocks * block_size_bytes
        cache_key = build_ec_cache_key(
            self._namespace_digest,
            identifier,
            modality,
            num_tokens,
            file_size_bytes,
        )
        if not self._secondary.contains(cache_key):
            return True

        entry = self._primary.alloc(identifier, num_blocks, shape)
        if entry is None:
            return False
        destinations = self._primary.block_views(entry.block_ids)
        try:
            accepted = self._secondary.submit_read(cache_key, destinations)
        except Exception:
            self._primary.discard(identifier)
            logger.warning(
                "Failed to submit encoder-cache filesystem load for %s",
                identifier,
                exc_info=True,
            )
            return True
        if not accepted:
            self._primary.discard(identifier)
            return True
        self._pending_reads[cache_key] = identifier
        return False

    def poll(self) -> tuple[list[ECReadResult], list[ECWriteResult]]:
        """Publish completed loads and release pins for completed stores."""
        read_completions = self._secondary.poll_completed_reads()
        for completion in read_completions:
            identifier = self._pending_reads.pop(completion.cache_key)
            if completion.success:
                self._primary.mark_ready(identifier)
            else:
                self._primary.discard(identifier)
                logger.warning(
                    "Encoder-cache filesystem load failed for %s: %s",
                    identifier,
                    completion.error,
                )

        write_completions = self._secondary.poll_completed_writes()
        for completion in write_completions:
            identifier = self._pending_writes.pop(completion.cache_key)
            self._primary.unpin(identifier)
            if not completion.success:
                logger.warning(
                    "Encoder-cache filesystem store failed for %s: %s",
                    identifier,
                    completion.error,
                )
        return read_completions, write_completions

    def has_pending(self) -> bool:
        """Poll once and report whether any tier transfer is still pending."""
        self.poll()
        return bool(self._pending_reads or self._pending_writes)

    def shutdown(self) -> None:
        """Drain transfers, finalize their state, and verify quiescence."""
        self._secondary.shutdown()
        self.poll()
        if self._pending_reads or self._pending_writes:
            raise RuntimeError("filesystem shutdown left encoder-cache jobs pending")
