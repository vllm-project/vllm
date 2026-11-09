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
        self._pending_read_identifiers: set[str] = set()
        self._prefetch_requests: dict[str, set[str]] = {}
        self._prefetch_waiters: dict[str, set[str]] = {}
        self._prefetch_holds: set[str] = set()

    def prefetch_entry(
        self,
        request_id: str,
        identifier: str,
        modality: str,
        num_tokens: int,
        shape: tuple[int, int],
        element_size: int,
    ) -> None:
        """Try one early read without waiting or reclaiming other prefetch holds."""
        self.poll()
        if (
            identifier not in self._pending_read_identifiers
            and identifier not in self._prefetch_holds
        ):
            self._ensure_primary_entry(
                identifier,
                modality,
                num_tokens,
                shape,
                element_size,
                reclaim_prefetch=False,
            )
        entry = self._primary.get(identifier)
        if entry is not None and entry.ready and identifier not in self._prefetch_holds:
            self._primary.pin(identifier)
            self._prefetch_holds.add(identifier)
        if (
            identifier in self._pending_read_identifiers
            or identifier in self._prefetch_holds
        ):
            self._prefetch_requests.setdefault(request_id, set()).add(identifier)
            self._prefetch_waiters.setdefault(identifier, set()).add(request_id)

    def release_prefetch(self, request_id: str, identifier: str | None = None) -> None:
        """Release a request's interest after consumption or cancellation."""
        identifiers = self._prefetch_requests.get(request_id)
        if identifiers is None:
            return
        released = tuple(identifiers) if identifier is None else (identifier,)
        for key in released:
            if key not in identifiers:
                continue
            identifiers.remove(key)
            waiters = self._prefetch_waiters[key]
            waiters.remove(request_id)
            if not waiters:
                del self._prefetch_waiters[key]
                if key in self._prefetch_holds:
                    self._prefetch_holds.remove(key)
                    self._primary.unpin(key)
        if not identifiers:
            del self._prefetch_requests[request_id]

    def _forget_prefetch(self, identifier: str) -> None:
        for request_id in tuple(self._prefetch_waiters.get(identifier, ())):
            self.release_prefetch(request_id, identifier)

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
        """Wait for submitted reads; allow recomputation when promotion cannot start."""
        entry = self._primary.get(identifier)
        if entry is not None and entry.ready:
            return True
        self.poll()
        return self._ensure_primary_entry(
            identifier,
            modality,
            num_tokens,
            shape,
            element_size,
            reclaim_prefetch=True,
        )

    def _ensure_primary_entry(
        self,
        identifier: str,
        modality: str,
        num_tokens: int,
        shape: tuple[int, int],
        element_size: int,
        *,
        reclaim_prefetch: bool,
    ) -> bool:
        entry = self._primary.get(identifier)
        if entry is not None:
            if entry.ready:
                return True
            return identifier not in self._pending_read_identifiers

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
        if entry is None and reclaim_prefetch and self._prefetch_holds:
            # Speculative ready entries must not prevent demand admission.
            for key in tuple(self._prefetch_holds):
                self._primary.unpin(key)
            self._prefetch_holds.clear()
            entry = self._primary.alloc(identifier, num_blocks, shape)
        if entry is None:
            # No read owns a destination: let has_cache_item report a miss.
            return True
        try:
            destinations = self._primary.block_views(entry.block_ids)
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
        self._pending_read_identifiers.add(identifier)
        return False

    def poll(self) -> tuple[list[ECReadResult], list[ECWriteResult]]:
        """Publish completed loads and release pins for completed stores."""
        read_completions = self._secondary.poll_completed_reads()
        for completion in read_completions:
            identifier = self._pending_reads.pop(completion.cache_key)
            self._pending_read_identifiers.remove(identifier)
            if completion.success:
                self._primary.mark_ready(identifier)
                if self._prefetch_waiters.get(identifier):
                    self._primary.pin(identifier)
                    self._prefetch_holds.add(identifier)
            else:
                self._primary.discard(identifier)
                self._forget_prefetch(identifier)
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
        for request_id in tuple(self._prefetch_requests):
            self.release_prefetch(request_id)
        if self._pending_reads or self._pending_writes:
            raise RuntimeError("filesystem shutdown left encoder-cache jobs pending")
