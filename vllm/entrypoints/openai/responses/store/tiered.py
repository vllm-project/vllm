from __future__ import annotations

import asyncio

from .base import (
    DiskEvictionResult,
    DiskEvictionStatus,
    MemoryEvictionResult,
    MemoryEvictionStatus,
    MemoryStoreEvictionStatus,
    SessionMetadata,
    SessionState,
    SessionStore,
)
from .disk import DiskCopyStatus, SQLiteSessionStore
from .memory import MemorySessionStore
from .metrics import ResponseStoreMetrics


class TieredSessionStore(SessionStore):
    """Memory 主存储，以及可选的 SQLite 二级恢复层。"""

    def __init__(
        self,
        memory_store: MemorySessionStore,
        disk_store: SQLiteSessionStore | None,
        num_shards: int = 64,
    ) -> None:
        if num_shards <= 0:
            raise ValueError("num_shards must be greater than 0")

        self._memory = memory_store
        self._disk = disk_store
        self._metrics = (
            disk_store.metrics if disk_store is not None else ResponseStoreMetrics()
        )

        # Tiered 的 save/get/delete 包含多个 await 和跨层操作，
        # 因此同一 session 需要一层复合操作锁。
        self._locks = [asyncio.Lock() for _ in range(num_shards)]

    async def save(
        self,
        session_id: str,
        response_id: str,
        delta_token_ids: list[int],
    ) -> None:
        if self._disk is None:
            await self._memory.save(session_id, response_id, delta_token_ids)
            return

        async with self._get_lock(session_id):
            # Session 可能已从 Memory 淘汰，只存在于 Disk。此时先恢复完整
            # 历史，再追加本轮 delta；正常 Memory hit 路径不会访问 SQLite。
            if not await self._memory.exists(session_id):
                disk_state = await self._disk.get_state(session_id)
                if disk_state is not None:
                    await self._memory.restore_if_absent(disk_state)

            await self._memory.save(session_id, response_id, delta_token_ids)

            if await self._disk.needs_snapshot(session_id):
                state = await self._memory.get_state_for_eviction(session_id)
                assert state is not None
                await self._disk.save_snapshot(state)
            else:
                # Disk.save 只 enqueue，不执行 SQLite I/O。
                await self._disk.save(session_id, response_id, delta_token_ids)

    async def get(self, session_id: str) -> list[int] | None:
        if self._disk is None:
            return await self._memory.get(session_id)

        async with self._get_lock(session_id):
            token_ids = await self._memory.get(session_id)
            if token_ids is not None:
                return token_ids

            state = await self._disk.get_state(session_id)
            if state is None:
                return None

            if await self._memory.restore_if_absent(state):
                return state.token_ids.copy()

            # Disk 读取期间 Memory 已被其他路径重新创建，以 Memory 为准。
            return await self._memory.get(session_id)

    async def exists(self, session_id: str) -> bool:
        """
        对外 exists：Memory 有则 True；否则只有 Disk 最新版本完整可恢复时 True。
        """
        if self._disk is None:
            return await self._memory.exists(session_id)

        async with self._get_lock(session_id):
            if await self._memory.exists(session_id):
                return True
            return await self._disk.exists(session_id)

    async def delete(self, session_id: str) -> bool:
        """
        逻辑删除整个 Session，两层都删。

        Memory 容量淘汰不是 Tiered.delete()：淘汰只应删除 Memory 层对象。
        """
        if self._disk is None:
            return await self._memory.delete(session_id)

        async with self._get_lock(session_id):
            memory_deleted = await self._memory.delete(session_id)
            disk_deleted = await self._disk.delete(session_id)
            return memory_deleted or disk_deleted

    async def list(self) -> list[SessionState]:
        if self._disk is None:
            return await self._memory.list()

        memory_states, disk_states = await asyncio.gather(
            self._memory.list(), self._disk.list()
        )

        merged = {state.session_id: state for state in disk_states}
        for memory_state in memory_states:
            disk_state = merged.get(memory_state.session_id)
            if disk_state is not None:
                memory_state.disk_idle_expires_at = disk_state.disk_idle_expires_at
                memory_state.disk_size_bytes = disk_state.disk_size_bytes
            # 在线 token / response 状态以 Memory 为准。
            merged[memory_state.session_id] = memory_state
        return list(merged.values())

    async def list_metadata(self) -> list[SessionMetadata]:
        if self._disk is None:
            return await self._memory.list_metadata()

        memory_metadata, disk_metadata = await asyncio.gather(
            self._memory.list_metadata(), self._disk.list_metadata()
        )

        merged = {metadata.session_id: metadata for metadata in disk_metadata}
        for memory_item in memory_metadata:
            disk_item = merged.get(memory_item.session_id)
            if disk_item is not None:
                memory_item.disk_idle_expires_at = disk_item.disk_idle_expires_at
                memory_item.disk_size_bytes = disk_item.disk_size_bytes
            merged[memory_item.session_id] = memory_item
        return list(merged.values())

    async def evict_memory_candidate(
        self,
        session_id: str,
        expected_response_id: str,
        expected_updated_at: int,
        now: int,
        force: bool = False,
    ) -> MemoryEvictionResult:
        """确认同版本 Disk 副本后，原子删除 Memory 副本。"""
        async with self._get_lock(session_id):
            candidate_status = await self._memory.check_eviction_candidate(
                session_id=session_id,
                expected_response_id=expected_response_id,
                expected_updated_at=expected_updated_at,
            )
            if candidate_status is not None:
                return self._eviction_result(
                    session_id,
                    expected_response_id,
                    self._map_memory_store_status(candidate_status),
                )

            if self._disk is None:
                return await self._evict_memory_without_disk(
                    session_id=session_id,
                    expected_response_id=expected_response_id,
                    expected_updated_at=expected_updated_at,
                    now=now,
                    force=force,
                )

            disk_result = await self._disk.ensure_eviction_copy(
                session_id=session_id,
                expected_response_id=expected_response_id,
                now=now,
            )

            eviction_status = MemoryEvictionStatus.EVICTED
            if disk_result.status is DiskCopyStatus.PENDING:
                return self._eviction_result(
                    session_id,
                    expected_response_id,
                    MemoryEvictionStatus.PERSISTENCE_PENDING,
                )
            if disk_result.status is DiskCopyStatus.FAILED:
                if not force:
                    return self._eviction_result(
                        session_id,
                        expected_response_id,
                        MemoryEvictionStatus.PERSISTENCE_FAILED,
                    )
                eviction_status = MemoryEvictionStatus.FORCE_EVICTED
            if disk_result.status is DiskCopyStatus.RESPONSE_CHANGED:
                return self._eviction_result(
                    session_id,
                    expected_response_id,
                    MemoryEvictionStatus.RESPONSE_CHANGED,
                )
            if disk_result.status is DiskCopyStatus.VERSION_CHANGED:
                return self._eviction_result(
                    session_id,
                    expected_response_id,
                    MemoryEvictionStatus.VERSION_CHANGED,
                )
            # Disk TTL 是软过期：过期副本仍可用于恢复，只是会优先进入
            # Disk cleanup。这样 Memory 淘汰不会制造不必要的重新 tokenizer。
            if disk_result.status not in {
                DiskCopyStatus.READY,
                DiskCopyStatus.EXPIRED,
            }:
                if not force:
                    return self._eviction_result(
                        session_id,
                        expected_response_id,
                        MemoryEvictionStatus.DISK_COPY_INVALID,
                    )
                eviction_status = MemoryEvictionStatus.FORCE_EVICTED

            memory_result = await self._memory.evict_if_unchanged(
                session_id=session_id,
                expected_response_id=expected_response_id,
                expected_updated_at=expected_updated_at,
            )
            if not memory_result.evicted:
                return self._eviction_result(
                    session_id,
                    expected_response_id,
                    self._map_memory_store_status(memory_result.status),
                )

            return self._eviction_result(
                session_id,
                expected_response_id,
                eviction_status,
                memory_result.freed_bytes,
            )

    async def evict_disk_candidate(
        self,
        session_id: str,
        expected_response_id: str,
    ) -> DiskEvictionResult:
        """条件删除 Disk 副本，不修改 Memory 副本。"""
        if self._disk is None:
            return DiskEvictionResult(
                session_id=session_id,
                response_id=expected_response_id,
                status=DiskEvictionStatus.NOT_FOUND,
            )

        async with self._get_lock(session_id):
            if await self._memory.exists(session_id):
                return DiskEvictionResult(
                    session_id=session_id,
                    response_id=expected_response_id,
                    status=DiskEvictionStatus.MEMORY_RESIDENT,
                )
            return await self._disk.evict_if_unchanged(
                session_id=session_id,
                expected_response_id=expected_response_id,
            )

    @staticmethod
    def _eviction_result(
        session_id: str,
        response_id: str,
        status: MemoryEvictionStatus,
        freed_bytes: int = 0,
    ) -> MemoryEvictionResult:
        return MemoryEvictionResult(
            session_id=session_id,
            response_id=response_id,
            status=status,
            freed_bytes=freed_bytes,
        )

    @staticmethod
    def _map_memory_store_status(
        status: MemoryStoreEvictionStatus,
    ) -> MemoryEvictionStatus:
        return {
            MemoryStoreEvictionStatus.NOT_FOUND: MemoryEvictionStatus.NOT_FOUND,
            MemoryStoreEvictionStatus.RESPONSE_CHANGED: (
                MemoryEvictionStatus.RESPONSE_CHANGED
            ),
            MemoryStoreEvictionStatus.ACCESSED_AFTER_SELECTION: (
                MemoryEvictionStatus.ACCESSED_AFTER_SELECTION
            ),
        }[status]

    async def is_disk_complete(self, session_id: str) -> bool:
        """供后续 Memory 淘汰模块判断该 Session 是否已经安全落盘。"""
        if self._disk is None:
            return False
        return await self._disk.is_complete(session_id)

    async def memory_used_bytes(self) -> int:
        return self._memory.used_bytes

    async def disk_used_bytes(self) -> int:
        if self._disk is None:
            return 0
        return await self._disk.used_bytes()

    async def close(self) -> None:
        if self._disk is not None:
            await self._disk.close()

    @property
    def memory_store(self) -> MemorySessionStore:
        return self._memory

    @property
    def disk_store(self) -> SQLiteSessionStore | None:
        return self._disk

    @property
    def disk_enabled(self) -> bool:
        return self._disk is not None

    @property
    def metrics(self) -> ResponseStoreMetrics:
        return self._metrics

    async def _evict_memory_without_disk(
        self,
        session_id: str,
        expected_response_id: str,
        expected_updated_at: int,
        now: int,
        force: bool,
    ) -> MemoryEvictionResult:
        state = await self._memory.get_state_for_eviction(session_id)
        if state is None:
            return self._eviction_result(
                session_id,
                expected_response_id,
                MemoryEvictionStatus.NOT_FOUND,
            )

        expired = (
            state.mem_idle_expires_at is not None and state.mem_idle_expires_at <= now
        )
        if not expired and not force:
            return self._eviction_result(
                session_id,
                expected_response_id,
                MemoryEvictionStatus.DISK_COPY_INVALID,
            )

        memory_result = await self._memory.evict_if_unchanged(
            session_id=session_id,
            expected_response_id=expected_response_id,
            expected_updated_at=expected_updated_at,
        )
        if not memory_result.evicted:
            return self._eviction_result(
                session_id,
                expected_response_id,
                self._map_memory_store_status(memory_result.status),
            )

        status = (
            MemoryEvictionStatus.EVICTED_REQUIRES_RERENDER
            if expired
            else MemoryEvictionStatus.FORCE_EVICTED
        )
        return self._eviction_result(
            session_id,
            expected_response_id,
            status,
            memory_result.freed_bytes,
        )

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        return self._locks[hash(session_id) % len(self._locks)]
