from __future__ import annotations

import sys
import time

from .base import (
    MemoryStoreEvictionResult,
    MemoryStoreEvictionStatus,
    SessionMetadata,
    SessionState,
    SessionStore,
)


class MemorySessionStore(SessionStore):
    """Memory 主存储"""

    _PY_INT_BYTES = sys.getsizeof(0)

    def __init__(
        self,
        max_capacity_bytes: int,
        mem_idle_ttl_seconds: int | None = None,
    ) -> None:
        if max_capacity_bytes <= 0:
            raise ValueError("max_capacity_bytes must be greater than 0")
        if mem_idle_ttl_seconds is not None and mem_idle_ttl_seconds <= 0:
            raise ValueError("mem_idle_ttl_seconds must be greater than 0")

        self._data: dict[str, SessionState] = {}
        self._max_capacity_bytes = max_capacity_bytes
        self._used_bytes = 0
        self._mem_idle_ttl_seconds = mem_idle_ttl_seconds

    async def save(
        self,
        session_id: str,
        response_id: str,
        delta_token_ids: list[int],
    ) -> None:
        state = self._data.get(session_id)
        now = time.time_ns() // 1_000_000

        if state is None:
            state = SessionState(
                session_id=session_id,
                response_id=response_id,
                token_ids=list(delta_token_ids),
                created_at=now,
                updated_at=now,
                mem_idle_expires_at=self._build_expire_time(now),
                memory_resident=True,
            )
            state.mem_size_bytes = self._estimate_state_size_bytes(state)
            self._data[session_id] = state
            self._used_bytes += state.mem_size_bytes
            return

        old_size = state.mem_size_bytes
        state.token_ids.extend(delta_token_ids)
        state.response_id = response_id
        state.updated_at = now
        state.mem_idle_expires_at = self._build_expire_time(now)
        state.mem_size_bytes = self._estimate_state_size_bytes(state)
        self._used_bytes += state.mem_size_bytes - old_size

    async def get(self, session_id: str) -> list[int] | None:
        state = self._data.get(session_id)
        if state is None:
            return None

        now = time.time_ns() // 1_000_000
        state.updated_at = now
        state.mem_idle_expires_at = self._build_expire_time(now)
        return state.token_ids.copy()

    async def exists(self, session_id: str) -> bool:
        """exists 不刷新 updated_at / idle TTL。"""
        return session_id in self._data

    async def delete(self, session_id: str) -> bool:
        state = self._data.pop(session_id, None)
        if state is None:
            return False

        self._used_bytes -= state.mem_size_bytes
        if self._used_bytes < 0:
            self._used_bytes = 0
        return True

    async def get_state_for_eviction(self, session_id: str) -> SessionState | None:
        """返回不刷新访问时间和 TTL 的 Session 快照。"""
        state = self._data.get(session_id)
        return None if state is None else self._copy_state(state)

    async def check_eviction_candidate(
        self,
        session_id: str,
        expected_response_id: str,
        expected_updated_at: int,
    ) -> MemoryStoreEvictionStatus | None:
        """检查候选版本；返回 None 表示仍可继续淘汰。"""
        state = self._data.get(session_id)
        if state is None:
            return MemoryStoreEvictionStatus.NOT_FOUND
        if state.response_id != expected_response_id:
            return MemoryStoreEvictionStatus.RESPONSE_CHANGED
        if state.updated_at != expected_updated_at:
            return MemoryStoreEvictionStatus.ACCESSED_AFTER_SELECTION
        return None

    async def evict_if_unchanged(
        self,
        session_id: str,
        expected_response_id: str,
        expected_updated_at: int,
    ) -> MemoryStoreEvictionResult:
        """候选版本未变化时原子删除 Memory 副本。"""
        state = self._data.get(session_id)
        if state is None:
            return MemoryStoreEvictionResult(MemoryStoreEvictionStatus.NOT_FOUND)
        if state.response_id != expected_response_id:
            return MemoryStoreEvictionResult(MemoryStoreEvictionStatus.RESPONSE_CHANGED)
        if state.updated_at != expected_updated_at:
            return MemoryStoreEvictionResult(
                MemoryStoreEvictionStatus.ACCESSED_AFTER_SELECTION
            )

        self._data.pop(session_id)
        self._used_bytes = max(0, self._used_bytes - state.mem_size_bytes)
        return MemoryStoreEvictionResult(
            MemoryStoreEvictionStatus.EVICTED,
            freed_bytes=state.mem_size_bytes,
        )

    async def list(self) -> list[SessionState]:
        """返回所有 Session 的一致性快照。"""
        return [self._copy_state(state) for state in self._data.values()]

    async def list_metadata(self) -> list[SessionMetadata]:
        return [
            SessionMetadata(
                session_id=state.session_id,
                response_id=state.response_id,
                created_at=state.created_at,
                updated_at=state.updated_at,
                mem_idle_expires_at=state.mem_idle_expires_at,
                disk_idle_expires_at=state.disk_idle_expires_at,
                mem_size_bytes=state.mem_size_bytes,
                disk_size_bytes=state.disk_size_bytes,
                memory_resident=state.memory_resident,
            )
            for state in self._data.values()
        ]

    async def restore_if_absent(self, state: SessionState) -> bool:
        """
        Tiered 从 Disk 恢复时使用。

        只在 Memory 仍不存在该 Session 时恢复，避免 Disk 旧快照覆盖并发写入
        产生的新 Memory 状态。
        """
        if state.session_id in self._data:
            return False

        now = time.time_ns() // 1_000_000
        restored = self._copy_state(state)
        restored.updated_at = now
        restored.mem_idle_expires_at = self._build_expire_time(now)
        restored.memory_resident = True
        restored.mem_size_bytes = self._estimate_state_size_bytes(restored)

        self._data[state.session_id] = restored
        self._used_bytes += restored.mem_size_bytes
        return True

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def max_capacity_bytes(self) -> int:
        return self._max_capacity_bytes

    @property
    def remaining_bytes(self) -> int:
        return max(0, self._max_capacity_bytes - self._used_bytes)

    @property
    def is_over_capacity(self) -> bool:
        return self._used_bytes > self._max_capacity_bytes

    def _build_expire_time(self, now: int) -> int | None:
        if self._mem_idle_ttl_seconds is None:
            return None
        return now + self._mem_idle_ttl_seconds * 1_000

    @classmethod
    def _estimate_state_size_bytes(cls, state: SessionState) -> int:
        """O(1) 逻辑容量估算，不逐个遍历历史 token_ids。"""
        return (
            sys.getsizeof(state)
            + sys.getsizeof(state.session_id)
            + sys.getsizeof(state.response_id)
            + sys.getsizeof(state.token_ids)
            + len(state.token_ids) * cls._PY_INT_BYTES
            + sys.getsizeof(state.created_at)
            + sys.getsizeof(state.updated_at)
            + sys.getsizeof(state.mem_idle_expires_at)
            + sys.getsizeof(state.disk_idle_expires_at)
            + sys.getsizeof(state.mem_size_bytes)
            + sys.getsizeof(state.disk_size_bytes)
            + sys.getsizeof(state.memory_resident)
        )

    @staticmethod
    def _copy_state(state: SessionState) -> SessionState:
        return SessionState(
            session_id=state.session_id,
            response_id=state.response_id,
            token_ids=state.token_ids.copy(),
            created_at=state.created_at,
            updated_at=state.updated_at,
            mem_idle_expires_at=state.mem_idle_expires_at,
            disk_idle_expires_at=state.disk_idle_expires_at,
            mem_size_bytes=state.mem_size_bytes,
            disk_size_bytes=state.disk_size_bytes,
            memory_resident=state.memory_resident,
        )
