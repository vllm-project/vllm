from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, unique


@unique
class MemoryStoreEvictionStatus(str, Enum):
    """Result of an atomic MemorySessionStore eviction."""

    EVICTED = "evicted"
    NOT_FOUND = "not_found"
    RESPONSE_CHANGED = "response_changed"
    ACCESSED_AFTER_SELECTION = "accessed_after_selection"


@dataclass(frozen=True, slots=True)
class MemoryStoreEvictionResult:
    """Result of an atomic MemorySessionStore eviction."""

    status: MemoryStoreEvictionStatus
    freed_bytes: int = 0

    def __post_init__(self) -> None:
        if self.freed_bytes < 0:
            raise ValueError("freed_bytes must be non-negative")

        if not self.evicted and self.freed_bytes != 0:
            raise ValueError("a non-evicted result cannot report freed bytes")

    @property
    def evicted(self) -> bool:
        return self.status is MemoryStoreEvictionStatus.EVICTED


@unique
class MemoryEvictionStatus(str, Enum):
    """Result of TieredSessionStore memory eviction coordination."""

    # 磁盘存在同版本有效副本，内存副本已删除。
    EVICTED = "evicted"

    # 磁盘副本已过期，内存副本已删除，后续需要重新渲染。
    EVICTED_REQUIRES_RERENDER = "evicted_requires_rerender"

    # 最大容量压力下强制删除，后续需要重新渲染。
    FORCE_EVICTED = "force_evicted"

    # 执行时内存记录已经不存在
    NOT_FOUND = "not_found"

    # response_id 已经变化，候选属于旧版本。
    RESPONSE_CHANGED = "response_changed"

    # 候选对应的落盘版本已经变化。
    VERSION_CHANGED = "version_changed"

    # 候选生成后 Session 又被访问
    ACCESSED_AFTER_SELECTION = "accessed_after_selection"

    # 磁盘副本不存在、版本落后或无法正常解密
    DISK_COPY_INVALID = "disk_copy_invalid"

    # 同版本磁盘持久化仍在进行
    PERSISTENCE_PENDING = "persistence_pending"

    # 磁盘持久化已经失败
    PERSISTENCE_FAILED = "persistence_failed"

    # 未被预期状态覆盖的内部错误
    ERROR = "error"

@dataclass(frozen=True, slots=True)
class MemoryEvictionResult:
    """Result of evicting one session from TieredSessionStore memory."""

    session_id: str
    response_id: str
    status: MemoryEvictionStatus
    freed_bytes: int = 0

    def __post_init__(self) -> None:
        if self.freed_bytes < 0:
            raise ValueError("freed_bytes must be non-negative")

        if not self.evicted and self.freed_bytes != 0:
            raise ValueError("a non-evicted result cannot report freed bytes")

    @property
    def evicted(self) -> bool:
        """Return whether the memory copy was actually deleted."""
        return self.status in {
            MemoryEvictionStatus.EVICTED,
            MemoryEvictionStatus.EVICTED_REQUIRES_RERENDER,
            MemoryEvictionStatus.FORCE_EVICTED,
        }

    @property
    def requires_rerender(self) -> bool:
        """Return whether a later request must rerender the session."""
        return self.status in {
            MemoryEvictionStatus.EVICTED_REQUIRES_RERENDER,
            MemoryEvictionStatus.FORCE_EVICTED,
        }


@unique
class DiskEvictionStatus(str, Enum):
    """Result of deleting one conditional Disk copy."""

    EVICTED = "evicted"
    NOT_FOUND = "not_found"
    RESPONSE_CHANGED = "response_changed"
    VERSION_CHANGED = "version_changed"
    MEMORY_RESIDENT = "memory_resident"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DiskEvictionResult:
    session_id: str
    response_id: str
    status: DiskEvictionStatus
    freed_bytes: int = 0

    def __post_init__(self) -> None:
        if self.freed_bytes < 0:
            raise ValueError("freed_bytes must be non-negative")
        if not self.evicted and self.freed_bytes != 0:
            raise ValueError("a non-evicted result cannot report freed bytes")

    @property
    def evicted(self) -> bool:
        return self.status is DiskEvictionStatus.EVICTED


@dataclass(slots=True)
class SessionMetadata:
    """Session eviction metadata; all timestamps are Unix milliseconds."""

    session_id: str
    response_id: str
    created_at: int
    updated_at: int
    mem_idle_expires_at: int | None = None
    disk_idle_expires_at: int | None = None
    mem_size_bytes: int = 0
    disk_size_bytes: int = 0
    memory_resident: bool = False


@dataclass(slots=True)
class SessionState:
    """一个 Session 对应一个状态对象。"""

    session_id: str
    response_id: str
    token_ids: list[int]

    # All timestamps, including idle expiry times, are Unix milliseconds.
    created_at: int
    updated_at: int

    mem_idle_expires_at: int | None = None
    disk_idle_expires_at: int | None = None

    mem_size_bytes: int = 0
    disk_size_bytes: int = 0
    memory_resident: bool = False


class SessionStore(ABC):
    """Session 状态统一存储接口。"""

    @abstractmethod
    async def save(
        self,
        session_id: str,
        response_id: str,
        delta_token_ids: list[int],
    ) -> None:
        """保存本轮新增 token。"""
        ...

    @abstractmethod
    async def get(self, session_id: str) -> list[int] | None:
        """返回当前会话完整 token_ids，不存在时返回 None。"""
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """判断指定 Session 是否存在且当前可安全读取。"""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """删除指定 Session；删除成功返回 True。"""
        ...

    @abstractmethod
    async def list(self) -> list[SessionState]:
        """返回当前 Store 中所有可读取 Session 的状态快照。"""
        ...

    @abstractmethod
    async def list_metadata(self) -> list[SessionMetadata]:
        """Return metadata snapshots without materializing token IDs."""
        ...
