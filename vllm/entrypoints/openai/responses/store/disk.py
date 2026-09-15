from __future__ import annotations

import array
import asyncio
import hmac
import sqlite3
import sys
import time
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path

from vllm.logger import init_logger

from .base import (
    DiskEvictionResult,
    DiskEvictionStatus,
    SessionMetadata,
    SessionState,
    SessionStore,
)
from .crypto import DataDecryptionError, FramedAESGCMCipher
from .key_provider import (
    EphemeralKeyProvider,
    FileKeyProvider,
    KeyProvider,
    StaticKeyProvider,
)
from .metrics import ResponseStoreMetrics

logger = init_logger(__name__)


@unique
class DiskCopyStatus(str, Enum):
    """Disk copy state observed by memory eviction."""

    READY = "ready"
    NOT_FOUND = "not_found"
    RESPONSE_CHANGED = "response_changed"
    VERSION_CHANGED = "version_changed"
    EXPIRED = "expired"
    PENDING = "pending"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class DiskCopyResult:
    status: DiskCopyStatus


@dataclass(slots=True)
class _PendingWrite:
    session_id: str
    response_id: str
    delta_token_ids: list[int]
    created_at: int
    updated_at: int
    disk_idle_expires_at: int | None
    version: int
    replace_existing: bool = False


@dataclass(frozen=True, slots=True)
class _CorruptDiskCopy:
    session_id: str
    write_version: int


@dataclass(frozen=True, slots=True)
class _ReadStateResult:
    state: SessionState | None = None
    corrupt_copy: _CorruptDiskCopy | None = None


class SQLiteSessionStore(SessionStore):
    """
    SQLite 二级存储。

    - 构造时清空 Session 表；持久密钥模式保留密钥元数据和轮换周期。
    - save() 只写进程内 pending，不做 SQLite I/O。
    - 后台单 Writer 按时间窗口合并同一 Session 的多次 save。
    - Token ID 以独立 AES-GCM 帧加密，增量写入不解密历史数据。
    - SQLite transaction 保证批次原子提交。
    - latest_version / committed_version 保证只恢复“最新且完整”的状态。
    - Memory miss 时若最新版本尚未落盘，立即返回 None 让上层重新渲染。
    """

    _TOKEN_TYPECODE = "I"  # uint32
    _LEGACY_SCHEMA_VERSION = 1
    _SCHEMA_VERSION = 2
    _KEY_CHECK_PLAINTEXT = b"vllm-responses-store-key-check-v1"
    _KEY_CHECK_ASSOCIATED_DATA = b"responses-store-metadata"
    KEY_ROTATION_INTERVAL_SECONDS = 90 * 24 * 60 * 60

    def __init__(
        self,
        db_path: str,
        disk_idle_ttl_seconds: int | None = None,
        write_interval_seconds: float = 0.05,
        key_provider: KeyProvider | None = None,
        metrics: ResponseStoreMetrics | None = None,
        persistent_key: bool = False,
        key_rotation_interval_seconds: int = KEY_ROTATION_INTERVAL_SECONDS,
    ) -> None:
        if not db_path:
            raise ValueError("db_path must not be empty")
        if disk_idle_ttl_seconds is not None and disk_idle_ttl_seconds <= 0:
            raise ValueError("disk_idle_ttl_seconds must be greater than 0")
        if write_interval_seconds <= 0:
            raise ValueError("write_interval_seconds must be greater than 0")
        if array.array(self._TOKEN_TYPECODE).itemsize != 4:
            raise RuntimeError("32-bit unsigned int array is required")
        if persistent_key and key_provider is None:
            raise ValueError("persistent_key requires a persistent key provider")
        if key_rotation_interval_seconds <= 0:
            raise ValueError("key_rotation_interval_seconds must be greater than 0")

        self._db_path = db_path
        self._disk_idle_ttl_seconds = disk_idle_ttl_seconds
        self._write_interval_seconds = write_interval_seconds
        self._persistent_key = persistent_key
        self._key_rotation_interval_seconds = key_rotation_interval_seconds
        self._key_file_provider = (
            key_provider if isinstance(key_provider, FileKeyProvider) else None
        )
        self._cipher = FramedAESGCMCipher(
            key_provider if key_provider is not None else EphemeralKeyProvider()
        )
        self._metrics = metrics if metrics is not None else ResponseStoreMetrics()

        # 创建 SQLite 数据库文件所在的目录
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # 同一 Session 在 pending 中最多一条，连续 save 会合并 delta。
        self._pending: dict[str, _PendingWrite] = {}

        # Store 级单调版本避免为每个历史 Session 永久保留计数器。
        self._version_counter = 0
        self._latest_versions: dict[str, int] = {}
        self._committed_versions: dict[str, int] = {}
        self._latest_response_ids: dict[str, str] = {}
        self._failed_versions: dict[str, int] = {}

        # delete 后，<= 此版本的已捕获 Writer 批次全部作废，避免删除复活。
        self._discard_through_versions: dict[str, int] = {}
        self._inflight_versions: dict[str, int] = {}

        self._meta_lock = asyncio.Lock()

        self._io_lock = asyncio.Lock()
        # Writer 通知机制
        self._writer_event = asyncio.Event()
        self._writer_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

        # sqlite3 是阻塞 API。connection 可跨 worker thread 使用，但所有
        # SQLite 操作仍通过 _io_lock 串行，并放入 asyncio.to_thread()。
        # 对调用方异步执行，但多个SQLite操作串行执行
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            self._initialize_database()
        except Exception:
            self._conn.close()
            raise

    async def save(
        self,
        session_id: str,
        response_id: str,
        delta_token_ids: list[int],
    ) -> None:
        """只进入 pending buffer；真正 SQLite I/O 由后台 Writer 完成。"""
        self._ensure_open()
        self._ensure_writer_started()
        now = time.time_ns() // 1_000_000

        async with self._meta_lock:
            version = self._allocate_version_locked()
            self._latest_versions[session_id] = version
            self._latest_response_ids[session_id] = response_id

            pending = self._pending.get(session_id)
            if pending is None:
                self._pending[session_id] = _PendingWrite(
                    session_id=session_id,
                    response_id=response_id,
                    delta_token_ids=list(delta_token_ids),
                    created_at=now,
                    updated_at=now,
                    disk_idle_expires_at=self._build_expire_time(now),
                    version=version,
                )
            else:
                pending.delta_token_ids.extend(delta_token_ids)
                pending.response_id = response_id
                pending.updated_at = now
                pending.disk_idle_expires_at = self._build_expire_time(now)
                pending.version = version

            self._writer_event.set()

    async def save_snapshot(self, state: SessionState) -> None:
        """使用 Memory 完整快照重建 Disk 副本。"""
        self._ensure_open()
        self._ensure_writer_started()
        now = time.time_ns() // 1_000_000

        async with self._meta_lock:
            version = self._allocate_version_locked()
            self._latest_versions[state.session_id] = version
            self._latest_response_ids[state.session_id] = state.response_id

            pending = self._pending.get(state.session_id)
            if pending is None:
                self._pending[state.session_id] = _PendingWrite(
                    session_id=state.session_id,
                    response_id=state.response_id,
                    delta_token_ids=list(state.token_ids),
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    disk_idle_expires_at=self._build_expire_time(now),
                    version=version,
                    replace_existing=True,
                )
            else:
                pending.response_id = state.response_id
                pending.delta_token_ids = list(state.token_ids)
                pending.created_at = state.created_at
                pending.updated_at = state.updated_at
                pending.disk_idle_expires_at = self._build_expire_time(now)
                pending.version = version
                pending.replace_existing = True

            self._writer_event.set()

    async def needs_snapshot(self, session_id: str) -> bool:
        async with self._meta_lock:
            latest = self._latest_versions.get(session_id)
            return (
                latest is None
                or self._committed_versions.get(session_id) != latest
                or self._failed_versions.get(session_id) == latest
            )

    async def used_bytes(self) -> int:
        """返回当前 SQLite Session 行记录的逻辑总字节数。"""
        async with self._io_lock:
            return await asyncio.to_thread(self._used_bytes_sync)

    @property
    def metrics(self) -> ResponseStoreMetrics:
        return self._metrics

    @property
    def automatic_key_rotation_enabled(self) -> bool:
        return self._persistent_key and self._key_file_provider is not None

    async def rotate_key_if_due(self, now: int | None = None) -> bool:
        """Atomically re-encrypt the complete database when its key is due."""
        self._ensure_open()
        if not self.automatic_key_rotation_enabled:
            return False

        rotation_time = int(time.time()) if now is None else now
        async with self._io_lock:
            return await asyncio.to_thread(
                self._rotate_key_if_due_sync,
                rotation_time,
            )

    async def get(self, session_id: str) -> list[int] | None:
        state = await self.get_state(session_id)
        return None if state is None else state.token_ids.copy()

    async def get_state(self, session_id: str) -> SessionState | None:
        """返回当前完整 Disk 状态；最新版本未落盘时返回 None。"""
        self._ensure_open()
        expected = await self._get_complete_version(session_id)
        if expected is None:
            return None

        async with self._io_lock:
            if await self._get_complete_version(session_id) != expected:
                return None

            read_result = await asyncio.to_thread(
                self._read_state_sync,
                session_id,
                expected,
                time.time_ns() // 1_000_000,
            )

            if read_result.corrupt_copy is not None:
                async with self._meta_lock:
                    await asyncio.to_thread(
                        self._delete_corrupt_copies_sync,
                        [read_result.corrupt_copy],
                    )
                    self._invalidate_corrupt_copy_locked(read_result.corrupt_copy)
                logger.warning(
                    "Removed corrupted ResponseStore SQLite copy for session %s",
                    session_id,
                )
                return None

        if read_result.state is None:
            return None
        if await self._get_complete_version(session_id) != expected:
            return None
        return read_result.state

    async def exists(self, session_id: str) -> bool:
        """只有最新版本完整落盘时才认为 Disk 中存在可恢复 Session。"""
        self._ensure_open()
        expected = await self._get_complete_version(session_id)
        if expected is None:
            return False

        async with self._io_lock:
            expected = await self._get_complete_version(session_id)
            if expected is None:
                return False
            exists = await asyncio.to_thread(self._exists_sync, session_id, expected)

        return exists and await self._get_complete_version(session_id) == expected

    async def delete(self, session_id: str) -> bool:
        """删除已落盘数据，同时取消 pending，并防止已捕获批次重新写回。"""
        self._ensure_open()

        # 先阻塞 Writer，再执行物理删除。只有 SQLite 删除成功后才清理
        # 版本元数据，避免删除失败留下无法读取、也无法淘汰的孤儿行。
        async with self._io_lock, self._meta_lock:
            current = self._latest_versions.get(session_id, 0)
            previous_discard = self._discard_through_versions.get(session_id)
            if current:
                self._discard_through_versions[session_id] = max(
                    current, self._discard_through_versions.get(session_id, 0)
                )

            try:
                deleted = await asyncio.to_thread(self._delete_sync, session_id)
            except Exception:
                if previous_discard is None:
                    self._discard_through_versions.pop(session_id, None)
                else:
                    self._discard_through_versions[session_id] = previous_discard
                raise

            had_pending = self._pending.pop(session_id, None) is not None
            had_latest = self._latest_versions.pop(session_id, None) is not None
            self._committed_versions.pop(session_id, None)
            self._latest_response_ids.pop(session_id, None)
            self._failed_versions.pop(session_id, None)
            self._prune_discard_tombstone_locked(session_id)

        return had_pending or had_latest or deleted

    async def list(self) -> list[SessionState]:
        """低频管理接口：只返回当前完整可恢复的 Disk 状态。"""
        self._ensure_open()

        # list 期间短暂阻塞新的 Disk enqueue，以获得稳定版本关系。
        async with self._io_lock, self._meta_lock:
            rows, corrupt_copies = await asyncio.to_thread(self._list_sync)
            if corrupt_copies:
                await asyncio.to_thread(
                    self._delete_corrupt_copies_sync, corrupt_copies
                )
                for corrupt_copy in corrupt_copies:
                    self._invalidate_corrupt_copy_locked(corrupt_copy)
                logger.warning(
                    "Removed %d corrupted ResponseStore SQLite copies",
                    len(corrupt_copies),
                )

            result: list[SessionState] = []
            for state, version in rows:
                if (
                    self._latest_versions.get(state.session_id) == version
                    and self._committed_versions.get(state.session_id) == version
                ):
                    result.append(state)
            return result

    async def list_metadata(self) -> list[SessionMetadata]:
        """Return complete-copy metadata without reading encrypted token data."""
        self._ensure_open()

        async with self._io_lock, self._meta_lock:
            rows = await asyncio.to_thread(self._list_metadata_sync)
            return [
                metadata
                for metadata, version in rows
                if (
                    self._latest_versions.get(metadata.session_id) == version
                    and self._committed_versions.get(metadata.session_id) == version
                )
            ]

    async def is_complete(self, session_id: str) -> bool:
        """
        给 Memory 淘汰模块使用的轻量接口。

        推荐只淘汰 is_complete(session_id) == True 的 Memory Session，
        这样不会出现 Memory 刚淘汰、Disk 最新版本还未提交的短暂空窗。
        """
        return await self._get_complete_version(session_id) is not None

    async def ensure_eviction_copy(
        self,
        session_id: str,
        expected_response_id: str,
        now: int,
    ) -> DiskCopyResult:
        """验证淘汰所需的同版本 Disk 副本，不刷新 Disk TTL。"""
        self._ensure_open()
        async with self._meta_lock:
            latest = self._latest_versions.get(session_id)
            if latest is None:
                return DiskCopyResult(DiskCopyStatus.NOT_FOUND)

            if self._latest_response_ids.get(session_id) != expected_response_id:
                return DiskCopyResult(DiskCopyStatus.RESPONSE_CHANGED)

            if self._failed_versions.get(session_id) == latest:
                return DiskCopyResult(DiskCopyStatus.FAILED)

            if self._committed_versions.get(session_id) != latest:
                self._writer_event.set()
                return DiskCopyResult(DiskCopyStatus.PENDING)

            expected_version = latest

        try:
            async with self._io_lock:
                status = await asyncio.to_thread(
                    self._inspect_eviction_copy_sync,
                    session_id,
                    expected_response_id,
                    expected_version,
                    now,
                )
        except sqlite3.Error:
            logger.exception("Failed to inspect ResponseStore SQLite copy")
            return DiskCopyResult(DiskCopyStatus.FAILED)

        async with self._meta_lock:
            if (
                self._latest_versions.get(session_id) != expected_version
                or self._committed_versions.get(session_id) != expected_version
                or self._latest_response_ids.get(session_id) != expected_response_id
            ):
                return DiskCopyResult(DiskCopyStatus.VERSION_CHANGED)

        return DiskCopyResult(status)

    async def evict_if_unchanged(
        self,
        session_id: str,
        expected_response_id: str,
    ) -> DiskEvictionResult:
        """response_id 未变化时删除 Disk 副本并返回实际大小。"""
        self._ensure_open()

        async with self._io_lock, self._meta_lock:
            latest = self._latest_versions.get(session_id)
            if latest is None:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.NOT_FOUND,
                )
            if self._latest_response_ids.get(session_id) != expected_response_id:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.RESPONSE_CHANGED,
                )
            if self._committed_versions.get(session_id) != latest:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    DiskEvictionStatus.VERSION_CHANGED,
                )

            status, freed_bytes = await asyncio.to_thread(
                self._delete_if_unchanged_sync,
                session_id,
                expected_response_id,
                latest,
            )
            if status is not DiskEvictionStatus.EVICTED:
                return DiskEvictionResult(
                    session_id,
                    expected_response_id,
                    status,
                )

            self._discard_through_versions[session_id] = max(
                latest,
                self._discard_through_versions.get(session_id, 0),
            )
            self._pending.pop(session_id, None)
            self._latest_versions.pop(session_id, None)
            self._committed_versions.pop(session_id, None)
            self._latest_response_ids.pop(session_id, None)
            self._failed_versions.pop(session_id, None)
            self._prune_discard_tombstone_locked(session_id)

            return DiskEvictionResult(
                session_id,
                expected_response_id,
                DiskEvictionStatus.EVICTED,
                freed_bytes=freed_bytes,
            )

    async def close(self) -> None:
        if self._closed:
            return

        # 不直接 cancel 正在 to_thread() 中执行的 SQLite 操作；取消 await
        # 并不能停止已经运行的工作线程。让 Writer 自然收尾后再 close。
        async with self._meta_lock:
            self._closing = True

        self._writer_event.set()

        task = self._writer_task
        self._writer_task = None
        if task is not None:
            await task

        async with self._io_lock:
            await asyncio.to_thread(self._conn.close)
        self._closed = True

    def _initialize_database(self) -> None:
        cursor = self._conn.cursor()
        key_managed_store_initialized = False
        pending_key_to_promote: bytes | None = None
        discard_pending_key = False
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(
                "PRAGMA synchronous=FULL"
                if self._persistent_key
                else "PRAGMA synchronous=NORMAL"
            )
            cursor.execute("BEGIN IMMEDIATE")

            if self._persistent_key:
                (
                    pending_key_to_promote,
                    discard_pending_key,
                ) = self._initialize_key_managed_database(cursor)
                key_managed_store_initialized = True
            else:
                self._initialize_ephemeral_database(cursor)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

        if pending_key_to_promote is not None:
            assert self._key_file_provider is not None
            self._key_file_provider.promote_pending_key(pending_key_to_promote)
        elif discard_pending_key:
            assert self._key_file_provider is not None
            try:
                self._key_file_provider.discard_pending_key()
            except RuntimeError:
                logger.warning(
                    "Unable to remove stale Responses store pending key",
                    exc_info=True,
                )

        if key_managed_store_initialized:
            logger.info(
                "Initialized Responses store with empty session state and "
                "persistent key metadata in %s",
                self._db_path,
            )

    def _initialize_ephemeral_database(self, cursor: sqlite3.Cursor) -> None:
        if self._table_exists(cursor, "store_metadata"):
            raise RuntimeError(
                "Refusing to reset a persistent Responses store without its key file"
            )

        cursor.execute("DROP TABLE IF EXISTS session_state")
        self._create_session_state_table(cursor)

    def _initialize_key_managed_database(
        self, cursor: sqlite3.Cursor
    ) -> tuple[bytes | None, bool]:
        metadata_exists = self._table_exists(cursor, "store_metadata")
        pending_key_to_promote: bytes | None = None
        discard_pending_key = False

        if metadata_exists:
            (
                pending_key_to_promote,
                discard_pending_key,
            ) = self._select_database_key(cursor)
            self._migrate_store_metadata(cursor, int(time.time()))
            self._validate_store_metadata(cursor)
        else:
            self._create_store_metadata(cursor)

        cursor.execute("DROP TABLE IF EXISTS session_state")
        self._create_session_state_table(cursor)
        return pending_key_to_promote, discard_pending_key

    @staticmethod
    def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
        row = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _create_session_state_table(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL,
                token_ids BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                disk_idle_expires_at INTEGER,
                disk_size_bytes INTEGER NOT NULL,
                write_version INTEGER NOT NULL
            )
            """
        )

    def _create_store_metadata(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE store_metadata (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                schema_version INTEGER NOT NULL,
                key_check BLOB NOT NULL,
                key_rotated_at INTEGER NOT NULL
            )
            """
        )
        key_check = self._cipher.encrypt(
            self._KEY_CHECK_PLAINTEXT,
            self._KEY_CHECK_ASSOCIATED_DATA,
        )
        cursor.execute(
            """
            INSERT INTO store_metadata (
                singleton_id, schema_version, key_check, key_rotated_at
            ) VALUES (1, ?, ?, ?)
            """,
            (self._SCHEMA_VERSION, key_check, int(time.time())),
        )

    def _read_store_metadata(
        self, cursor: sqlite3.Cursor
    ) -> tuple[int, bytes, int | None]:
        columns = {
            str(row[1])
            for row in cursor.execute("PRAGMA table_info(store_metadata)").fetchall()
        }
        if not {"singleton_id", "schema_version", "key_check"} <= columns:
            raise RuntimeError("Persistent Responses store metadata is incompatible")

        has_rotation_time = "key_rotated_at" in columns
        row = cursor.execute(
            "SELECT schema_version, key_check, key_rotated_at "
            "FROM store_metadata WHERE singleton_id = 1"
            if has_rotation_time
            else "SELECT schema_version, key_check, NULL "
            "FROM store_metadata WHERE singleton_id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Persistent Responses store metadata is missing")

        schema_version = int(row[0])
        if schema_version not in {
            self._LEGACY_SCHEMA_VERSION,
            self._SCHEMA_VERSION,
        }:
            raise RuntimeError(
                "Unsupported Responses store schema version: "
                f"{schema_version}; expected {self._LEGACY_SCHEMA_VERSION} "
                f"or {self._SCHEMA_VERSION}"
            )
        if schema_version == self._SCHEMA_VERSION and not has_rotation_time:
            raise RuntimeError(
                "Persistent Responses store metadata is missing key_rotated_at"
            )
        rotated_at = None if row[2] is None else int(row[2])
        return schema_version, bytes(row[1]), rotated_at

    def _validate_store_metadata(self, cursor: sqlite3.Cursor) -> int:
        schema_version, encrypted_check, rotated_at = self._read_store_metadata(cursor)
        if schema_version != self._SCHEMA_VERSION or rotated_at is None:
            raise RuntimeError("Persistent Responses store metadata was not migrated")
        if not self._key_matches(encrypted_check, self._cipher):
            raise ValueError(
                "Responses store key does not match the persistent database"
            )
        return rotated_at

    def _select_database_key(self, cursor: sqlite3.Cursor) -> tuple[bytes | None, bool]:
        _, encrypted_check, _ = self._read_store_metadata(cursor)
        if self._key_matches(encrypted_check, self._cipher):
            discard_pending = (
                self._key_file_provider is not None
                and self._key_file_provider.get_pending_key() is not None
            )
            return None, discard_pending

        if self._key_file_provider is not None:
            pending_key = self._key_file_provider.get_pending_key()
            if pending_key is not None:
                pending_cipher = FramedAESGCMCipher(StaticKeyProvider(pending_key))
                if self._key_matches(encrypted_check, pending_cipher):
                    self._cipher = pending_cipher
                    return pending_key, False

        raise ValueError("Responses store key does not match the persistent database")

    def _migrate_store_metadata(self, cursor: sqlite3.Cursor, now: int) -> None:
        schema_version, _, rotated_at = self._read_store_metadata(cursor)
        if schema_version == self._SCHEMA_VERSION:
            if rotated_at is None:
                raise RuntimeError(
                    "Persistent Responses store metadata has no rotation time"
                )
            return

        cursor.execute("ALTER TABLE store_metadata ADD COLUMN key_rotated_at INTEGER")
        cursor.execute(
            """
            UPDATE store_metadata
            SET schema_version = ?, key_rotated_at = ?
            WHERE singleton_id = 1
            """,
            (self._SCHEMA_VERSION, now),
        )

    def _key_matches(
        self,
        encrypted_check: bytes,
        cipher: FramedAESGCMCipher,
    ) -> bool:
        try:
            key_check = cipher.decrypt(
                encrypted_check,
                self._KEY_CHECK_ASSOCIATED_DATA,
            )
        except DataDecryptionError:
            return False
        return hmac.compare_digest(key_check, self._KEY_CHECK_PLAINTEXT)

    def _rotate_key_if_due_sync(self, now: int) -> bool:
        key_provider = self._key_file_provider
        assert key_provider is not None

        cursor = self._conn.cursor()
        committed = False
        new_key: bytes | None = None
        try:
            _, encrypted_check, rotated_at = self._read_store_metadata(cursor)
            pending_key = key_provider.get_pending_key()
            if pending_key is not None:
                pending_cipher = FramedAESGCMCipher(StaticKeyProvider(pending_key))
                if self._key_matches(encrypted_check, pending_cipher):
                    self._cipher = pending_cipher
                    key_provider.promote_pending_key(pending_key)
                elif self._key_matches(encrypted_check, self._cipher):
                    key_provider.discard_pending_key()
                else:
                    raise ValueError(
                        "Neither active nor pending Responses store key matches "
                        "the persistent database"
                    )

            if rotated_at is None:
                raise RuntimeError(
                    "Persistent Responses store metadata has no rotation time"
                )
            if not self._key_matches(encrypted_check, self._cipher):
                raise ValueError(
                    "Responses store key does not match the persistent database"
                )
            if now < rotated_at + self._key_rotation_interval_seconds:
                return False

            new_key = key_provider.stage_new_key()
            new_cipher = FramedAESGCMCipher(StaticKeyProvider(new_key))

            cursor.execute("BEGIN IMMEDIATE")
            _, encrypted_check, rotated_at = self._read_store_metadata(cursor)
            if rotated_at is None:
                raise RuntimeError(
                    "Persistent Responses store metadata has no rotation time"
                )
            if not self._key_matches(encrypted_check, self._cipher):
                raise ValueError("Responses store key changed before rotation")
            if now < rotated_at + self._key_rotation_interval_seconds:
                self._conn.rollback()
                key_provider.discard_pending_key()
                return False

            rotated_sessions = self._reencrypt_all_sessions(
                cursor,
                old_cipher=self._cipher,
                new_cipher=new_cipher,
            )
            new_key_check = new_cipher.encrypt(
                self._KEY_CHECK_PLAINTEXT,
                self._KEY_CHECK_ASSOCIATED_DATA,
            )
            cursor.execute(
                """
                UPDATE store_metadata
                SET key_check = ?, key_rotated_at = ?
                WHERE singleton_id = 1
                """,
                (new_key_check, now),
            )
            self._conn.commit()
            committed = True
            self._cipher = new_cipher

            key_provider.promote_pending_key(new_key)
            logger.info(
                "Rotated Responses store key for %d persisted sessions in %s",
                rotated_sessions,
                self._db_path,
            )
            return True
        except Exception:
            if not committed:
                self._conn.rollback()
                if new_key is not None:
                    try:
                        key_provider.discard_pending_key()
                    except RuntimeError:
                        logger.warning(
                            "Unable to remove failed Responses store pending key",
                            exc_info=True,
                        )
            raise
        finally:
            cursor.close()

    def _reencrypt_all_sessions(
        self,
        cursor: sqlite3.Cursor,
        old_cipher: FramedAESGCMCipher,
        new_cipher: FramedAESGCMCipher,
    ) -> int:
        rotated_sessions = 0
        last_session_id: str | None = None
        try:
            while True:
                if last_session_id is None:
                    rows = cursor.execute(
                        """
                        SELECT session_id, response_id, token_ids
                        FROM session_state
                        ORDER BY session_id
                        LIMIT 128
                        """
                    ).fetchall()
                else:
                    rows = cursor.execute(
                        """
                        SELECT session_id, response_id, token_ids
                        FROM session_state
                        WHERE session_id > ?
                        ORDER BY session_id
                        LIMIT 128
                        """,
                        (last_session_id,),
                    ).fetchall()
                if not rows:
                    break

                for session_id_value, response_id_value, encrypted_blob in rows:
                    session_id = str(session_id_value)
                    response_id = str(response_id_value)
                    encryption_context = self._encryption_context(session_id)
                    plaintext = old_cipher.decrypt(
                        bytes(encrypted_blob),
                        encryption_context,
                    )
                    encryption_started_ns = time.perf_counter_ns()
                    new_encrypted_blob = new_cipher.encrypt(
                        plaintext,
                        encryption_context,
                    )
                    self._metrics.record_encryption(
                        plaintext_bytes=len(plaintext),
                        duration_ns=(time.perf_counter_ns() - encryption_started_ns),
                    )
                    disk_size = self._estimate_disk_size_bytes(
                        session_id,
                        response_id,
                        new_encrypted_blob,
                    )
                    cursor.execute(
                        """
                        UPDATE session_state
                        SET token_ids = ?, disk_size_bytes = ?
                        WHERE session_id = ?
                        """,
                        (new_encrypted_blob, disk_size, session_id),
                    )
                    rotated_sessions += 1
                last_session_id = str(rows[-1][0])
        except DataDecryptionError as exc:
            raise DataDecryptionError(
                "Unable to rotate corrupted Responses store data"
            ) from exc
        return rotated_sessions

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("SQLiteSessionStore is closing or already closed")

    def _ensure_writer_started(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            # 创建后台Writer Task
            self._writer_task = asyncio.create_task(
                self._writer_loop(), name="responses-sqlite-session-writer"
            )

    def _allocate_version_locked(self) -> int:
        self._version_counter += 1
        return self._version_counter

    def _prune_discard_tombstone_locked(self, session_id: str) -> None:
        discarded_through = self._discard_through_versions.get(session_id)
        if discarded_through is None:
            return

        inflight_version = self._inflight_versions.get(session_id)
        if inflight_version is not None and inflight_version <= discarded_through:
            return

        self._discard_through_versions.pop(session_id, None)

    def _invalidate_corrupt_copy_locked(self, corrupt_copy: _CorruptDiskCopy) -> None:
        session_id = corrupt_copy.session_id
        if session_id not in self._committed_versions:
            return

        latest = self._latest_versions.get(session_id, corrupt_copy.write_version)
        self._discard_through_versions[session_id] = max(
            latest,
            self._discard_through_versions.get(session_id, 0),
        )
        self._pending.pop(session_id, None)
        self._latest_versions.pop(session_id, None)
        self._committed_versions.pop(session_id, None)
        self._latest_response_ids.pop(session_id, None)
        self._failed_versions.pop(session_id, None)
        self._prune_discard_tombstone_locked(session_id)

    async def _writer_loop(self) -> None:
        while True:
            await self._writer_event.wait()
            if not self._closing:
                # 短合并窗口：多个 save 尽量进入一个 SQLite transaction。
                await asyncio.sleep(self._write_interval_seconds)

            async with self._meta_lock:
                batch = self._pending
                self._pending = {}
                self._writer_event.clear()
                self._inflight_versions.update(
                    (session_id, pending.version)
                    for session_id, pending in batch.items()
                )

            if not batch:
                if self._closing:
                    return
                continue

            try:
                try:
                    async with self._io_lock:
                        # 只过滤 delete 已明确作废的批次。不能因为出现更新版本就
                        # 跳过旧批次，否则下一批 delta 会失去落盘基础。
                        async with self._meta_lock:
                            valid = {
                                sid: p
                                for sid, p in batch.items()
                                if p.version
                                > self._discard_through_versions.get(sid, 0)
                            }

                        if valid:
                            await asyncio.to_thread(self._write_batch_sync, valid)

                        async with self._meta_lock:
                            for sid, pending in valid.items():
                                if pending.version > (
                                    self._discard_through_versions.get(sid, 0)
                                ):
                                    self._committed_versions[sid] = pending.version
                                    failed_version = self._failed_versions.get(sid)
                                    if (
                                        failed_version is not None
                                        and failed_version <= pending.version
                                    ):
                                        self._failed_versions.pop(sid, None)

                except Exception:
                    logger.exception("Failed to flush ResponseStore SQLite batch")
                    if not self._closing:
                        await self._mark_failed_batch(batch)
            finally:
                async with self._meta_lock:
                    for session_id, pending in batch.items():
                        if self._inflight_versions.get(session_id) == pending.version:
                            self._inflight_versions.pop(session_id, None)
                        self._prune_discard_tombstone_locked(session_id)

            if self._closing:
                async with self._meta_lock:
                    if not self._pending:
                        return
                    self._writer_event.set()

    async def _mark_failed_batch(self, batch: dict[str, _PendingWrite]) -> None:
        async with self._meta_lock:
            for session_id in batch:
                latest = self._latest_versions.get(session_id)
                if latest is None:
                    continue
                if latest <= self._discard_through_versions.get(session_id, 0):
                    continue

                # Newer deltas depend on the failed batch. Drop them from Disk;
                # Memory remains authoritative and the next save writes a snapshot.
                self._pending.pop(session_id, None)
                self._failed_versions[session_id] = latest

    async def _get_complete_version(self, session_id: str) -> int | None:
        async with self._meta_lock:
            latest = self._latest_versions.get(session_id)
            if latest is None or self._committed_versions.get(session_id) != latest:
                return None
            if latest <= self._discard_through_versions.get(session_id, 0):
                return None
            return latest

    def _write_batch_sync(self, batch: dict[str, _PendingWrite]) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            for pending in batch.values():
                row = cursor.execute(
                    """
                    SELECT token_ids, created_at FROM session_state
                    WHERE session_id = ?
                    """,
                    (pending.session_id,),
                ).fetchone()

                plaintext_blob = self._encode_token_ids(pending.delta_token_ids)
                encryption_started_ns = time.perf_counter_ns()
                delta_blob = self._cipher.encrypt(
                    plaintext_blob,
                    self._encryption_context(pending.session_id),
                )
                self._metrics.record_encryption(
                    plaintext_bytes=len(plaintext_blob),
                    duration_ns=time.perf_counter_ns() - encryption_started_ns,
                )
                if row is None or pending.replace_existing:
                    token_blob = delta_blob
                    created_at = pending.created_at
                else:
                    # 不解密历史 token；直接追加独立密文帧。
                    token_blob = bytes(row[0]) + delta_blob
                    created_at = int(row[1])

                disk_size = self._estimate_disk_size_bytes(
                    pending.session_id, pending.response_id, token_blob
                )

                cursor.execute(
                    """
                    INSERT INTO session_state (
                        session_id, response_id, token_ids, created_at, updated_at,
                        disk_idle_expires_at, disk_size_bytes, write_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        response_id = excluded.response_id,
                        token_ids = excluded.token_ids,
                        updated_at = excluded.updated_at,
                        disk_idle_expires_at = excluded.disk_idle_expires_at,
                        disk_size_bytes = excluded.disk_size_bytes,
                        write_version = excluded.write_version
                    """,
                    (
                        pending.session_id,
                        pending.response_id,
                        token_blob,
                        created_at,
                        pending.updated_at,
                        pending.disk_idle_expires_at,
                        disk_size,
                        pending.version,
                    ),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _read_state_sync(
        self, session_id: str, expected_version: int, now: int
    ) -> _ReadStateResult:
        cursor = self._conn.cursor()
        try:
            row = cursor.execute(
                """
                SELECT response_id, token_ids, created_at,
                       disk_size_bytes, write_version
                FROM session_state WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None or int(row[4]) != expected_version:
                return _ReadStateResult()

            try:
                token_ids = self._decrypt_token_ids(session_id, row[1])
            except DataDecryptionError:
                return _ReadStateResult(
                    corrupt_copy=_CorruptDiskCopy(session_id, expected_version)
                )

            disk_expires_at = self._build_expire_time(now)
            cursor.execute(
                """
                UPDATE session_state
                SET updated_at = ?, disk_idle_expires_at = ?
                WHERE session_id = ? AND write_version = ?
                """,
                (now, disk_expires_at, session_id, expected_version),
            )
            self._conn.commit()

            return _ReadStateResult(
                state=SessionState(
                    session_id=session_id,
                    response_id=str(row[0]),
                    token_ids=token_ids,
                    created_at=int(row[2]),
                    updated_at=now,
                    disk_idle_expires_at=disk_expires_at,
                    disk_size_bytes=int(row[3]),
                )
            )
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _exists_sync(self, session_id: str, expected_version: int) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM session_state
            WHERE session_id = ? AND write_version = ? LIMIT 1
            """,
            (session_id, expected_version),
        ).fetchone()
        return row is not None

    def _inspect_eviction_copy_sync(
        self,
        session_id: str,
        expected_response_id: str,
        expected_version: int,
        now: int,
    ) -> DiskCopyStatus:
        row = self._conn.execute(
            """
            SELECT response_id, token_ids, disk_idle_expires_at, write_version
            FROM session_state WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return DiskCopyStatus.NOT_FOUND
        if int(row[3]) != expected_version:
            return DiskCopyStatus.VERSION_CHANGED
        if str(row[0]) != expected_response_id:
            return DiskCopyStatus.RESPONSE_CHANGED

        try:
            self._decrypt_token_ids(session_id, row[1])
        except DataDecryptionError:
            return DiskCopyStatus.INVALID

        expires_at = None if row[2] is None else int(row[2])
        if expires_at is not None and expires_at <= now:
            return DiskCopyStatus.EXPIRED
        return DiskCopyStatus.READY

    def _delete_if_unchanged_sync(
        self,
        session_id: str,
        expected_response_id: str,
        expected_version: int,
    ) -> tuple[DiskEvictionStatus, int]:
        cursor = self._conn.cursor()
        try:
            row = cursor.execute(
                """
                SELECT response_id, disk_size_bytes, write_version
                FROM session_state WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return DiskEvictionStatus.NOT_FOUND, 0
            if str(row[0]) != expected_response_id:
                return DiskEvictionStatus.RESPONSE_CHANGED, 0
            if int(row[2]) != expected_version:
                return DiskEvictionStatus.VERSION_CHANGED, 0

            cursor.execute(
                """
                DELETE FROM session_state
                WHERE session_id = ? AND response_id = ? AND write_version = ?
                """,
                (session_id, expected_response_id, expected_version),
            )
            if cursor.rowcount == 0:
                return DiskEvictionStatus.VERSION_CHANGED, 0

            self._conn.commit()
            return DiskEvictionStatus.EVICTED, int(row[1])
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _delete_sync(self, session_id: str) -> bool:
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                "DELETE FROM session_state WHERE session_id = ?", (session_id,)
            )
            deleted = cursor.rowcount > 0
            self._conn.commit()
            return deleted
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _delete_corrupt_copies_sync(
        self, corrupt_copies: list[_CorruptDiskCopy]
    ) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.executemany(
                """
                DELETE FROM session_state
                WHERE session_id = ? AND write_version = ?
                """,
                [(copy.session_id, copy.write_version) for copy in corrupt_copies],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def _used_bytes_sync(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(disk_size_bytes), 0) FROM session_state"
        ).fetchone()
        return int(row[0])

    def _list_sync(
        self,
    ) -> tuple[list[tuple[SessionState, int]], list[_CorruptDiskCopy]]:
        rows = self._conn.execute(
            """
            SELECT session_id, response_id, token_ids, created_at, updated_at,
                   disk_idle_expires_at, disk_size_bytes, write_version
            FROM session_state
            """
        ).fetchall()

        result: list[tuple[SessionState, int]] = []
        corrupt_copies: list[_CorruptDiskCopy] = []
        for row in rows:
            session_id = str(row[0])
            write_version = int(row[7])
            try:
                token_ids = self._decrypt_token_ids(session_id, row[2])
            except DataDecryptionError:
                corrupt_copies.append(_CorruptDiskCopy(session_id, write_version))
                continue

            state = SessionState(
                session_id=session_id,
                response_id=str(row[1]),
                token_ids=token_ids,
                created_at=int(row[3]),
                updated_at=int(row[4]),
                disk_idle_expires_at=None if row[5] is None else int(row[5]),
                disk_size_bytes=int(row[6]),
            )
            result.append((state, write_version))
        return result, corrupt_copies

    def _list_metadata_sync(self) -> list[tuple[SessionMetadata, int]]:
        rows = self._conn.execute(
            """
            SELECT session_id, response_id, created_at, updated_at,
                   disk_idle_expires_at, disk_size_bytes, write_version
            FROM session_state
            """
        ).fetchall()

        return [
            (
                SessionMetadata(
                    session_id=str(row[0]),
                    response_id=str(row[1]),
                    created_at=int(row[2]),
                    updated_at=int(row[3]),
                    disk_idle_expires_at=(None if row[4] is None else int(row[4])),
                    disk_size_bytes=int(row[5]),
                ),
                int(row[6]),
            )
            for row in rows
        ]

    def _build_expire_time(self, now: int) -> int | None:
        if self._disk_idle_ttl_seconds is None:
            return None
        return now + self._disk_idle_ttl_seconds * 1_000

    @classmethod
    def _encode_token_ids(cls, token_ids: list[int]) -> bytes:
        values = array.array(cls._TOKEN_TYPECODE, token_ids)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    @classmethod
    def _decode_token_ids(cls, data: bytes) -> list[int]:
        if len(data) % 4 != 0:
            raise ValueError("invalid token_ids blob length")
        values = array.array(cls._TOKEN_TYPECODE)
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()

    def _decrypt_token_ids(self, session_id: str, data: bytes) -> list[int]:
        try:
            plaintext = self._cipher.decrypt(
                bytes(data), self._encryption_context(session_id)
            )
            return self._decode_token_ids(plaintext)
        except DataDecryptionError:
            raise
        except (TypeError, ValueError) as exc:
            raise DataDecryptionError("invalid encrypted token data") from exc

    @staticmethod
    def _encryption_context(session_id: str) -> bytes:
        return session_id.encode("utf-8")

    @staticmethod
    def _estimate_disk_size_bytes(
        session_id: str, response_id: str, token_blob: bytes
    ) -> int:
        """逻辑大小估算，不追求精确等于 SQLite page 占用。"""
        fixed_integer_bytes = 5 * 8
        return (
            len(session_id.encode("utf-8"))
            + len(response_id.encode("utf-8"))
            + len(token_blob)
            + fixed_integer_bytes
        )
