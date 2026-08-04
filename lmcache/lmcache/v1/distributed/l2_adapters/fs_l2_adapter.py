# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header).  Each ObjectKey maps to a separate ``.data`` file whose
name encodes all key fields so it can be reversed on startup.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import asyncio
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.memory_management import MemoryObj

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

_KEY_SEP = "@"
# ``@`` in both ``model_name`` and ``cache_salt`` is rejected by
# ObjectKey.__post_init__, so splitting on ``@`` is unambiguous.
# Kept in sync with native_connector_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp.
_PATH_SLASH_REPLACEMENT = "-SEP-"
_FILE_EXT = ".data"


def _readinto_full(
    f,  # typing: IO[bytes]
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Loop readinto() until *buf* is full or EOF.

    A single ``readinto()`` may return fewer bytes than
    *len(buf)* even when more data is available.  This
    helper keeps reading until the buffer is completely
    filled or the file reaches EOF.

    Returns:
        Total number of bytes read.
    """
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


async def _async_readinto_full(
    f,  # aiofiles async file handle
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Async version of :func:`_readinto_full`."""
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = await f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


def _object_key_to_filename(key: ObjectKey) -> str:
    """Build a reversible, filesystem-safe filename.

    Unsalted::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data

    Salted (trailing ``cache_salt``)::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data

    ``kv_rank`` is written in ``0x`` prefixed hex so each byte
    of the bitmap ``(ws<<24)|(rank<<16)|(local_ws<<8)|local``
    is directly readable. ``object_group_id`` is written in plain hex.
    """
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = (
        f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}{_FILE_EXT}"
    return f"{base}{_FILE_EXT}"


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Reverse ``_object_key_to_filename``.

    Accepts both the 4-field unsalted shape and the 5-field salted
    shape (trailing ``cache_salt``). Returns ``None`` for anything
    else. Since ``model_name`` is guaranteed not to contain ``@``,
    plain ``split`` suffices — no marker, no rsplit.
    """
    if not filename.endswith(_FILE_EXT):
        return None
    stem = filename[: -len(_FILE_EXT)]
    parts = stem.split(_KEY_SEP)
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        return None

    model_name = safe_model.replace(_PATH_SLASH_REPLACEMENT, "/")
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
        kv_rank = int(kv_rank_str, 16)
        object_group_id = int(object_group_str, 16)
        # ObjectKey.__post_init__ raises ValueError when the decoded
        # model_name / cache_salt violate the forbidden-char or length
        # invariants (e.g. a stray file from another tool on disk).
        # The contract here is to return None for anything unparsable,
        # so keep the constructor inside the try block.
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
        )
    except ValueError:
        return None


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
    ):
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")
        relative_tmp_dir = d.get("relative_tmp_dir", None)
        if relative_tmp_dir is not None:
            if not isinstance(relative_tmp_dir, str):
                raise ValueError("relative_tmp_dir must be a string")
        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS L2 adapter config fields:\n"
            "- base_path (str): directory for KV cache "
            "files (required)\n"
            "- relative_tmp_dir (str): relative sub-dir "
            "for temp files (optional, same as "
            "fs_connector_relative_tmp_dir)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)"
        )


class FSL2Adapter(L2AdapterInterface):
    """
    File-system backed L2 adapter with async I/O via *aiofiles*.

    Each file stores **only** the raw tensor bytes (no metadata
    header), which gives maximum I/O throughput.  The file name
    itself encodes the full ``ObjectKey`` so it is reversible.

    Thread safety is ensured via a lock for shared bookkeeping
    and an asyncio event loop running on a dedicated daemon
    thread.
    """

    _DELETE_CONCURRENCY = 64

    def __init__(self, config: FSL2AdapterConfig):
        super().__init__()
        self._config = config
        base = config.base_path
        self._base_path = Path(base)
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Temp-file strategy aligned with FSConnector:
        # if relative_tmp_dir is set, write to a sub-dir;
        # otherwise fall back to a .tmp suffix.
        self._relative_tmp_dir: Optional[Path] = None
        if config.relative_tmp_dir is not None:
            self._relative_tmp_dir = Path(config.relative_tmp_dir)
            if (
                self._relative_tmp_dir.is_absolute()
                or ".." in self._relative_tmp_dir.parts
            ):
                raise ValueError("Invalid relative_tmp_dir: " + config.relative_tmp_dir)
            (self._base_path / self._relative_tmp_dir).mkdir(
                parents=False, exist_ok=True
            )

        # I/O tuning options aligned with FSConnector
        self._read_ahead_size = config.read_ahead_size
        self._use_odirect = config.use_odirect
        self._os_disk_bs = 0
        if self._use_odirect:
            stat = os.statvfs(self._base_path)
            self._os_disk_bs = stat.f_bsize

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(keys, task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # No-op: the FS adapter tracks no per-key locks — a delete
        # racing the lookup→load window degrades the load to a miss.
        pass

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_load(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Status Interface
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for the FS L2 adapter."""
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "FSL2Adapter",
            "base_path": str(self._base_path),
            "use_odirect": self._use_odirect,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete each key's backing file and notify listeners.

        Args:
            keys: The object keys to delete.

        Note:
            No per-key locks: a delete racing a load turns that load
            into a miss; racing a store of the same key may leave the
            key re-stored.
        """
        if not keys:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._execute_delete(keys),
                self._loop,
            )
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("FSL2Adapter delete failed: %s", e)
            return
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains byte totals via ``_notify_keys_stored`` /
    # ``_notify_keys_deleted``, but the FS adapter declares no max capacity
    # (default 0) so ``supports_global_eviction`` returns ``False`` and
    # ``usage_fraction == -1.0`` — the eviction controller treats this as
    # "no eviction signal" and skips the adapter entirely.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("FSL2Adapter closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def _key_to_path(self, key: ObjectKey) -> Path:
        return self._base_path / _object_key_to_filename(key)

    async def _key_exists_on_disk(
        self,
        key: ObjectKey,
    ) -> bool:
        """Check whether the file for *key* exists on disk.

        Uses ``aiofiles.os.path.exists`` so the check is
        non-blocking and always reflects the real FS state,
        which is critical for multi-node shared-FS setups.
        """
        path = self._key_to_path(key)
        return await aiofiles.os.path.exists(path)

    def _key_to_file_and_tmp_path(self, key: ObjectKey) -> tuple[Path, Path]:
        """Return ``(final_path, tmp_path)``.

        When ``relative_tmp_dir`` is configured, the temp file
        is placed under that sub-directory (same behaviour as
        ``FSConnector._get_file_and_tmp_path``).  Otherwise a
        ``.tmp`` suffix is used.
        """
        fname = _object_key_to_filename(key)
        final = self._base_path / fname
        if self._relative_tmp_dir is not None:
            tmp = self._base_path / self._relative_tmp_dir / fname
        else:
            tmp = final.with_suffix(".tmp")
        return final, tmp

    # ---- O_DIRECT helpers -----------------------------------------------

    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf*.

        Returns the number of bytes actually read.
        Runs in an executor (not on the event loop).
        """
        fd = -1
        size = len(dst_buf)
        try:
            aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
            if not aligned:
                logger.warning(
                    "Cannot use O_DIRECT for %s, size is not aligned.",
                    file_path,
                )
                with open(file_path, "rb") as f:
                    return _readinto_full(f, dst_buf)

            fd = os.open(
                str(file_path),
                os.O_RDONLY | getattr(os, "O_DIRECT", 0),
            )
            with os.fdopen(fd, "rb", buffering=0) as fdo:
                fd = -1  # now managed by fdopen
                return _readinto_full(fdo, dst_buf)
        except Exception:
            logger.exception("Failed to O_DIRECT read %s", file_path)
            return 0
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _write_with_odirect(self, file_path: Path, buf: bytes) -> None:
        """Synchronous O_DIRECT write of *buf*.

        Runs in an executor (not on the event loop).
        """
        fd = -1
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_WRONLY | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            os.write(fd, buf)
        except Exception:
            logger.exception("Failed to O_DIRECT write %s", file_path)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    # ---- store ----------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk
                if await aiofiles.os.path.exists(file_path):
                    continue
                buf = obj.byte_array
                size = len(buf)

                try:
                    # Decide whether O_DIRECT is usable
                    do_odirect = self._use_odirect
                    if do_odirect:
                        aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                        if not aligned:
                            logger.warning(
                                "Cannot use O_DIRECT for "
                                "writing size %d, not "
                                "aligned to block size "
                                "%d.",
                                size,
                                self._os_disk_bs,
                            )
                            do_odirect = False

                    if do_odirect:
                        await self._loop.run_in_executor(
                            None,
                            self._write_with_odirect,
                            tmp_path,
                            buf,
                        )
                    else:
                        async with aiofiles.open(tmp_path, "wb") as f:
                            await f.write(buf)

                    await aiofiles.os.replace(tmp_path, file_path)
                    bytes_written += size
                    stored_keys.append(key)
                    stored_sizes.append(size)
                    logger.debug(
                        "FSL2Adapter stored key %s (%d bytes)",
                        file_path.name,
                        size,
                    )
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to store %s",
                        file_path,
                    )
                    if await aiofiles.os.path.exists(tmp_path):
                        await aiofiles.os.unlink(tmp_path)
                    success = False
        except Exception:
            logger.exception(
                "FSL2Adapter store task %s failed",
                task_id,
            )
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()

    # ---- lookup ---------------------------------------------------------

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            if not await self._key_exists_on_disk(key):
                continue
            bitmap.set(i)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

    # ---- load -----------------------------------------------------------

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            file_path = self._key_to_path(key)
            try:
                dst_buf = objects[i].byte_array
                expected = len(dst_buf)
                num_read: Optional[int] = None

                # O_DIRECT path (sync, via executor)
                if self._use_odirect:
                    num_read = await self._loop.run_in_executor(
                        None,
                        self._read_with_odirect,
                        file_path,
                        dst_buf,
                    )
                    if num_read != expected:
                        logger.warning(
                            "Incomplete O_DIRECT read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read or 0,
                        )
                    else:
                        bitmap.set(i)
                        logger.debug(
                            "FSL2Adapter loaded key %s (%d bytes, O_DIRECT)",
                            file_path.name,
                            num_read,
                        )
                    continue

                # Standard async path with optional
                # read-ahead
                expected = len(dst_buf)
                async with aiofiles.open(file_path, "rb") as f:
                    if self._read_ahead_size is None:
                        num_read = await _async_readinto_full(f, dst_buf)
                    else:
                        if not isinstance(dst_buf, memoryview):
                            dst_buf = memoryview(dst_buf)
                        # Trigger readahead with a
                        # small initial read
                        ra = self._read_ahead_size
                        n_head = await _async_readinto_full(f, dst_buf[:ra])
                        if n_head == ra:
                            n_tail = await _async_readinto_full(f, dst_buf[ra:])
                            num_read = n_head + n_tail
                        else:
                            num_read = n_head

                    if num_read != expected:
                        logger.warning(
                            "Incomplete read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read,
                        )
                        continue

                    bitmap.set(i)
                    logger.debug(
                        "FSL2Adapter loaded key %s (%d bytes)",
                        file_path.name,
                        num_read,
                    )
            except FileNotFoundError:
                continue
            except Exception:
                logger.exception(
                    "FSL2Adapter failed to load %s",
                    file_path,
                )
                continue

        loaded_keys = [keys[i] for i in bitmap.get_indices_list()]
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    # ---- delete ---------------------------------------------------------

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Unlink each key's file concurrently; return removed keys + sizes."""
        sem = asyncio.Semaphore(self._DELETE_CONCURRENCY)

        async def _delete_one(key: ObjectKey) -> tuple[ObjectKey, int] | None:
            file_path = self._key_to_path(key)
            async with sem:
                try:
                    size = (await aiofiles.os.stat(file_path)).st_size
                    await aiofiles.os.unlink(file_path)
                except FileNotFoundError:
                    return None
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to delete %s",
                        file_path,
                    )
                    return None
            return key, size

        results = await asyncio.gather(*(_delete_one(k) for k in keys))

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for res in results:
            if res is not None:
                key, size = res
                deleted_keys.append(key)
                deleted_sizes.append(size)
        return deleted_keys, deleted_sizes


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an FSL2Adapter from config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("fs", _create_fs_adapter)
