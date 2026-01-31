# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import shutil
import signal
import threading
import time
from collections.abc import Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import filelock
import lmdb
from typing_extensions import Self

from vllm import envs
from vllm.distributed.device_communicators.shm_object_storage import MsgpackSerde
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.utils.system_utils import decorate_logs, get_mp_context, set_process_title

from .inputs import MultiModalKwargsItem
from .processing.processor import ResolvedPromptUpdate

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

    from vllm.config import VllmConfig

    from .cache import MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem

logger = init_logger(__name__)

OPEN_ENVS_LOCK = threading.Lock()
OPEN_ENVS = dict[str, lmdb.Environment]()
REGISTERED_FORK_HANDLER = False


def _on_fork():
    with OPEN_ENVS_LOCK:
        for env in OPEN_ENVS.values():
            env.close()
        OPEN_ENVS.clear()


def ensure_lmdb_env(path: str, **kwargs) -> lmdb.Environment:
    """Opens or reuses an LMDB environment."""
    global REGISTERED_FORK_HANDLER

    with OPEN_ENVS_LOCK:
        # A given LMDB environment can only be opened once per process,
        # and not across forks.
        if existing_env := OPEN_ENVS.get(path):
            logger.debug("Reusing existing LMDB environment at %s", path)
            return existing_env
        else:
            lmdb_env = lmdb.Environment(
                path=path,
                **kwargs,
            )

            OPEN_ENVS[path] = lmdb_env
            if not REGISTERED_FORK_HANDLER:
                os.register_at_fork(after_in_child=_on_fork)
                REGISTERED_FORK_HANDLER = True
            return lmdb_env


class LmdbMultiModalCache:
    """LMDB-based multi-modal processor cache."""

    CACHES_DIR = os.path.join(envs.VLLM_CACHE_ROOT, "mm_caches")

    DB_TIMESTAMPS_AND_HASHES = b"timestamps_and_hashes"
    DB_HASH_TO_TIMESTAMP = b"hash_to_timestamp"
    DB_HASH_TO_OBJECT = b"hash_to_object"
    DB_HASH_TO_PROMPT_UPDATES = b"hash_to_prompt_updates"

    MAP_SIZE_MULTIPLIER = 2
    MINIMUM_MAP_SIZE = GiB_bytes

    INT_SIZE = 4  # Used for both timestamps (seconds) and chunk indices
    LMDB_PAGE_HEADER_SIZE = 16
    LMDB_PAGE_ID_SIZE = 8
    HASH_ITEM_KEY = "lmdb_mm_hash"

    EVICTOR_READER_CHECK_INTERVAL = 60.0  # seconds
    EVICTOR_BATCH_SIZE_FRACTION = 0.6  # Relative to the max page ids per page
    EVICTOR_MIN_UTILIZATION = 0.5  # Start evicting at 50% utilization
    EVICTOR_MAX_UTILIZATION = 1.0  # Reach maximum duty cycle at 100% utilization
    EVICTOR_MAX_INTERVAL = 15.0  # seconds
    EVICTOR_MAX_DUTY_CYCLE = 0.1  # 10%

    def __init__(
        self,
        cache_dir: str,
        cache_size: int,
        min_eviction_age: int,
        max_object_size: int,
    ) -> None:
        super().__init__()
        os.makedirs(cache_dir, exist_ok=True)

        self._cache_dir = cache_dir
        self._cache_size = cache_size
        self._min_eviction_age = min_eviction_age
        self._max_object_size = max_object_size
        # LMDB can require additional space beyond the maximum amoun of data
        # we're storing, so allocate extra space. (Once the map size is
        # exhausted, even eviction is liable to fail.)
        map_size = max(
            self._cache_size * self.MAP_SIZE_MULTIPLIER, self.MINIMUM_MAP_SIZE
        )
        self.lmdb_env = ensure_lmdb_env(
            cache_dir, map_size=map_size, max_dbs=4, writemap=True, map_async=True
        )

        # Large objects are stored in chunks to fit within LMDB page size limits and
        # avoid pathological free list fragmentation.
        overflow_page_data_size = (
            cast(int, self.lmdb_env.stat()["psize"]) - self.LMDB_PAGE_HEADER_SIZE
        )
        self._max_chunk_size = overflow_page_data_size

        # The maximum number of LMDB page IDs that can fit in a single overflow page.
        # Used to try to keep the freed page IDs in a transaction within a single page.
        self._max_page_ids_per_page = (
            overflow_page_data_size // self.LMDB_PAGE_ID_SIZE - 1
        )

        # (timestamp, hash) => ()
        self.timestamps_and_hashes = self.lmdb_env.open_db(
            self.DB_TIMESTAMPS_AND_HASHES
        )

        # hash => timestamp
        self.hash_to_timestamp = self.lmdb_env.open_db(self.DB_HASH_TO_TIMESTAMP)

        # (hash, chunk_index) => MultiModalKwargsItem
        self.hash_to_object = self.lmdb_env.open_db(self.DB_HASH_TO_OBJECT)

        # (hash, chunk_index) => prompt_updates
        self.hash_to_prompt_updates = self.lmdb_env.open_db(
            self.DB_HASH_TO_PROMPT_UPDATES
        )

        self._serde = MsgpackSerde()
        self._scratch_buffers: list[bytearray] = []

    @contextmanager
    def scratch_buffer(self):
        if self._scratch_buffers:
            buffer = self._scratch_buffers.pop()
        else:
            buffer = bytearray(self._max_object_size)
        try:
            with memoryview(buffer) as mv:
                yield mv
        finally:
            self._scratch_buffers.append(buffer)

    @classmethod
    def ensure_cache_id(cls):
        if envs.VLLM_MM_LMDB_CACHE_ID is None:
            os.environ["VLLM_MM_LMDB_CACHE_ID"] = uuid4().hex
            assert envs.VLLM_MM_LMDB_CACHE_ID is not None, "Cache ID must be set now."

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> Self:
        # The cache ID must be set by now.
        cache_id = envs.VLLM_MM_LMDB_CACHE_ID
        assert cache_id is not None, "LMDB cache ID must be set."

        mm_config = vllm_config.model_config.get_multimodal_config()
        return cls(
            cache_dir=os.path.join(cls.CACHES_DIR, cache_id),
            cache_size=int(
                mm_config.mm_processor_cache_gb * GiB_bytes,
            ),
            min_eviction_age=(mm_config.mm_lmdb_cache_min_eviction_age),
            max_object_size=(mm_config.mm_lmdb_cache_max_object_size_mb * MiB_bytes),
        )

    def utilization(self, txn: lmdb.Transaction) -> float:
        database_size = 0
        for db in (
            self.timestamps_and_hashes,
            self.hash_to_timestamp,
            self.hash_to_object,
            self.hash_to_prompt_updates,
        ):
            stats = txn.stat(db=db)
            database_size += stats["psize"] * (
                stats["branch_pages"] + stats["leaf_pages"] + stats["overflow_pages"]
            )
        return database_size / self._cache_size

    def int2bytes(self, value: int) -> bytes:
        # Use big-endian to ensure lexicographical ordering
        return value.to_bytes(self.INT_SIZE, byteorder="big", signed=False)

    def bytes2int(self, value: bytes) -> int:
        # Use big-endian to ensure lexicographical ordering
        assert len(value) == self.INT_SIZE
        return int.from_bytes(value, byteorder="big", signed=False)

    def serialize(self, item: object) -> bytearray:
        value, value_size, metadata, md_size = self._serde.serialize(item)

        if value_size + md_size > self._max_object_size:
            raise ValueError(
                f"Object size {value_size} exceeds maximum allowed "
                f"size of {self._max_object_size} bytes."
            )

        buf = bytearray(value_size + md_size)
        buf[0:md_size] = metadata
        idx = md_size
        for chunk in value if isinstance(value, list) else [value]:
            chunk_size = len(chunk)
            buf[idx : idx + chunk_size] = chunk
            idx += chunk_size
        return buf

    def deserialize(self, item: memoryview) -> object:
        return self._serde.deserialize(item)

    def get_chunked_object(
        self,
        db: lmdb._Database,
        txn: lmdb.Transaction,
        key: bytes,
        buffer: memoryview,
    ) -> memoryview:
        with txn.cursor(db=db) as cursor:
            if not cursor.set_key(key + self.int2bytes(0)):
                raise ValueError(f"Key {key!r} not found in LMDB cache.")

            chunk_index = 0
            offset = 0
            while cursor.key()[-self.INT_SIZE :] == self.int2bytes(chunk_index):
                chunk_value = cursor.value()
                buffer[offset : offset + len(chunk_value)] = chunk_value
                offset += len(chunk_value)
                chunk_index += 1
                if not cursor.next():
                    break

            return buffer[0:offset]

    def put_chunked_object(
        self, db: lmdb._Database, txn: lmdb.Transaction, key: bytes, value: memoryview
    ) -> None:
        with txn.cursor(db=db) as cursor:
            cursor.putmulti(
                (
                    (
                        key + self.int2bytes(i),
                        value[offset : offset + self._max_chunk_size],
                    )
                    for i, offset in enumerate(
                        range(0, len(value), self._max_chunk_size)
                    )
                )
            )

    def delete_chunked_object(
        self, db: lmdb._Database, txn: lmdb.Transaction, key: bytes
    ) -> int:
        """
        Deletes a chunked object from the given LMDB database and returns the number of
        chunks deleted.
        """

        with txn.cursor(db=db) as cursor:
            if not cursor.set_key(key + self.int2bytes(0)):
                return 0

            chunk_index = 0
            while cursor.key()[-self.INT_SIZE :] == self.int2bytes(chunk_index):
                chunk_index += 1
                cursor.delete()  # Deletes and advances the cursor

            return chunk_index

    def begin_write(self) -> "LmdbWriteTransaction":
        return LmdbWriteTransaction(self)

    def begin_read(self):
        return LmdbReadTransaction(self)

    def lock_and_clear_stale_caches(self) -> filelock.UnixFileLock | None:
        cache_id = os.path.basename(self._cache_dir)
        lock = filelock.UnixFileLock(self._cache_dir + ".lock")
        try:
            lock.acquire(blocking=False)
        except filelock.Timeout:
            # Another process is using this cache.
            logger.debug(
                "LMDB cache %s is currently in use by another process.", self._cache_dir
            )
            return None

        # Clean up any existing caches that are not locked.
        for entry in os.scandir(self.CACHES_DIR):
            if (
                not entry.is_file()
                or not entry.name.endswith(".lock")
                or entry.name == f"{cache_id}.lock"
            ):
                continue

            other_cache = os.path.join(self.CACHES_DIR, entry.name[: -len(".lock")])
            try:
                with filelock.FileLock(entry.path, blocking=False):
                    if os.path.exists(other_cache):
                        logger.info("Cleaning up stale cache at %s", other_cache)
                        try:
                            shutil.rmtree(other_cache)
                        except Exception as e:
                            logger.error(
                                "Failed to remove stale cache at %s: %s", other_cache, e
                            )
                    try:
                        os.unlink(entry.path)
                    except Exception as e:
                        logger.error(
                            "Failed to remove stale cache lock file at %s: %s",
                            entry.path,
                            e,
                        )

            except filelock.Timeout:
                # Another process is using the cache.
                logger.debug(
                    "Cache %s is currently in use by another process.", other_cache
                )

        return lock

    def start_evictor(
        self, maybe_event: "Event | None" = None
    ) -> multiprocessing.Process:
        evictor = get_mp_context().Process(
            name="LMDBEvictor",
            target=self._evictor_main,
            args=(
                maybe_event,
                self._cache_dir,
                self._cache_size,
                self._min_eviction_age,
                self._max_object_size,
            ),
            daemon=True,
        )
        evictor.start()
        return evictor

    @classmethod
    def _evictor_main(cls, maybe_event: "Event | None", *cache_init_args):
        set_process_title("LMDBEvictor")
        decorate_logs()

        cache = cls(*cache_init_args)
        next_reader_check = 0.0

        got_signal = False
        wait_for_signal = threading.Semaphore(0)
        signal.signal(signal.SIGUSR1, lambda signum, frame: wait_for_signal.release())

        if maybe_event:
            maybe_event.set()

        while True:
            if os.getppid() == 1:
                # Parent process has exited.
                break

            if got_signal or time.monotonic() >= next_reader_check:
                stale_readers = cache.lmdb_env.reader_check()
                if stale_readers > 0:
                    logger.warning("Removed %d stale LMDB readers.", stale_readers)
                next_reader_check = time.monotonic() + cls.EVICTOR_READER_CHECK_INTERVAL

            if got_signal:
                items, delay = cache.evict_once(min_utilization=0.0)
                logger.info(
                    "Forced eviction removed %d items from LMDB cache. "
                    "Next eviction in %.2f seconds.",
                    items,
                    delay,
                )
            else:
                _, delay = cache.evict_once()

            got_signal = wait_for_signal.acquire(timeout=delay)

    def evict_once(
        self,
        batch_size: int = 0,
        min_utilization: float = EVICTOR_MIN_UTILIZATION,
        max_utilization: float = EVICTOR_MAX_UTILIZATION,
        max_interval: float = EVICTOR_MAX_INTERVAL,
        max_duty_cycle: float = EVICTOR_MAX_DUTY_CYCLE,
    ) -> tuple[int, float]:
        """Evict items from the cache."""

        # By default, try to keep all the evicted pages within a single overflow page.
        # (Since this is the lower bound of the batch size, leave margin for items as
        # well as any DB pages the transaction touches.)
        batch_size = batch_size or int(
            self._max_page_ids_per_page * self.EVICTOR_BATCH_SIZE_FRACTION
        )
        with self.lmdb_env.begin(write=False) as txn:
            utilization = self.utilization(txn)

        evicted_items = 0
        delay = max_interval

        if utilization >= min_utilization:
            current_timestamp = int(time.time())

            with self.lmdb_env.begin(write=True) as txn:
                evict_start = time.perf_counter()

                with txn.cursor(db=self.timestamps_and_hashes) as cursor:
                    if cursor.first():
                        while evicted_items < batch_size:
                            combined_key = cursor.key()
                            if not combined_key:
                                # No more items to evict.
                                break

                            timestamp = self.bytes2int(combined_key[0 : self.INT_SIZE])

                            # Don't evict any items newer than the minimum eviction age.
                            if current_timestamp - timestamp < self._min_eviction_age:
                                break

                            # Delete the item from all dbs.
                            mm_hash_key = combined_key[self.INT_SIZE :]
                            evicted_items += self.delete_chunked_object(
                                self.hash_to_object, txn, mm_hash_key
                            )
                            evicted_items += self.delete_chunked_object(
                                self.hash_to_prompt_updates, txn, mm_hash_key
                            )
                            txn.delete(mm_hash_key, db=self.hash_to_timestamp)
                            cursor.delete()

            eviction_duration = time.perf_counter() - evict_start

            # Calculate how long to wait before the next eviction, approaching
            # the maximum duty cycle as we approach the maximum utilization.
            adjusted_utilization = max(
                0.0,
                min(
                    1.0,
                    (utilization - min_utilization)
                    / (max_utilization - min_utilization),
                ),
            )

            # Square the utilization to have a more gradual increase in duty cycle.
            duty_cycle = max(
                1e-6, adjusted_utilization * adjusted_utilization * max_duty_cycle
            )

            # Ensure eviction duration is at least 1ms to avoid extremely short delays.
            clamped_batch_duration = max(0.001, eviction_duration)

            # Example: If it took 2ms to evict one batch and our duty cycle is 1%, wait
            #
            #    2ms / 1% - 2ms = 198ms
            #
            # before evicting another batch.
            delay = (
                min(
                    max_interval,
                    clamped_batch_duration / duty_cycle - clamped_batch_duration,
                )
                if evicted_items > 0
                else max_interval
            )

            logger.debug(
                "LMDB cache utilization is %.2f. "
                "Evicted %d items in %.3fs. "
                "Next eviction in %.3fs.",
                utilization,
                evicted_items,
                eviction_duration,
                delay,
            )
        else:
            logger.debug(
                "LMDB cache utilization is %.2f (< %.2f)", utilization, min_utilization
            )

        return evicted_items, delay

    def clear(self) -> None:
        with self.lmdb_env.begin(write=True) as txn:
            txn.drop(self.timestamps_and_hashes, delete=False)
            txn.drop(self.hash_to_timestamp, delete=False)
            txn.drop(self.hash_to_object, delete=False)
            txn.drop(self.hash_to_prompt_updates, delete=False)


class LmdbWriteTransaction(AbstractContextManager):
    def __init__(
        self,
        cache: LmdbMultiModalCache,
    ) -> None:
        super().__init__()
        self._cache = cache
        self._read_txn = self._cache.lmdb_env.begin(write=False, buffers=True)
        self._inserts_enabled: bool | None = None
        self._write_queue = list[tuple[bytes, tuple[bytes, bytes] | None]]()
        self._scratch_buffer: bytearray | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if not exc_type:
                self._process_writes()
        finally:
            self._read_txn.abort()

    @property
    def read_txn(self) -> lmdb.Transaction:
        assert self._read_txn is not None, "Transaction not started"
        return self._read_txn

    @property
    def inserts_enabled(self) -> bool:
        if self._inserts_enabled is None:
            self._inserts_enabled = self._cache.utilization(self._read_txn) < 1.0
        return self._inserts_enabled

    def is_cached_item(self, mm_hash: str) -> bool:
        mm_hash_key = mm_hash.encode("utf-8")

        return (
            self._read_txn.get(mm_hash_key, db=self._cache.hash_to_timestamp)
            is not None
        )

    def get_and_update_item(
        self, mm_item: "MultiModalProcessorCacheInItem", mm_hash: str
    ) -> "MultiModalProcessorCacheOutItem":
        mm_hash_key = mm_hash.encode("utf-8")

        if mm_item is None:
            # Item is cached, so just update the timestamps for the hash.
            self._write_queue.append((mm_hash_key, None))

            with self._cache.scratch_buffer() as buffer:
                cached_prompt_updates = self._cache.get_chunked_object(
                    self._cache.hash_to_prompt_updates,
                    self._read_txn,
                    mm_hash_key,
                    buffer,
                )
                return None, cast(
                    Sequence[ResolvedPromptUpdate],
                    self._cache.deserialize(cached_prompt_updates),
                )
        if not self.inserts_enabled:
            # Cache is too full, do not cache new items.
            return mm_item

        # Item is not cached, serialize it and add it to the write queue.
        try:
            serialized_object = self._cache.serialize(mm_item[0])
            serialized_prompt_updates = self._cache.serialize(mm_item[1])
        except ValueError:
            # Object is too large to cache.
            return mm_item

        self._write_queue.append(
            (mm_hash_key, (serialized_object, serialized_prompt_updates))
        )
        return None, mm_item[1]

    def _process_writes(self):
        if not self._write_queue:
            # Nothing to write.
            return

        with (
            self._cache.lmdb_env.begin(write=True) as write_txn,
            write_txn.cursor(
                db=self._cache.hash_to_timestamp
            ) as hash_to_timestamp_cursor,
        ):
            current_timestamp_bytes = self._cache.int2bytes(int(time.time()))
            for mm_hash_key, serialized_pair in self._write_queue:
                if hash_to_timestamp_cursor.set_key(mm_hash_key):
                    # The item is already in the cache, delete the old (timestamp, hash)
                    existing_timestamp = hash_to_timestamp_cursor.value()
                    write_txn.delete(
                        existing_timestamp + mm_hash_key,
                        db=self._cache.timestamps_and_hashes,
                    )
                elif serialized_pair is None:
                    # The item was evicted in the meantime, so we need to retrieve the
                    # serialized values from the read transaction.
                    with self._cache.scratch_buffer() as buffer:
                        serialized_mm_item = self._cache.get_chunked_object(
                            self._cache.hash_to_object,
                            self._read_txn,
                            mm_hash_key,
                            buffer,
                        )

                        self._cache.put_chunked_object(
                            self._cache.hash_to_object,
                            write_txn,
                            mm_hash_key,
                            serialized_mm_item,
                        )

                        serialized_prompt_updates = self._cache.get_chunked_object(
                            self._cache.hash_to_prompt_updates,
                            self._read_txn,
                            mm_hash_key,
                            buffer,
                        )

                        self._cache.put_chunked_object(
                            self._cache.hash_to_prompt_updates,
                            write_txn,
                            mm_hash_key,
                            serialized_prompt_updates,
                        )
                else:
                    with memoryview(serialized_pair[0]) as mv:
                        self._cache.put_chunked_object(
                            self._cache.hash_to_object, write_txn, mm_hash_key, mv
                        )

                    with memoryview(serialized_pair[1]) as mv:
                        self._cache.put_chunked_object(
                            self._cache.hash_to_prompt_updates,
                            write_txn,
                            mm_hash_key,
                            mv,
                        )

                # Now update the timestamp entries.
                hash_to_timestamp_cursor.put(mm_hash_key, current_timestamp_bytes)
                write_txn.put(
                    current_timestamp_bytes + mm_hash_key,
                    b"",
                    db=self._cache.timestamps_and_hashes,
                )


class LmdbReadTransaction(AbstractContextManager):
    def __init__(
        self,
        cache: LmdbMultiModalCache,
    ) -> None:
        self._cache = cache
        self._txn = self._cache.lmdb_env.begin(write=False, buffers=True)
        self._scratch_buffer: memoryview | None = None
        self._scratch_buffer_ctx: AbstractContextManager[memoryview] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._txn.abort()
        if self._scratch_buffer_ctx is not None:
            self._scratch_buffer_ctx.__exit__(exc_type, exc_value, traceback)
            self._scratch_buffer = None

    @property
    def scratch_buffer(self) -> memoryview:
        if self._scratch_buffer is None:
            self._scratch_buffer_ctx = self._cache.scratch_buffer()
            self._scratch_buffer = self._scratch_buffer_ctx.__enter__()
        return self._scratch_buffer

    def is_cached_item(self, mm_hash: str) -> bool:
        mm_hash_bytes = mm_hash.encode("utf-8")

        return (
            self._txn.get(mm_hash_bytes, db=self._cache.hash_to_timestamp) is not None
        )

    def get_item(self, mm_hash: str) -> MultiModalKwargsItem:
        mm_hash_bytes = mm_hash.encode("utf-8")
        item = self._cache.get_chunked_object(
            self._cache.hash_to_object, self._txn, mm_hash_bytes, self.scratch_buffer
        )
        return cast(MultiModalKwargsItem, self._cache.deserialize(item))
