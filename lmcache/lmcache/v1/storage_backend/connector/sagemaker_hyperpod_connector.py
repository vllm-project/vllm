# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from multiprocessing import shared_memory
from typing import AsyncIterator, List, Optional, Tuple
import asyncio
import json
import time
import urllib.parse

# Third Party
import aiohttp
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


# Constants
METADATA_SIZE_BYTES = 28  # RemoteMetadata is 7 int32 fields
METADATA_SHAPE_DIMS = 4  # Number of shape dimensions in metadata
DEFAULT_CHUNK_SIZE_BYTES = 65536  # 64KB default for streaming
HTTP_OK = 200
HTTP_NO_CONTENT = 204
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409


class Priorities(IntEnum):
    """Priority levels for job execution in the priority queue."""

    LEASE = 0  # Highest priority - lease acquisition/release
    PREFETCH = auto()  # Medium priority - prefetching data
    PUT = auto()  # Lower priority - storing data


@dataclass
class LeaseInfo:
    """Information about a lease obtained from ai-toolkit daemon.

    A lease represents temporary exclusive access to cached data in shared memory.
    The daemon manages leases to prevent data from being evicted while in use.
    """

    lease_id: str
    offsets: List[Tuple[int, int]]  # (offset, length) pairs in shared memory


class SageMakerHyperPodConnector(RemoteConnector):
    """
    SageMaker HyperPod remote connector for communicating with KV cache daemon
     in SageMaker HyperPod.

    This connector provides high-performance access to KV cache data stored in
    a remote SageMaker HyperPod service using:
    - Shared memory (data plane) - zero-copy access via shared memory segment
    - HTTP (control plane) - lease acquisition, release, and PUT operations

    The connector uses a lease-based protocol with immediate release after all reads
    """

    def __init__(
        self,
        sagemaker_hyperpod_url: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        bucket_name: str,
        shared_memory_name: Optional[str],
        max_concurrent_requests: int,
        max_connections: int,
        max_connections_per_host: int,
        timeout_ms: int,
        lease_ttl_s: float = 10.0,
        put_stream_chunk_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
        max_lease_size_mb: Optional[float] = None,
        **kwargs,  # Accept and ignore unused legacy parameters
    ):
        """
        Initialize SageMaker HyperPod connector.

        Args:
            sagemaker_hyperpod_url: Base URL of the ai-toolkit daemon
            loop: Event loop for async operations
            local_cpu_backend: Backend for local memory allocation
            bucket_name: Bucket name for KV storage namespace
            shared_memory_name: Name of shared memory segment
            (if None, shared memory disabled)
            max_concurrent_requests: Maximum concurrent control plane requests
            max_connections: Maximum total HTTP connections in pool
            max_connections_per_host: Maximum HTTP connections per host
            timeout_ms: Timeout for lease acquisition requests
            lease_ttl_s: Server-side lease timeout (default: 10s)
            put_stream_chunk_bytes: Chunk size for
            streaming PUT requests (default: 64KB)
            **kwargs: Unused legacy parameters (ignored for backward compatibility)
        """
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Core configuration
        self.base_url = sagemaker_hyperpod_url.rstrip("/")
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.bucket_name = bucket_name
        self.shared_memory_name = shared_memory_name
        self.lease_ttl_s = lease_ttl_s
        self.put_stream_chunk_bytes = max(1024, put_stream_chunk_bytes)  # Minimum 1KB
        self.max_lease_size_bytes = (
            int(max_lease_size_mb * 1024 * 1024) if max_lease_size_mb else None
        )

        # HTTP configuration
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.max_connections = max(1, max_connections)
        self.max_connections_per_host = max(1, max_connections_per_host)
        self.timeout_ms = max(100, timeout_ms)  # Minimum 100ms

        # HTTP session (lazy initialized)
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()

        # Concurrency control
        self.control_inflight = asyncio.Semaphore(self.max_concurrent_requests)
        self.put_inflight = asyncio.Semaphore(self.max_concurrent_requests)
        self.pq_executor = AsyncPQExecutor(loop)

        # Shared memory
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        if self.shared_memory_name:
            self._init_shared_memory()

        # Observability
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Statistics for monitoring
        self.stats = {
            "get_success": 0,
            "get_failure": 0,
            "put_success": 0,
            "put_failure": 0,
            "lease_acquired": 0,
            "lease_released": 0,
            "lease_release_failed": 0,
        }

        logger.info(
            f"SageMaker HyperPod Connector initialized: url={self.base_url}, "
            f"bucket={self.bucket_name}, shared_memory={self.shared_memory_name}, "
            f"connections={self.max_connections}, lease_ttl={lease_ttl_s}s"
        )

    def _init_shared_memory(self):
        """Initialize shared memory connection to ai-toolkit daemon."""
        try:
            self.shared_memory_obj = shared_memory.SharedMemory(
                name=self.shared_memory_name, create=False
            )
            self.shared_memory_map = memoryview(self.shared_memory_obj.buf)
            size_mb = len(self.shared_memory_map) / (1024**2)
            logger.info(
                f"Shared memory opened: {self.shared_memory_name} ({size_mb:.2f} MB)"
            )
        except FileNotFoundError:
            logger.error(
                f"Shared memory segment '{self.shared_memory_name}' not found. "
                "Ensure ai-toolkit daemon is running."
            )
            self.shared_memory_map = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize shared memory: {e}")
            self.shared_memory_map = None
            raise

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session with connection pooling is initialized."""
        if self.http_session is None:
            async with self.session_lock:
                if self.http_session is None:  # Double-check locking
                    connector = aiohttp.TCPConnector(
                        limit=self.max_connections,
                        limit_per_host=self.max_connections_per_host,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=30,
                        connect=5,
                        sock_read=10,
                    )

                    self.http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": "LMCache-SageMaker-HyperPod/1.0"},
                    )
                    logger.info(
                        f"HTTP session created with {self.max_connections} "
                        f"max connections"
                    )

        return self.http_session

    async def _http_request(
        self,
        method: str,
        url: str,
        data=None,
        params=None,
        timeout: float = 5.0,
        headers=None,
        gate: Optional[asyncio.Semaphore] = None,
    ):
        """Execute HTTP request with optional semaphore gate."""
        if gate is None:
            return await self._http_request_impl(
                method, url, data=data, params=params, timeout=timeout, headers=headers
            )
        async with gate:
            return await self._http_request_impl(
                method, url, data=data, params=params, timeout=timeout, headers=headers
            )

    async def _http_request_impl(
        self,
        method: str,
        url: str,
        data=None,
        params=None,
        timeout: float = 5.0,
        headers=None,
    ):
        """Execute HTTP request with connection pooling and error handling."""
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.request(
                method,
                url,
                data=data,
                params=params,
                timeout=request_timeout,
                headers=headers,
            ) as response:
                # Parse JSON response if available
                body_json = None
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    try:
                        body_json = await response.json()
                    except aiohttp.ContentTypeError as e:
                        logger.warning(
                            f"JSON parsing failed for {method} {url}:"
                            f"invalid content-type - {e}"
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON parsing failed for {method} {url}:"
                            f"malformed JSON - {e}"
                        )
                    except Exception as e:
                        logger.warning(f"JSON parsing failed for {method} {url}: {e}")

                return {
                    "status": response.status,
                    "json": body_json,
                }

        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} timeout: {url}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP {method} client error: {url} - {e}")
            return None
        except Exception as e:
            logger.error(f"HTTP {method} failed: {url} - {e}")
            return None

    def _key_to_string(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to URL-safe string format."""
        key_str = key.to_string()
        return urllib.parse.quote(key_str, safe="")

    async def _release_lease(self, key: CacheEngineKey, lease_id: str) -> bool:
        """
        Release a lease to free server resources immediately.

        Args:
            key: The cache key
            lease_id: The lease ID to release

        Returns:
            True if release successful, False on error
        """
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/leases/{lease_id}/release"

        try:
            result = await self._http_request(
                "POST",
                url,
                timeout=5.0,
                gate=self.control_inflight,
            )

            if result and result["status"] == HTTP_OK:
                self.stats["lease_released"] += 1
                logger.debug(f"Lease released: key={key_str}, lease_id={lease_id}")
                return True
            else:
                status = result["status"] if result else "TIMEOUT"
                self.stats["lease_release_failed"] += 1
                logger.warning(
                    f"Lease release failed: key={key_str}, lease_id={lease_id}, "
                    f"status={status}"
                )
                return False

        except Exception as e:
            self.stats["lease_release_failed"] += 1
            logger.warning(
                f"Lease release error: key={key_str}, lease_id={lease_id} - {e}"
            )
            return False

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """
        Acquire a lease for the given key.

        A lease prevents the daemon from evicting data while we're reading it.
        The response includes offset information for shared memory access.

        Args:
            key: The cache key to acquire lease for

        Returns:
            LeaseInfo if successful, None otherwise
        """
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
        params = {
            "timeout_ms": self.timeout_ms,
            "ttl_s": self.lease_ttl_s,
        }

        result = await self._http_request(
            "POST",
            url,
            params=params,
            timeout=self.timeout_ms / 1000.0,
            gate=self.control_inflight,
        )

        if not result or result["status"] != HTTP_OK or not result["json"]:
            logger.debug(f"Lease acquisition failed: key={key_str}")
            return None

        lease_data = result["json"]
        offsets = [(o["offset"], o["len"]) for o in lease_data.get("offsets", [])]

        if not offsets:
            logger.debug(f"Lease has no offsets: key={key_str}")
            return None

        lease_info = LeaseInfo(
            lease_id=lease_data["id"],
            offsets=offsets,
        )

        total_size = sum(length for _, length in offsets)

        if (
            self.max_lease_size_bytes is not None
            and total_size > self.max_lease_size_bytes
        ):
            logger.warning(
                f"Lease size {total_size / 1024:.2f} KB exceeds limit "
                f"{self.max_lease_size_bytes / 1024:.2f} KB, releasing"
            )
            await self._release_lease(key, lease_info.lease_id)
            return None

        self.stats["lease_acquired"] += 1

        logger.debug(
            f"Lease acquired: key={key_str}, lease_id={lease_info.lease_id}, "
            f"size={total_size / 1024:.2f} KB, blocks={len(offsets)}"
        )

        return lease_info

    async def _executor_submit_lease_acquisition(
        self, key: CacheEngineKey
    ) -> Optional[LeaseInfo]:
        """Submit lease acquisition to priority executor."""
        return await self.pq_executor.submit_job(
            self._acquire_lease,
            key=key,
            priority=Priorities.LEASE,
        )

    def _read_from_shared_memory(
        self, key: CacheEngineKey, lease_info: LeaseInfo
    ) -> Optional[MemoryObj]:
        """
        Read data from shared memory using lease offsets.

        Data format: [RemoteMetadata header (28 bytes)] + [KV cache payload]
        Data may be fragmented across multiple blocks in shared memory.

        Args:
            key: The cache key being read
            lease_info: Lease information with memory offsets

        Returns:
            MemoryObj containing the data, or None on error
        """
        if self.shared_memory_map is None:
            logger.error("Shared memory not available")
            return None

        if not lease_info.offsets:
            logger.error("No offsets in lease")
            return None

        shm_size = len(self.shared_memory_map)
        for offset, length in lease_info.offsets:
            if offset < 0 or length < 0:
                logger.error(
                    f"Invalid offset or length: offset={offset}, length={length}"
                )
                return None
            if offset + length > shm_size:
                logger.error(
                    f"Offset out of bounds: offset={offset}, length={length}, "
                    f"shm_size={shm_size}"
                )
                return None

        memory_obj = None
        try:
            # Validate total size
            total_size = sum(length for _, length in lease_info.offsets)
            if total_size < METADATA_SIZE_BYTES:
                logger.error(f"Insufficient data for metadata: {total_size} bytes")
                return None

            # Read metadata header (may span multiple blocks)
            header = self._read_bytes_from_offsets(
                lease_info.offsets, 0, METADATA_SIZE_BYTES
            )
            if len(header) < METADATA_SIZE_BYTES:
                logger.error("Failed to read complete metadata header")
                return None

            # Parse metadata
            metadata = RemoteMetadata.deserialize(header)
            if metadata.length <= 0:
                logger.error(f"Invalid payload length: {metadata.length}")
                return None

            # Restore original shape (remove padding zeros)
            actual_shape = self._parse_shape(metadata.shapes[0])

            # Allocate local CPU memory
            memory_obj = self.local_cpu_backend.allocate(
                actual_shape,
                metadata.dtypes[0],
                metadata.fmt,
            )
            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key.to_string()}")
                return None

            # Get writable view
            view = self._get_writable_view(memory_obj.byte_array)

            # Copy payload data from shared memory (skip header)
            copied = self._copy_bytes_from_offsets(
                lease_info.offsets, METADATA_SIZE_BYTES, metadata.length, view
            )

            if copied != metadata.length:
                logger.error(
                    f"Data size mismatch: expected {metadata.length}, got {copied}"
                )
                memory_obj.ref_count_down()
                return None

            logger.debug(
                "Read from shared memory: key=%s, shape=%s, dtype=%s, size=%s bytes",
                key.to_string(),
                actual_shape,
                metadata.dtypes[0],
                metadata.length,
            )

            return memory_obj

        except Exception as e:
            logger.error(
                f"Error reading from shared memory: key={key.to_string()} - {e}"
            )
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    def _read_bytes_from_offsets(
        self, offsets: List[Tuple[int, int]], skip_bytes: int, read_bytes: int
    ) -> bytearray:
        """Read bytes from shared memory offsets, skipping initial bytes."""
        if self.shared_memory_map is None:
            logger.error("Shared memory not available")
            return bytearray()

        result = bytearray(read_bytes)
        filled = 0
        bytes_to_skip = skip_bytes
        shm_size = len(self.shared_memory_map)

        for offset, length in offsets:
            if filled >= read_bytes:
                break

            # Skip header bytes in first chunk(s)
            if bytes_to_skip > 0:
                if length <= bytes_to_skip:
                    bytes_to_skip -= length
                    continue
                offset += bytes_to_skip
                length -= bytes_to_skip
                bytes_to_skip = 0

            if length <= 0:
                continue

            take = min(read_bytes - filled, length)

            if offset < 0 or take <= 0:
                logger.error(f"Invalid read parameters: offset={offset}, take={take}")
                break
            if offset + take > shm_size:
                logger.error(
                    f"Read would exceed shared memory bounds: "
                    f"offset={offset}, take={take}, shm_size={shm_size}"
                )
                break

            result[filled : filled + take] = self.shared_memory_map[
                offset : offset + take
            ]
            filled += take

        return result

    def _copy_bytes_from_offsets(
        self,
        offsets: List[Tuple[int, int]],
        skip_bytes: int,
        copy_bytes: int,
        dest_view: memoryview,
    ) -> int:
        """Copy bytes from shared memory offsets to destination view."""
        if self.shared_memory_map is None:
            logger.error("Shared memory not available")
            return 0

        copied = 0
        bytes_to_skip = skip_bytes
        shm_size = len(self.shared_memory_map)

        for offset, length in offsets:
            if copied >= copy_bytes:
                break

            # Skip header bytes
            if bytes_to_skip > 0:
                if length <= bytes_to_skip:
                    bytes_to_skip -= length
                    continue
                offset += bytes_to_skip
                length -= bytes_to_skip
                bytes_to_skip = 0

            if length <= 0:
                continue

            take = min(copy_bytes - copied, length)

            if offset < 0 or take <= 0:
                logger.error(f"Invalid copy parameters: offset={offset}, take={take}")
                break
            if offset + take > shm_size:
                logger.error(
                    f"Copy would exceed shared memory bounds: "
                    f"offset={offset}, take={take}, shm_size={shm_size}"
                )
                break

            dest_view[copied : copied + take] = self.shared_memory_map[
                offset : offset + take
            ]
            copied += take

        return copied

    @staticmethod
    def _parse_shape(shape: torch.Size) -> torch.Size:
        """Parse shape from metadata, removing padding zeros."""
        actual_shape_list: List[int] = []
        for dim in shape:
            if dim == 0 and len(actual_shape_list) > 0:
                break
            actual_shape_list.append(dim)
        return torch.Size(actual_shape_list) if actual_shape_list else torch.Size([1])

    @staticmethod
    def _get_writable_view(byte_array) -> memoryview:
        """Get a writable memoryview from byte array."""
        if isinstance(byte_array, memoryview):
            view = byte_array
            if getattr(view, "format", None) == "<B":
                view = view.cast("B")
        else:
            view = memoryview(byte_array)
        return view

    async def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if a key exists in remote storage.

        Acquires a lease, checks existence, then releases immediately.
        """
        lease = await self._executor_submit_lease_acquisition(key)
        if lease is None:
            return False

        try:
            return True
        finally:
            await self._release_lease(key, lease.lease_id)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if a key exists in remote storage (sync wrapper)."""
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(future.result())

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Retrieve KV cache data for the given key with metrics reporting.

        Flow:
        1. Acquire a new lease
        2. Read from shared memory using lease offsets
        3. Release lease immediately (in finally block)
        4. Report metrics

        Args:
            key: The cache key to retrieve

        Returns:
            MemoryObj containing the KV cache data, or None if not found
        """
        begin = time.perf_counter()

        lease_info = await self._executor_submit_lease_acquisition(key)

        if lease_info is None:
            self.stats["get_failure"] += 1
            logger.debug(f"GET failed (no lease): key={key.to_string()}")
            return None

        try:
            memory_obj = self._read_from_shared_memory(key, lease_info)

            if memory_obj is not None:
                end = time.perf_counter()
                obj_size = memory_obj.get_size()

                # Report metrics for successful get
                self._stats_monitor.update_interval_remote_time_to_get(
                    (end - begin) * 1000
                )
                self._stats_monitor.update_interval_remote_read_metrics(obj_size)

                self.stats["get_success"] += 1
                logger.debug(
                    f"GET success: key={key.to_string()}, "
                    f"shape={memory_obj.get_shape()}, "
                    f"size={obj_size / 1e6:.3f} MBytes in {(end - begin) * 1000:.3f}ms"
                )
            else:
                self.stats["get_failure"] += 1
                logger.error(
                    f"Failed to read from shared memory: key={key.to_string()}"
                )

            return memory_obj

        except Exception as e:
            self.stats["get_failure"] += 1
            logger.error(f"GET error: key={key.to_string()} - {e}")
            return None

        finally:
            # Always release lease immediately after read
            await self._release_lease(key, lease_info.lease_id)

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Get multiple keys in parallel. Metrics reported per individual get."""
        tasks = [self.get(key) for key in keys]
        return await asyncio.gather(*tasks)

    def support_batched_put(self) -> bool:
        """Indicate support for batched PUT operations."""
        return True

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        """Store multiple objects in parallel. Metrics reported per individual put."""
        await asyncio.gather(
            *(self.put(key, mem) for key, mem in zip(keys, memory_objs, strict=True))
        )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to ai-toolkit (queued with priority)."""
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Internal PUT operation - sends data via HTTP streaming."""
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}"

        begin = time.perf_counter()

        try:
            # Build streaming payload (header + data)
            payload_len, payload_iter = self._build_put_stream(memory_obj)

            logger.debug(
                f"PUT: key={key_str}, size={payload_len / 1024:.2f} KB, "
                f"shape={memory_obj.get_shape()}"
            )

            # Send HTTP PUT request with streaming
            result = await self._http_request(
                "PUT",
                url,
                data=payload_iter,
                timeout=self.timeout_ms / 1000.0,
                headers={"Content-Length": str(payload_len)},
                gate=self.put_inflight,
            )

            end = time.perf_counter()

            if result and result["status"] == HTTP_OK:
                self.stats["put_success"] += 1

                # Report metrics for successful put
                self._stats_monitor.update_interval_remote_time_to_put(
                    (end - begin) * 1000
                )
                self._stats_monitor.update_interval_remote_write_metrics(payload_len)

                logger.info(
                    f"PUT success: key={key_str}, size={payload_len / 1024:.2f} KB "
                    f"in {(end - begin) * 1000:.3f}ms"
                )
            elif result and result["status"] == HTTP_CONFLICT:
                # 409 Conflict = key already exists (not an error)
                self.stats["put_success"] += 1

                # Still report metrics for conflict case
                self._stats_monitor.update_interval_remote_time_to_put(
                    (end - begin) * 1000
                )
                self._stats_monitor.update_interval_remote_write_metrics(payload_len)

                logger.debug(f"PUT skipped (already exists): key={key_str}")
            else:
                status = result["status"] if result else "TIMEOUT"
                self.stats["put_failure"] += 1
                logger.error(f"PUT failed: key={key_str}, status={status}")

        except Exception as e:
            self.stats["put_failure"] += 1
            logger.error(f"PUT exception: key={key_str} - {e}")

    def _build_put_stream(self, memory_obj: MemoryObj) -> Tuple[int, AsyncIterator]:
        """
        Build streaming payload: [RemoteMetadata (28 bytes)] + [KV cache data]

        Args:
            memory_obj: The memory object to stream

        Returns:
            Tuple of (total_length, async_generator)
        """
        # Prepare data view
        kv_view = self._get_writable_view(memory_obj.byte_array)
        kv_len = len(kv_view)

        # Prepare metadata
        shape = list(memory_obj.get_shape())
        padded_shape = (shape + [0] * METADATA_SHAPE_DIMS)[:METADATA_SHAPE_DIMS]

        metadata = RemoteMetadata(
            kv_len,
            [torch.Size(padded_shape)],
            [memory_obj.get_dtype()],
            memory_obj.get_memory_format(),
        )

        # Serialize metadata header
        header = bytearray(METADATA_SIZE_BYTES)
        metadata.serialize_into(header)
        header_bytes = bytes(header)

        total_len = len(header_bytes) + kv_len
        chunk_size = self.put_stream_chunk_bytes

        async def generator() -> AsyncIterator:
            # First yield header
            yield header_bytes
            # Then yield data in chunks
            offset = 0
            while offset < kv_len:
                next_offset = min(kv_len, offset + chunk_size)
                yield kv_view[offset:next_offset]
                offset = next_offset

        return total_len, generator()

    def support_batched_async_contains(self) -> bool:
        """Indicate support for batched async contains operation."""
        return True

    async def _batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        """
        Check existence of keys sequentially until first miss.

        Args:
            lookup_id: Lookup identifier (for logging/tracking)
            keys: List of keys to check
            pin: Whether to pin data (unused, for API compatibility)

        Returns:
            Number of consecutive hits from the start
        """
        num_hits = 0
        for key in keys:
            lease = None
            try:
                lease = await self._executor_submit_lease_acquisition(key)
                if lease is None:
                    break

                num_hits += 1

            except Exception as exc:
                logger.debug(f"Lease acquisition failed for {key}: {exc}")
                break
            finally:
                if lease is not None:
                    await self._release_lease(key, lease.lease_id)

        return num_hits

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        """Check existence of multiple keys (queued with priority)."""
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.LEASE,
        )

    def support_batched_get_non_blocking(self) -> bool:
        """Indicate support for non-blocking batched GET."""
        return True

    async def _batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        """Prefetch multiple keys and filter out None results."""
        results = await self.batched_get(keys)
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        """Prefetch multiple keys (queued with priority)."""
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    def support_batched_get(self) -> bool:
        """Indicate support for batched GET operations."""
        return True

    async def list(self) -> List[str]:
        """List operation not supported by ai-toolkit."""
        return []

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """Remove operation not supported by ai-toolkit."""
        return True

    def support_ping(self) -> bool:
        """Indicate ping operation is not supported."""
        return False

    async def ping(self) -> int:
        """Ping operation not implemented."""
        raise NotImplementedError(
            "Ping operation not supported by SageMaker HyperPod connector"
        )

    async def close(self):
        """Clean up all resources and log statistics."""
        # Log final statistics
        logger.info(
            f"SageMaker HyperPod Connector Statistics: "
            f"GET(ok/fail)={self.stats['get_success']}/{self.stats['get_failure']}, "
            f"PUT(ok/fail)={self.stats['put_success']}/{self.stats['put_failure']}, "
            f"leases(acq/rel/fail)={self.stats['lease_acquired']}/"
            f"{self.stats['lease_released']}/{self.stats['lease_release_failed']}"
        )

        # Shutdown priority queue executor
        try:
            await self.pq_executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")

        # Close HTTP session
        if self.http_session is not None:
            try:
                await self.http_session.close()
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {e}")
            self.http_session = None

        # Release shared memory
        if self.shared_memory_map is not None:
            self.shared_memory_map = None

        if self.shared_memory_obj is not None:
            try:
                self.shared_memory_obj.close()
            except Exception as e:
                logger.warning(f"Error closing shared memory object: {e}")
            self.shared_memory_obj = None

        logger.info("SageMaker HyperPod connector closed")
