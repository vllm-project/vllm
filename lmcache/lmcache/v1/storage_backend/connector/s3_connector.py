# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional
from urllib.parse import quote as url_quote
import asyncio
import ctypes

# Third Party
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import ClientTlsContext, TlsConnectionOptions, TlsContextOptions

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


# zero copy helper for S3 upload
class MemoryViewStream:
    def __init__(self, mv: bytes):
        # casting does not copy
        # we just get a uint8 view
        self.mv = memoryview(mv).cast("B")
        self.offset = 0

    def read(self, size=None):
        if size is None:
            size = len(self.mv) - self.offset
        if size < 0:
            size = 0

        end = min(self.offset + size, len(self.mv))
        result = self.mv[self.offset : end]
        self.offset = end
        # CRT/Python accepts memoryview
        return result

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.mv) + offset
        return self.offset

    def tell(self):
        return self.offset

    def __len__(self):
        return len(self.mv)


class S3Connector(RemoteConnector):
    """
    S3 remote connector
    """

    def __init__(
        self,
        s3_endpoint: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        s3_num_io_threads: int,
        s3_prefer_http2: bool,
        s3_region: str,
        s3_enable_s3express: bool,
        disable_tls: bool,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        if not s3_endpoint.startswith("s3://"):
            raise ValueError("S3 url must start with 's3://'")

        self.s3_part_size = self.full_chunk_size_bytes

        self.s3_endpoint = s3_endpoint.removeprefix("s3://")
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        self.s3_num_io_threads = s3_num_io_threads
        self.s3_prefer_http2 = s3_prefer_http2
        self.s3_region = s3_region
        self.s3_enable_s3express = s3_enable_s3express

        event_loop_group = io.EventLoopGroup(s3_num_io_threads)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        if aws_access_key_id and aws_secret_access_key:
            logger.info("Using explicit AWS credentials passed to S3Connector")
            self.credentials_provider = auth.AwsCredentialsProvider.new_static(
                aws_access_key_id,
                aws_secret_access_key,
            )
        else:
            logger.info(
                "No credentials provider, trying to use credentials from environment"
            )
            self.credentials_provider = auth.AwsCredentialsProvider.new_default_chain(
                client_bootstrap
            )

        tls_opts = None

        if self.s3_prefer_http2:
            # Use HTTP/2 multiplexing if possible.
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            try:
                tls_opts.set_alpn_list(["h2", "http/1.1"])
            except Exception:
                tls_opts = None

        signing_config = None
        if self.s3_enable_s3express:
            signing_config = auth.AwsSigningConfig(
                algorithm=auth.AwsSigningAlgorithm.V4_S3EXPRESS,
                region=self.s3_region,
                service="s3",
                credentials_provider=self.credentials_provider,
            )

        # turn off TLS for non-AWS services
        # regular and directory/express buckets both use TLS by default
        turn_off_tls = (
            s3.S3RequestTlsMode.DISABLED if disable_tls else s3.S3RequestTlsMode.ENABLED
        )
        logger.info("Initializing S3 client")
        self.s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=s3_region,
            enable_s3express=s3_enable_s3express,
            tls_connection_options=tls_opts,
            tls_mode=turn_off_tls,
            signing_config=signing_config,
        )

        # TODO(Jiayi): We need to handle cache consistency issues in a systematic way
        # across all connectors.
        # We assume S3 cache is never evicted and read-only for now.
        # the object size cache does not need protection because
        # asyncio scheduling is cooperative and not preemptive
        self.object_size_cache: dict[str, int] = {}

        # Circuit breaker for connection failures
        self.connection_failures = 0
        self.max_connection_failures = 3
        self.connection_disabled = False

        self.pq_executor = AsyncPQExecutor(loop)

    def _format_safe_path(self, key_str: str) -> str:
        """
        Generate a safe HTTP path for the S3 key.
        Flattens the key by replacing slashes with underscores and URL-encodes
        any special characters.
        """
        flat_key_str = key_str.replace("/", "_")
        return "/" + url_quote(flat_key_str)

    # TODO(Jiayi): optimize this with async
    def _get_object_size(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        try:
            s3_req.finished_future.result()
        except Exception as e:
            # 404 (not found) is expected when checking if object exists
            if got["status"] == 404:
                logger.debug("Object not found: %s", key_str)
            else:
                logger.debug("Exception in `_get_object_size`: %s", e)
            return 0
        if got["err"] or got["status"] != 200:
            if got["status"] != 404:  # Don't warn for 404, it's expected
                logger.warning(
                    "Encountering error in S3 HEAD request with error code: %s",
                    got["status"],
                )
            return 0
        return got["len"] if got["len"] is not None else 0

    # exactly the same as _get_object_size just awaiting an asyncio.Future
    # instead of a concurrent.futures.Future
    async def _get_object_size_async(self, key_str: str) -> int:
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        req = HttpRequest("HEAD", self._format_safe_path(key_str), headers)

        got = {"len": None, "status": None, "err": None}

        def on_headers(status_code, headers, **kwargs):
            got["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        got["len"] = int(value)
                    except Exception:
                        pass

        def on_done(error=None, **kwargs):
            got["err"] = error

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
        )

        try:
            await asyncio.wrap_future(s3_req.finished_future)
        except Exception as e:
            # 404 (not found) is expected when checking if object exists
            if got["status"] == 404:
                logger.debug("Object not found: %s", key_str)
            else:
                logger.debug("Exception in `_get_object_size_async`: %s", e)
            return 0
        if got["err"] or got["status"] != 200:
            if got["status"] != 404:  # Don't warn for 404, it's expected
                logger.warning(
                    "Encountering error in S3 HEAD request with error code: %s",
                    got["status"],
                )
            return 0
        return got["len"] if got["len"] is not None else 0

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        # Circuit breaker: if connection is disabled, return False
        if self.connection_disabled:
            return False

        key_str = key.to_string()
        if key_str in self.object_size_cache:
            return self.object_size_cache[key_str] > 0
        cache_size = self._get_object_size(key_str)
        if cache_size > 0:
            self.object_size_cache[key_str] = cache_size
            return True
        return False

    def _write_mem_obj(self, mem_obj: MemoryObj, data: bytes, offset: int):
        ctypes.memmove(mem_obj.data_ptr + offset, data, len(data))

    def _s3_download(
        self,
        key_str: str,
        mem_obj: MemoryObj,
    ) -> "s3.S3Request":
        """
        Download a file from S3.
        """
        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)

        # TODO(Jiayi): Enable more finegrained data partition
        # range_header = f"bytes={start_byte}-{end_byte}"
        # headers.add("Range", range_header)

        req = HttpRequest("GET", self._format_safe_path(key_str), headers)

        def on_body(chunk, offset, **kwargs):
            # Directly write chunk to the memory object at the correct offset
            self._write_mem_obj(mem_obj, chunk, offset)

        # NOTE(Jiayi): Run in crt threads (not this thread) with GIL
        # See https://github.com/awslabs/aws-crt-python/blob/4250709624119de1af3ca86816e1a154fcac7cc8/source/common.c#L51
        def on_done(error=None, status_code=None, **kwargs):
            ok = (status_code in (200, 206)) or (status_code is None)
            if error or not ok:
                raise RuntimeError(
                    f"Failed to download {key_str} from S3: {error or status_code}"
                )

        # TODO(Jiayi): Need to support offset to enable zero-copy
        # More concretely, we need to get the shared memory offset.
        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.GET_OBJECT,
            request=req,
            on_body=on_body,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )

        return s3_req

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # Circuit breaker: if connection is disabled, return None immediately
        if self.connection_disabled:
            logger.debug(
                "S3 connection disabled. Skipping download for %s", key.to_string()
            )
            return None

        key_str = key.to_string()

        obj_size = self.object_size_cache.get(key_str, None)

        if obj_size is None:
            obj_size = await self._get_object_size_async(key_str)
            if obj_size <= 0:
                self.object_size_cache[key_str] = 0
                return None
            self.object_size_cache[key_str] = obj_size

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )

        if memory_obj is None:
            return None

        # Check if stored size matches expected size
        if obj_size != memory_obj.get_size():
            logger.error(
                "Size mismatch for %s: S3 has %s bytes, "
                "but current config expects %s bytes. "
                "This usually means the data was stored with different chunk_size "
                "or model configuration. Please use matching config or clear S3.",
                key_str,
                obj_size,
                memory_obj.get_size(),
            )
            memory_obj.ref_count_down()
            return None

        s3_req = self._s3_download(
            key_str=key_str,
            mem_obj=memory_obj,
        )

        try:
            # use blocking_timeout_sec in config to control the timeout
            await asyncio.wrap_future(s3_req.finished_future)

            # Reset failure counter on success
            self._reset_connection_failures()

            return memory_obj
        except Exception as e:
            error_msg = str(e)

            # Update connection failures and check if it's a connection error
            is_connection_error = self._update_connection_failures(error_msg)

            if not is_connection_error:
                # Log non-connection errors
                logger.error("Failed to download %s from S3: %s", key_str, e)

            memory_obj.ref_count_down()
            return None

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # Circuit breaker: if connection is disabled, return all None
        if self.connection_disabled:
            logger.debug(
                "S3 connection disabled. Skipping batched download for %s keys",
                len(keys),
            )
            return [None] * len(keys)

        memory_objs: List[Optional[MemoryObj]] = []
        futures = []
        future_to_memobj_idx = []

        for idx, key in enumerate(keys):
            key_str = key.to_string()

            obj_size = self.object_size_cache.get(key_str, None)

            if obj_size is None:
                obj_size = await self._get_object_size_async(key_str)
                if obj_size <= 0:
                    self.object_size_cache[key_str] = 0
                    memory_objs.append(None)
                    continue
                self.object_size_cache[key_str] = obj_size

            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shapes,
                self.meta_dtypes,
                self.meta_fmt,
            )

            if not memory_obj:
                memory_objs.append(None)
                continue

            # Check if stored size matches expected size
            if obj_size != memory_obj.get_size():
                logger.error(
                    "Size mismatch for %s: S3 has %s bytes, "
                    "but current config expects %s bytes. "
                    "Skipping this key.",
                    key_str,
                    obj_size,
                    memory_obj.get_size(),
                )
                memory_obj.ref_count_down()
                memory_objs.append(None)
                continue

            memory_objs.append(memory_obj)

            s3_req = self._s3_download(
                key_str=key_str,
                mem_obj=memory_obj,
            )
            fut = asyncio.wrap_future(s3_req.finished_future)
            futures.append(fut)
            future_to_memobj_idx.append(len(memory_objs) - 1)

        # Use return_exceptions to prevent one failure from stopping all downloads
        results = await asyncio.gather(*futures, return_exceptions=True)

        had_success = False

        for future_idx, result in enumerate(results):
            memobj_idx = future_to_memobj_idx[future_idx]

            if isinstance(result, Exception):
                error_msg = str(result)

                is_connection_error = self._update_connection_failures(error_msg)

                if not is_connection_error:
                    # Log non-connection errors
                    logger.error(
                        "Failed to download key at index %s: %s",
                        memobj_idx,
                        error_msg,
                    )
                # Release the memory object for failed download
                memobj = memory_objs[memobj_idx]
                if memobj is not None:
                    memobj.ref_count_down()
                    memory_objs[memobj_idx] = None
            else:
                had_success = True

        if had_success:
            self._reset_connection_failures()

        return memory_objs

    def _s3_upload(
        self,
        key_str: str,
        memory_obj: MemoryObj,
    ) -> "s3.S3Request":
        """
        Upload a file to S3.
        """
        # Zero-copy approach using MemoryViewStream
        stream = MemoryViewStream(memory_obj.byte_array)
        # Calculate total length from the memoryview
        total_len = len(stream)

        headers = HttpHeaders()
        headers.add("Host", self.s3_endpoint)
        headers.add("Content-Length", str(total_len))
        headers.add("Content-Type", "application/octet-stream")

        req = HttpRequest(
            "PUT", self._format_safe_path(key_str), headers, body_stream=stream
        )

        done = {"err": None, "status": None}

        def on_done(error=None, status_code=None, **kwargs):
            done["err"] = error
            done["status"] = status_code

            if done["err"] or done["status"] not in (200, 201):
                raise RuntimeError(f"Upload failed in S3Connector: {done}")

        s3_req = s3.S3Request(
            client=self.s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            credential_provider=self.credentials_provider,
            region=self.s3_region,
            on_done=on_done,
        )
        return s3_req

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Store data to S3
        """
        # Circuit breaker: if connection is disabled, just log and return
        if self.connection_disabled:
            logger.debug(
                "S3 connection disabled due to repeated failures. "
                "Skipping upload for %s",
                key.to_string(),
            )
            return

        key_str = key.to_string()

        # Check if the chunk size matches expected S3 part size
        if memory_obj.get_physical_size() != self.s3_part_size:
            logger.error(
                "Cannot upload %s: chunk size %s "
                "bytes does not match S3 part size %s bytes. "
                "Partial/unfull chunks are not supported.",
                key_str,
                memory_obj.get_physical_size(),
                self.s3_part_size,
            )
            return

        try:
            logger.debug("Uploading %s to S3", key_str)
            s3_req = self._s3_upload(key_str, memory_obj)
            await asyncio.wrap_future(s3_req.finished_future)

            self.object_size_cache[key_str] = memory_obj.get_physical_size()
            logger.debug("Uploaded %s to S3 successfully", key_str)

            # Reset failure counter on success
            self._reset_connection_failures()
        except Exception as e:
            error_msg = str(e)

            # Update connection failures and check if it's a connection error
            is_connection_error = self._update_connection_failures(error_msg)

            if not is_connection_error:
                # Log non-connection errors
                logger.error("Failed to upload %s to S3: %s", key_str, e)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return await self.pq_executor.submit_job(
            self._put,
            key=key,
            memory_obj=memory_obj,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        # Circuit breaker: if connection is disabled, return 0
        if self.connection_disabled:
            return 0

        num_hit_counts = 0
        for key in keys:
            key_str = key.to_string()
            cached_size = self.object_size_cache.get(key_str, None)
            if cached_size is not None:
                if cached_size > 0:
                    num_hit_counts += 1
                    continue
                else:
                    return num_hit_counts

            obj_size = await self._get_object_size_async(key_str)
            if not obj_size > 0:
                self.object_size_cache[key_str] = 0
                return num_hit_counts

            self.object_size_cache[key_str] = obj_size
            num_hit_counts += 1

        return num_hit_counts

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # batched get is already a coroutine
        result = await self.batched_get(keys)
        return [r for r in result if r is not None]

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def list(self) -> List[str]:
        raise NotImplementedError

    def support_ping(self) -> bool:
        return False

    # TODO(Jiayi): This needs to be implemented.
    async def ping(self) -> int:
        raise NotImplementedError

    def support_batched_get(self) -> bool:
        return True

    def _update_connection_failures(self, error_msg: str) -> bool:
        # Check if it's a connection error
        is_connection_error = (
            "CONNECTION_REFUSED" in error_msg
            or "SOCKET" in error_msg
            or "DNS" in error_msg
            or "TIMEOUT" in error_msg
        )

        if is_connection_error:
            self.connection_failures += 1
            logger.error(
                "S3 connection error (%s/%s): %s",
                self.connection_failures,
                self.max_connection_failures,
                error_msg,
            )

            if self.connection_failures >= self.max_connection_failures:
                self.connection_disabled = True
                logger.error(
                    "S3 connection disabled after %s "
                    "consecutive failures. "
                    "All future S3 operations will be skipped.",
                    self.max_connection_failures,
                )

        return is_connection_error

    def _reset_connection_failures(self):
        """Reset connection failure counter on successful operation."""
        if self.connection_failures > 0:
            logger.info("S3 connection recovered")
            self.connection_failures = 0

    async def close(self):
        await self.pq_executor.shutdown(wait=True)
