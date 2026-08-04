# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional, Union, no_type_check
import asyncio
import ctypes
import os
import random
import string
import time

# Third Party
import eic
import yaml

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class PerformanceTimer:
    def __init__(self, key, op):
        self.key = key
        self.size = 0
        self.op = op
        self.start_times = {}
        self.elapsed_times = {}

    def set_size(self, size):
        self.size = size

    def start(self, operation_name):
        self.start_times[operation_name] = time.perf_counter()

    # unit: us
    def stop(self, operation_name):
        if operation_name not in self.start_times:
            raise ValueError(f"operation {operation_name} not found")

        end_time = time.perf_counter()
        start_time = self.start_times[operation_name]
        elapsed_time = end_time - start_time
        self.elapsed_times[operation_name] = elapsed_time * 1000000
        del self.start_times[operation_name]
        return elapsed_time

    def get_elapsed_time(self, operation_name):
        return self.elapsed_times.get(operation_name)

    def get_all_elapsed_times(self):
        return self.elapsed_times

    def debug_all_elapsed_times(self):
        logger.debug(f"== Perf key:{self.key} =========")
        logger.debug(f"== Perf op {self.op} size {self.size} =========")
        for op, item in self.elapsed_times.items():
            logger.debug(f"Step: {op} cost {item:.2f} us")
        logger.debug(f"== Perf op {self.op} size {self.size} =========")


class FlexibleDRAMMemoryPool:
    def __init__(self, conn):
        self._init = False
        self.connection = conn
        self.used_mem = {}

    def allocate(self, size):
        ptr = self.connection.allocate_managed_buffer(size)
        if ptr == 0:
            logger.error(f"fail to allocate dram pool, ptr {ptr}, size {size}")
            return ptr
        logger.debug(f"allocate dram pool: ptr {ptr}, size {size}")
        self.used_mem[ptr] = size
        return ptr

    def deallocate(self, ptr):
        size = self.used_mem.get(ptr)
        if size is not None:
            logger.debug(f"deallocate dram pool: ptr {ptr}, size {size}")
            self.connection.free_managed_buffer(ptr, size)
            del self.used_mem[ptr]


def _make_dir(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info(f"create dir '{path}' success")
    except OSError as e:
        logger.error(f"create dir '{path}' error {e}")


class EICConnector(RemoteConnector):
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(
        self,
        endpoint: str,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(memory_allocator.config, memory_allocator.metadata)

        logger.info("init EICConnector")
        logger.info(f"try connect to eic: {endpoint}")

        self.loop = loop
        self.memory_allocator = memory_allocator

        # Initialize pq_executor early to avoid AttributeError
        try:
            self.pq_executor = AsyncPQExecutor(loop)
            logger.info("AsyncPQExecutor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncPQExecutor: {e}")
            raise

        self.cudaError_t = ctypes.c_int
        self.cudaMemcpyDeviceToHost = 2
        self.cudaMemcpyHostToDevice = 1
        self.cuda_lib = ctypes.CDLL("libcudart.so")
        self.cuda_lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.cuda_lib.cudaMemcpy.restype = self.cudaError_t

        self.enable_compare = False

        config_file = os.getenv("LMCACHE_CONFIG_FILE")
        if config_file is None:
            raise ValueError("LMCACHE_CONFIG_FILE environment variable is not set")
        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        logger.info(f"eic remote_url: {remote_url}")

        eic_instance_id = config.get("eic_instance_id", None)
        logger.info(f"eic instance_id: {eic_instance_id}")

        eic_thread_num = config.get("eic_thread_num", 6)
        logger.info(f"eic thread_num: {eic_thread_num}")

        eic_log_dir = config.get("eic_log_dir", None)
        logger.info(f"eic log_dir: {eic_log_dir}")

        eic_log_level = config.get("eic_log_level", 1)
        logger.info(f"eic log_level: {eic_log_level}")

        eic_trans_type = config.get("eic_trans_type", 3)
        logger.info(f"eic trans_type: {eic_trans_type}")

        self.eic_kv_ttl = config.get("eic_kv_ttl", -1)
        logger.info(f"eic eic_kv_ttl: {self.eic_kv_ttl}")

        self.eic_kv_ns = config.get("eic_kv_ns", "")
        logger.info(f"eic eic_kv_ns: {self.eic_kv_ns}")

        eic_flag_file = config.get("eic_flag_file", None)
        logger.info(f"eic flag_file: {eic_flag_file}")

        _make_dir(eic_log_dir)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        init_option.flag_file = eic_flag_file
        endpoint = endpoint.removeprefix("eic://").removesuffix("/")
        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            raise RuntimeError(
                f"Failed to initialize eic client with error code: {ret}"
            )
        else:
            logger.info(f"init eic client success, ret: {ret}")

        self.trans_type = eic.TransportType(eic_trans_type)

        # Register memory in rdma and gdr scenarios
        if self.trans_type == eic.TransportType.TRANSPORT_GDR:
            if not isinstance(self.memory_allocator, LocalCPUBackend):
                raise RuntimeError("memory_allocator must be LocalCPUBackend")
            allocator = self.memory_allocator.memory_allocator
            if not isinstance(allocator, MixedMemoryAllocator):
                raise RuntimeError(
                    "memory_allocator.memory_allocator must be MixedMemoryAllocator"
                )

            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                mem_pool = allocator.pin_allocator.buffer
                meminfo = eic.MemoryInfo()
                meminfo.type = eic.MemoryType.MEMORY_CUDA
                meminfo.cuda_id = 0

                vals = eic.IOBuffers()
                vals.append(
                    mem_pool.data_ptr(),
                    mem_pool.numel() * mem_pool.element_size(),
                    True,
                )

                if self.connection.register_memory(vals, meminfo):
                    logger.info("register mixed memory pin buffer success")
                else:
                    logger.error("fail to register mixed memory pin buffer")
                    exit(1)
            else:
                logger.error("mixed memory pin buffer is None")
                exit(1)
        self.prebuilt_connection()

    def prebuilt_connection(self) -> None:
        def random_string(N):
            return self.eic_kv_ns.join(
                random.choices(string.ascii_uppercase + string.digits, k=N)
            )

        try:
            for i in range(2048):
                key = random_string(30)
                self._exists_sync(key)
            logger.info("eic prebuilt connection finish")
        except Exception as e:
            logger.error(f"Error in prebuilt connection thread: {e}")

    def delete_sync(self, key: str) -> bool:
        keys = eic.StringVector()
        keys.append(key)
        status_code, _ = self.connection.mdel(keys)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug(f"eic delete {key} failed, status_code {status_code}")
            return False
        return True

    def _exists_sync(self, key_str: str) -> bool:
        keys = eic.StringVector()
        keys.append(key_str)
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(keys, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug(f"eic exists {key_str} failed, status_code {status_code}")
            return False

        err_code = exist_outcome.status_codes[0]
        success = err_code == eic.StatusCode.SUCCESS
        if success:
            logger.debug(f"eic exists {key_str} success")
        else:
            logger.debug(
                f"eic exists {key_str} failed, status_code {status_code} "
                "err_code {err_code}"
            )
        return success

    async def _exists(self, key: CacheEngineKey) -> bool:
        return self._exists_sync(key.to_string())

    async def exists(self, key: CacheEngineKey) -> bool:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self._exists_sync(key.to_string())

    async def get_meta(self, key_str: str) -> Optional[RemoteMetadata]:
        perf_timer = PerformanceTimer(key_str, "get_meta")
        perf_timer.start("total_cost")

        # Get Meta: generate meta keys and vals
        meta_keys = eic.StringVector()
        meta_vals = eic.IOBuffers()

        # Get Meta: generate meta buffer tensor
        perf_timer.start("alloc_mem")
        meta_key = key_str + "_meta"
        meta_size = self.remote_metadata_bytes
        meta_bytes = bytearray(meta_size)
        meta_bytes_ptr = self.bytes_get_ptr(meta_bytes)

        meta_keys.append(meta_key)
        meta_vals.append(meta_bytes_ptr, meta_size, False)

        perf_timer.set_size(meta_size)
        perf_timer.stop("alloc_mem")

        # Get Meta: recv meta buffer tensor
        perf_timer.start("eic_mget")
        get_option = eic.GetOption()
        get_option.ns = self.eic_kv_ns
        status_code, meta_vals, get_outcome = self.connection.mget(
            meta_keys, get_option, meta_vals
        )
        err_code = get_outcome.status_codes[0]
        if status_code != eic.StatusCode.SUCCESS or err_code != eic.StatusCode.SUCCESS:
            if err_code == eic.StatusCode.KEY_NOT_EXIST:
                logger.debug(
                    f"eic mget meta {key_str} failed, status_code {status_code}"
                    " err_code {err_code}"
                )
            else:
                logger.error(
                    f"eic mget meta {key_str} failed, status_code {status_code}"
                    " err_code {err_code}"
                )
            return None
        else:
            logger.debug(f"eic mget meta {key_str} success")

        perf_timer.stop("eic_mget")

        perf_timer.start("serialize")
        meta = RemoteMetadata.deserialize(meta_bytes[:meta_size])
        perf_timer.stop("serialize")

        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

        return meta

    async def get_data(self, key_str: str, meta: RemoteMetadata) -> Optional[MemoryObj]:
        perf_timer = PerformanceTimer(key_str, "get_data")
        perf_timer.start("total_cost")
        perf_timer.start("alloc_obj")
        memory_obj = self.memory_allocator.allocate(
            meta.shapes,
            meta.dtypes,
            meta.fmt,
        )
        if memory_obj is None:
            logger.error(
                f"fail to allocate memory during remote receive key {key_str} length"
                " {meta.length}"
            )
            return None
        perf_timer.stop("alloc_obj")

        perf_timer.start("alloc_mem")
        obj_size = memory_obj.get_size()
        data_ptr = memory_obj.tensor.data_ptr()
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        data_keys.append(key_str)

        perf_timer.set_size(obj_size)
        perf_timer.stop("alloc_mem")

        try:
            if self.trans_type == eic.TransportType.TRANSPORT_GDR:
                data_vals.append(data_ptr, obj_size, True)
            else:
                data_vals.append(data_ptr, obj_size, False)

            perf_timer.start("eic_mget")
            get_option = eic.GetOption()
            get_option.ns = self.eic_kv_ns
            status_code, data_vals, get_outcome = self.connection.mget(
                data_keys, get_option, data_vals
            )
            err_code = get_outcome.status_codes[0]
            if (
                status_code != eic.StatusCode.SUCCESS
                or err_code != eic.StatusCode.SUCCESS
            ):
                logger.error(
                    f"eic mget data {key_str} failed, status_code {status_code} "
                    f"err_code {err_code}"
                )
                memory_obj.ref_count_down()
                return None
            else:
                logger.debug(f"eic mget data {key_str} success")
        except Exception as e:
            logger.error(
                f"eic mget data {key_str} raised exception: {e}", exc_info=True
            )
            memory_obj.ref_count_down()
            return None

        perf_timer.stop("eic_mget")

        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

        return memory_obj

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        meta = await self.get_meta(key_str)
        if meta is None:
            return None
        data = await self.get_data(key_str, meta)
        return data

    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def bytes_get_ptr(self, mv: Union[bytearray, memoryview, bytes]) -> int:
        if isinstance(mv, bytes):
            pointer = ctypes.cast(ctypes.c_char_p(mv), ctypes.POINTER(ctypes.c_char))
            ptr = ctypes.addressof(pointer.contents)
            return ptr
        return ctypes.addressof(ctypes.c_char.from_buffer(mv))

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return self._put_sync(key, memory_obj)

    def _put_sync(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        logger.debug(f"eic put {key_str}")

        perf_timer = PerformanceTimer(key_str, "put_data")
        perf_timer.start("total_cost")
        kv_bytes = memory_obj.byte_array
        kv_tensor = memory_obj.tensor
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()
        value_size = memory_obj.get_physical_size()

        logger.debug(
            f"eic put {key_str} data len {len(kv_bytes)} value_size {value_size}"
        )

        if kv_tensor is None:
            logger.error(f"Memory object tensor is None for key {key_str}")
            return

        perf_timer.start("serialize")

        # generate meta bytes
        remote_meta = RemoteMetadata(
            self.remote_metadata_bytes, kv_shapes, kv_dtypes, memory_format
        )

        logger.debug(f"eic meta {key_str} remote_meta{remote_meta}")

        meta_bytes = remote_meta.serialize()

        perf_timer.stop("serialize")

        perf_timer.start("trans_address")
        # generate meta & data ptr
        meta_ptr = self.bytes_get_ptr(meta_bytes)
        meta_size = len(meta_bytes)
        data_ptr = kv_tensor.data_ptr()
        data_size = len(kv_bytes)
        perf_timer.set_size(data_size)
        perf_timer.stop("trans_address")

        logger.debug(
            f"eic put {key_str} meta ptr {meta_ptr} len {meta_size} data ptr {data_ptr}"
            " len {data_size}"
        )

        perf_timer.start("eic_mset")
        keys = eic.StringVector()
        vals = eic.IOBuffers()

        # set meta key & value
        meta_key = key_str + "_meta"
        keys.append(meta_key)
        vals.append(meta_ptr, meta_size, False)

        # set data key & value
        keys.append(key_str)
        if self.trans_type == eic.TransportType.TRANSPORT_GDR:
            vals.append(data_ptr, data_size, True)
        else:
            vals.append(data_ptr, data_size, False)

        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_kv_ns
        set_option.ttl_second = self.eic_kv_ttl

        status_code, set_outcome = self.connection.mset(keys, vals, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {key_str} failed, status_code {status_code}")

        err_code = set_outcome.status_codes[0]
        if err_code == eic.StatusCode.SUCCESS:
            logger.debug(f"eic put meta key {meta_key} success")
        else:
            logger.error(f"eic put meta key {meta_key} failed, err_code {err_code}")

        err_code = set_outcome.status_codes[1]
        if err_code == eic.StatusCode.SUCCESS:
            logger.debug(f"eic put data key {key_str} success")
        else:
            logger.error(f"eic put data key {key_str} failed, err_code {err_code}")

        perf_timer.stop("eic_mset")
        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        if not keys or not memory_objs:
            return

        # Prepare all keys and values for batch mset
        eic_keys = eic.StringVector()
        eic_vals = eic.IOBuffers()
        # Keep references to meta_bytes to prevent dangling pointers
        meta_list = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            key_str = key.to_string()
            logger.debug(f"eic batched_put processing {key_str}")

            # Get memory object data
            kv_bytes = memory_obj.byte_array
            kv_tensor = memory_obj.tensor
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()
            if kv_tensor is None:
                logger.error(f"Memory object tensor is None for key {key_str}")
                return

            remote_meta = RemoteMetadata(
                self.remote_metadata_bytes, kv_shapes, kv_dtypes, memory_format
            )
            meta_bytes = remote_meta.serialize()
            meta_list.append(meta_bytes)
            meta_ptr = self.bytes_get_ptr(meta_bytes)
            meta_size = len(meta_bytes)
            data_ptr = kv_tensor.data_ptr()
            data_size = len(kv_bytes)

            logger.info(
                "eic batched_put %s, shapes: %s, dtypes: %s, fmt: %s",
                key_str,
                kv_shapes,
                kv_dtypes,
                memory_format,
            )

            # Add meta key & value
            meta_key = key_str + "_meta"
            eic_keys.append(meta_key)
            eic_vals.append(meta_ptr, meta_size, False)
            # Add data key & value
            eic_keys.append(key_str)
            if self.trans_type == eic.TransportType.TRANSPORT_GDR:
                eic_vals.append(data_ptr, data_size, True)
            else:
                eic_vals.append(data_ptr, data_size, False)

        set_option = eic.SetOption()
        set_option.ns = self.eic_kv_ns
        set_option.ttl_second = self.eic_kv_ttl

        set_status_code, set_outcome = self.connection.mset(
            eic_keys, eic_vals, set_option
        )

        if set_status_code != eic.StatusCode.SUCCESS:
            logger.error(
                f"eic batched_put mset data failed, status_code {set_status_code}"
            )
            return
        for i, key_str in enumerate(eic_keys):
            meta_key = key_str + "_meta"

            outcome_err_code = set_outcome.status_codes[i]
            log_key = meta_key if i % 2 == 0 else key_str
            if outcome_err_code == eic.StatusCode.SUCCESS:
                logger.debug(f"eic batched_put {log_key} success")
            else:
                logger.error(
                    f"eic batched_put {log_key} failed, err_code {outcome_err_code}"
                )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    def support_batched_put(self) -> bool:
        return False

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not keys:
            return 0

        # Convert all keys to strings at once
        key_strings = eic.StringVector()
        for key in keys:
            key_strings.append(key.to_string())

        # Use mexist to check all keys at once
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(key_strings, exist_option)

        if status_code != eic.StatusCode.SUCCESS:
            logger.error(
                f"eic batched_async_contains mexist failed, status_code {status_code}"
            )
            return 0

        # Count consecutive hits from the beginning
        num_hit_counts = 0
        for i, key in enumerate(keys):
            status_code = exist_outcome.status_codes[i]
            if status_code != eic.StatusCode.SUCCESS:
                logger.debug(
                    f"eic batched_async_contains {key.to_string()} miss,"
                    " err_code {status_code}"
                )
                break
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return results

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_get,
            keys=keys,
            priority=Priorities.GET,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # calling self.get will create a circular dependency
        results = await asyncio.gather(*(self._get(key) for key in keys))
        return [r for r in results if r is not None]

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def close(self):
        if hasattr(self, "pq_executor") and self.pq_executor is not None:
            await self.pq_executor.shutdown(wait=True)
        if self.connection:
            self.connection = None
        logger.info("closed the eic connection")

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass
