# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import reduce
from typing import List, Optional, Union, no_type_check
import asyncio
import ctypes
import operator

# Third Party
import infinistore
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

# reuse
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

MAX_BUFFER_SIZE = 40 << 20  # 40MB
METADATA_BYTES_LEN = 28
MAX_BUFFER_CNT = 16


def _get_ptr(mv: Union[bytearray, memoryview]) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class InfinistoreConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        dev_name: str,
        link_type: str,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(memory_allocator.config, memory_allocator.metadata)

        config = infinistore.ClientConfig(
            host_addr=host,
            service_port=port,
            log_level="info",
            connection_type=infinistore.TYPE_RDMA,
            ib_port=1,
            link_type=link_type,
            dev_name=dev_name,
        )

        self.rdma_conn = infinistore.InfinityConnection(config)

        self.loop = loop
        self.rdma_conn.connect()

        self.send_buffers = []
        self.recv_buffers = []
        self.send_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=MAX_BUFFER_CNT)
        self.recv_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=MAX_BUFFER_CNT)

        self.buffer_size = MAX_BUFFER_SIZE
        self.memory_allocator = memory_allocator

        for i in range(MAX_BUFFER_CNT):
            send_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(send_buffer), self.buffer_size)
            self.send_buffers.append(send_buffer)
            self.send_queue.put_nowait(i)

            recv_buffer = bytearray(self.buffer_size)
            self.rdma_conn.register_mr(_get_ptr(recv_buffer), self.buffer_size)
            self.recv_buffers.append(recv_buffer)
            self.recv_queue.put_nowait(i)

    async def exists(self, key: CacheEngineKey) -> bool:
        def blocking_io():
            return self.rdma_conn.check_exist(key.to_string())

        return await self.loop.run_in_executor(None, blocking_io)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self.rdma_conn.check_exist(key.to_string())

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        buf_idx = await self.recv_queue.get()
        buffer = self.recv_buffers[buf_idx]
        try:
            await self.rdma_conn.rdma_read_cache_async(
                [(key_str, 0)], self.buffer_size, _get_ptr(buffer)
            )
        except Exception as e:
            logger.warning(f"get failed: {e}")
            self.recv_queue.put_nowait(buf_idx)
            return None

        metadata = RemoteMetadata.deserialize(buffer)

        num_elements = reduce(operator.mul, metadata.shapes[0])
        assert len(metadata.dtypes) == 1
        temp_tensor = torch.frombuffer(
            buffer,
            dtype=metadata.dtypes[0],
            offset=METADATA_BYTES_LEN,
            count=num_elements,
        ).reshape(metadata.shapes[0])

        memory_obj = self.memory_allocator.allocate(
            metadata.shapes[0],
            metadata.dtypes[0],
            metadata.fmt,
        )

        assert memory_obj is not None
        assert memory_obj.tensor is not None

        # deep copy to pinned memory
        # and hot cache will reference this memory obj
        memory_obj.tensor.copy_(temp_tensor)

        logger.debug(f"get key: {key_str} done, {memory_obj.get_shape()}")
        self.recv_queue.put_nowait(buf_idx)

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()

        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()

        buf_idx = await self.send_queue.get()
        buffer = self.send_buffers[buf_idx]

        RemoteMetadata(
            len(kv_bytes), kv_shapes, kv_dtypes, memory_format
        ).serialize_into(buffer)

        buffer[METADATA_BYTES_LEN : METADATA_BYTES_LEN + len(kv_bytes)] = kv_bytes

        size = memory_obj.get_physical_size()

        if size + METADATA_BYTES_LEN > self.buffer_size:
            raise ValueError(
                f"Value size ({size + METADATA_BYTES_LEN} bytes)"
                f"exceeds the maximum allowed size"
                f"({self.buffer_size} bytes). Please decrease chunk_size."
            )
        try:
            await self.rdma_conn.rdma_write_cache_async(
                [(key_str, 0)], METADATA_BYTES_LEN + size, _get_ptr(buffer)
            )
        except Exception as e:
            logger.warning(f"exception happens in rdma_write_cache_async kv_bytes {e}")
            return
        finally:
            self.send_queue.put_nowait(buf_idx)

        logger.debug(f"put key: {key.to_string()}, {memory_obj.get_shape()}")

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        self.rdma_conn.close()
        logger.info("Closed the infinistore connection")
