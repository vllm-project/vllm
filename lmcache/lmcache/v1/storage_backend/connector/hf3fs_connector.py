# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
# Authors:
#   Wenwen Chen <wenwen.chen@samsung.com>
#   Ruyi Zhang <ruyi.zhang@samsung.com>


# Standard
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional
import asyncio
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

HF3FS_AVAILABLE = True
try:
    # Third Party
    from hf3fs_fuse.io import (
        deregister_fd,
        extract_mount_point,
        make_ioring,
        make_iovec,
        register_fd,
    )
except ImportError:
    HF3FS_AVAILABLE = False

logger = init_logger(__name__)


class HF3fsConnector(FSConnector):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        mount_point: str,
        iov_size: int,
        ior_entries: int,
        io_depth: int,
        numa_id: int,
        io_thread_num: int,
        plugin_name: Optional[str] = None,
        base_paths_str: Optional[str] = None,
    ):
        """
        Args:
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
            mount_point: Mount point of 3FS
            iov_size: Shared memory size for Iov
            ior_entries: Max num of concurrent requests that can be submitted in Ior
            io_depth: Control with I/O depth in ior
            numa_id:  NUMA ID for Ior shared memory
            io_thread_num: Number of io thread
            plugin_name: Plugin instance name
                (e.g. "hf3fs", "hf3fs.primary")
            base_paths_str: Comma-separated base paths
                (legacy, passed from adapter when using hf3fs://URL)
        """
        if not HF3FS_AVAILABLE:
            logger.error(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package"
            )
            raise ImportError(
                "hf3fs_fuse.io is not available. Please install the hf3fs_fuse package."
            )
        super().__init__(
            loop,
            local_cpu_backend,
            local_cpu_backend.config,
            plugin_name,
            base_paths_str,
        )

        self.config = Hf3fsClientConfig(
            mount_point=mount_point,
            iov_size=iov_size,
            ior_entries=ior_entries,
            io_depth=io_depth,
            numa_id=numa_id,
        )

        # check mount point
        try:
            hf3fs_mount_point = extract_mount_point(mount_point)
        except Exception as e:
            logger.error(f"Failed to extract 3FS mount point: {e}")
            raise Exception(f"Failed to extract 3FS mount point: {e}") from e

        if hf3fs_mount_point != mount_point:
            logger.error(
                f"Invalid 3FS mount point, config:{mount_point},"
                f" extract:{hf3fs_mount_point}"
            )
            raise Exception(
                f"Invalid 3FS mount point, config:{mount_point},"
                f" extract:{hf3fs_mount_point}"
            )

        self.io_executor = ThreadPoolExecutor(
            max_workers=io_thread_num, thread_name_prefix="HF3fsConnector_executor"
        )
        # create Hf3fsClient instance in main thread
        self.client = Hf3fsClient.get_instance(self.config)

    def get_client(self):
        return Hf3fsClient.get_instance(self.config)

    def _close(self):
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Failed to close HF3fsConnector,{e}")

        if hasattr(self, "io_executor") and self.io_executor is not None:
            try:
                self.io_executor.shutdown(wait=True)
            except Exception as e:
                logger.error(f"Failed to close HF3fsConnector,{e}")

    async def close(self):
        await asyncio.to_thread(self._close)
        logger.info("Finish close HF3fsConnector")

    def __del__(self) -> None:
        self._close()
        logger.info("Finish delete HF3fsConnector")

    def support_batched_get(self) -> bool:
        return False

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return await self.loop.run_in_executor(self.io_executor, self.sync_get, key)

    def sync_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)
        client = self.get_client()
        memory_obj = None

        try:
            with Hf3fsFile(file_path, os.O_RDONLY, client) as f:
                meta_len = None
                data_len = None
                if self.save_chunk_meta:
                    # Read metadata buffer first to get shape, dtype, fmt
                    # to be able to allocate memory object for the data and read into it
                    md_buffer = bytearray(self.remote_metadata_bytes)
                    md_view = memoryview(md_buffer)
                    meta_len = f.read(md_view, len(md_view), 0)
                    if meta_len != len(md_view):
                        raise RuntimeError(
                            f"Partial read meta {len(md_view)} got {meta_len}"
                        )

                    # Deserialize metadata and allocate memory
                    metadata = RemoteMetadata.deserialize(md_buffer)
                    memory_obj = self.local_cpu_backend.allocate(
                        metadata.shapes, metadata.dtypes, metadata.fmt
                    )
                else:
                    memory_obj = self.local_cpu_backend.allocate(
                        self.meta_shapes, self.meta_dtypes, self.meta_fmt
                    )
                if memory_obj is None:
                    logger.debug("Memory allocation failed during async disk load.")
                    return None

                # Read the actual data into allocated memory
                data_buffer = memory_obj.byte_array
                if isinstance(data_buffer, memoryview):
                    data_view = memory_obj.byte_array
                    if data_view.format == "<B":
                        data_view = data_view.cast("B")
                        logger.debug("convert memory_obj.byte_array format to B")
                else:
                    data_view = memoryview(data_buffer)
                    logger.debug("convert memory_obj.byte_array to memoryview")

                if self.save_chunk_meta:
                    data_len = f.read(data_view, len(data_view), meta_len)
                    if data_len != len(data_view):
                        raise RuntimeError(
                            f"Partial read data {len(data_view)} got {data_len}"
                        )
                else:
                    # logger.info("get, before read", file_path)
                    data_len = f.read(data_view, len(data_view), 0)

                memory_obj = self.reshape_partial_chunk(memory_obj, data_len)
            return memory_obj

        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error(f"Failed to read from file {file_path}: {str(e)}")
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    def support_batched_put(self) -> bool:
        return False

    def support_batched_get_non_blocking(self) -> bool:
        return False

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return await self.loop.run_in_executor(
            self.io_executor, self.sync_put, key, memory_obj
        )

    def sync_put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to file system"""
        final_path, temp_path = self._get_file_and_tmp_path(key)
        client = self.get_client()

        try:
            # Prepare metadata
            buffer = memoryview(memory_obj.byte_array)
            metadata = (
                RemoteMetadata(
                    len(buffer),
                    memory_obj.get_shapes(),
                    memory_obj.get_dtypes(),
                    memory_obj.get_memory_format(),
                )
                if self.save_chunk_meta
                else None
            )

            size = len(buffer)
            # Use standard sync I/O
            # Write to file (metadata + data)
            with Hf3fsFile(
                temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, client
            ) as f:
                md_len = 0
                if metadata is not None:
                    md = metadata.serialize()
                    md_len = len(md)
                    # write metadata
                    f.write(md, md_len, 0)

                # write data
                f.write(buffer.tobytes(), size, md_len)
                # logger.info(f"num_write = {size}, key chunk_hash={key.chunk_hash}")

                # Atomically rename temp file to final destination
                Hf3fsFile.rename(temp_path, final_path)
        except Exception as e:
            logger.error(f"Failed to write file {final_path}: {str(e)}")
            Hf3fsFile.remove(temp_path)  # Remove corrupted file
            raise

    def remove_sync(self, key: CacheEngineKey) -> bool:
        file_path = self._get_file_path(key)
        return Hf3fsFile.remove(file_path)

    def support_batched_contains(self) -> bool:
        return False

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system"""
        file_path = self._get_file_path(key)
        return Hf3fsFile.exists(file_path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system synchronized"""
        file_path = self._get_file_path(key)
        return Hf3fsFile.exists(file_path)


class Hf3fsClientConfig:
    def __init__(
        self,
        mount_point: str,
        iov_size: int = 200 * (1 << 20),  # 200MB
        ior_entries: int = 256,
        io_depth: int = 0,
        timeout: int = 200,
        numa_id: int = -1,
    ):
        self.mount_point = mount_point
        self.iov_size = iov_size
        self.ior_entries = ior_entries
        self.io_depth = io_depth
        self.timeout = timeout
        self.numa_id = numa_id


class Hf3fsClient:
    """Singleton mode, a instance per thread"""

    _thread_local = threading.local()

    @classmethod
    def get_instance(cls, config: Hf3fsClientConfig):
        """Initialize the HF3FS client"""

        # check whether the instance exist
        if not hasattr(cls._thread_local, "_instance"):
            # init the instance
            cls._thread_local._instance = cls(config)
            logger.info(
                f"Create {cls._thread_local._instance.client_name} instance success"
            )
        else:
            logger.debug(
                f"Hf3fsClient instance already exist in {threading.get_ident()}"
            )
        return cls._thread_local._instance

    def __init__(self, config: Hf3fsClientConfig):
        self.config = config
        self.client_name = f"3FSClient_{os.getpid()}_{threading.get_ident()}"
        self._closed = False  #
        # shared memory
        try:
            self.shm_read = SharedMemory(
                name=f"{self.client_name}_rd", size=self.config.iov_size, create=True
            )
            self.shm_write = SharedMemory(
                name=f"{self.client_name}_wt", size=self.config.iov_size, create=True
            )
        except Exception as e:
            logger.error(f"{self.client_name} failed to create share memory {e}")
            self.close()
            raise
        logger.debug(f"{self.client_name} create share memory successfully")

        try:
            self.iov_read = make_iovec(
                shm=self.shm_read,
                hf3fs_mount_point=self.config.mount_point,
                numa=self.config.numa_id,
            )
            self.iov_write = make_iovec(
                shm=self.shm_write,
                hf3fs_mount_point=self.config.mount_point,
                numa=self.config.numa_id,
            )
        except Exception as e:
            logger.error(f"{self.client_name} failed to create iov: {e}")
            self.close()
            raise
        logger.debug(f"{self.client_name} create iov successfully")

        # read I/O ring
        try:
            self.ior_read = make_ioring(
                self.config.mount_point,
                self.config.ior_entries,
                for_read=True,
                timeout=self.config.timeout,
                numa=self.config.numa_id,
            )
        except Exception as e:
            logger.error(f"{self.client_name} failed to create ior_read: {e}")
            self.close()
            raise
        logger.debug(f"{self.client_name} create ior_read successfully")

        # write I/O ring
        try:
            self.ior_write = make_ioring(
                self.config.mount_point,
                self.config.ior_entries,
                for_read=False,
                timeout=self.config.timeout,
                numa=self.config.numa_id,
            )
        except Exception as e:
            logger.error(f"{self.client_name} failed to create ior_write: {e}")
            self.close()
            raise

        logger.debug(f"{self.client_name} create ior_write successfully")
        # lock for read and write
        self.lock_read = threading.RLock()
        self.lock_write = threading.RLock()
        logger.debug(f"{self.client_name} initialized end ")

    def close(self):
        if self._closed:
            logger.debug(f"{self.client_name} already closed, skip")
            return
        logger.debug(f"{self.client_name} close begin")
        try:
            #  clear I/O resource first, then share memory
            if hasattr(self, "ior_read") and self.ior_read is not None:
                del self.ior_read
                self.ior_read = None
            if hasattr(self, "ior_write") and self.ior_write is not None:
                del self.ior_write
                self.ior_write = None
            if hasattr(self, "iov_read") and self.iov_read is not None:
                del self.iov_read
                self.iov_read = None
            if hasattr(self, "iov_write") and self.iov_write is not None:
                del self.iov_write
                self.iov_write = None

            # clear share memory
            if hasattr(self, "shm_read") and self.shm_read is not None:
                try:
                    self.shm_read.close()
                    self.shm_read.unlink()
                except FileNotFoundError:
                    # already release, ignore it
                    logger.debug(f"{self.client_name} shm_read already unlinked")
                except Exception as e:
                    logger.error(f"{self.client_name} error closing shm_read: {e}")
                finally:
                    self.shm_read = None
            if hasattr(self, "shm_write") and self.shm_write is not None:
                try:
                    self.shm_write.close()
                    self.shm_write.unlink()
                except FileNotFoundError:
                    # already release, ignore it
                    logger.debug(f"{self.client_name} shm_write already unlinked")
                except Exception as e:
                    logger.error(f"{self.client_name} error closing shm_write: {e}")
                finally:
                    self.shm_write = None

            self._closed = True
            logger.info(f"{self.client_name} release successfully")
        except Exception as e:
            logger.error(f"{self.client_name} error during close: {e}")
            self._closed = True  # Mark this flag even it exception

    def __del__(self):
        try:
            logger.debug(f"{self.client_name} __del__ begin")
            self.close()
        except Exception as e:
            # do not throw exception in __del__
            logger.error(f"{self.client_name} error in __del__: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """auto close when exit"""
        self.close()
        return False  # False, exceptions are not suppressed

    def _read(self, fd: int, buffer: memoryview, length: int, file_offset: int) -> int:
        assert length <= self.config.iov_size
        total_bytes_read = 0

        # prepare
        self.ior_read.prepare(self.iov_read[:length], True, fd, file_offset)
        # submit
        try:
            done = self.ior_read.submit().wait(min_results=1)[0]
        except Exception as e:
            logger.error(f"{self.client_name} read submit failed: {e}")
            raise RuntimeError(f"{self.client_name} read submit failed") from e

        total_bytes_read = done.result

        # read from shm
        buffer[:total_bytes_read] = self.shm_read.buf[:total_bytes_read]
        return total_bytes_read

    def read(self, fd: int, buffer: memoryview, length: int, start: int = 0) -> int:
        with self.lock_read:
            offset = 0
            total_bytes_read = 0
            iov_size = self.config.iov_size

            while offset < length:
                sub_len = min(iov_size, length - offset)
                bytes_read = self._read(
                    fd, buffer[offset : offset + sub_len], sub_len, offset + start
                )
                if bytes_read > 0:
                    total_bytes_read += bytes_read
                    offset += bytes_read
                if bytes_read != sub_len:
                    break
            if total_bytes_read != length:
                logger.warning(
                    f"{self.client_name} read: requested {length}, "
                    f"got {total_bytes_read}"
                )
            return total_bytes_read

    def _write(self, fd: int, buffer: memoryview, length: int, file_offset: int) -> int:
        assert length <= self.config.iov_size
        total_bytes_written = 0

        # write shm
        self.shm_write.buf[:length] = buffer[:length]

        # prepare
        self.ior_write.prepare(self.iov_write[:length], False, fd, file_offset)
        # userdata= buffer)
        # submit
        try:
            done = self.ior_write.submit().wait(min_results=1)[0]
        except Exception as e:
            logger.error(f"{self.client_name} write submit failed: {e}")
            raise RuntimeError(f"{self.client_name} write submit failed") from e

        total_bytes_written = done.result
        return total_bytes_written

    def write(self, fd: int, buffer: memoryview, length: int, start: int = 0) -> int:
        with self.lock_write:
            offset = 0
            total_bytes_written = 0
            iov_size = self.config.iov_size

            while offset < length:
                sub_len = min(iov_size, length - offset)
                written = self._write(
                    fd, buffer[offset : offset + sub_len], sub_len, offset + start
                )
                if written > 0:
                    total_bytes_written += written
                    offset += written
                if written != sub_len:
                    break
            if total_bytes_written != length:
                logger.error(
                    f"{self.client_name} write: requested {length}, "
                    f"got {total_bytes_written}"
                )
                raise RuntimeError(
                    f"{self.client_name} write: requested {length}, "
                    f"got {total_bytes_written}"
                )

            return total_bytes_written


class Hf3fsFile:
    def __init__(self, fname: Path, flags: int, client: Hf3fsClient):
        self.fd = -1
        self.fname = fname
        self.flags = flags
        self.client = client

    def open(self):
        try:
            self.fd = os.open(self.fname, self.flags)
            register_fd(self.fd)
            logger.debug(f"Open file {self.fname} with flag {self.flags} successfully")
        except Exception as e:
            logger.error(f"Open file {self.fname} with flag {self.flags} failed {e}")
            raise

    def close(self):
        if not hasattr(self, "fd") or self.fd == -1:
            return
        try:
            deregister_fd(self.fd)
            os.close(self.fd)
            logger.debug(f"Close file ({self.fname}) successfully")
        except Exception as e:
            logger.error(f"Close file {self.fname} failed {e}")
            raise
        self.fd = -1
        self.fname = ""

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type:
            logger.error(f"Hf3fsFile closed with exception: {exc_val}")
        return False  # False, exceptions are not suppressed

    @staticmethod
    def exists(fname: Path) -> bool:
        # check if file exists
        return os.path.exists(fname)

    @staticmethod
    def remove(fname: Path) -> bool:
        try:
            os.remove(fname)
            return True
        except OSError as e:
            logger.error(f"Failed to remove file {fname}: {e}")
            return False

    @staticmethod
    def rename(old_fname: Path, new_fname: Path):
        try:
            os.rename(old_fname, new_fname)
        except OSError as e:
            logger.warning(f"rename {old_fname} to {new_fname} failed, {e}")
            raise
        return

    def write(self, data: memoryview, length: int, file_offset: int = 0) -> int:
        return self.client.write(self.fd, data, length, file_offset)

    def read(self, buffer: memoryview, length: int, file_offset: int = 0) -> int:
        return self.client.read(self.fd, buffer, length, file_offset)

    def get_client_name(self) -> str:
        return self.client.client_name
