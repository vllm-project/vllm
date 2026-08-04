# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import List, Optional, Tuple, no_type_check
import asyncio
import os

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


class FSConnector(RemoteConnector):
    """File system based connector that stores data in local files.

    Data is stored in the following format:
    - Each key is stored as a separate file
    - File content: metadata (remote_metadata_bytes) + serialized data
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
        plugin_name: Optional[str] = None,
        base_paths_str: Optional[str] = None,
    ):
        """
        Args:
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
            config: Lmcache engine config
            plugin_name: Plugin instance name
                (e.g. "fs", "fs.primary")
            base_paths_str: Comma-separated base paths
                (legacy, passed from adapter when using
                fs:// URL)
        """
        # initialize base class
        super().__init__(
            local_cpu_backend.config,
            local_cpu_backend.metadata,
        )

        base_path = base_paths_str
        if base_path is None:
            # Resolve from extra_config
            extra_config = config.extra_config if config else None
            if extra_config is not None:
                key_prefix = plugin_name or "fs"
                base_path = extra_config.get(
                    "remote_storage_plugin.%s.base_path" % key_prefix
                )
            if base_path is None:
                if extra_config is not None:
                    base_path = extra_config.get("fs_base_path")
            if base_path is None:
                raise ValueError(
                    "FS connector requires base_path via URL or extra_config"
                )

        # Parse comma separated paths
        self.base_paths = (
            [Path(p.strip()) for p in base_path.split(",")]
            if "," in base_path
            else [Path(base_path)]
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        relative_tmp_dir = (
            None
            if config is None
            else config.get_extra_config_value("fs_connector_relative_tmp_dir", None)
        )
        self.relative_tmp_dir = None
        if relative_tmp_dir is not None:
            self.relative_tmp_dir = Path(relative_tmp_dir)
            assert not self.relative_tmp_dir.is_absolute()

        self.read_ahead_size = (
            None
            if config is None
            else config.get_extra_config_value("fs_connector_read_ahead_size", None)
        )

        self.use_odirect = (
            False
            if config is None
            else config.get_extra_config_value("fs_connector_use_odirect", False)
        )
        self.os_disk_bs = 0
        if self.use_odirect:
            # save_chunk_meta is useful if save_unfull_chunk is True, since partial
            # chunk will be saved. When loading partial chunk, we need to know
            # data size and shape. However, chunk meta is short (28 bytes) which
            # is not aligned to disk block size (512 or 4096).
            # Therefore, we disable O_DIRECT if save_chunk_meta is True.
            # TODO: support O_DIRECT for save_chunk_meta by
            # padding meta data to 4096.
            if self.save_chunk_meta:
                logger.warning("Cannot use O_DIRECT if save_chunk_meta enabled.")
                self.use_odirect = False
            else:
                stat = os.statvfs(self.base_paths[0])
                self.os_disk_bs = stat.f_bsize

        logger.info(
            "Initialized FSConnector with base paths %s, "
            "relative tmp dir: %s, "
            "read ahead size: %s, "
            "use O_DIRECT: %s",
            self.base_paths,
            self.relative_tmp_dir,
            self.read_ahead_size,
            self.use_odirect,
        )
        # Create directories for all paths
        for path in self.base_paths:
            path.mkdir(parents=True, exist_ok=True)
            if self.relative_tmp_dir is not None:
                (path / self.relative_tmp_dir).mkdir(parents=False, exist_ok=True)

    def _get_base_path(self, key: CacheEngineKey) -> Path:
        """Get file base path for the given key"""
        if len(self.base_paths) == 1:
            base_path = self.base_paths[0]
        else:
            # Calculate hash value and modulo to select path
            hash_val = abs(key.chunk_hash)
            idx = hash_val % len(self.base_paths)
            base_path = self.base_paths[idx]

        return base_path

    def _get_file_name(self, key: CacheEngineKey) -> str:
        return key.to_string().replace("/", "-SEP-") + ".data"

    def _get_file_path(self, key: CacheEngineKey) -> Path:
        """Get file path for the given key"""
        base_path = self._get_base_path(key)
        file_name = self._get_file_name(key)
        return base_path / file_name

    def _get_file_and_tmp_path(self, key: CacheEngineKey) -> Tuple[Path, Path]:
        """Get file and tmp path for the given key"""
        base_path = self._get_base_path(key)
        file_name = self._get_file_name(key)
        file_path = base_path / file_name
        if self.relative_tmp_dir is not None:
            tmp_path = base_path / self.relative_tmp_dir / file_name
        else:
            tmp_path = file_path.with_suffix(".tmp")
        return file_path, tmp_path

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system"""
        file_path = self._get_file_path(key)
        return await aiofiles.os.path.exists(file_path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in file system synchronized"""
        file_path = self._get_file_path(key)
        return os.path.exists(file_path)

    def _get_with_odirect(self, file_path: Path) -> Optional[MemoryObj]:
        """Synchronous direct IO read, executed in a thread."""
        fd = -1
        memory_obj: Optional[MemoryObj] = None
        try:
            memory_obj = self.local_cpu_backend.allocate(
                self.meta_shapes, self.meta_dtypes, self.meta_fmt
            )
            if memory_obj is None:
                logger.debug("Memory allocation failed.")
                return None

            buffer = memory_obj.byte_array
            size = len(buffer)

            fblock_aligned = (
                self.os_disk_bs is not None
                and self.os_disk_bs > 0
                and size % self.os_disk_bs == 0
            )
            if not fblock_aligned:
                logger.warning(
                    "Cannot use O_DIRECT for %s, size is not aligned.", file_path
                )
                with open(file_path, "rb") as f:
                    num_read = f.readinto(buffer)
            else:
                fd = os.open(file_path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    # The fd is now managed by the file object, so we "forget" it
                    # to prevent closing it in the finally block.
                    fd = -1
                    num_read = fdo.readinto(buffer)

            memory_obj = self.reshape_partial_chunk(memory_obj, num_read)
            return memory_obj

        except Exception as e:
            logger.error("Failed to read from file %s: %s", file_path, str(e))
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Get data from file system"""
        file_path = self._get_file_path(key)

        if self.use_odirect and not self.save_chunk_meta:
            return await self.loop.run_in_executor(
                None, self._get_with_odirect, file_path
            )

        memory_obj = None
        try:
            async with aiofiles.open(file_path, "rb") as f:
                if self.save_chunk_meta:
                    # Read metadata buffer first to get shape, dtype, fmt
                    # to be able to allocate memory object for the data and read into it
                    md_buffer = bytearray(self.remote_metadata_bytes)
                    num_read = await f.readinto(md_buffer)
                    if num_read != len(md_buffer):
                        raise RuntimeError(
                            f"Partial read meta {len(md_buffer)} got {num_read}"
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
                buffer = memory_obj.byte_array
                if self.save_chunk_meta:
                    # if save chunk meta, read meta will trigger
                    # read ahead if fs supported
                    num_read = await f.readinto(buffer)
                    if num_read != len(buffer):
                        raise RuntimeError(
                            f"Partial read data {len(buffer)} got {num_read}"
                        )
                else:
                    if self.read_ahead_size is None:
                        num_read = await f.readinto(buffer)
                    else:
                        if not isinstance(buffer, memoryview):
                            buffer = memoryview(buffer)

                        # trigger read head if fs supported
                        num_read_ahead = await f.readinto(
                            buffer[: self.read_ahead_size]
                        )
                        assert num_read_ahead <= self.read_ahead_size

                        # if num_read_ahead == self.read_ahead_size,
                        # means there may still be some remaining content
                        if num_read_ahead == self.read_ahead_size:
                            num_read_tail = await f.readinto(
                                buffer[self.read_ahead_size :]
                            )
                            assert num_read_tail is not None
                            num_read = num_read_ahead + num_read_tail
                        else:
                            num_read = num_read_ahead
                    # reshape and check
                    assert num_read is not None
                    memory_obj = self.reshape_partial_chunk(memory_obj, num_read)

            return memory_obj

        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                logger.error("Failed to read from file %s: %s", file_path, str(e))
            if memory_obj is not None:
                memory_obj.ref_count_down()
            return None

    def _put_with_odirect(self, file_path: Path, buffer: bytes) -> None:
        fd = -1
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_WRONLY | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            os.write(fd, buffer)
        except Exception as e:
            logger.error("Failed to write to file %s: %s", file_path, e)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data to file system"""
        final_path, temp_path = self._get_file_and_tmp_path(key)

        try:
            # Prepare metadata
            buffer = memory_obj.byte_array
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
            do_use_odirect = self.use_odirect
            if do_use_odirect:
                fblock_aligned = self.os_disk_bs > 0 and size % self.os_disk_bs == 0
                if not fblock_aligned:
                    logger.warning(
                        "Cannot use O_DIRECT for writing size %s, "
                        "which is not aligned to block size %s.",
                        size,
                        self.os_disk_bs,
                    )
                    do_use_odirect = False

            if do_use_odirect:
                # Use Direct I/O
                await self.loop.run_in_executor(
                    None, self._put_with_odirect, temp_path, buffer
                )
            else:
                # Use standard async I/O
                # Write to file (metadata + data)
                async with aiofiles.open(temp_path, "wb") as f:
                    if metadata is not None:
                        await f.write(metadata.serialize())
                    await f.write(buffer)

            # Atomically rename temp file to final destination
            await aiofiles.os.replace(temp_path, final_path)

        except Exception as e:
            logger.error("Failed to write file %s: %s", final_path, str(e))
            if await aiofiles.os.path.exists(temp_path):
                await aiofiles.os.unlink(temp_path)  # Remove corrupted file
            raise

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """
        Remove the file associated with the given key.

        Args:
            key: The key to remove.

        Returns:
            bool: True if the file was successfully removed, False otherwise.
        """
        file_path = self._get_file_path(key)
        try:
            os.remove(file_path)
            return True
        except OSError as e:
            logger.error("Failed to remove file %s: %s", file_path, e)
            return False

    @no_type_check
    async def list(self) -> List[str]:
        """List all keys in file system"""
        keys = []
        for base_path in self.base_paths:
            keys.extend([f.stem for f in base_path.glob("*.data")])
        return keys

    async def close(self):
        """Clean up resources"""
        logger.info("Closed the file system connector")
