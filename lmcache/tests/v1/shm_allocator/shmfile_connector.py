# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import asyncio
import os
import socket
import subprocess

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import (
    LocalCPUBackend,
)

logger = init_logger(__name__)

# extra_config key prefix for shmfile connector
_SHMFS_PREFIX = "shmfs."


class ShmFileConnector(RemoteConnector):
    """Connector that uses a C++ subprocess to read/write
    files via POSIX shared memory.

    Supports two communication modes:
    - **PIPE mode** (default): spawn a local subprocess and
      communicate via stdin/stdout. Used in unit tests.
    - **TCP mode**: connect to a remote shm_file_worker that
      is already listening on host:port. Used in IDC
      environments where the worker runs on the host machine
      while LMCache runs inside a container.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig] = None,
    ):
        super().__init__(
            local_cpu_backend.config,
            local_cpu_backend.metadata,
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Read shmfs.* parameters from extra_config
        storage_dir = self._shmfs_cfg(config, "storage_dir", "/tmp/lmcache_shmfs")
        worker_binary = self._shmfs_cfg(config, "worker_binary")
        worker_addr = self._shmfs_cfg(config, "worker_addr")

        self.storage_dir: str = storage_dir or "/tmp/lmcache_shmfs"
        os.makedirs(self.storage_dir, exist_ok=True)

        # Resolve shm_name: shmfs.shm_name -> shm_name -> allocator
        allocator = self.local_cpu_backend.memory_allocator
        shm_name = (
            self._shmfs_cfg(config, "shm_name")
            or self._extra_cfg(config, "shm_name")
            or getattr(allocator, "shm_name", None)
        )
        self.shm_name = shm_name
        if not self.shm_name:
            raise ValueError(
                "ShmFileConnector requires shm_name. "
                "Set extra_config shmfs.shm_name or "
                "shm_name, or use "
                "MixedMemoryAllocator(shm_name=...)."
            )

        self.shm_size = getattr(allocator, "size", 0)
        if self.shm_size <= 0:
            raise ValueError("ShmFileConnector requires allocator.size > 0")

        # Cache base address for offset calculation
        base_ptr = getattr(allocator, "buffer", None)
        if base_ptr is None:
            raise ValueError("ShmFileConnector requires allocator.buffer")
        self.base_addr = base_ptr.data_ptr()

        # Communication mode: TCP or PIPE
        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self._sock_buf = b""

        if worker_addr:
            # TCP mode: connect to a remote worker
            self._init_tcp(worker_addr)
        else:
            # PIPE mode: spawn a local subprocess
            self._init_pipe(worker_binary)

        # Attach to shared memory
        resp = self._send_cmd(
            "ATTACH %s %d %d",
            self.shm_name,
            self.shm_size,
            self.base_addr,
        )
        if not resp.startswith("OK"):
            raise RuntimeError("shm_file_worker ATTACH failed: %s" % resp)

        logger.info(
            "ShmFileConnector: attached shm=%s size=%d base_addr=%d mode=%s",
            self.shm_name,
            self.shm_size,
            self.base_addr,
            "tcp" if self._sock else "pipe",
        )

    # -- Config helpers -------------------------------------------

    @staticmethod
    def _shmfs_cfg(
        config: Optional[LMCacheEngineConfig],
        key: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Read extra_config['shmfs.<key>']."""
        if config is None:
            return default
        return config.get_extra_config_value(_SHMFS_PREFIX + key, default)

    @staticmethod
    def _extra_cfg(
        config: Optional[LMCacheEngineConfig],
        key: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Read extra_config['<key>'] (no prefix)."""
        if config is None:
            return default
        return config.get_extra_config_value(key, default)

    # -- Worker init helpers --------------------------------------

    def _init_tcp(self, worker_addr: str):
        """Connect to a remote worker via TCP."""
        host, port_str = worker_addr.rsplit(":", 1)
        port = int(port_str)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        logger.info(
            "ShmFileConnector: connected to worker at %s",
            worker_addr,
        )

    def _init_pipe(self, worker_binary: Optional[str]):
        """Spawn a local worker subprocess."""
        if worker_binary is None:
            worker_binary = os.environ.get("SHM_FILE_WORKER_BIN", "shm_file_worker")
        self._proc = subprocess.Popen(
            [worker_binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    # -- Low-level communication ----------------------------------

    def _send_cmd(self, fmt: str, *args) -> str:
        """Send a command and return the response line."""
        cmd = (fmt % args) + "\n"
        if self._sock is not None:
            return self._send_cmd_tcp(cmd)
        return self._send_cmd_pipe(cmd)

    def _send_cmd_pipe(self, cmd: str) -> str:
        """Send via stdin/stdout PIPE."""
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(cmd)
        self._proc.stdin.flush()
        return self._proc.stdout.readline().strip()

    def _send_cmd_tcp(self, cmd: str) -> str:
        """Send via TCP socket."""
        assert self._sock is not None
        self._sock.sendall(cmd.encode("utf-8"))
        # Read until newline
        while b"\n" not in self._sock_buf:
            data = self._sock.recv(4096)
            if not data:
                return "ERROR connection closed"
            self._sock_buf += data
        line, self._sock_buf = self._sock_buf.split(b"\n", 1)
        return line.decode("utf-8").strip()

    def _get_file_name(self, key: CacheEngineKey) -> str:
        return key.to_string().replace("/", "-SEP-") + ".data"

    def _file_path(self, key: CacheEngineKey) -> str:
        return os.path.join(self.storage_dir, self._get_file_name(key))

    # -- Connector interface --------------------------------------

    async def exists(self, key: CacheEngineKey) -> bool:
        path = self._file_path(key)
        return await asyncio.to_thread(os.path.isfile, path)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return os.path.isfile(self._file_path(key))

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Write memory_obj data to a file via the worker."""
        tensor = memory_obj.tensor
        assert tensor is not None
        buf_ptr = tensor.data_ptr()
        buf_size = tensor.numel() * tensor.element_size()

        path = self._file_path(key)
        resp = await asyncio.to_thread(
            self._send_cmd,
            "WRITE %s %d %d",
            path,
            buf_ptr,
            buf_size,
        )
        if not resp.startswith("OK"):
            raise RuntimeError("shm_file_worker WRITE failed: %s" % resp)

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Read a file into a newly allocated memory_obj
        via the worker subprocess."""
        path = self._file_path(key)
        if not os.path.isfile(path):
            return None

        if not self.meta_shapes or not self.meta_dtypes or not self.meta_fmt:
            logger.error("Metadata not available for get")
            return None

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )
        if memory_obj is None:
            return None

        tensor = memory_obj.tensor
        if tensor is None:
            memory_obj.ref_count_down()
            return None

        buf_ptr = tensor.data_ptr()
        buf_size = tensor.numel() * tensor.element_size()

        resp = await asyncio.to_thread(
            self._send_cmd,
            "READ %s %d %d",
            path,
            buf_ptr,
            buf_size,
        )
        if not resp.startswith("OK"):
            memory_obj.ref_count_down()
            logger.error("shm_file_worker READ failed: %s", resp)
            return None

        parts = resp.split()
        bytes_read = int(parts[1]) if len(parts) > 1 else 0
        if bytes_read <= 0:
            memory_obj.ref_count_down()
            return None

        try:
            return self.reshape_partial_chunk(memory_obj, bytes_read)
        except Exception:
            logger.error(
                "reshape_partial_chunk failed for key %s",
                key,
            )
            memory_obj.ref_count_down()
            return None

    async def list(self) -> List[str]:
        files = await asyncio.to_thread(os.listdir, self.storage_dir)
        return [
            f.replace(".data", "").replace("-SEP-", "/")
            for f in files
            if f.endswith(".data")
        ]

    async def close(self):
        try:
            self._send_cmd("QUIT")
        except Exception:
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            self._proc = None
        logger.info("ShmFileConnector closed")
