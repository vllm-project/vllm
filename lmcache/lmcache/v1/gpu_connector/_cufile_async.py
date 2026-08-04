# SPDX-License-Identifier: Apache-2.0
"""Minimal ctypes wrapper around the cuFile async C API.

kvikio's ``raw_read_async`` / ``raw_write_async`` work but on ext4 +
real GDS they leave ~60% read throughput on the table compared to the
bare C path (``cuFileReadAsync`` + ``cuFileStreamRegister`` + batched
submit + single ``cudaStreamSynchronize``). This module exposes the
same C primitives directly from Python so callers that batch
submissions can match the C-direct throughput.

Surface:

- :func:`register_buffer` / :func:`deregister_buffer` — wrap
  ``cuFileBufRegister`` / ``cuFileBufDeregister`` on a torch tensor.
- :func:`register_stream` / :func:`deregister_stream` — wrap
  ``cuFileStreamRegister`` / ``cuFileStreamDeregister`` on a raw
  CUDA stream handle.
- :class:`AsyncHandle` — opens a file with ``O_DIRECT`` (required by
  cuFile on ext4) and registers the cuFile handle. ``read_async`` /
  ``write_async`` enqueue an async IO on a stream and return a
  :class:`Submission`. Callers run ``cudaStreamSynchronize`` once to
  drain a batch; :meth:`Submission.bytes_done` returns the actual
  byte count after the sync.

This module is intentionally narrow: no thread pool, no future
abstraction, no LRU. It is the layer :class:`GDSContext`
(``lmcache.v1.gpu_connector.gds_context``) uses to talk to libcufile on
the GDS DMA fast path.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional
import ctypes
import os

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    from cufile.bindings import CUfileError

# ``cufile.bindings`` dlopens ``libcufile.so`` at import time, which is absent
# on CPU-only / macOS hosts. Importing this module (transitively pulled in by
# the CLI command discovery via ``storage_manager``) must not trigger that, so
# every cufile symbol is imported lazily inside the function that uses it and
# the dlopen happens only when GDS is actually exercised. This mirrors the
# lazy ``import cufile`` in the legacy ``GdsBackend``.

# --- Declare ctypes signatures for the async symbols (see cufile.h) --


def _declare_signatures() -> None:
    """Set argtypes/restype on libcufile symbols. Idempotent."""
    # Third Party
    from cufile.bindings import CUfileError, libcufile

    if getattr(libcufile.cuFileReadAsync, "argtypes", None):
        return
    libcufile.cuFileReadAsync.argtypes = [
        ctypes.c_void_p,  # CUfileHandle_t fh
        ctypes.c_void_p,  # void *bufPtr_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *bufPtr_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p
        ctypes.c_void_p,  # CUstream stream
    ]
    libcufile.cuFileReadAsync.restype = CUfileError

    libcufile.cuFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    libcufile.cuFileWriteAsync.restype = CUfileError

    libcufile.cuFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libcufile.cuFileStreamRegister.restype = CUfileError

    libcufile.cuFileStreamDeregister.argtypes = [ctypes.c_void_p]
    libcufile.cuFileStreamDeregister.restype = CUfileError


_driver_opened = False


def _ensure_driver_open() -> None:
    """Idempotently open the cuFile driver and declare async signatures."""
    global _driver_opened
    if _driver_opened:
        return
    # Third Party
    from cufile.bindings import cuFileDriverOpen

    cuFileDriverOpen()
    _declare_signatures()
    _driver_opened = True


def close_driver() -> None:
    """Close the cuFile driver. Optional — useful in tests."""
    global _driver_opened
    if not _driver_opened:
        return
    # Third Party
    from cufile.bindings import cuFileDriverClose

    try:
        cuFileDriverClose()
    finally:
        _driver_opened = False


def _check(err: "CUfileError", op: str) -> None:
    """Convert a non-zero ``CUfileError_t`` into a Python exception."""
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: cuFileError(err={err.err}, cu_err={err.cu_err})"
        )


# --- Handle registration -------------------------------------------


def register_handle(fd: int) -> Any:
    """Register an open fd with cuFile and return the ``CUfileHandle_t``.

    Opens the cuFile driver on first use. The returned handle is accepted
    directly as the first argument of ``cuFileReadAsync`` / ``cuFileWriteAsync``.
    """
    _ensure_driver_open()
    # Third Party
    from cufile.bindings import cuFileHandleRegister

    return cuFileHandleRegister(fd)


def deregister_handle(handle: Any) -> None:
    """Reverse of :func:`register_handle` (``cuFileHandleDeregister``)."""
    # Third Party
    from cufile.bindings import cuFileHandleDeregister

    cuFileHandleDeregister(handle)


# --- Buffer / stream registration ----------------------------------


def register_buffer(buf: torch.Tensor) -> None:
    """Register a device tensor with cuFile for GDS DMA.

    Must be called before any ``read_async`` / ``write_async`` whose
    ``buf_base`` falls inside this tensor's allocation. Implicitly
    opens the cuFile driver on first use.

    Uses ``libcufile.cuFileBufRegister`` directly (not the
    ``cufile.bindings`` wrapper) because the wrapper hides the error
    code by raising internally — we want the raw status so callers
    see ``cuFileError(err=…, cu_err=…)`` instead of a Python re-raise.
    """
    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on CUDA")
    # Third Party
    from cufile.bindings import libcufile

    nbytes = buf.numel() * buf.element_size()
    _check(
        libcufile.cuFileBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(0),
        ),
        "cuFileBufRegister",
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`."""
    # Third Party
    from cufile.bindings import libcufile

    _check(
        libcufile.cuFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "cuFileBufDeregister",
    )


# cuFileStreamRegister flags (cufile.h): declare the buffer offset, file offset,
# and size are all set at submission time (CU_FILE_STREAM_FIXED_* = 0x1|0x2|0x4).
# Worth ~12% higher read throughput vs 0x0 in our benchmark (write unchanged).
# PAGE_ALIGNED_INPUTS (0x8) is omitted -- transfer sizes are not always 4 KiB.
_STREAM_REGISTER_FLAGS = 0x7


def register_stream(raw_stream: int) -> None:
    """Register a CUDA stream with cuFile.

    ``raw_stream`` is the integer ``CUstream`` handle — get it via
    ``torch_dev.current_stream().cuda_stream``.

    Optional for correctness (``read_async`` / ``write_async`` also take the
    stream per call). We register with the FIXED_* flags (0x7): cuFile still
    reads the size/offset pointers at stream-execution time -- so their storage
    must stay alive and unchanged until completion (see ``Submission``) -- but
    promising the values are fixed at submission lets cuFile skip per-op setup,
    worth ~12% higher read throughput in our benchmark.
    """
    # Third Party
    from cufile.bindings import libcufile

    _ensure_driver_open()
    _check(
        libcufile.cuFileStreamRegister(
            ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
        ),
        "cuFileStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream`."""
    # Third Party
    from cufile.bindings import libcufile

    _check(
        libcufile.cuFileStreamDeregister(ctypes.c_void_p(raw_stream)),
        "cuFileStreamDeregister",
    )


# --- AsyncHandle + Submission --------------------------------------


class Submission:
    """One in-flight ``cuFileReadAsync`` / ``cuFileWriteAsync``.

    Holds the host-side ``size_p`` / ``file_offset_p`` /
    ``bufPtr_offset_p`` / ``bytes_done_p`` storage that cuFile writes
    into asynchronously. These ctypes objects MUST stay alive until
    the stream actually executes the op — keep the :class:`Submission`
    reference (or stash it in a list) until after the stream sync.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(
        self,
        size: int,
        file_offset: int,
        buf_offset: int,
    ) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        """Bytes actually transferred. Valid only AFTER the stream sync."""
        return self._bytes_done.value


class AsyncHandle:
    """Open file + cuFile handle wrapper.

    Opens with ``O_DIRECT`` (required for cuFile's GDS fast path on
    ext4). Optionally pre-allocates the file via ``posix_fallocate``.
    """

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        path: str,
        writable: bool = False,
        fallocate_size: Optional[int] = None,
        mode: int = 0o644,
    ) -> None:
        flags = os.O_DIRECT
        if writable:
            flags |= os.O_CREAT | os.O_RDWR
        else:
            flags |= os.O_RDONLY
        self.path = path
        self.writable = writable
        self._fd = os.open(path, flags, mode)
        try:
            if fallocate_size is not None and writable:
                os.posix_fallocate(self._fd, 0, fallocate_size)
            self._handle = register_handle(self._fd)
        except Exception:
            os.close(self._fd)
            raise

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: Any,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered cuFile handle.

        For callers that open + register the file themselves (e.g. a slab that
        must be created, truncated, and ``posix_fallocate``d before
        ``cuFileHandleRegister``) and just need an ``AsyncHandle`` around the
        result.
        """
        obj = cls.__new__(cls)
        obj._fd = fd
        obj._handle = handle
        obj.path = path
        obj.writable = writable
        return obj

    @property
    def fd(self) -> int:
        return self._fd

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``cuFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g.
        ``buf.data_ptr()``). ``buf_offset`` is the byte offset within
        that registration that the data should land at.
        """
        # Third Party
        from cufile.bindings import libcufile

        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            libcufile.cuFileReadAsync(
                self._handle,
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "cuFileReadAsync",
        )
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``cuFileWriteAsync`` on the stream."""
        # Third Party
        from cufile.bindings import libcufile

        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        _check(
            libcufile.cuFileWriteAsync(
                self._handle,
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub._size),
                ctypes.byref(sub._file_offset),
                ctypes.byref(sub._buf_offset),
                ctypes.byref(sub._bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            "cuFileWriteAsync",
        )
        return sub

    def close(self) -> None:
        """Deregister the cuFile handle and close the fd."""
        if self._fd < 0:
            return
        try:
            deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "AsyncHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
