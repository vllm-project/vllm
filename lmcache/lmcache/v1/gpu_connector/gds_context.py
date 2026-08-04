# SPDX-License-Identifier: Apache-2.0
"""Process-global GPUDirect Storage data path for the GDS L1 tier.

The actual GPU<->slab DMA goes through the platform GDS library -- cuFile on
NVIDIA, hipFile on AMD ROCm -- reached via the :mod:`_gds_async` dispatch shim
(imported here as ``ca``), so this module is platform-agnostic.

One :class:`GDSContext` per worker process owns the slab file, its GDS handle,
the registered GPU staging buffers, and the stream-ordered GDS submissions.
Created once at startup by :func:`initialize_gds_context`, reached via
:func:`get_gds_context`. :meth:`GDSContext.register_gpu_buffer` registers a
staging buffer; :meth:`GDSContext.transfer_async` moves a chunk between that
buffer and the slab. No POSIX fallback -- if the GDS library is unavailable,
construction fails loudly. The slab is cleared on init, so it does not survive
a restart (GDS L1 is treated like DRAM).
"""

# Standard
from dataclasses import dataclass, field
from typing import Optional
import bisect
import enum
import functools
import os
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import _gds_async as ca
from lmcache.v1.memory_management import GDSMemoryObject

logger = init_logger(__name__)

_SLAB_FILENAME = "lmcache_gds_slab.bin"
_CUFILE_ALIGNMENT = 4096
# A single GDS buffer registration / DMA is capped at 16 MiB (both cuFile and
# hipFile); larger buffers and chunks are registered and transferred in
# <=16 MiB regions.
_MAX_CUFILE_REGION = 16 * 1024 * 1024
# GDS submissions to accumulate before recording a completion event and
# draining finished ones (keeps the live submission set bounded).
_SUBMISSION_CHECKPOINT_EVERY = 64


class SlabDirection(enum.Enum):
    """Direction of a GDS slab transfer. GPUDirect DMAs run straight between GPU
    memory and the slab *file* (no host buffer), so directions are file I/O
    (READ/WRITE), not host<->device (H2D/D2H)."""

    READ = enum.auto()  # slab file -> GPU buffer
    WRITE = enum.auto()  # GPU buffer -> slab file


@dataclass
class _StreamSubmissions:
    """Per-stream GDS submissions, kept alive until their DMA has run.

    Submissions accumulate in ``uncommitted``, move to ``inflight`` behind a
    CUDA event on the stream, and drop once it completes. Per-stream because an
    event only orders work on its own stream.
    """

    uncommitted: list[ca.Submission] = field(default_factory=list)
    inflight: list[tuple[torch.Event, list[ca.Submission]]] = field(
        default_factory=list
    )
    ops_since_checkpoint: int = 0


class GDSContext:
    """Per-process GDS context owning the slab file and its DMA path.

    The singleton always exists but is inert until :meth:`initialize` creates
    the slab and registers the GDS handle (flipping :attr:`initialized`).
    While off, ``register_gpu_buffer`` is a no-op.
    """

    #: Whether :meth:`initialize` has completed (GDS L1 is active).
    initialized: bool = False

    def __init__(self) -> None:
        # ``initialized`` defaults to False via the class attribute; it is
        # flipped to True by ``initialize``.
        self._slab_size = 0
        self._slab_path = ""
        self._slab_handle: Optional[ca.AsyncHandle] = None
        # Per-stream in-flight submissions (keyed by raw ``CUstream``), released
        # once a CUDA event recorded on that stream completes. Guarded by
        # ``_submissions_lock`` (see ``_record_submission``).
        self._submissions_lock = threading.Lock()
        self._submissions: dict[int, _StreamSubmissions] = {}
        # Registry of GDS-registered GPU regions and the streams they run on
        self._registry_lock = threading.Lock()
        self._buffers: list[torch.Tensor] = []
        self._base_ptrs: list[int] = []
        self._nbytes: list[int] = []
        self._registered_streams: set[int] = set()  # maintained for close()

    def initialize(self, config: GdsL1Config) -> None:
        """Create + clear the slab and register it with the GDS library.

        Args:
            config: GDS tier config. ``size_in_bytes`` sizes the preallocated
                slab (rounded up to 4 KiB) at
                ``<file_location>/lmcache_gds_slab.bin`` (one per process);
                ``use_direct_io`` opens it with ``O_DIRECT``.

        Raises:
            Exception: Whatever the GDS library (cuFile/hipFile) raises if GDS
                is unavailable.
        """
        self._slab_size = (config.size_in_bytes + _CUFILE_ALIGNMENT - 1) & ~(
            _CUFILE_ALIGNMENT - 1
        )

        # One shared slab per process (the GDSContext is a process-global
        # singleton used by every GPU instance).
        selected = config.file_location
        os.makedirs(selected, exist_ok=True)
        self._slab_path = os.path.join(selected, _SLAB_FILENAME)

        self._open_and_register_slab(config.use_direct_io)
        self.initialized = True

    # --- Public API ---------------------------------------------------

    def register_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Register a staging buffer (and its stream) with the GDS library.

        Registered as contiguous <=16 MiB regions (the GDS buffer-registration
        cap); :meth:`transfer_async` splits transfers at these boundaries.

        Args:
            buffer: Contiguous CUDA staging buffer, 4 KiB-aligned in size.
        """
        if not self.initialized:
            return
        raw_stream = torch_dev.current_stream().cuda_stream
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        with self._registry_lock:
            if raw_stream not in self._registered_streams:
                ca.register_stream(raw_stream)
                self._registered_streams.add(raw_stream)
            for start in range(0, nbytes, _MAX_CUFILE_REGION):
                self._register_region_locked(
                    buf[start : min(start + _MAX_CUFILE_REGION, nbytes)]
                )

    def deregister_gpu_buffer(self, buffer: torch.Tensor) -> None:
        """Reverse of :meth:`register_gpu_buffer`: deregister its regions + stream.

        Args:
            buffer: The buffer passed to :meth:`register_gpu_buffer`.
        """
        if not self.initialized:
            return
        stream = torch_dev.current_stream()
        raw_stream = stream.cuda_stream
        # No in-flight DMA on this stream may still reference the buffer.
        stream.synchronize()
        buf = buffer.view(torch.uint8)
        nbytes = buf.numel()
        with self._registry_lock:
            for start in range(0, nbytes, _MAX_CUFILE_REGION):
                self._deregister_region_locked(
                    buf[start : min(start + _MAX_CUFILE_REGION, nbytes)]
                )
            if raw_stream in self._registered_streams:
                try:
                    ca.deregister_stream(raw_stream)
                except Exception as e:
                    logger.warning(
                        "GDSContext.deregister_gpu_buffer: deregister_stream: %s", e
                    )
                self._registered_streams.discard(raw_stream)
        # Stream is synced above, so its submissions' DMAs are done -- drop them.
        with self._submissions_lock:
            self._submissions.pop(raw_stream, None)

    def transfer_async(
        self,
        memory_obj: GDSMemoryObject,
        gpu_buffer: torch.Tensor,
        direction: SlabDirection,
    ) -> None:
        """DMA a chunk between ``gpu_buffer`` and its slab region.

        ``READ`` pulls slab -> ``gpu_buffer``; ``WRITE`` pushes the reverse.
        Split at registered-region boundaries (each GDS DMA must stay within
        one <=16 MiB region), so any chunk size works. Stream-ordered, no sync.

        Args:
            memory_obj: The chunk; ``slab_offset`` / ``get_size()`` give the
                file offset and length.
            gpu_buffer: A slice of a registered staging buffer; its first
                ``get_size()`` bytes are transferred.
            direction: :attr:`SlabDirection.READ` or ``.WRITE``.
        """
        slab_op = (
            self._slab_read if direction is SlabDirection.READ else self._slab_write
        )
        nbytes = memory_obj.get_size()
        buf = gpu_buffer.view(torch.uint8)
        pos = 0
        while pos < nbytes:
            base_ptr, dev_offset, region_nbytes = self._resolve_buffer(buf[pos:])
            seg_len = min(nbytes - pos, region_nbytes - dev_offset)
            slab_op(memory_obj.slab_offset + pos, seg_len, dev_offset, base_ptr)
            pos += seg_len

    def close(self) -> None:
        """Sync the stream, deregister GDS state, and close the slab handle."""
        if self._buffers:
            torch_dev.synchronize(device=self._buffers[0].device)
        with self._submissions_lock:
            self._submissions.clear()
        # Deregister any regions/streams still live (per-instance teardown via
        # ``deregister_gpu_buffer`` normally clears these first; this is the
        # shutdown sweep for anything left).
        with self._registry_lock:
            for buf in self._buffers:
                try:
                    ca.deregister_buffer(buf)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_buffer: %s", e)
            self._buffers.clear()
            self._base_ptrs.clear()
            self._nbytes.clear()
            for raw_stream in list(self._registered_streams):
                try:
                    ca.deregister_stream(raw_stream)
                except Exception as e:
                    logger.warning("GDSContext.close: deregister_stream: %s", e)
            self._registered_streams.clear()
        if self._slab_handle is not None:
            try:
                self._slab_handle.close()
            except Exception as e:
                logger.warning("GDSContext.close: slab handle close failed: %s", e)
            self._slab_handle = None

    # --- Internal -----------------------------------------------------

    def _open_and_register_slab(self, use_direct_io: bool) -> None:
        """Create, truncate, preallocate the slab file and register it with GDS.

        Args:
            use_direct_io: Open with ``O_DIRECT`` (required for the GDS fast path).
        """
        # Create, truncate, and fallocate via a regular (non-O_DIRECT) fd.
        creator_fd = os.open(
            self._slab_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644
        )
        try:
            os.posix_fallocate(creator_fd, 0, self._slab_size)
        finally:
            os.close(creator_fd)
        flags = os.O_RDWR
        if use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(self._slab_path, flags)
        try:
            handle = ca.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        self._slab_handle = ca.AsyncHandle.from_fd(
            fd, handle, self._slab_path, writable=True
        )
        logger.info(
            "GDSContext: slab created at %s (%.1f GiB, O_DIRECT=%s), GDS "
            "handle registered",
            self._slab_path,
            self._slab_size / (1 << 30),
            use_direct_io,
        )

    def _register_region_locked(self, buffer: torch.Tensor) -> None:
        """GDS-register one <=16 MiB region (caller holds the lock)."""
        nbytes = buffer.numel() * buffer.element_size()
        base = buffer.data_ptr()
        ca.register_buffer(buffer)
        idx = bisect.bisect_left(self._base_ptrs, base)
        self._buffers.insert(idx, buffer)
        self._base_ptrs.insert(idx, base)
        self._nbytes.insert(idx, nbytes)
        logger.info(
            "GDSContext: registered %d bytes at 0x%x via GDS (total registrations: %d)",
            nbytes,
            base,
            len(self._buffers),
        )

    def _deregister_region_locked(self, buffer: torch.Tensor) -> None:
        """Deregister one region with the GDS library (caller holds the lock).

        Args:
            buffer: A staging-buffer slot previously registered.
        """
        base = buffer.data_ptr()
        idx = bisect.bisect_left(self._base_ptrs, base)
        try:
            ca.deregister_buffer(self._buffers[idx])
        except Exception as e:
            logger.warning("GDSContext: deregister_buffer: %s", e)
        del self._buffers[idx]
        del self._base_ptrs[idx]
        del self._nbytes[idx]

    def _resolve_buffer(self, gpu_buffer: torch.Tensor) -> tuple[int, int, int]:
        """Locate the registered region ``gpu_buffer`` starts in.

        Returns ``(base_ptr, dev_offset, region_nbytes)``; ``region_nbytes -
        dev_offset`` is the room left in the region, which :meth:`transfer_async`
        uses to cut DMAs at region boundaries. Callers always pass a pointer
        inside a registered region.
        """
        ptr = gpu_buffer.data_ptr()
        # Held briefly so a concurrent deregister can't mutate the parallel
        # lists mid-lookup.
        with self._registry_lock:
            idx = bisect.bisect_right(self._base_ptrs, ptr) - 1
            base = self._base_ptrs[idx]
            nbytes = self._nbytes[idx]
        offset = ptr - base
        return base, offset, nbytes

    def _slab_read(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one async GDS read against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_read: slab handle not open")
        stream_handle = torch_dev.current_stream().cuda_stream
        sub = self._slab_handle.read_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )
        self._record_submission(sub)

    def _slab_write(
        self, slab_offset: int, size: int, dev_offset: int, buf_base: int
    ) -> None:
        """Submit one async GDS write against the slab handle (stream-ordered)."""
        if self._slab_handle is None:
            raise RuntimeError("GDSContext._slab_write: slab handle not open")
        stream_handle = torch_dev.current_stream().cuda_stream
        sub = self._slab_handle.write_async(
            buf_base, size, slab_offset, dev_offset, stream_handle
        )
        self._record_submission(sub)

    def _record_submission(self, sub: "ca.Submission") -> None:
        """Track an in-flight submission so its ctypes storage outlives the DMA.

        Accumulated per (current) stream; every ``_SUBMISSION_CHECKPOINT_EVERY``
        ops a CUDA event is recorded and completed batches are released.
        """
        stream = torch_dev.current_stream()
        raw_stream = stream.cuda_stream
        with self._submissions_lock:
            st = self._submissions.get(raw_stream)
            if st is None:
                st = self._submissions[raw_stream] = _StreamSubmissions()
            st.uncommitted.append(sub)
            st.ops_since_checkpoint += 1
            if st.ops_since_checkpoint >= _SUBMISSION_CHECKPOINT_EVERY:
                self._checkpoint_submissions_locked(st, stream)

    def _checkpoint_submissions_locked(
        self, st: _StreamSubmissions, stream: "torch.Stream"
    ) -> None:
        """Close ``st``'s current batch behind a CUDA event on ``stream`` and
        drop earlier batches whose event has completed. Hold
        ``self._submissions_lock``.
        """
        if st.uncommitted:
            event = torch_dev.Event()
            event.record(stream)
            st.inflight.append((event, st.uncommitted))
            st.uncommitted = []
        st.ops_since_checkpoint = 0
        st.inflight = [
            (event, subs) for (event, subs) in st.inflight if not event.query()
        ]


@functools.cache
def get_gds_context() -> GDSContext:
    """Return the process-global :class:`GDSContext` singleton (created empty on
    first access). Consult :attr:`GDSContext.initialized` to tell whether GDS L1
    is active."""
    return GDSContext()


def initialize_gds_context(config: Optional[GdsL1Config]) -> GDSContext:
    """Set up the process-global :class:`GDSContext` (once, at startup).

    ``config=None`` leaves it uninitialized (GDS L1 disabled); otherwise the
    slab is created and registered. Returns the singleton.
    """
    context = get_gds_context()
    if config is not None:
        context.initialize(config)
    return context
