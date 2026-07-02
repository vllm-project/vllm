# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-process shared mmap for the routed-experts slot buffer.

The routing slot buffer is written by the GPU worker (the ``output_rank``
worker, which alone returns ``ModelRunnerOutput``) and read by the scheduler
(EngineCore) to return routing per request. Both live on the same node for the
supported parallelism (TP/DP/EP; PP=1, CP=1), so they share one ``MAP_SHARED``
file in ``/dev/shm`` keyed by ``instance_id`` (and DP rank).

The creator/attacher handshake mirrors
``vllm.v1.kv_offload.cpu.shared_offload_region.SharedOffloadRegion`` — the
proven pattern KV offload already uses for scheduler<->worker sharing — but
this region is a single flat plane (routing is replicated across ranks and only
one rank writes), not the per-worker strided layout KV needs. We therefore
extract just the ~30-line handshake here rather than reuse that class.
"""

import contextlib
import mmap
import os
import time

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger

logger = init_logger(__name__)

_WAIT_TIMEOUT_S = 30.0


def _wait_for_file_size(fd: int, expected_size: int, timeout: float) -> None:
    """Spin-wait until the file reaches ``expected_size`` (creator truncated)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for shared routing mmap to reach "
                f"{expected_size} bytes"
            )
        time.sleep(0.005)


class SharedRoutingRegion:
    """Flat ``MAP_SHARED`` ndarray shared across scheduler + worker on a node.

    Create-or-attach with no ordering dependency: whichever process runs first
    ``O_EXCL``-creates the file and ``ftruncate``s it (becomes the unlinking
    owner); the others get ``FileExistsError``, open it, and wait for the size.
    This matters because (in uniproc) the worker's ``initialize_from_config``
    runs INSIDE ``EngineCore.__init__`` BEFORE the scheduler builds its manager,
    so neither side can be assumed to create first. Both ``mmap(MAP_SHARED)``
    the whole file as one flat ``ndarray`` (no per-rank stride). Only the
    creating process unlinks on close.
    """

    def __init__(
        self,
        path: str,
        shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> None:
        self.path = path
        self._dtype = np.dtype(dtype)
        self._nbytes = int(np.prod(shape)) * self._dtype.itemsize
        self._creator = False
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None

        try:
            # First arrival creates + sizes the file.
            self.fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(self.fd, self._nbytes)
            self._creator = True
            logger.info("Created routing mmap %s (%.2f GB)", path, self._nbytes / 1e9)
        except FileExistsError:
            # Someone created it first; open and wait until it is fully sized.
            self.fd = os.open(path, os.O_RDWR)
            _wait_for_file_size(self.fd, self._nbytes, _WAIT_TIMEOUT_S)
            logger.info("Opened routing mmap %s", path)

        self.mmap_obj = mmap.mmap(
            self.fd,
            self._nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.array: np.ndarray = np.frombuffer(self.mmap_obj, dtype=dtype).reshape(
            shape
        )

    def close(self) -> None:
        """Release the mapping; the creator also unlinks the file. Idempotent."""
        self.array = None  # type: ignore[assignment]
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close routing mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close routing fd", exc_info=True)
            self.fd = None
        if self._creator:
            try:
                os.unlink(self.path)
                logger.info("Removed routing mmap %s", self.path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.warning(
                    "Failed to unlink routing mmap %s", self.path, exc_info=True
                )
            self._creator = False


def shared_routing_mmap_path(instance_id: str, dp_rank: int) -> str:
    """Stable per-instance, per-DP-rank path so DP ranks never collide."""
    return f"/dev/shm/vllm_routed_experts_{instance_id}_dp{dp_rank}.mmap"


class RoutedExpertsWorkerWriter:
    """Worker-side writer that scatters this step's routing into the shared
    slot buffer the scheduler reads.

    **Attach-only, lazy.** The scheduler's ``RoutedExpertsManager`` is the sole
    creator (and unlinker) of the ``/dev/shm`` file; this writer never creates
    it. Attaching is deferred to the first ``scatter`` because the worker's
    ``initialize_from_config`` runs (in uniproc) inside ``EngineCore.__init__``
    BEFORE the scheduler builds its manager — so the file does not exist yet at
    worker init, but always exists by the first forward step. This keeps
    ownership unambiguous (manager creates+unlinks, worker only maps) and avoids
    a /dev/shm leak from the worker winning a create race.

    Only the ``output_rank`` worker constructs one (routing is replicated across
    ranks; a single writer avoids redundant writes). The scatter mirrors
    ``RoutedExpertsManager.store_batch`` exactly — ``slot_buffer[slot_mapping]
    = routing`` — so int32 capture ids narrow into the uint8/uint16 slot dtype
    the same way, and the flat slot layout is identical.

    The scatter MUST be called only after the routing D2H into the pinned
    staging buffer has completed (the CUDA event sync); it is a plain CPU
    fancy-index assignment, not a CUDA op, so it cannot be ordered by a stream.
    """

    def __init__(
        self,
        instance_id: str,
        dp_rank: int,
        slot_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        block_size: int,
    ) -> None:
        self._path = shared_routing_mmap_path(instance_id, dp_rank)
        self._slot_shape = slot_shape
        self._dtype = np.dtype(dtype)
        self._block_size = block_size
        self._nbytes = int(np.prod(slot_shape)) * self._dtype.itemsize
        self._fd: int | None = None
        self._mmap_obj: mmap.mmap | None = None
        self._array: np.ndarray | None = None

    def _ensure_attached(self, create: bool = False) -> None:
        if self._array is not None:
            return
        # Attach-only by default: the manager (scheduler-side
        # RoutedExpertsManager) creates the file. The worker may reach here
        # before the manager has created it (its register_kv_caches can run
        # during EngineCore init, before the scheduler builds the manager), so
        # wait for the path to APPEAR, then for it to reach full size.
        #
        # ``create=True`` (NIXL sidecar): the worker MUST have the mapping live
        # at register_kv_caches time (to RDMA-register the region before the
        # handshake), which runs BEFORE the scheduler builds its manager. So the
        # worker creates+sizes the file itself; whichever side loses the
        # O_CREAT|O_EXCL race just opens the existing file. The worker never
        # unlinks — the manager owns the lifecycle.
        if create:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.ftruncate(fd, self._nbytes)
            except FileExistsError:
                fd = os.open(self._path, os.O_RDWR)
                _wait_for_file_size(fd, self._nbytes, _WAIT_TIMEOUT_S)
        else:
            deadline = time.monotonic() + _WAIT_TIMEOUT_S
            while True:
                try:
                    fd = os.open(self._path, os.O_RDWR)
                    break
                except FileNotFoundError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Timed out waiting for shared routing mmap to appear "
                            f"at {self._path}"
                        ) from None
                    time.sleep(0.005)
            _wait_for_file_size(fd, self._nbytes, _WAIT_TIMEOUT_S)
        self._fd = fd
        self._mmap_obj = mmap.mmap(
            fd,
            self._nbytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self._array = np.frombuffer(self._mmap_obj, dtype=self._dtype).reshape(
            self._slot_shape
        )

    def scatter(self, routing_data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Scatter one step's routing into the shared slot buffer.

        ``routing_data[i]`` (shape ``(layers, top_k)``) is written to slot
        ``slot_mapping[i]``. Mirrors ``store_batch``; numpy narrows the int32
        capture ids into the slot dtype on assignment.

        Fast path: when the slots are one ascending contiguous run (common for
        a single request's prefill chunk, whose freshly-allocated blocks give
        consecutive slots), a slice assign replaces the random scatter — a
        sequential memcpy with prefetch-friendly page faults instead of
        per-element bounded random writes into the demand-paged mmap. A cheap
        O(1) span check gates the O(n) confirmation, so the non-contiguous
        (decode / multi-request) case pays almost nothing before falling back.
        """
        self._ensure_attached()
        assert self._array is not None
        n = len(slot_mapping)
        if (
            n
            and int(slot_mapping[-1]) - int(slot_mapping[0]) == n - 1
            and bool(np.all(np.diff(slot_mapping) == 1))
        ):
            s0 = int(slot_mapping[0])
            self._array[s0 : s0 + n] = routing_data
        else:
            self._array[slot_mapping] = routing_data

    def attach(self, create: bool = False) -> None:
        """Force the lazy attach now (e.g. so the region can be RDMA-registered).

        Idempotent; normally ``scatter`` attaches on first use, but the Mooncake
        bridge / NIXL sidecar need the mapping live at ``register_kv_caches``
        time. ``create=True`` lets the worker create+size the file itself when
        it may run before the scheduler's manager (NIXL handshake needs the
        region registered up front); the worker never unlinks.
        """
        self._ensure_attached(create=create)

    def region_base_address(self) -> int:
        """Address of the flat shared slot buffer (for Mooncake register/IO).

        The buffer is one contiguous ``MAP_SHARED`` plane, so block ``b``'s
        routing row lives at ``base + b * block_row_nbytes()`` — the same
        block-offset addressing the Mooncake KV path uses.
        """
        self._ensure_attached()
        assert self._array is not None
        return self._array.ctypes.data

    def region_nbytes(self) -> int:
        """Total byte length of the shared slot buffer."""
        return self._nbytes

    def block_row_nbytes(self) -> int:
        """Bytes of one block's routing row (``block_size * layers * top_k``).

        ``slot_shape`` is ``(num_blocks * block_size, layers, top_k)``; one
        block packs ``block_size`` contiguous slot rows.
        """
        per_slot = int(np.prod(self._slot_shape[1:])) * self._dtype.itemsize
        return per_slot * self._block_size

    def close(self) -> None:
        """Release the mapping (never unlinks — the manager owns the file)."""
        self._array = None
        if self._mmap_obj is not None:
            with contextlib.suppress(Exception):
                self._mmap_obj.close()
            self._mmap_obj = None
        if self._fd is not None:
            with contextlib.suppress(Exception):
                os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
