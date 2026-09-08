# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public RDNA4 all-reduce communicator and measured transport router.

The communicator owns HIP resource setup and selects between mapped-memory and
direct-P2P transports. GPU implementations stay in ``flydsl_kernels``: the
``tp2`` module owns the two-rank family, while ``mapped`` and ``p2p`` own the
TP4/TP8 transport families.
"""

import importlib
import os
from contextlib import contextmanager, suppress
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

DEFAULT_MAX_SIZE = 128 * 1024 * 1024
HIP_IPC_MIN_ALLOC_SIZE = 64 * 1024
TP2_MAPPED_BF16_MAX_SIZE = 64 * 1024
MAPPED_ALL_REDUCE_MAX_SIZE = {4: 192 * 1024 + 16, 8: 384 * 1024}
P2P_GRAPH_MIN_SIZE = {
    2: 64 * 1024 + 16,
    4: 1024 * 1024,
    8: 384 * 1024 + 16,
}
TP2_P2P_GRAPH_MAX_SIZE = 128 * 1024 * 1024
TP2_P2P_SMALL_MAX_SIZE = 1024 * 1024
TP2_P2P_EAGER_MIN_SIZE = 1024 * 1024
TP4_EAGER_PYNCCL_SIZES = frozenset({32 * 1024})
TP4_P2P_PULL_GRAPH_RANGE = (1024 * 1024, 128 * 1024 * 1024)
TP4_P2P_TWO_SHOT_MIN_SIZE = 1024 * 1024
TP4_P2P_TWO_SHOT_LARGE_CHUNK_SIZE = 16 * 1024 * 1024
TP4_P2P_TWO_SHOT_SMALL_CHUNK_SIZE = 64 * 1024 * 1024
TP4_P2P_FOUR_BLOCK_MIN_SIZE = 1024 * 1024
TP4_P2P_TWO_BLOCK_MIN_SIZE = 48 * 1024 * 1024
TP4_P2P_PULL_FULL_WGP_MIN_SIZE = 32 * 1024 * 1024 - 64
TP8_P2P_REDUCED_LOCAL_COPY_THREADS_MIN_SIZE = 34 * 1024 * 1024
TP8_PYNCCL_GRAPH_RANGE = (960 * 1024, 1408 * 1024)
MAX_GRAPH_ALL_REDUCE_CALLS = 65536
_MAX_P2P_BLOCKS = 80


class _Route(Enum):
    """Measured implementation route for one all-reduce invocation."""

    TP2_MAPPED = "tp2_mapped"
    MAPPED = "mapped"
    P2P = "p2p"


def _is_rdna4_flydsl_available() -> bool:
    """Return whether FlyDSL exposes the stable collective-kernel primitives."""
    try:
        fx = importlib.import_module("flydsl.expr")
        rocdl = fx.rocdl
        return all(
            hasattr(rocdl, name)
            for name in (
                "MemoryOrder",
                "SyncScope",
                "global_load",
                "global_store",
                "sleep",
            )
        )
    except Exception:
        logger.debug("FlyDSL collective primitive probe failed", exc_info=True)
        return False


class RDNA4AllReduce:
    """RDNA4 all-reduce using vLLM-owned FlyDSL kernels."""

    _HIP_IPC_HANDLE_BYTES = 64
    _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
    _HIP_DEVICE_MALLOC_UNCACHED = 0x3
    _HIP_HOST_REGISTER_PORTABLE = 0x1
    _HIP_HOST_REGISTER_MAPPED = 0x2
    _hip = None
    _hipIpcMemHandle_t = None

    # Signal struct layout (each field alignas(128)):
    #   uint32_t start[_MAX_P2P_BLOCKS][8]
    #   uint32_t end[_MAX_P2P_BLOCKS][8]
    #   uint32_t flag[_MAX_P2P_BLOCKS]
    # Struct size padded to 128-byte alignment.
    _SIGNAL_SIZE = ((_MAX_P2P_BLOCKS * 8 * 4) * 2 + _MAX_P2P_BLOCKS * 4 + 127) & ~127

    @classmethod
    def _load_hip(cls):
        if cls._hip is not None:
            return cls._hip
        import ctypes

        sonames = (
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
        )
        candidates = []
        rocm_path = os.environ.get("ROCM_PATH")
        if rocm_path:
            candidates.extend(
                os.path.join(rocm_path, lib_dir, name)
                for lib_dir in ("lib", "lib64")
                for name in sonames
            )
        candidates.extend(sonames)

        for name in candidates:
            try:
                cls._hip = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if cls._hip is None:
            raise RuntimeError("Failed to load HIP runtime library")

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_byte * cls._HIP_IPC_HANDLE_BYTES)]

        cls._hipIpcMemHandle_t = hipIpcMemHandle_t

        cls._hip.hipIpcGetMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(hipIpcMemHandle_t),
            ctypes.c_void_p,
        ]
        cls._hip.hipIpcOpenMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            hipIpcMemHandle_t,
            ctypes.c_uint,
        ]
        cls._hip.hipIpcCloseMemHandle.restype = ctypes.c_int
        cls._hip.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        cls._hip.hipGetErrorString.restype = ctypes.c_char_p
        cls._hip.hipGetErrorString.argtypes = [ctypes.c_int]
        cls._hip.hipExtMallocWithFlags.restype = ctypes.c_int
        cls._hip.hipExtMallocWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls._hip.hipFree.restype = ctypes.c_int
        cls._hip.hipFree.argtypes = [ctypes.c_void_p]
        cls._hip.hipMemset.restype = ctypes.c_int
        cls._hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cls._hip.hipHostRegister.restype = ctypes.c_int
        cls._hip.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cls._hip.hipHostGetDevicePointer.restype = ctypes.c_int
        cls._hip.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        cls._hip.hipHostUnregister.restype = ctypes.c_int
        cls._hip.hipHostUnregister.argtypes = [ctypes.c_void_p]
        return cls._hip

    @classmethod
    def _hip_check(cls, err: int, *, what: str):
        if int(err) == 0:
            return
        hip = cls._load_hip()
        try:
            s = hip.hipGetErrorString(int(err))
            msg = s.decode("utf-8", errors="replace") if s else f"hipError({err})"
        except Exception:
            msg = f"hipError({err})"
        raise RuntimeError(f"{what} failed: {msg}")

    @classmethod
    def _get_mem_handle_bytes(cls, base_ptr: int) -> bytes:
        import ctypes

        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()
        err = hip.hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(h), cls._HIP_IPC_HANDLE_BYTES))

    @classmethod
    def _open_mem_handle(cls, handle_bytes: bytes) -> int:
        import ctypes

        if len(handle_bytes) != cls._HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"Expected {cls._HIP_IPC_HANDLE_BYTES}B handle")
        hip = cls._load_hip()
        h = cls._hipIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(h), bytes(handle_bytes), cls._HIP_IPC_HANDLE_BYTES)
        out_ptr = ctypes.c_void_p()
        err = hip.hipIpcOpenMemHandle(
            ctypes.byref(out_ptr),
            h,
            ctypes.c_uint(int(cls._HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)),
        )
        cls._hip_check(err, what="hipIpcOpenMemHandle")
        return int(out_ptr.value)

    @classmethod
    def _close_mem_handle(cls, base_ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipIpcCloseMemHandle(ctypes.c_void_p(int(base_ptr)))
        cls._hip_check(err, what="hipIpcCloseMemHandle")

    @classmethod
    def _alloc_uncached(cls, size: int) -> int:
        """Allocate zero-initialised uncached device memory (hipDeviceMallocUncached).

        Returns the raw device pointer as int.
        """
        import ctypes

        hip = cls._load_hip()
        buf = ctypes.c_void_p()
        err = hip.hipExtMallocWithFlags(
            ctypes.byref(buf),
            ctypes.c_size_t(size),
            ctypes.c_uint(cls._HIP_DEVICE_MALLOC_UNCACHED),
        )
        cls._hip_check(err, what="hipExtMallocWithFlags")
        err = hip.hipMemset(buf, 0, ctypes.c_size_t(size))
        cls._hip_check(err, what="hipMemset")
        return int(buf.value)

    @classmethod
    def _free_device_mem(cls, ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipFree(ctypes.c_void_p(ptr))
        cls._hip_check(err, what="hipFree")

    @classmethod
    def _register_host_mapping(cls, host_ptr: int, size: int) -> int:
        import ctypes

        hip = cls._load_hip()
        flags = cls._HIP_HOST_REGISTER_PORTABLE | cls._HIP_HOST_REGISTER_MAPPED
        err = hip.hipHostRegister(
            ctypes.c_void_p(host_ptr), ctypes.c_size_t(size), ctypes.c_uint(flags)
        )
        cls._hip_check(err, what="hipHostRegister")
        device_ptr = ctypes.c_void_p()
        err = hip.hipHostGetDevicePointer(
            ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
        )
        cls._hip_check(err, what="hipHostGetDevicePointer")
        return int(device_ptr.value)

    @classmethod
    def _unregister_host_mapping(cls, host_ptr: int) -> None:
        import ctypes

        hip = cls._load_hip()
        err = hip.hipHostUnregister(ctypes.c_void_p(host_ptr))
        cls._hip_check(err, what="hipHostUnregister")

    @staticmethod
    def _gather_object_list_via_broadcast(group, shard_data):
        import torch.distributed as dist

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        all_data = [[None] for _ in range(world_size)]
        all_data[rank][0] = shard_data
        ranks = sorted(dist.get_process_group_ranks(group=group))
        for i, r in enumerate(ranks):
            dist.broadcast_object_list(all_data[i], src=r, group=group, device="cpu")
        return [all_data[i][0] for i in range(world_size)]

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        self.disabled = True
        self._meta_ptr = None
        self._meta_bases: list[int | None] = []
        self._tp2_mapped = None
        self._mapped = None
        self._p2p_launchers = {}
        self._p2p_blocks_override = 0
        self._p2p_copy_load_nontemporal = None
        self._p2p_tp8_local_copy_threads = 0
        self._p2p_ready = False
        self._IS_CAPTURING = False
        self._captured_outputs: list[torch.Tensor] = []
        self._captured_inputs: list[torch.Tensor] = []
        self._graph_call_cursor = 0
        self._capture_base_index = 0
        self._gpu_graph_output_ptrs_array = None
        self._gpu_graph_input_ptrs_array = None
        self._graph_output_bases: list[int] = []
        self._graph_input_bases: list[int] = []
        self._eager_input_bases: list[int] = []
        self._eager_input_ptrs: dict[int, torch.Tensor] = {}
        self._eager_output_bases: list[int] = []
        self._eager_output_ptrs: dict[int, torch.Tensor] = {}
        self.fully_connected = False
        self.group = group
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.max_size = int(max_size)

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        if dist.get_backend(group) == dist.Backend.NCCL:
            raise ValueError("RDNA4AllReduce requires a CPU process group")

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size not in {2, 4, 8}:
            logger.info_once(
                "RDNA4AllReduce is disabled for world size %d", self.world_size
            )
            return
        if not current_platform.is_rocm():
            return
        from vllm.platforms.rocm import on_rdna4

        if not on_rdna4():
            logger.info_once("RDNA4AllReduce requires gfx1200 or gfx1201")
            return
        if not _is_rdna4_flydsl_available():
            logger.info_once(
                "RDNA4AllReduce requires FlyDSL with stable collective primitives"
            )
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.info_once("RDNA4AllReduce requires a single-node process group")
            return
        if self.max_size <= 0:
            raise ValueError(f"max_size must be positive, got {self.max_size}")

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_index}")

        props = torch.cuda.get_device_properties(self.device)
        if self.world_size == 2 and "gfx1201" in getattr(props, "gcnArchName", ""):
            from .rdna4_all_reduce_mapped_tp2 import RDNA4TP2MappedAllReduce

            self._tp2_mapped = RDNA4TP2MappedAllReduce(
                group=self.group,
                device=self.device,
                max_numel=min(
                    self.max_size,
                    TP2_MAPPED_BF16_MAX_SIZE,
                )
                // torch.bfloat16.itemsize,
                register_host_mapping=self._register_host_mapping,
                unregister_host_mapping=self._unregister_host_mapping,
            )
            if self._tp2_mapped.disabled:
                self._tp2_mapped = None
        elif self.world_size in (4, 8) and "gfx1201" in getattr(
            props, "gcnArchName", ""
        ):
            from .rdna4_all_reduce_mapped import RDNA4MappedAllReduce

            mapped_max_size = min(
                self.max_size,
                MAPPED_ALL_REDUCE_MAX_SIZE[self.world_size],
            )
            self._mapped = RDNA4MappedAllReduce(
                group=self.group,
                device=self.device,
                max_size=mapped_max_size,
                register_host_mapping=self._register_host_mapping,
                unregister_host_mapping=self._unregister_host_mapping,
                threads=1024 if self.world_size == 4 else 512,
            )
            if self._mapped.disabled:
                self._mapped = None

        group_devices = [None] * self.world_size
        dist.all_gather_object(group_devices, device_index, group=group)
        local_p2p = all(
            peer == device_index
            or torch.cuda.can_device_access_peer(device_index, int(peer))
            for peer in group_devices
        )
        p2p_status = [None] * self.world_size
        dist.all_gather_object(p2p_status, local_p2p, group=group)
        self.fully_connected = all(bool(value) for value in p2p_status)
        if not self.fully_connected:
            if self._tp2_mapped is not None or self._mapped is not None:
                self.disabled = False
                return
            logger.warning_once("RDNA4AllReduce requires full P2P access for TP4/TP8")
            return

        # Pre-initialize resource attributes so close() is safe on partial init failure.
        self._meta_bases = [None] * self.world_size

        # gfx1201 requires an uncached allocation exported through HIP IPC to
        # end on a 64 KiB boundary.
        required_size = self._SIGNAL_SIZE + 2 * self.max_size
        alloc_size = (
            (required_size + HIP_IPC_MIN_ALLOC_SIZE - 1)
            // HIP_IPC_MIN_ALLOC_SIZE
            * HIP_IPC_MIN_ALLOC_SIZE
        )
        self._meta_ptr = self._alloc_uncached(alloc_size)

        my_meta_bytes = self._get_mem_handle_bytes(self._meta_ptr)
        all_meta = self._gather_object_list_via_broadcast(
            self.group, (my_meta_bytes, 0)
        )

        self._meta_bases = [None] * self.world_size
        self._sg_ptrs = [0] * 8
        self._input_buffer_ptrs = [0] * 8
        self._tmp_ptrs = [0] * 8
        for r in range(self.world_size):
            hb, off = all_meta[r]
            base_ptr = (
                self._meta_ptr
                if r == self.rank
                else int(self._open_mem_handle(bytes(hb)))
            )
            if r != self.rank:
                self._meta_bases[r] = base_ptr
            sg_ptr = base_ptr + off
            input_ptr = sg_ptr + self._SIGNAL_SIZE
            tmp_ptr = input_ptr + self.max_size
            if r < 8:
                self._sg_ptrs[r] = sg_ptr
                self._input_buffer_ptrs[r] = input_ptr
                self._tmp_ptrs[r] = tmp_ptr
        for i in range(self.world_size, 8):
            self._sg_ptrs[i] = self._sg_ptrs[0]
            self._input_buffer_ptrs[i] = self._input_buffer_ptrs[0]
            self._tmp_ptrs[i] = self._tmp_ptrs[0]
        self._self_sg = self._sg_ptrs[self.rank]
        self._gpu_sg_ptrs_array = torch.tensor(
            self._sg_ptrs[:8], dtype=torch.int64, device=self.device
        )

        ws, rk = self.world_size, self.rank
        rotated_input_buf_ptrs = [
            self._input_buffer_ptrs[(rk + i) % ws] for i in range(8)
        ]
        self._gpu_input_buffer_ptrs_array = torch.tensor(
            rotated_input_buf_ptrs, dtype=torch.int64, device=self.device
        )

        rotated_tmp_ptrs = [self._tmp_ptrs[(rk + i) % ws] for i in range(8)]
        self._gpu_tmp_ptrs_array = torch.tensor(
            rotated_tmp_ptrs, dtype=torch.int64, device=self.device
        )
        self._gpu_graph_output_ptrs_array = torch.empty(
            (MAX_GRAPH_ALL_REDUCE_CALLS, 8),
            dtype=torch.int64,
            device=self.device,
        )
        self._gpu_graph_input_ptrs_array = torch.empty(
            (MAX_GRAPH_ALL_REDUCE_CALLS, 8),
            dtype=torch.int64,
            device=self.device,
        )

        self._threads = 1024 if self.world_size == 8 else 512
        self._p2p_ready = True

        self.disabled = False

    def close(self):
        """Release IPC memory handles for peer GPU buffers."""
        if self._tp2_mapped is not None:
            with suppress(Exception):
                self._tp2_mapped.close()
            self._tp2_mapped = None
        if self._mapped is not None:
            with suppress(Exception):
                self._mapped.close()
            self._mapped = None
        for b in getattr(self, "_meta_bases", []):
            if b is not None:
                with suppress(Exception):
                    self._close_mem_handle(int(b))
        self._meta_bases = []
        for b in getattr(self, "_graph_output_bases", []):
            with suppress(Exception):
                self._close_mem_handle(int(b))
        self._graph_output_bases = []
        for b in getattr(self, "_graph_input_bases", []):
            with suppress(Exception):
                self._close_mem_handle(int(b))
        self._graph_input_bases = []
        for b in getattr(self, "_eager_input_bases", []):
            with suppress(Exception):
                self._close_mem_handle(int(b))
        self._eager_input_bases = []
        self._eager_input_ptrs = {}
        for b in getattr(self, "_eager_output_bases", []):
            with suppress(Exception):
                self._close_mem_handle(int(b))
        self._eager_output_bases = []
        self._eager_output_ptrs = {}
        if getattr(self, "_meta_ptr", None):
            with suppress(Exception):
                self._free_device_mem(self._meta_ptr)
            self._meta_ptr = None
        self._p2p_ready = False
        self.disabled = True

    @contextmanager
    def capture(self):
        """Record and IPC-register graph inputs and outputs after capture."""
        try:
            self._IS_CAPTURING = True
            self._captured_outputs = []
            self._captured_inputs = []
            self._capture_base_index = self._graph_call_cursor
            yield
        finally:
            self._IS_CAPTURING = False
            if self._captured_outputs:
                self._register_graph_tensors()
                self._graph_call_cursor += len(self._captured_outputs)

    def _register_graph_tensors(self) -> None:
        """Populate direct peer input/output pointers recorded during capture."""
        outputs = self._captured_outputs
        inputs = self._captured_inputs
        if len(inputs) != len(outputs):
            raise RuntimeError("RDNA4 graph input/output call counts differ")
        end_index = self._capture_base_index + len(outputs)
        if end_index > MAX_GRAPH_ALL_REDUCE_CALLS:
            raise RuntimeError(
                "RDNA4 graph captured too many all-reduces: "
                f"{end_index} > {MAX_GRAPH_ALL_REDUCE_CALLS}"
            )

        segments = []
        for segment in torch.cuda.memory_snapshot():
            address = int(segment.get("address", 0))
            size = int(segment.get("total_size", 0))
            if address and size:
                segments.append((address, address + size))

        handle_cache: dict[int, bytes] = {}
        local_metadata = []
        for input_, output in zip(inputs, outputs):
            call_metadata = []
            for tensor in (input_, output):
                pointer = int(tensor.data_ptr())
                matches = [start for start, end in segments if start <= pointer < end]
                if not matches:
                    raise RuntimeError(
                        f"Could not locate graph tensor 0x{pointer:x} in ROCm allocator"
                    )
                base = max(matches)
                handle = handle_cache.get(base)
                if handle is None:
                    handle = self._get_mem_handle_bytes(base)
                    handle_cache[base] = handle
                call_metadata.append((handle, pointer - base))
            local_metadata.append(call_metadata)

        all_metadata = self._gather_object_list_via_broadcast(
            self.group,
            local_metadata,
        )
        if any(len(peer) != len(outputs) for peer in all_metadata):
            raise RuntimeError("RDNA4 graph all-reduce call counts differ across ranks")

        opened: dict[tuple[int, bytes], int] = {}
        input_pointer_rows = []
        output_pointer_rows = []
        for call_index, output in enumerate(outputs):
            rows = ([], [])
            local_tensors = (inputs[call_index], output)
            for tensor_kind, row in enumerate(rows):
                for peer_offset in range(self.world_size):
                    peer = (self.rank + peer_offset) % self.world_size
                    handle, offset = all_metadata[peer][call_index][tensor_kind]
                    handle = bytes(handle)
                    if peer == self.rank:
                        pointer = int(local_tensors[tensor_kind].data_ptr())
                    else:
                        key = (peer, handle)
                        base = opened.get(key)
                        if base is None:
                            base = self._open_mem_handle(handle)
                            opened[key] = base
                            if tensor_kind == 0:
                                self._graph_input_bases.append(base)
                            else:
                                self._graph_output_bases.append(base)
                        pointer = base + int(offset)
                    row.append(pointer)
                row.extend([row[0]] * (8 - self.world_size))
            input_pointer_rows.append(rows[0])
            output_pointer_rows.append(rows[1])

        input_pointer_tensor = torch.tensor(
            input_pointer_rows, dtype=torch.int64, device=self.device
        )
        output_pointer_tensor = torch.tensor(
            output_pointer_rows, dtype=torch.int64, device=self.device
        )
        assert self._gpu_graph_input_ptrs_array is not None
        assert self._gpu_graph_output_ptrs_array is not None
        self._gpu_graph_input_ptrs_array[self._capture_base_index : end_index].copy_(
            input_pointer_tensor
        )
        self._gpu_graph_output_ptrs_array[self._capture_base_index : end_index].copy_(
            output_pointer_tensor
        )
        torch.cuda.synchronize(self.device)

    def _get_eager_tensor_ptrs(
        self,
        tensor: torch.Tensor,
        cache: dict[int, torch.Tensor],
        bases: list[int],
    ) -> int | None:
        """Return a cached rotated IPC pointer row for a steady eager tensor.

        Registration is deliberately bounded: serving reuses a small set of
        activation buffers, while an unbounded data-pointer cache would pin
        allocator segments and peer mappings.
        """
        pointer = int(tensor.data_ptr())
        cached = cache.get(pointer)
        if cached is not None:
            return int(cached.data_ptr())
        if len(cache) >= 32:
            # Reclaim the bounded registration window collectively.  The GPU
            # and CPU barriers make it safe to close mappings that a peer's
            # previously enqueued kernel may have read.
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            for peer_base in bases:
                self._close_mem_handle(int(peer_base))
            bases.clear()
            cache.clear()

        matches = []
        for segment in torch.cuda.memory_snapshot():
            address = int(segment.get("address", 0))
            size = int(segment.get("total_size", 0))
            if address and address <= pointer < address + size:
                matches.append(address)
        if not matches:
            return None
        base = max(matches)
        metadata = (self._get_mem_handle_bytes(base), pointer - base)
        all_metadata = self._gather_object_list_via_broadcast(self.group, metadata)

        row = []
        for peer_offset in range(self.world_size):
            peer = (self.rank + peer_offset) % self.world_size
            handle, offset = all_metadata[peer]
            if peer == self.rank:
                peer_pointer = pointer
            else:
                peer_base = self._open_mem_handle(bytes(handle))
                bases.append(peer_base)
                peer_pointer = peer_base + int(offset)
            row.append(peer_pointer)
        row.extend([row[0]] * (8 - self.world_size))
        pointer_tensor = torch.tensor(row, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        cache[pointer] = pointer_tensor
        return int(pointer_tensor.data_ptr())

    def _get_eager_input_ptrs(self, inp: torch.Tensor) -> int | None:
        return self._get_eager_tensor_ptrs(
            inp, self._eager_input_ptrs, self._eager_input_bases
        )

    def _get_eager_output_ptrs(self, out: torch.Tensor) -> int | None:
        return self._get_eager_tensor_ptrs(
            out, self._eager_output_ptrs, self._eager_output_bases
        )

    def __del__(self):
        with suppress(Exception):
            self.close()

    _SUPPORTED_WORLD_SIZES = {2, 4, 8}

    def _select_route(self, inp: torch.Tensor, *, graph: bool) -> _Route | None:
        """Select the measured transport for an input and execution mode."""
        if self.disabled or self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return None
        if self._tp2_mapped is not None and self._tp2_mapped.should_use(inp):
            # The mapped-host TP2 kernel wins under graph replay, but its
            # launch/synchronization floor loses to PyNCCL in eager mode.
            return _Route.TP2_MAPPED if graph else None
        if self._mapped is not None and self._mapped.should_use(inp):
            if (
                not graph
                and self.world_size == 4
                and inp.nbytes in TP4_EAGER_PYNCCL_SIZES
            ):
                return None
            return _Route.MAPPED
        if self._p2p_route_supported(inp, graph=graph):
            return _Route.P2P
        return None

    def should_use(self, inp: torch.Tensor) -> bool:
        """Return whether the measured route uses an RDNA4 transport."""
        return self._select_route(inp, graph=self._IS_CAPTURING) is not None

    def should_use_graph(self, inp: torch.Tensor) -> bool:
        """Return whether graph capture will use an RDNA4 kernel for ``inp``."""
        return self._select_route(inp, graph=True) is not None

    def _p2p_route_supported(self, inp: torch.Tensor, *, graph: bool) -> bool:
        if not self._p2p_tensor_supported(inp):
            return False
        if graph:
            return True
        if self.world_size == 2:
            return (
                inp.nbytes >= TP2_P2P_EAGER_MIN_SIZE
                and inp.nbytes <= TP2_P2P_SMALL_MAX_SIZE
            ) or inp.nbytes >= 2 * 1024 * 1024
        return self.world_size == 4 and inp.nbytes >= TP4_P2P_FOUR_BLOCK_MIN_SIZE

    def _p2p_tensor_supported(self, inp: torch.Tensor) -> bool:
        if not self._p2p_ready:
            return False
        if inp.device != self.device or inp.dtype != torch.bfloat16:
            return False
        if self.world_size not in P2P_GRAPH_MIN_SIZE:
            return False
        inp_size = inp.nbytes
        if inp_size < P2P_GRAPH_MIN_SIZE[self.world_size]:
            return False
        if self.world_size == 2 and inp_size > TP2_P2P_GRAPH_MAX_SIZE:
            return False
        if (
            self.world_size == 8
            and TP8_PYNCCL_GRAPH_RANGE[0] <= inp_size <= TP8_PYNCCL_GRAPH_RANGE[1]
        ):
            return False
        if self.world_size != 2 and (inp.numel() // 8) % 4:
            return False
        if not 0 < inp_size <= self.max_size or inp_size % 16 != 0:
            return False
        if inp.data_ptr() % 16 != 0 or not is_weak_contiguous(inp):
            return False
        return self.fully_connected

    def should_custom_ar(self, inp: torch.Tensor, **_: object) -> bool:
        """Compatibility alias for communicator benchmarks."""
        return self.should_use(inp)

    def _run_p2p_tp4_push_rsag(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        stream_ptr: int | None,
        direct_output_ptrs_address: int | None = None,
    ) -> torch.Tensor | None:
        if self.world_size != 4 or inp.dtype != torch.bfloat16:
            return None
        pack_count = inp.numel() // 8
        if pack_count % self.world_size:
            return None
        threads = self._threads
        blocks = self._p2p_blocks_override
        if not blocks:
            if inp.nbytes >= TP4_P2P_TWO_BLOCK_MIN_SIZE:
                blocks = 2
            elif inp.nbytes >= TP4_P2P_FOUR_BLOCK_MIN_SIZE:
                blocks = 4
            else:
                blocks = 8
        direct_output = direct_output_ptrs_address is not None
        copy_load_nontemporal = self._p2p_copy_load_nontemporal
        if copy_load_nontemporal is None:
            copy_load_nontemporal = inp.nbytes >= TP4_P2P_TWO_BLOCK_MIN_SIZE
        key = (
            "tp4_push",
            blocks,
            threads,
            direct_output,
            copy_load_nontemporal,
        )
        launcher = self._p2p_launchers.get(key)
        if launcher is None:
            from .flydsl_kernels.rdna4_all_reduce_p2p import (
                make_p2p_tp4_push_rsag_launcher,
            )

            launcher = make_p2p_tp4_push_rsag_launcher(
                blocks=blocks,
                threads=threads,
                direct_output=direct_output,
                copy_load_nontemporal=copy_load_nontemporal,
            )
            self._p2p_launchers[key] = launcher
        stream = (
            torch.cuda.current_stream(self.device)
            if stream_ptr is None
            else torch.cuda.ExternalStream(stream_ptr)
        )
        from flydsl.expr.typing import Int32, Int64

        launcher(
            Int32(self.rank),
            Int64(self._self_sg),
            Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
            Int64(
                int(self._gpu_input_buffer_ptrs_array.data_ptr())
                if direct_output_ptrs_address is None
                else direct_output_ptrs_address
            ),
            Int64(int(self._gpu_tmp_ptrs_array.data_ptr())),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int32(inp.numel()),
            stream=stream,
        )
        return out

    def _run_p2p_tp2_one_shot(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        stream_ptr: int | None,
        input_ptrs_address: int | None = None,
    ) -> torch.Tensor | None:
        if self.world_size != 2 or inp.dtype != torch.bfloat16:
            return None
        if inp.nbytes < 8 * 1024 * 1024:
            blocks, threads = 8, 512
        elif inp.nbytes < 16 * 1024 * 1024:
            blocks, threads = 16, 512
        else:
            blocks, threads = 32, 256
        chunk_packs = 32768 if inp.nbytes >= 64 * 1024 * 1024 else 16384
        direct_input = input_ptrs_address is not None
        key = ("tp2_one_shot", blocks, threads, chunk_packs, direct_input)
        launcher = self._p2p_launchers.get(key)
        if launcher is None:
            from .flydsl_kernels.rdna4_all_reduce_tp2 import (
                make_p2p_tp2_one_shot_launcher,
            )

            launcher = make_p2p_tp2_one_shot_launcher(
                blocks=blocks,
                threads=threads,
                chunk_packs=chunk_packs,
                direct_input=direct_input,
            )
            self._p2p_launchers[key] = launcher
        stream = (
            torch.cuda.current_stream(self.device)
            if stream_ptr is None
            else torch.cuda.ExternalStream(stream_ptr)
        )
        from flydsl.expr.typing import Int32, Int64

        launcher(
            Int32(self.rank),
            Int64(self._self_sg),
            Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
            Int64(
                input_ptrs_address
                if direct_input
                else int(self._gpu_input_buffer_ptrs_array.data_ptr())
            ),
            Int64(
                input_ptrs_address
                if direct_input
                else int(self._gpu_tmp_ptrs_array.data_ptr())
            ),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int32(inp.numel()),
            stream=stream,
        )
        return out

    def _run_p2p_tp4_pull(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        stream_ptr: int | None,
        output_ptrs_address: int,
        input_ptrs_address: int,
    ) -> torch.Tensor | None:
        if self.world_size != 4 or inp.dtype != torch.bfloat16:
            return None
        pack_count = inp.numel() // 8
        if pack_count % self.world_size:
            return None
        threads = self._threads
        blocks = self._p2p_blocks_override
        if not blocks:
            if inp.nbytes >= TP4_P2P_PULL_FULL_WGP_MIN_SIZE:
                blocks = 32
                threads = 256
            else:
                blocks = 8
        copy_load_nontemporal = bool(self._p2p_copy_load_nontemporal)
        two_shot = inp.nbytes >= TP4_P2P_TWO_SHOT_MIN_SIZE
        direct_input = two_shot
        chunk_bytes = 32 * 1024 * 1024
        if two_shot:
            chunk_bytes = (
                TP4_P2P_TWO_SHOT_SMALL_CHUNK_SIZE
                if inp.nbytes <= TP4_P2P_TWO_SHOT_SMALL_CHUNK_SIZE
                else TP4_P2P_TWO_SHOT_LARGE_CHUNK_SIZE
            )
        tail_safe = two_shot and inp.nbytes % (4 * 512) != 0
        key = (
            "tp4_pull",
            blocks,
            threads,
            copy_load_nontemporal,
            two_shot,
            chunk_bytes,
            tail_safe,
            direct_input,
        )
        launcher = self._p2p_launchers.get(key)
        if launcher is None:
            from .flydsl_kernels.rdna4_all_reduce_p2p import (
                make_p2p_tp4_pull_launcher,
            )

            launcher = make_p2p_tp4_pull_launcher(
                blocks=blocks,
                threads=threads,
                copy_load_nontemporal=copy_load_nontemporal,
                two_shot=two_shot,
                chunk_bytes=chunk_bytes,
                tail_safe=tail_safe,
                direct_input=direct_input,
            )
            self._p2p_launchers[key] = launcher
        stream = (
            torch.cuda.current_stream(self.device)
            if stream_ptr is None
            else torch.cuda.ExternalStream(stream_ptr)
        )
        from flydsl.expr.typing import Int32, Int64

        launcher(
            Int32(self.rank),
            Int64(self._self_sg),
            Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
            Int64(output_ptrs_address),
            Int64(
                input_ptrs_address
                if direct_input
                else int(self._gpu_tmp_ptrs_array.data_ptr())
            ),
            Int64(int(inp.data_ptr())),
            Int64(int(out.data_ptr())),
            Int32(inp.numel()),
            stream=stream,
        )
        return out

    def _run_p2p_hierarchical_tp8(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        *,
        stream_ptr: int | None,
        direct_output_ptrs_address: int,
    ) -> torch.Tensor | None:
        if self.world_size != 8 or inp.dtype != torch.bfloat16:
            return None
        pack_count = inp.numel() // 8
        if pack_count % 4:
            return None
        threads = self._threads
        blocks = self._p2p_blocks_override or (
            3 if 576 * 1024 <= inp.nbytes <= 704 * 1024 else 4
        )
        copy_load_nontemporal = bool(self._p2p_copy_load_nontemporal)
        local_copy_threads = self._p2p_tp8_local_copy_threads
        if not local_copy_threads:
            # Keep the 1024-thread cross-half reduction, but reduce concurrent
            # same-half transactions once each peer chunk is large enough.
            local_copy_threads = (
                256
                if inp.nbytes >= TP8_P2P_REDUCED_LOCAL_COPY_THREADS_MIN_SIZE
                else threads
            )
        key = (
            "hierarchical",
            8,
            blocks,
            threads,
            copy_load_nontemporal,
            local_copy_threads,
        )
        launcher = self._p2p_launchers.get(key)
        if launcher is None:
            from .flydsl_kernels.rdna4_all_reduce_p2p import (
                make_p2p_hierarchical_tp8_launcher,
            )

            launcher = make_p2p_hierarchical_tp8_launcher(
                blocks=blocks,
                threads=threads,
                copy_load_nontemporal=copy_load_nontemporal,
                local_copy_threads=local_copy_threads,
            )
            self._p2p_launchers[key] = launcher
        stream = (
            torch.cuda.current_stream(self.device)
            if stream_ptr is None
            else torch.cuda.ExternalStream(stream_ptr)
        )
        from flydsl.expr.typing import Int32, Int64

        launcher(
            Int32(self.rank),
            Int64(self._self_sg),
            Int64(int(self._gpu_sg_ptrs_array.data_ptr())),
            Int64(int(self._gpu_input_buffer_ptrs_array.data_ptr())),
            Int64(int(self._gpu_tmp_ptrs_array.data_ptr())),
            Int64(int(inp.data_ptr())),
            Int64(direct_output_ptrs_address),
            Int32(inp.numel()),
            stream=stream,
        )
        return out

    def custom_all_reduce(
        self,
        inp,
        *,
        out=None,
        stream_ptr: int | None = None,
    ):
        """Unified all-reduce (eager and cudagraph).

        Returns None when the input is not supported by the custom kernel
        (caller should fall back to NCCL).
        """
        route = self._select_route(inp, graph=self._IS_CAPTURING)
        if route is _Route.TP2_MAPPED:
            assert self._tp2_mapped is not None
            return self._tp2_mapped.all_reduce(
                inp,
                out=out,
                stream_ptr=stream_ptr,
            )
        if route is _Route.MAPPED:
            assert self._mapped is not None
            return self._mapped.all_reduce(
                inp,
                out=out,
                stream_ptr=stream_ptr,
            )
        if route is not _Route.P2P:
            return None

        graph_capturing = (
            self._IS_CAPTURING and torch.cuda.is_current_stream_capturing()
        )
        if self._IS_CAPTURING and not graph_capturing:
            return torch.empty_like(inp)

        if out is None:
            out = torch.empty_like(inp)

        N = int(out.numel())

        if int(inp.numel()) != N:
            raise ValueError("inp.numel must equal out.numel")
        if not is_weak_contiguous(inp):
            raise ValueError("input tensor must be weak-contiguous")
        if not is_weak_contiguous(out):
            raise ValueError("output tensor must be weak-contiguous")
        if out.device != self.device or out.data_ptr() % 16 != 0:
            raise ValueError("output tensor must be on the local GPU and 16B-aligned")
        if inp.dtype != out.dtype:
            raise ValueError("inp/out dtype mismatch")

        if graph_capturing:
            capture_index = self._capture_base_index + len(self._captured_outputs)
            if capture_index >= MAX_GRAPH_ALL_REDUCE_CALLS:
                raise RuntimeError(
                    "RDNA4 graph captured more than "
                    f"{MAX_GRAPH_ALL_REDUCE_CALLS} all-reduces"
                )
            self._captured_outputs.append(out)
            self._captured_inputs.append(inp)
            assert self._gpu_graph_input_ptrs_array is not None
            assert self._gpu_graph_output_ptrs_array is not None
            output_ptrs_address = (
                int(self._gpu_graph_output_ptrs_array.data_ptr())
                + capture_index * 8 * torch.int64.itemsize
            )
            input_ptrs_address = (
                int(self._gpu_graph_input_ptrs_array.data_ptr())
                + capture_index * 8 * torch.int64.itemsize
            )
            if self.world_size == 2:
                result = self._run_p2p_tp2_one_shot(
                    inp,
                    out,
                    stream_ptr=stream_ptr,
                    input_ptrs_address=input_ptrs_address,
                )
            elif self.world_size == 8:
                result = self._run_p2p_hierarchical_tp8(
                    inp,
                    out,
                    stream_ptr=stream_ptr,
                    direct_output_ptrs_address=output_ptrs_address,
                )
            else:
                if (
                    TP4_P2P_PULL_GRAPH_RANGE[0]
                    <= inp.nbytes
                    <= (TP4_P2P_PULL_GRAPH_RANGE[1])
                ):
                    result = self._run_p2p_tp4_pull(
                        inp,
                        out,
                        stream_ptr=stream_ptr,
                        output_ptrs_address=output_ptrs_address,
                        input_ptrs_address=input_ptrs_address,
                    )
                else:
                    result = self._run_p2p_tp4_push_rsag(
                        inp,
                        out,
                        stream_ptr=stream_ptr,
                        direct_output_ptrs_address=output_ptrs_address,
                    )
            if result is not None:
                return result
            raise RuntimeError("RDNA4 graph P2P dispatch has no matching kernel")
        if self.world_size == 2:
            eager_input_ptrs = self._get_eager_input_ptrs(inp)
            if eager_input_ptrs is None:
                return None
            result = self._run_p2p_tp2_one_shot(
                inp,
                out,
                stream_ptr=stream_ptr,
                input_ptrs_address=eager_input_ptrs,
            )
            if result is not None:
                return result
        elif self.world_size == 4:
            eager_input_ptrs = self._get_eager_input_ptrs(inp)
            eager_output_ptrs = self._get_eager_output_ptrs(out)
            if eager_input_ptrs is not None and eager_output_ptrs is not None:
                result = self._run_p2p_tp4_pull(
                    inp,
                    out,
                    stream_ptr=stream_ptr,
                    output_ptrs_address=eager_output_ptrs,
                    input_ptrs_address=eager_input_ptrs,
                )
                if result is not None:
                    return result
        return None

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        """Perform an out-of-place all-reduce or return ``None``."""
        return self.custom_all_reduce(inp)


__all__ = ["RDNA4AllReduce"]
