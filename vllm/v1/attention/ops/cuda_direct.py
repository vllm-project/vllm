# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA driver calls on raw handles, and Triton kernels launched by their handle.

The PCP + DCP overlaps (`kv_ce_overlap`, `pcp_dcp_overlap`) issue a few small
operations per layer from the thread that runs the model. At short contexts that
thread's time is the step time, and a tensor operation or a Triton launch spends
tens of microseconds on its arguments before anything reaches the device. Here the
same work is one driver call on addresses that were computed once:

* `CudaDriver`: copies between device addresses (`CopyList`), events, stream
  memory operations (`Int32Table`), fills and readbacks.
* `DirectKernel`: a Triton kernel compiled and loaded once, when its owner is
  built, and then launched with `cuLaunchKernel` from a parameter block that is
  only edited in place. Where a kernel's build cannot be driven this way, the
  overlap that needs it stays off and its steps take the serial path.
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Sequence
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

# Bytes a pointer parameter is a multiple of when its kernel was specialized on
# alignment (Triton's assumption for a tensor whose address is one).
POINTER_ALIGNMENT = 16


class _MemOpWriteValue(ctypes.Structure):
    _fields_ = [
        ("operation", ctypes.c_uint),
        ("address", ctypes.c_uint64),
        ("value", ctypes.c_uint64),
        ("flags", ctypes.c_uint),
        ("alias", ctypes.c_uint64),
    ]


class _MemOpParams(ctypes.Union):
    """`CUstreamBatchMemOpParams`."""

    _fields_ = [("write_value", _MemOpWriteValue), ("pad", ctypes.c_uint64 * 6)]


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    """`CUmemcpyAttributes`."""

    _fields_ = [
        ("src_access_order", ctypes.c_int),
        ("src_loc_hint", _MemLocation),
        ("dst_loc_hint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


class CopyList:
    """`(destination, source, bytes)` copies of device addresses with the driver's
    argument arrays built once; empty copies are dropped."""

    __slots__ = ("copies", "count", "destinations", "sources", "sizes", "nbytes")

    def __init__(self, copies: list[tuple[int, int, int]]) -> None:
        self.copies = [copy for copy in copies if copy[2] > 0]
        self.count = count = len(self.copies)
        self.nbytes = sum(size for _, _, size in self.copies)
        destinations, sources, sizes = zip(*self.copies) if count else ((), (), ())
        self.destinations = (ctypes.c_uint64 * count)(*destinations)
        self.sources = (ctypes.c_uint64 * count)(*sources)
        self.sizes = (ctypes.c_size_t * count)(*sizes)


class CudaDriver:
    """The CUDA driver calls the overlaps issue themselves on raw handles. Each is
    one driver call, with none of the per-call work of a tensor operation."""

    NOT_READY = 600
    EVENT_DISABLE_TIMING = 0x2
    OP_WRITE_VALUE_32 = 2
    SRC_ACCESS_ORDER_STREAM = 0x1

    def __init__(self) -> None:
        driver = ctypes.CDLL("libcuda.so.1")
        handle, u64, size = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
        uint = ctypes.c_uint
        signatures: dict[str, list[Any]] = {
            "cuMemcpyAsync": [u64, u64, size, handle],
            "cuStreamBatchMemOp_v2": [handle, uint, handle, uint],
            "cuEventCreate": [ctypes.POINTER(handle), uint],
            "cuEventRecord": [handle, handle],
            "cuStreamWaitEvent": [handle, handle, uint],
            "cuEventQuery": [handle],
            "cuLaunchKernel": [handle] + [uint] * 7 + [handle] * 3,
            "cuMemsetD8Async": [u64, ctypes.c_ubyte, size, handle],
            "cuMemcpyDtoHAsync_v2": [handle, u64, size, handle],
        }
        for name, argtypes in signatures.items():
            function = getattr(driver, name)
            function.argtypes, function.restype = argtypes, ctypes.c_int
        self._memcpy = driver.cuMemcpyAsync
        self._batch_mem_op = driver.cuStreamBatchMemOp_v2
        self._event_create = driver.cuEventCreate
        self._record = driver.cuEventRecord
        self._wait_event = driver.cuStreamWaitEvent
        self._query = driver.cuEventQuery
        self._launch = driver.cuLaunchKernel
        self._memset = driver.cuMemsetD8Async
        self._read = driver.cuMemcpyDtoHAsync_v2
        # Newer than the calls above: without it no kernel is launched directly.
        self._param_info = getattr(driver, "cuFuncGetParamInfo", None)
        if self._param_info is not None:
            self._param_info.argtypes = [handle, size] + [ctypes.POINTER(size)] * 2
            self._param_info.restype = ctypes.c_int
        # One call for several copies, where the driver has it (CUDA 13).
        self._memcpy_batch = getattr(driver, "cuMemcpyBatchAsync_v2", None)
        if self._memcpy_batch is not None:
            self._memcpy_batch.argtypes = [handle] * 3 + [size] + [handle] * 2
            self._memcpy_batch.argtypes += [size, handle]
            self._memcpy_batch.restype = ctypes.c_int
            self._attributes = _MemcpyAttributes(self.SRC_ACCESS_ORDER_STREAM)
            self._attributes_ref = ctypes.byref(self._attributes)
            self._attribute_index = (size * 1)(0)

    @staticmethod
    def check(result: int) -> None:
        if result != 0:
            raise RuntimeError(f"CUDA driver call failed with error {result}")

    def event(self) -> int:
        event = ctypes.c_void_p()
        self.check(self._event_create(ctypes.byref(event), self.EVENT_DISABLE_TIMING))
        assert event.value is not None
        return event.value

    def record(self, event: int, stream: int) -> None:
        self.check(self._record(event, stream))

    def wait_event(self, stream: int, event: int) -> None:
        self.check(self._wait_event(stream, event, 0))

    def done(self, event: int) -> bool:
        """Whether the work before the event's last record completed."""
        result = self._query(event)
        if result == self.NOT_READY:
            return False
        self.check(result)
        return True

    def memcpy_list(self, copies: CopyList, stream: int) -> None:
        """Enqueue the copies on `stream`, which access their sources in stream
        order: one driver call where the driver takes a batch."""
        count = copies.count
        if count == 0:
            return
        if count > 1 and self._memcpy_batch is not None:
            result = self._memcpy_batch(
                copies.destinations,
                copies.sources,
                copies.sizes,
                count,
                self._attributes_ref,
                self._attribute_index,
                1,
                stream,
            )
            if result == 0:
                return
            # A rejected batch enqueued nothing: copy one by one from now on.
            self._memcpy_batch = None
        for destination, source, size in copies.copies:
            self.check(self._memcpy(destination, source, size, stream))

    def fill(self, address: int, byte: int, nbytes: int, stream: int) -> None:
        """Set `nbytes` of device memory to `byte` in stream order."""
        self.check(self._memset(address, byte, nbytes, stream))

    def read(self, host: int, address: int, nbytes: int, stream: int) -> None:
        """Copy device memory to page-locked host memory in stream order, without
        blocking."""
        self.check(self._read(host, address, nbytes, stream))

    def launch(
        self,
        function: int,
        grid_x: int,
        grid_y: int,
        threads: int,
        shared: int,
        stream: int,
        arguments: Any,
    ) -> None:
        self.check(
            self._launch(
                function, grid_x, grid_y, 1, threads, 1, 1, shared, stream, arguments, 0
            )
        )

    def param_count(self, function: int, limit: int = 256) -> int:
        """Parameters of a loaded kernel function."""
        if self._param_info is None:
            raise RuntimeError("the CUDA driver cannot describe kernel parameters")
        offset, size = ctypes.c_size_t(), ctypes.c_size_t()
        count = 0
        while count < limit and (
            self._param_info(function, count, ctypes.byref(offset), ctypes.byref(size))
            == 0
        ):
            count += 1
        return count


@functools.cache
def cuda_driver() -> CudaDriver:
    return CudaDriver()


class Int32Table:
    """A contiguous int32 device tensor written in stream order with one batch of
    stream memory operations, which are built once: a write only fills in the
    values.

    `before` gives the device addresses of 32-bit words outside the tensor that
    are written as its elements `-len(before)` to `-1`: one batch then carries
    them and the head of the tensor.
    """

    # 64-bit words of one `CUstreamBatchMemOpParams`, and the one holding the value.
    _WORDS = 6
    _VALUE = 2

    def __init__(
        self, driver: CudaDriver, tensor: torch.Tensor, before: Sequence[int] = ()
    ) -> None:
        assert tensor.dtype == torch.int32 and tensor.is_contiguous()
        base = tensor.data_ptr()
        addresses = [*before, *(base + 4 * index for index in range(tensor.numel()))]
        self.driver = driver
        self.tensor = tensor
        self._before = len(before)
        self._params = (_MemOpParams * len(addresses))()
        for param, address in zip(self._params, addresses):
            param.write_value.operation = driver.OP_WRITE_VALUE_32
            param.write_value.address = address
        self._words = (ctypes.c_uint64 * (self._WORDS * len(addresses))).from_buffer(
            self._params
        )
        self._stride = ctypes.sizeof(_MemOpParams)

    def write(self, stream: int, start: int, values: list[int]) -> None:
        """Write `values` to elements `[start, start + len(values))` on `stream`."""
        first = start + self._before
        words, at = self._words, self._WORDS * first + self._VALUE
        for value in values:
            words[at] = value
            at += self._WORDS
        self.driver.check(
            self.driver._batch_mem_op(
                stream,
                len(values),
                ctypes.byref(self._params, self._stride * first),
                0,
            )
        )


class OnStream:
    """`torch.cuda.stream` for a stream of the current device, without querying
    the device and its current stream."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream

    def __enter__(self) -> None:
        self.previous = current_stream()
        torch.cuda.set_stream(self.stream)

    def __exit__(self, *exc: object) -> None:
        torch.cuda.set_stream(self.previous)


_SCALAR_TYPES = ("i32", "i64", "u32", "u64")
# Shared memory per block above which a launch needs an attribute of its own.
_MAX_PLAIN_SHARED_BYTES = 228 << 10


class KernelLaunch:
    """One parameter block of a `DirectKernel`: `set` edits a value in place and
    a call enqueues the kernel. Pointers are device addresses."""

    __slots__ = ("kernel", "slots", "arguments", "index", "masks")

    def __init__(self, kernel: DirectKernel) -> None:
        self.kernel = kernel
        count = len(kernel.names) + 2
        # One 8-byte slot per parameter; the two trailing pointers Triton appends
        # (its global and profile scratch) stay null.
        self.slots = (ctypes.c_uint64 * count)()
        self.arguments = (ctypes.c_void_p * count)()
        base = ctypes.addressof(self.slots)
        for position in range(count):
            self.arguments[position] = base + 8 * position
        self.index = kernel.index
        self.masks = kernel.masks

    def set(self, name: str, value: int) -> None:
        position = self.index[name]
        if value & self.masks[position]:
            raise ValueError(
                f"{self.kernel.name}: {name} must be a multiple of "
                f"{POINTER_ALIGNMENT} bytes"
            )
        # Little-endian: a 32-bit parameter is the low half of its slot.
        self.slots[position] = value

    def update(self, **values: int) -> KernelLaunch:
        for name, value in values.items():
            self.set(name, value)
        return self

    def __call__(self, stream: int, grid_x: int, grid_y: int = 1) -> None:
        kernel = self.kernel
        kernel.driver.check(
            kernel.driver._launch(
                kernel.function,
                grid_x,
                grid_y,
                1,
                kernel.threads,
                1,
                1,
                kernel.shared,
                stream,
                self.arguments,
                0,
            )
        )


class DirectKernel:
    """A Triton kernel compiled and loaded when its owner is built, and launched
    by its function handle.

    `pointers` gives the element type of every pointer parameter; every other
    parameter that is not a `tl.constexpr` is an integer annotated in the
    kernel's signature and listed in its `do_not_specialize`, so the kernel has
    one specialization whatever the values. `constexprs` are that
    specialization's constants. A pointer the kernel may specialize on alignment
    must be a multiple of `POINTER_ALIGNMENT`; `KernelLaunch.set` checks it.

    `direct` is False where the build cannot be driven this way (no device, the
    Triton interpreter, a Triton whose compiled kernels look different). The
    overlap that needs the kernel is then unavailable and its steps take the
    serial path.
    """

    def __init__(
        self,
        jit_fn: Any,
        pointers: dict[str, torch.dtype],
        **constexprs: int,
    ) -> None:
        self.jit_fn = jit_fn
        self.name = getattr(jit_fn, "__name__", str(jit_fn))
        self.constexprs = constexprs
        self.direct = False
        self.names: tuple[str, ...] = ()
        self.index: dict[str, int] = {}
        self.masks: list[int] = []
        self.function = 0
        self.threads = self.shared = 0
        try:
            self._load(pointers)
        except Exception as error:  # noqa: BLE001
            logger.warning_once(
                "Triton kernel %s cannot be launched directly: %s",
                self.name,
                str(error),
            )

    def _load(self, pointers: dict[str, torch.dtype]) -> None:
        params = [param for param in self.jit_fn.params if not param.is_constexpr]
        arguments: list[Any] = []
        for param in params:
            if param.name in pointers:
                arguments.append(pointers[param.name])
                continue
            if not param.annotation_type or not param.do_not_specialize:
                raise ValueError(f"{param.name} is not a fixed-type integer")
            arguments.append(0)
        compiled = self.jit_fn.warmup(*arguments, grid=(1,), **self.constexprs)
        # Loads the binary; the handle lives as long as `compiled` does.
        _ = compiled.run
        metadata = compiled.metadata
        signature = [
            (name, kind)
            for name, kind in compiled.src.signature.items()
            if kind != "constexpr"
        ]
        names = tuple(name for name, _ in signature)
        if names != tuple(param.name for param in params):
            raise ValueError("unexpected kernel signature")
        masks = []
        for param, (name, kind) in zip(params, signature):
            if kind.startswith("*"):
                aligned = not param.do_not_specialize_on_alignment
                masks.append(POINTER_ALIGNMENT - 1 if aligned else 0)
            elif kind in _SCALAR_TYPES:
                masks.append(0)
            else:
                raise ValueError(f"{name} has unsupported type {kind}")
        if (
            getattr(metadata, "global_scratch_size", 0)
            or getattr(metadata, "profile_scratch_size", 0)
            or getattr(metadata, "launch_pdl", False)
            or getattr(metadata, "launch_cooperative_grid", False)
            or getattr(metadata, "num_ctas", 1) != 1
            or int(metadata.shared) > _MAX_PLAIN_SHARED_BYTES
        ):
            raise ValueError("the kernel needs Triton's launch attributes")
        driver = cuda_driver()
        function = int(compiled.function)
        if driver.param_count(function) != len(names) + 2:
            raise ValueError("unexpected kernel parameter count")
        self.compiled = compiled
        self.driver = driver
        self.function = function
        self.threads = 32 * int(metadata.num_warps)
        self.shared = int(metadata.shared)
        self.names = names
        self.index = {name: position for position, name in enumerate(names)}
        self.masks = masks + [0, 0]
        self.direct = True

    def bind(self, **values: int) -> KernelLaunch:
        """A parameter block of its own, with `values` set."""
        assert self.direct
        return KernelLaunch(self).update(**values)


def direct_launch(
    jit_fn: Any, pointers: dict[str, torch.dtype], **constexprs: int
) -> DirectKernel | None:
    """The kernel loaded for direct launches, or None where it cannot be."""
    kernel = DirectKernel(jit_fn, pointers, **constexprs)
    return kernel if kernel.direct else None
