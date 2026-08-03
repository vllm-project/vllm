# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load model weights from disk using DirectIO, reduce page cache overhead."""

import ctypes
import ctypes.util
import glob
import json
import os
import struct
import time

import libaio
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

IO_MEM_ALIGN = 512  # Linux support 512 bytes as logical block size for many years
IO_DEPTH = 128  # The maximum IO depth

_SAFETENSORS_DTYPE_TO_TORCH = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


class _Malloc:
    # aligned memory allocator via glibc posix_memalign function for direct IO
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _libc.posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _libc.posix_memalign.restype = ctypes.c_int
    _libc.free.argtypes = [ctypes.c_void_p]
    _libc.free.restype = None

    def __init__(self, alignment=IO_MEM_ALIGN):
        self.alignment = alignment

    def aligned_alloc(self, size: int) -> int:
        buf = ctypes.c_void_p()
        ret = self._libc.posix_memalign(ctypes.byref(buf), self.alignment, size)
        if ret != 0:
            raise MemoryError(f"posix_memalign failed: errno={ret}")
        return buf.value

    def free(self, addr: int = None):
        if addr is not None:
            self._libc.free(ctypes.c_void_p(addr))


_malloc = _Malloc()


class _InflightWeight:
    __slots__ = (
        "param",
        "loaded_weight",
        "offset",
        "length",
        "iomem_size",
        "iomem",
        "ioctx",
    )

    def __init__(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        offset: int = 0,
        length: int = 0,
        iomem_size: int = 0,
        iomem: int = 0,
        ioctx: libaio.AIOBlock = None,
    ):
        self.param = param
        self.loaded_weight = loaded_weight
        self.offset = offset
        self.length = length
        self.iomem_size = iomem_size
        self.iomem = iomem
        self.ioctx = ioctx


def _copy_to_param(inflight_weight: _InflightWeight) -> None:
    # TODO (zhenwei):the final goal is to reduce any data copy operation from the disk
    #                to the GPU memory.
    # 1, NVMe is widely used in modern servers, and there is almost no alignment
    #    requirement. See NVMe specification v1.3 section 4.4 Scatter Gather List (SGL):
    #    A controller may support byte or Dword alignment and granularity of Data Blocks
    # 2, Linux kernel NVMe core supports DMA alignment as 3 since commit 52fde2c07da606
    #    ("nvme: set dma alignment to dword")
    # 3, Linux block generic layer supports P2P since commit a3ebb59eee2e
    #    ("Merge tag 'vfio-v6.19-rc1' of https://github.com/awilliam/linux-vfio")
    # 4, however, there is still limitation that the alignment must be logical block
    #    size. See linux/block/fops.c. working on it, and implement the final solution
    #    (this compatibility code will be required for a long time).
    loaded_weight = inflight_weight.loaded_weight
    _storage = loaded_weight.untyped_storage()
    _offset = loaded_weight.storage_offset()
    _shape = loaded_weight.shape
    _stride = loaded_weight.stride()

    # [aligned header] + weight + [aligned footer] from iomem, skip header via offset
    iomem = inflight_weight.iomem + inflight_weight.offset
    iobuf = (ctypes.c_uint8 * inflight_weight.length).from_address(iomem)
    iomem_storage = torch.UntypedStorage.from_buffer(
        buffer=iobuf, dtype=loaded_weight.dtype, byte_order="native"
    )

    # use the loaded data to replace the underlying storage of loaded_weight
    loaded_weight.set_(iomem_storage, _offset, loaded_weight.shape, _stride)

    # the only copy operation in current implementation, optimize it next step
    param = inflight_weight.param
    if param.requires_grad:
        param.requires_grad_(False)
        param.copy_(loaded_weight)
        param.requires_grad_(True)
    else:
        param.copy_(loaded_weight)

    # restore original storage to avoid memory leak
    loaded_weight.set_(_storage, _offset, _shape, _stride)
    _malloc.free(inflight_weight.iomem)


def _make_ioctxs(inflight_weights: list[_InflightWeight]) -> list[libaio.AIOBlock]:
    ioctxs = []

    for inflight_weight in inflight_weights:
        loaded_weight = inflight_weight.loaded_weight
        copy_attr = getattr(loaded_weight, "copy_attr", None)
        if copy_attr is None:
            raise ValueError("DirectIO: missing copy_attr")

        fd = copy_attr["fd"]
        file_offset = copy_attr["file_offset"]
        file_length = copy_attr["file_length"]

        align_mask = ~(IO_MEM_ALIGN - 1)
        io_start = file_offset & align_mask  # align down
        io_end = (file_offset + file_length + IO_MEM_ALIGN - 1) & align_mask  # align up
        io_mem_size = io_end - io_start

        iomem = _malloc.aligned_alloc(io_mem_size)
        ioctx = libaio.AIOBlock(
            mode=libaio.AIOBLOCK_MODE_READ,
            target_file=fd,
            buffer_list=[(ctypes.c_uint8 * io_mem_size).from_address(iomem)],
            offset=io_start,
        )
        inflight_weight.offset = file_offset - io_start
        inflight_weight.length = file_length
        inflight_weight.iomem_size = io_mem_size
        inflight_weight.iomem = iomem
        inflight_weight.ioctx = ioctx
        ioctxs.append(ioctx)

    return ioctxs


def _drain_io(ctx: libaio.AIOContext, inflight_weights: list[_InflightWeight]):
    inflight_weights_nr = len(inflight_weights)
    inflight_weights_issued = 0
    inflight_weights_completed = 0
    inflight_weights_map = {}
    _next_progress_milestone = 20

    while inflight_weights_completed < inflight_weights_nr:
        inflight_nr = inflight_weights_issued - inflight_weights_completed

        if inflight_weights_issued < inflight_weights_nr and inflight_nr < IO_DEPTH:
            remaining_nr = inflight_weights_nr - inflight_weights_issued
            to_issue_nr = min(IO_DEPTH - inflight_nr, remaining_nr)
            to_issue_weights = inflight_weights[
                inflight_weights_issued : inflight_weights_issued + to_issue_nr
            ]
            ioctxs = _make_ioctxs(to_issue_weights)
            ctx.submit(ioctxs)

            for inflight_weight in to_issue_weights:
                inflight_weights_map[id(inflight_weight.ioctx)] = inflight_weight

            inflight_weights_issued += to_issue_nr

        events = ctx.getEvents(min_nr=1, nr=IO_DEPTH)
        for event, res, res2 in events:
            inflight_weight = inflight_weights_map[id(event)]
            if res < 0:
                raise OSError(f"DirectIO: read error: {-res} ({os.strerror(-res)})")

            if res < inflight_weight.length:
                raise OSError(
                    f"DirectIO: EOF expected {inflight_weight.length} agaist {res}"
                )

            _copy_to_param(inflight_weight)
            inflight_weights_map.pop(id(event))
            inflight_weights_completed += 1

        _percent = inflight_weights_completed * 100 // inflight_weights_nr
        if _percent >= _next_progress_milestone:
            logger.info(
                "DirectIO: loading progress %d%% (%d/%d)",
                _percent,
                inflight_weights_completed,
                inflight_weights_nr,
            )
            _next_progress_milestone = (_percent // 20 + 1) * 20


def direct_io_copy_weight(param: nn.Parameter, loaded_weight: torch.Tensor):
    copy_attr = getattr(loaded_weight, "copy_attr", None)
    if copy_attr is None:
        raise ValueError("DirectIO: missing copy_attr")

    inflight_weights = copy_attr["inflight_weights"]
    inflight_weights.append(_InflightWeight(param=param, loaded_weight=loaded_weight))


class DirectIOModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._fds = []  # type: list[int]

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _parse_safetensors_header(
        self,
        file_path: str,
        fd: int,
        ctx: libaio.AIOContext,
        inflight_weights: list[_InflightWeight],
    ):
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = json.loads(f.read(header_size))

        data_start = 8 + header_size
        weights = {}
        for name, info in header_json.items():
            if name == "__metadata__":
                continue

            torch_dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(info["dtype"])
            if torch_dtype is None:
                raise ValueError(
                    f"Unsupported dtype '{info['dtype']}' for weight '{name}'"
                )

            offsets = info["data_offsets"]
            offsets = info["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or offsets[0] < 0
                or offsets[1] < 0
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"Invalid data_offsets {offsets} for weight '{name}'")

            file_length = offsets[1] - offsets[0]
            file_size = os.fstat(fd).st_size
            if data_start + offsets[1] > file_size:
                raise ValueError(
                    f"data_offsets for weight '{name}' exceed file size {file_size}"
                )

            # placeholder of virtual memory, it records the meta information only
            weight = torch.empty(info["shape"], dtype=torch_dtype)
            weight.copy_attr = {
                "copy_weight": direct_io_copy_weight,
                "fd": fd,
                "file_offset": data_start + offsets[0],
                "file_length": file_length,
                "shape": info["shape"],
                "ctx": ctx,
                "inflight_weights": inflight_weights,
            }

            weights[name] = weight
        return weights

    def _get_weights_iterator(
        self,
        weight_files: list[str],
        fd_map: dict[str, int],
        ctx: int,
        inflight_weights: list[_InflightWeight],
    ):
        weights = {}
        for f in weight_files:
            weights.update(
                self._parse_safetensors_header(f, fd_map[f], ctx, inflight_weights)
            )
        yield from weights.items()

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        logger.info("DirectIO: start to load weights")
        start = time.perf_counter()

        try:
            weight_files = sorted(
                glob.glob(os.path.join(model_config.model, "*.safetensors"))
            )
            if not weight_files:
                raise RuntimeError(
                    f"No safetensors files found in {model_config.model}"
                )

            logger.debug("DirectIO: loading %s", weight_files)

            fd_map = {}
            for f in weight_files:
                fd_map[f] = os.open(f, os.O_RDONLY | os.O_DIRECT)
                self._fds.append(fd_map[f])

            inflight_weights = []  # type: list[_InflightWeight]

            with libaio.AIOContext(IO_DEPTH) as ctx:
                loaded_weights = model.load_weights(
                    self._get_weights_iterator(
                        weight_files, fd_map, ctx, inflight_weights
                    )
                )
                _drain_io(ctx, inflight_weights)

            elapsed = time.perf_counter() - start
            logger.info("DirectIO: load weights in %.2fs", elapsed)

            if loaded_weights is not None:
                weights_not_loaded = {
                    n for n, _ in model.named_parameters()
                } - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        f"DirectIO: weights not loaded: {weights_not_loaded}"
                    )
        finally:
            for fd in self._fds:
                os.close(fd)
            self._fds.clear()