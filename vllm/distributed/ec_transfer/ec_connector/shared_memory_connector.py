# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import struct
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Shared-memory wire format (little-endian):
#
#   [0:4)       magic "ECv1"
#   [4:12)      payload_size (8 bytes, little-endian)
#   [12:48)     tensor header (ndim, dtype_code, shape[8])
#   [48:...)     raw tensor bytes (flattened uint8 view)
#   [last byte] ACK (0=pending, 1=done)
#
# payload_size covers the tensor header plus raw bytes (excludes magic, size
# field, and ACK). tensor_header = ndim (i32) + dtype_code (i32) + shape[8]
# (8 x i32).
_MAGIC = b"ECv1"
_MAGIC_BYTES = len(_MAGIC)
SIZE_FIELD_BYTES = 8
_HEADER_PREFIX_BYTES = _MAGIC_BYTES + SIZE_FIELD_BYTES
_NDIM_SHAPE_SLOTS = 8
_HEADER_SIZE = 4 + 4 + 4 * _NDIM_SHAPE_SLOTS
ACK_DONE = 1
ACK_PENDING = 0

# EC cache tensors follow model_config.dtype (fp16/bf16/fp32 in practice).
# float64 is included for completeness; quantized/int dtypes are not supported.
_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _shm_name(identifier: str) -> str:
    """Map mm identifier to a POSIX-safe shared memory name."""
    return hashlib.sha256(identifier.encode()).hexdigest()


def _payload_size(num_bytes: int) -> int:
    return _HEADER_SIZE + num_bytes


def _total_shm_size(payload_size: int) -> int:
    return _HEADER_PREFIX_BYTES + payload_size + 1


def _ack_index(payload_size: int) -> int:
    return _HEADER_PREFIX_BYTES + payload_size


def _validate_payload_size(payload_size: int, shm_size: int) -> bool:
    if payload_size < _HEADER_SIZE:
        return False
    return payload_size <= shm_size - _HEADER_PREFIX_BYTES - 1


def _try_unpack_header(header: bytes) -> tuple[torch.dtype, tuple[int, ...]] | None:
    try:
        return _unpack_header(header)
    except ValueError:
        return None


def _is_readable_shm_file(path: str) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.pread(fd, _MAGIC_BYTES, 0) != _MAGIC:
            return False
        size_buf = os.pread(fd, SIZE_FIELD_BYTES, _MAGIC_BYTES)
        if len(size_buf) < SIZE_FIELD_BYTES:
            return False
        payload_size = int.from_bytes(size_buf, "little")
        if not _validate_payload_size(payload_size, os.fstat(fd).st_size):
            return False
        header = os.pread(fd, _HEADER_SIZE, _HEADER_PREFIX_BYTES)
        if len(header) < _HEADER_SIZE:
            return False
        return _try_unpack_header(header) is not None
    finally:
        os.close(fd)


def _pack_header(tensor: torch.Tensor) -> bytes:
    ndim = tensor.dim()
    if ndim > _NDIM_SHAPE_SLOTS:
        raise ValueError(f"EC cache tensor ndim={ndim} exceeds max {_NDIM_SHAPE_SLOTS}")
    dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported EC cache dtype: {tensor.dtype}")
    shape = list(tensor.shape) + [0] * (_NDIM_SHAPE_SLOTS - ndim)
    return struct.pack("<ii8i", ndim, dtype_code, *shape)


def _unpack_header(header: bytes | memoryview) -> tuple[torch.dtype, tuple[int, ...]]:
    ndim, dtype_code, *shape_slots = struct.unpack("<ii8i", header[:_HEADER_SIZE])
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported EC cache dtype code: {dtype_code}")
    return dtype, tuple(shape_slots[:ndim])


def _shm_path(name: str) -> str:
    return f"/dev/shm/{name}"


def _write_shm_segment(buf: memoryview | bytearray, cpu_tensor: torch.Tensor) -> int:
    """Write magic + flat EC cache payload into ``buf``. Returns payload size."""
    buf[0:_MAGIC_BYTES] = _MAGIC
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    header = _pack_header(cpu_tensor)
    raw = cpu_tensor.view(torch.uint8).numpy()
    payload_size = _payload_size(raw.nbytes)
    buf[_MAGIC_BYTES:_HEADER_PREFIX_BYTES] = payload_size.to_bytes(
        SIZE_FIELD_BYTES, "little"
    )
    payload_offset = _HEADER_PREFIX_BYTES
    buf[payload_offset : payload_offset + _HEADER_SIZE] = header
    data_offset = payload_offset + _HEADER_SIZE
    data_end = data_offset + raw.nbytes
    buf[data_offset:data_end] = memoryview(raw).cast("B")
    buf[_ack_index(payload_size)] = ACK_PENDING
    return payload_size


def _read_tensor_from_payload(
    buf: memoryview | bytearray,
    payload_offset: int,
    payload_size: int,
    *,
    sync_device: bool = True,
) -> torch.Tensor:
    from vllm.platforms import current_platform

    header_end = payload_offset + _HEADER_SIZE
    dtype, shape = _unpack_header(buf[payload_offset:header_end])
    data_offset = header_end
    num_bytes = payload_size - _HEADER_SIZE
    # Clone immediately so the returned tensor does not pin the SHM mapping.
    cpu_tensor = (
        torch.frombuffer(buf[data_offset : data_offset + num_bytes], dtype=dtype)
        .reshape(shape)
        .clone()
    )
    device_type = current_platform.device_type
    if device_type == "cpu":
        return cpu_tensor
    device_tensor = cpu_tensor.to(device_type, non_blocking=True)
    if sync_device and device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return device_tensor


@dataclass
class ECSharedMemoryConnectorMetadata(ECConnectorMetadata):
    mm_hashes: list[str] = field(default_factory=list)

    def add_mm_hash(self, mm_hash: str) -> None:
        self.mm_hashes.append(mm_hash)


class ECSharedMemoryConnector(ECConnectorBase):
    """EC connector backed by POSIX shared memory segments."""

    supports_ec_connector_cache_manager = True

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")
        self._mm_hashes_need_loads: set[str] = set()
        self._pending_sends: set[str] = set()
        self._pending_recvs: set[str] = set()
        mc = vllm_config.model_config
        encoder_slot_bytes = mc.get_hidden_size() * mc.dtype.itemsize
        ec_capacity_embeds = transfer_config.get_ec_connector_capacity_embeds()
        scheduler_slot_budget_bytes = encoder_slot_bytes * ec_capacity_embeds
        logger.info(
            "ECSharedMemoryConnector init: scheduler_slot_budget=%.3f MiB "
            "(ec_connector_capacity_embeds=%d, encoder_slot_bytes=%d)",
            scheduler_slot_budget_bytes / MiB_bytes,
            ec_capacity_embeds,
            encoder_slot_bytes,
        )

    @staticmethod
    def _serialize_cache(tensor: torch.Tensor) -> bytes:
        cpu_tensor = tensor.detach().contiguous().cpu()
        header = _pack_header(cpu_tensor)
        raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
        return header + raw

    @staticmethod
    def _deserialize_cache(payload: bytes | memoryview) -> torch.Tensor:
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        return _read_tensor_from_payload(payload, 0, len(payload))

    @staticmethod
    def _unlink_shm(mm_hash: str) -> None:
        try:
            shm = shared_memory.SharedMemory(name=_shm_name(mm_hash))
        except OSError:
            return
        try:
            shm.unlink()
        finally:
            shm.close()

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECSharedMemoryConnectorMetadata)
        for mm_hash in metadata.mm_hashes:
            if mm_hash in encoder_cache:
                continue
            try:
                shm = shared_memory.SharedMemory(name=_shm_name(mm_hash))
            except (FileNotFoundError, PermissionError, OSError):
                continue
            try:
                if bytes(shm.buf[:_MAGIC_BYTES]) != _MAGIC:
                    continue
                payload_size = int.from_bytes(
                    shm.buf[_MAGIC_BYTES:_HEADER_PREFIX_BYTES], "little"
                )
                if not _validate_payload_size(payload_size, shm.size):
                    continue
                payload_offset = _HEADER_PREFIX_BYTES
                tensor = _read_tensor_from_payload(
                    shm.buf, payload_offset, payload_size
                )
                encoder_cache[mm_hash] = tensor
                payload_end = _ack_index(payload_size)
                if shm.size > payload_end:
                    shm.buf[payload_end] = ACK_DONE
                self._pending_recvs.add(mm_hash)
            finally:
                shm.close()

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        if get_tensor_model_parallel_rank() != 0:
            return
        tensor = encoder_cache[mm_hash]
        cpu_tensor = tensor.detach().contiguous().cpu()
        payload_size = _payload_size(cpu_tensor.view(torch.uint8).numel())
        total_size = _total_shm_size(payload_size)
        shm_name = _shm_name(mm_hash)
        shm = None
        try:
            try:
                shm = shared_memory.SharedMemory(
                    name=shm_name, create=True, size=total_size
                )
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=shm_name)
                if shm.size < total_size:
                    shm.close()
                    shm = None
                    self._unordered_resize_shm(mm_hash, total_size)
                    shm = shared_memory.SharedMemory(name=shm_name)
            _write_shm_segment(shm.buf, cpu_tensor)
            self._pending_sends.add(mm_hash)
        finally:
            if shm is not None:
                shm.close()

    def _unordered_resize_shm(self, mm_hash: str, new_size: int) -> None:
        self._unlink_shm(mm_hash)
        shared_memory.SharedMemory(
            name=_shm_name(mm_hash), create=True, size=new_size
        ).close()

    @staticmethod
    def _is_send_acknowledged(mm_hash: str) -> bool | None:
        path = _shm_path(_shm_name(mm_hash))
        if not os.path.exists(path):
            return None
        fd = os.open(path, os.O_RDONLY)
        try:
            if os.pread(fd, _MAGIC_BYTES, 0) != _MAGIC:
                return False
            size_buf = os.pread(fd, SIZE_FIELD_BYTES, _MAGIC_BYTES)
            if len(size_buf) < SIZE_FIELD_BYTES:
                return False
            payload_size = int.from_bytes(size_buf, "little")
            if not _validate_payload_size(payload_size, os.fstat(fd).st_size):
                return False
            ack_buf = os.pread(fd, 1, _ack_index(payload_size))
            return len(ack_buf) == 1 and ack_buf[0] == ACK_DONE
        finally:
            os.close(fd)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self.role != ECConnectorRole.WORKER:
            return None, None
        if not self.is_producer:
            recving, self._pending_recvs = self._pending_recvs, set()
            return None, recving or None

        finished: set[str] = set()
        for mm_hash in list(self._pending_sends):
            ack = self._is_send_acknowledged(mm_hash)
            if ack is None:
                self._pending_sends.discard(mm_hash)
            elif ack:
                finished.add(mm_hash)
                self._pending_sends.discard(mm_hash)
        return finished or None, None

    def free_physical_cache(self, mm_hash: str) -> None:
        self._pending_sends.discard(mm_hash)
        self._pending_recvs.discard(mm_hash)
        if get_tensor_model_parallel_rank() != 0:
            return
        self._unlink_shm(mm_hash)

    def has_cache_item(self, identifier: str) -> bool:
        path = _shm_path(_shm_name(identifier))
        if not os.path.exists(path):
            return False
        return _is_readable_shm_file(path)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if not self.is_consumer:
            return
        if not self.has_cache_item(mm_hash):
            return
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = ECSharedMemoryConnectorMetadata()
        for mm_hash in self._mm_hashes_need_loads:
            meta.add_mm_hash(mm_hash)
        self._mm_hashes_need_loads.clear()
        return meta
