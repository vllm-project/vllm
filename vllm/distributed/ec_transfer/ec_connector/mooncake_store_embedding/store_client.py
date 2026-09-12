# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin Mooncake Store client for embedding objects.

Tensor codec adapted from vLLM PR #47302 (0d1f71f5d36c).
"""

from __future__ import annotations

import ctypes
import math
import struct
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import Any

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    MOONCAKE_TENSOR_METADATA_NBYTES,
    EmbeddingPoolKey,
    TensorSpec,
)
from vllm.distributed.mooncake_store import MooncakeStoreConfig, setup_mooncake_store
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

# A failed transfer can still use registered memory after the binding returns.
# Poisoned clients retain their owners until process exit, including if normal
# worker teardown drops the connector. New operations are rejected.
_UNSAFE_CLIENTS: list[Any] = []


# Values from Mooncake v0.3.12 / v0.3.12.post1 types.h (ErrorCode).
# Python exposes integer returns, not this enum.
class _MooncakeErrorCode(IntEnum):
    NO_AVAILABLE_HANDLE = -200
    REPLICA_IS_NOT_READY = -703
    OBJECT_NOT_FOUND = -704
    OBJECT_ALREADY_EXISTS = -705
    LEASE_EXPIRED = -707
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    TENANT_QUOTA_EXCEEDED = -1700


# For partial RAM reads and single-replica writes, these rejections occur
# before submission or after completed I/O. Transfer timeouts can leave I/O
# in flight; all other failures retain registered buffers and poison the client.
_SAFE_IO_REJECTIONS = frozenset(
    {
        _MooncakeErrorCode.NO_AVAILABLE_HANDLE,
        _MooncakeErrorCode.REPLICA_IS_NOT_READY,
        _MooncakeErrorCode.OBJECT_NOT_FOUND,
        _MooncakeErrorCode.OBJECT_ALREADY_EXISTS,
        _MooncakeErrorCode.RPC_FAIL,
        _MooncakeErrorCode.RPC_TIMEOUT,
    }
)
# Ranged GET checks the lease after transfer completion; PUT rejects tenant
# quotas before allocating replicas or submitting transfers.
_SAFE_GET_REJECTIONS = _SAFE_IO_REJECTIONS | {_MooncakeErrorCode.LEASE_EXPIRED}
_SAFE_PUT_REJECTIONS = _SAFE_IO_REJECTIONS | {_MooncakeErrorCode.TENANT_QUOTA_EXCEEDED}


_MOONCAKE_TENSOR_OBJECT_MAGIC = 0x4D4F4F4E
_MOONCAKE_TENSOR_OBJECT_VERSION = 1
_MOONCAKE_TENSOR_HEADER_FORMAT = "<IHHiiIIQQ"
_MOONCAKE_TENSOR_HEADER_NBYTES = struct.calcsize(_MOONCAKE_TENSOR_HEADER_FORMAT)
_MOONCAKE_TENSOR_GLOBAL_SHAPE_OFFSET = _MOONCAKE_TENSOR_HEADER_NBYTES
_MOONCAKE_TENSOR_LOCAL_SHAPE_OFFSET = _MOONCAKE_TENSOR_GLOBAL_SHAPE_OFFSET + 64
_MOONCAKE_DTYPE_TO_TORCH_DTYPE = {
    0: "torch.float32",
    11: "torch.float16",
    12: "torch.bfloat16",
}
_TORCH_DTYPE_TO_MOONCAKE_DTYPE = {
    value: key for key, value in _MOONCAKE_DTYPE_TO_TORCH_DTYPE.items()
}
_SUPPORTED_EMBEDDING_DTYPE_NBYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
}


def create_mooncake_embedding_store_client() -> MooncakeEmbeddingStoreClient:
    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ObjectDataType,
            ReplicateConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Install mooncake with Store support to enable cross_encoder_cache."
        ) from e

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils

    config = MooncakeStoreConfig.load_from_config()
    if config.enable_offload:
        raise ValueError(
            "cross_encoder_cache supports a RAM Store; disable enable_offload"
        )
    store = MooncakeDistributedStore()
    local_ip = get_ip()
    local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
    setup_mooncake_store(store, config, local_hostname)

    logger.info(
        "Initialized embedding Mooncake store mode=%s global_segment_size=%d "
        "local_buffer_size=%d",
        config.mode,
        config.global_segment_size,
        config.local_buffer_size,
    )
    replicate_config = ReplicateConfig()
    replicate_config.data_type = ObjectDataType.TENSOR
    return MooncakeEmbeddingStoreClient(store, replicate_config=replicate_config)


class EmbeddingStoreError(RuntimeError):
    """A Store failure that must propagate to the worker."""


class EmbeddingStoreOperationError(EmbeddingStoreError):
    """A rejected or completed operation that permits normal Encoder fallback."""


class MooncakeEmbeddingStoreClient:
    """Wraps Mooncake object and buffer APIs used by embedding transfer."""

    def __init__(self, store: Any, replicate_config: Any | None = None):
        self._lifetime_lock = threading.Lock()
        self._unsafe_owners: list[Any] = []
        self._poisoned = False
        self.store = store
        self.replicate_config = replicate_config

    def _check_healthy(self) -> None:
        if self._poisoned:
            raise EmbeddingStoreError(
                "Mooncake I/O completion is unconfirmed; worker must exit"
            )

    @contextmanager
    def _registered_io(
        self,
        buffers: list[tuple[Any, int, int]],
        *,
        safe_rejections: frozenset[_MooncakeErrorCode],
    ) -> Iterator[Callable[[Any], None]]:
        self._check_healthy()
        registered = []
        submitted = False
        confirmed = False

        def confirm(results: Any) -> None:
            nonlocal confirmed

            def terminal(value: Any) -> bool:
                if isinstance(value, (list, tuple)):
                    return bool(value) and all(terminal(x) for x in value)
                return type(value) is int and (value >= 0 or value in safe_rejections)

            if not terminal(results):
                raise EmbeddingStoreError(
                    "Mooncake I/O completion is unconfirmed; retaining buffers"
                )
            confirmed = True

        try:
            for owner, addr, size in buffers:
                if owner is None:
                    raise ValueError("registered I/O requires a buffer owner")
                if addr not in registered:
                    ret = self.store.register_buffer(addr, size)
                    if ret != 0:
                        raise EmbeddingStoreOperationError(
                            f"Failed to register embedding buffer: {ret}"
                        )
                    registered.append(addr)
            submitted = True
            yield confirm
        except BaseException as error:
            if submitted and not confirmed:
                with self._lifetime_lock:
                    self._unsafe_owners.extend(owner for owner, _, _ in buffers)
                    if not self._poisoned:
                        self._poisoned = True
                        _UNSAFE_CLIENTS.append(self)
                raise EmbeddingStoreError(
                    "Mooncake I/O completion is unconfirmed; retaining buffers"
                ) from error
            raise
        finally:
            if not submitted or confirmed:
                try:
                    for addr in reversed(registered):
                        self.unregister_tensor(addr)
                except BaseException:
                    with self._lifetime_lock:
                        self._unsafe_owners.extend(owner for owner, _, _ in buffers)
                        if not self._poisoned:
                            self._poisoned = True
                            _UNSAFE_CLIENTS.append(self)
                    raise

    def close(self) -> None:
        """Close the supported Mooncake binding after all I/O is drained."""
        self._check_healthy()
        ret = self.store.close()
        if ret != 0:
            raise EmbeddingStoreError(
                f"failed to close embedding Mooncake Store: {ret}"
            )

    def exists(self, pool_key: EmbeddingPoolKey) -> bool:
        return self.batch_exists([pool_key])[0]

    def batch_exists(self, pool_keys: list[EmbeddingPoolKey]) -> list[bool]:
        self._check_healthy()
        if not pool_keys:
            return []

        keys = [pool_key.to_string() for pool_key in pool_keys]
        states = self.store.batch_is_exist(keys)
        if len(states) != len(keys):
            raise RuntimeError(
                "Mooncake Store returned an unexpected number of lookup results"
            )
        return [state == 1 for state in states]

    def get_tensor_meta(self, pool_key: EmbeddingPoolKey) -> TensorSpec:
        buffer = (ctypes.c_ubyte * MOONCAKE_TENSOR_METADATA_NBYTES)()
        self.get_tensor_payload(
            pool_key,
            ctypes.addressof(buffer),
            MOONCAKE_TENSOR_METADATA_NBYTES,
            0,
            owner=buffer,
        )
        return _decode_mooncake_tensor_metadata(pool_key, bytes(buffer))

    def put_tensor(self, pool_key: EmbeddingPoolKey, tensor: Any) -> None:
        self._check_healthy()
        if not tensor.is_contiguous():
            raise EmbeddingStoreOperationError("embedding tensor must be contiguous")
        metadata = _encode_mooncake_tensor_metadata(tensor)
        header = (ctypes.c_ubyte * len(metadata)).from_buffer_copy(metadata)
        header_ptr = ctypes.addressof(header)
        data_size = tensor.numel() * tensor.element_size()
        buffers = [
            (tensor, tensor.data_ptr(), data_size),
            (header, header_ptr, len(metadata)),
        ]
        with self._registered_io(
            buffers, safe_rejections=_SAFE_PUT_REJECTIONS
        ) as confirm:
            results = self.store.batch_put_from_multi_buffers(
                [pool_key.to_string()],
                [[header_ptr, tensor.data_ptr()]],
                [[len(metadata), data_size]],
                self.replicate_config,
            )
            if len(results) != 1:
                raise EmbeddingStoreOperationError(
                    "Mooncake put returned an unexpected number of results"
                )
            confirm(results)
            if results[0] < 0:
                raise EmbeddingStoreOperationError(
                    f"failed to put embedding tensor for {pool_key.to_string()}"
                )

    def unregister_tensor(self, addr: int) -> None:
        try:
            ret = self.store.unregister_buffer(addr)
        except Exception as error:
            raise EmbeddingStoreError(
                f"could not confirm buffer unregistration addr={addr:#x}"
            ) from error
        if ret != 0:
            raise EmbeddingStoreError(
                f"failed to unregister embedding buffer addr={addr:#x}: {ret}"
            )

    def get_tensor_payload(
        self,
        pool_key: EmbeddingPoolKey,
        addr: int,
        size: int,
        src_offset: int,
        *,
        owner: Any,
    ) -> int:
        with self._registered_io(
            [(owner, addr, size)], safe_rejections=_SAFE_GET_REJECTIONS
        ) as confirm:
            key = pool_key.to_string()
            results = self.store.get_into_ranges(
                [addr],
                [[key]],
                [[[0]]],
                [[[src_offset]]],
                [[[size]]],
            )
            confirm(results)
            result = _single_range_result(results)
            if result != size:
                raise EmbeddingStoreOperationError(
                    "failed to get embedding tensor payload for "
                    f"{pool_key.to_string()}: {result}"
                )
            return result


def _single_range_result(results: Any) -> int:
    try:
        return int(results[0][0][0])
    except (IndexError, TypeError, ValueError):
        return -1


def _decode_mooncake_tensor_metadata(
    pool_key: EmbeddingPoolKey,
    metadata: bytes,
) -> TensorSpec:
    if len(metadata) < MOONCAKE_TENSOR_METADATA_NBYTES:
        raise EmbeddingStoreOperationError(
            f"embedding tensor metadata is too small: {len(metadata)}"
        )
    (
        magic,
        version,
        header_size,
        dtype,
        ndim,
        layout_kind,
        _reserved_flags,
        data_offset,
        data_bytes,
    ) = struct.unpack_from(_MOONCAKE_TENSOR_HEADER_FORMAT, metadata, 0)
    if (
        magic != _MOONCAKE_TENSOR_OBJECT_MAGIC
        or version != _MOONCAKE_TENSOR_OBJECT_VERSION
        or header_size != MOONCAKE_TENSOR_METADATA_NBYTES
    ):
        raise EmbeddingStoreOperationError(
            f"invalid Mooncake tensor metadata header for {pool_key.to_string()}"
        )
    if ndim <= 0 or ndim > 8:
        raise EmbeddingStoreOperationError(
            f"invalid embedding tensor ndim for {pool_key.to_string()}: {ndim}"
        )
    if dtype not in _MOONCAKE_DTYPE_TO_TORCH_DTYPE:
        raise EmbeddingStoreOperationError(
            f"unsupported Mooncake tensor dtype for {pool_key.to_string()}: {dtype}"
        )
    dtype_name = _MOONCAKE_DTYPE_TO_TORCH_DTYPE[dtype]
    if layout_kind != 0:
        raise EmbeddingStoreOperationError(
            "unsupported embedding tensor layout for "
            f"{pool_key.to_string()}: {layout_kind}"
        )
    if data_offset != MOONCAKE_TENSOR_METADATA_NBYTES:
        raise EmbeddingStoreOperationError(
            "invalid embedding tensor data offset for "
            f"{pool_key.to_string()}: {data_offset}"
        )
    global_shape = struct.unpack_from(
        "<8q",
        metadata,
        _MOONCAKE_TENSOR_GLOBAL_SHAPE_OFFSET,
    )
    local_shape = struct.unpack_from(
        "<8q",
        metadata,
        _MOONCAKE_TENSOR_LOCAL_SHAPE_OFFSET,
    )
    if global_shape != local_shape:
        raise EmbeddingStoreOperationError(
            f"embedding tensor global/local shape mismatch for {pool_key.to_string()}"
        )
    shape = tuple(int(dim) for dim in local_shape[:ndim])
    if any(dim <= 0 for dim in shape):
        raise EmbeddingStoreOperationError(
            f"invalid embedding tensor shape for {pool_key.to_string()}: {shape}"
        )
    expected_data_bytes = (
        math.prod(shape) * _SUPPORTED_EMBEDDING_DTYPE_NBYTES[dtype_name]
    )
    if data_bytes <= 0 or data_bytes != expected_data_bytes:
        raise EmbeddingStoreOperationError(
            "embedding tensor shape/dtype do not match payload size for "
            f"{pool_key.to_string()}: expected={expected_data_bytes} "
            f"actual={data_bytes}"
        )
    return TensorSpec(
        shape=shape,
        dtype=dtype_name,
        nbytes=int(data_bytes),
    )


def _encode_mooncake_tensor_metadata(tensor: Any) -> bytes:
    dtype = str(tensor.dtype)
    if dtype not in _TORCH_DTYPE_TO_MOONCAKE_DTYPE:
        raise EmbeddingStoreOperationError(
            f"unsupported embedding tensor dtype: {dtype}"
        )
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) > 8:
        raise EmbeddingStoreOperationError(
            f"embedding tensor has too many dimensions: {len(shape)}"
        )
    nbytes = tensor.numel() * tensor.element_size()
    header = struct.pack(
        _MOONCAKE_TENSOR_HEADER_FORMAT,
        _MOONCAKE_TENSOR_OBJECT_MAGIC,
        _MOONCAKE_TENSOR_OBJECT_VERSION,
        MOONCAKE_TENSOR_METADATA_NBYTES,
        _TORCH_DTYPE_TO_MOONCAKE_DTYPE[dtype],
        len(shape),
        0,
        0,
        MOONCAKE_TENSOR_METADATA_NBYTES,
        nbytes,
    )
    dims = shape + (-1,) * (8 - len(shape))
    tensor_shape = struct.pack("<8q", *dims)
    axes = b"\0" * (32 * 4)
    metadata = header + tensor_shape + tensor_shape + struct.pack("<II", 0, 0) + axes
    if len(metadata) != MOONCAKE_TENSOR_METADATA_NBYTES:
        raise EmbeddingStoreOperationError(
            f"invalid Mooncake tensor metadata size: {len(metadata)}"
        )
    return metadata
