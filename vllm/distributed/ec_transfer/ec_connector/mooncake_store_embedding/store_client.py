# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin Mooncake Store client for embedding objects.

Originally adapted from vLLM PR #47302 (0d1f71f5d36c). The v3 key namespace
uses a compact header for complete 2D embeddings, not that PR's tensor format.
"""

from __future__ import annotations

import ctypes
import struct
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import Any

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
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
# Magic, dtype, token count, hidden dimension. Format versioning is in the key.
_MOONCAKE_TENSOR_HEADER = struct.Struct("<IIQQ")
_MOONCAKE_DTYPE_TO_TORCH_DTYPE = {
    0: "torch.float32",
    11: "torch.float16",
    12: "torch.bfloat16",
}
_TORCH_DTYPE_TO_MOONCAKE_DTYPE = {
    value: key for key, value in _MOONCAKE_DTYPE_TO_TORCH_DTYPE.items()
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
    ) -> Iterator[Callable[[int], None]]:
        self._check_healthy()
        registered = []
        submitted = False
        confirmed = False

        def confirm(result: int) -> None:
            nonlocal confirmed
            if type(result) is not int or (
                result < 0 and result not in safe_rejections
            ):
                raise EmbeddingStoreError(
                    "Mooncake I/O completion is unconfirmed; retaining buffers"
                )
            confirmed = True

        try:
            for owner, addr, size in buffers:
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
                        self._unregister_buffer(addr)
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

    def exists(self, pool_key: str) -> bool:
        return self.batch_exists([pool_key])[0]

    def batch_exists(self, keys: list[str]) -> list[bool]:
        self._check_healthy()
        if not keys:
            return []

        states = self.store.batch_is_exist(keys)
        if len(states) != len(keys):
            raise RuntimeError(
                "Mooncake Store returned an unexpected number of lookup results"
            )
        return [state == 1 for state in states]

    def load_tensor(
        self, pool_key: str, expected: TensorSpec, device: torch.device | str
    ) -> torch.Tensor:
        buffer = (ctypes.c_ubyte * _MOONCAKE_TENSOR_HEADER.size)()
        self._get_range(
            pool_key,
            ctypes.addressof(buffer),
            _MOONCAKE_TENSOR_HEADER.size,
            0,
            owner=buffer,
        )
        _validate_mooncake_tensor_metadata(pool_key, bytes(buffer), expected)
        target = torch.empty(
            expected.shape,
            dtype=getattr(torch, expected.dtype.removeprefix("torch.")),
            device=device,
        )
        self._get_range(
            pool_key,
            target.data_ptr(),
            expected.nbytes,
            _MOONCAKE_TENSOR_HEADER.size,
            owner=target,
        )
        return target

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
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
            [result] = self.store.batch_put_from_multi_buffers(
                [pool_key],
                [[header_ptr, tensor.data_ptr()]],
                [[len(metadata), data_size]],
                self.replicate_config,
            )
            confirm(result)
            if result < 0:
                raise EmbeddingStoreOperationError(
                    f"failed to put embedding tensor for {pool_key}"
                )

    def _unregister_buffer(self, addr: int) -> None:
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

    def _get_range(
        self,
        pool_key: str,
        addr: int,
        size: int,
        src_offset: int,
        *,
        owner: Any,
    ) -> int:
        with self._registered_io(
            [(owner, addr, size)], safe_rejections=_SAFE_GET_REJECTIONS
        ) as confirm:
            [[[result]]] = self.store.get_into_ranges(
                [addr],
                [[pool_key]],
                [[[0]]],
                [[[src_offset]]],
                [[[size]]],
            )
            confirm(result)
            if result != size:
                raise EmbeddingStoreOperationError(
                    f"failed to get embedding tensor payload for {pool_key}: {result}"
                )
            return result


def _validate_mooncake_tensor_metadata(
    pool_key: str,
    metadata: bytes,
    expected: TensorSpec,
) -> None:
    magic, dtype, num_tokens, hidden_dim = _MOONCAKE_TENSOR_HEADER.unpack(metadata)
    if magic != _MOONCAKE_TENSOR_OBJECT_MAGIC:
        raise EmbeddingStoreOperationError(
            f"invalid Mooncake tensor metadata header for {pool_key}"
        )
    if _MOONCAKE_DTYPE_TO_TORCH_DTYPE.get(dtype) != expected.dtype:
        raise EmbeddingStoreOperationError(
            f"unexpected Mooncake tensor dtype for {pool_key}: {dtype}"
        )
    if (num_tokens, hidden_dim) != expected.shape:
        raise EmbeddingStoreOperationError(
            f"embedding tensor shape mismatch for {pool_key}: expected={expected.shape}"
        )


def _encode_mooncake_tensor_metadata(tensor: torch.Tensor) -> bytes:
    dtype = str(tensor.dtype)
    if dtype not in _TORCH_DTYPE_TO_MOONCAKE_DTYPE:
        raise EmbeddingStoreOperationError(
            f"unsupported embedding tensor dtype: {dtype}"
        )
    if tensor.ndim != 2:
        raise EmbeddingStoreOperationError(
            f"embedding tensor must be 2D: shape={tuple(tensor.shape)}"
        )
    return _MOONCAKE_TENSOR_HEADER.pack(
        _MOONCAKE_TENSOR_OBJECT_MAGIC,
        _TORCH_DTYPE_TO_MOONCAKE_DTYPE[dtype],
        *tensor.shape,
    )
