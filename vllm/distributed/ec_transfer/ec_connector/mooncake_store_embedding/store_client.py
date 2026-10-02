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
from collections.abc import Callable, Iterator, Mapping
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
    INVALID_PARAMS = -600
    INVALID_REPLICA = -702
    REPLICA_IS_NOT_READY = -703
    OBJECT_NOT_FOUND = -704
    OBJECT_ALREADY_EXISTS = -705
    LEASE_EXPIRED = -707
    RPC_FAIL = -900
    RPC_TIMEOUT = -901
    TENANT_QUOTA_EXCEEDED = -1700


# For RAM batch reads and single-replica writes, these rejections occur
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
# Batch GET validates capacity/replicas before I/O and checks leases afterwards.
# PUT rejects tenant quotas before allocating replicas or submitting transfers.
_SAFE_GET_REJECTIONS = _SAFE_IO_REJECTIONS | {
    _MooncakeErrorCode.INVALID_PARAMS,
    _MooncakeErrorCode.INVALID_REPLICA,
    _MooncakeErrorCode.LEASE_EXPIRED,
}
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


def create_mooncake_embedding_store_client(
    read_buffer_bytes: int = 128 * 1024**2,
) -> MooncakeEmbeddingStoreClient:
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
    return MooncakeEmbeddingStoreClient(
        store, replicate_config=replicate_config, read_buffer_bytes=read_buffer_bytes
    )


class EmbeddingStoreError(RuntimeError):
    """A Store failure that must propagate to the worker."""


class EmbeddingStoreOperationError(EmbeddingStoreError):
    """A rejected or completed operation that permits normal Encoder fallback."""


def _batch_get_completed_safely(results: Any, capacities: list[int]) -> bool:
    return (
        isinstance(results, list)
        and len(results) == len(capacities)
        and all(
            type(result) is int
            and (0 <= result <= capacity or result in _SAFE_GET_REJECTIONS)
            for result, capacity in zip(results, capacities, strict=True)
        )
    )


class MooncakeEmbeddingStoreClient:
    """Wraps Mooncake object and buffer APIs used by embedding transfer."""

    def __init__(
        self,
        store: Any,
        replicate_config: Any | None = None,
        *,
        read_buffer_bytes: int = 128 * 1024**2,
    ):
        self._lifetime_lock = threading.Lock()
        self._unsafe_owners: list[Any] = []
        self._poisoned = False
        self.store = store
        self.replicate_config = replicate_config
        self._read_buffer_bytes = read_buffer_bytes
        self._read_buffer: torch.Tensor | None = None
        self._put_header: ctypes.Array[ctypes.c_char] | None = None

    def _check_healthy(self) -> None:
        if self._poisoned:
            raise EmbeddingStoreError(
                "Mooncake I/O completion is unconfirmed; worker must exit"
            )

    def _poison(self, owners: list[Any]) -> None:
        with self._lifetime_lock:
            self._unsafe_owners.extend(owners)
            if not self._poisoned:
                self._poisoned = True
                _UNSAFE_CLIENTS.append(self)

    @contextmanager
    def _registered_io(
        self,
        buffers: list[tuple[Any, int, int]],
    ) -> Iterator[Callable[[int], None]]:
        self._check_healthy()
        registered = []
        submitted = False
        confirmed = False

        def confirm(result: int) -> None:
            nonlocal confirmed
            if type(result) is not int or (
                result < 0 and result not in _SAFE_PUT_REJECTIONS
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
                self._poison([owner for owner, _, _ in buffers])
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
                    self._poison([owner for owner, _, _ in buffers])
                    raise

    def close(self) -> None:
        """Close the supported Mooncake binding after all I/O is drained."""
        self._check_healthy()
        if self._read_buffer is not None:
            try:
                self._unregister_buffer(self._read_buffer.data_ptr())
            except BaseException:
                self._poison([self._read_buffer])
                raise
            self._read_buffer = None
        if self._put_header is not None:
            try:
                self._unregister_buffer(ctypes.addressof(self._put_header))
            except BaseException:
                self._poison([self._put_header])
                raise
            self._put_header = None
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

    def load_tensors(
        self, expected: Mapping[str, TensorSpec], device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        """Synchronously load runner inputs into independently owned tensors."""
        self._check_healthy()
        device = torch.device(device)
        loaded: dict[str, torch.Tensor] = {}
        chunk: dict[str, TensorSpec] = {}
        chunk_bytes = 0
        for key, spec in expected.items():
            size = _MOONCAKE_TENSOR_HEADER.size + spec.nbytes
            if size > self._read_buffer_bytes:
                logger.warning(
                    "Skipping Store GET for %s: object bytes=%d exceed read buffer=%d",
                    key,
                    size,
                    self._read_buffer_bytes,
                )
                continue
            if chunk_bytes + size > self._read_buffer_bytes:
                loaded.update(self._load_chunk(chunk, device))
                chunk = {}
                chunk_bytes = 0
            chunk[key] = spec
            chunk_bytes += size
        if chunk:
            loaded.update(self._load_chunk(chunk, device))
        return loaded

    def _load_chunk(
        self, expected: dict[str, TensorSpec], device: torch.device
    ) -> dict[str, torch.Tensor]:
        self._check_healthy()
        if self._read_buffer is None:
            buffer = torch.empty(
                self._read_buffer_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
            ret = self.store.register_buffer(buffer.data_ptr(), buffer.nbytes)
            if ret != 0:
                raise EmbeddingStoreOperationError(
                    f"Failed to register embedding read buffer: {ret}"
                )
            self._read_buffer = buffer
        buffer = self._read_buffer
        base = buffer.data_ptr()
        sizes = [
            _MOONCAKE_TENSOR_HEADER.size + spec.nbytes for spec in expected.values()
        ]
        offsets: list[int] = []
        end = 0
        for size in sizes:
            offsets.append(end)
            end += size
        try:
            results = self.store.batch_get_into(
                list(expected), [base + offset for offset in offsets], sizes
            )
            if not _batch_get_completed_safely(results, sizes):
                raise EmbeddingStoreError("Unsafe batch GET results")
        except BaseException as error:
            self._poison([buffer])
            raise EmbeddingStoreError(
                "Mooncake I/O completion is unconfirmed; retaining read buffer"
            ) from error

        valid: list[tuple[str, TensorSpec, int]] = []
        for (key, spec), offset, size, result in zip(
            expected.items(), offsets, sizes, results, strict=True
        ):
            if result != size:
                logger.warning(
                    "Store GET skipped for %s: result=%d, expected bytes=%d",
                    key,
                    result,
                    size,
                )
                continue
            try:
                _validate_mooncake_tensor_metadata(
                    key,
                    ctypes.string_at(base + offset, _MOONCAKE_TENSOR_HEADER.size),
                    spec,
                )
            except EmbeddingStoreOperationError as error:
                logger.warning("Store GET skipped for %s: %s", key, error)
                continue
            valid.append((key, spec, offset + _MOONCAKE_TENSOR_HEADER.size))
        if not valid:
            return {}
        # Allocate every destination before submitting any asynchronous copy.
        targets = {
            key: torch.empty(
                spec.shape,
                dtype=getattr(torch, spec.dtype.removeprefix("torch.")),
                device=device,
            )
            for key, spec, _ in valid
        }
        ready = torch.Event() if device.type == "cuda" else None
        try:
            for key, spec, offset in valid:
                targets[key].view(torch.uint8).view(-1).copy_(
                    buffer[offset : offset + spec.nbytes],
                    non_blocking=ready is not None,
                )
            if ready is not None:
                ready.record(torch.accelerator.current_stream(device))
                ready.synchronize()
        except BaseException as error:
            if ready is not None:
                self._poison([buffer, *targets.values(), ready])
                raise EmbeddingStoreError(
                    "Embedding copy completion is unconfirmed; retaining buffers"
                ) from error
            raise
        return targets

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
        self._check_healthy()
        if not tensor.is_contiguous():
            raise EmbeddingStoreOperationError("embedding tensor must be contiguous")
        metadata = _encode_mooncake_tensor_metadata(tensor)
        # The single publisher reuses this header only after PUT completion.
        if self._put_header is None:
            header = ctypes.create_string_buffer(len(metadata))
            ret = self.store.register_buffer(ctypes.addressof(header), len(metadata))
            if ret != 0:
                raise EmbeddingStoreOperationError(
                    f"Failed to register embedding header: {ret}"
                )
            self._put_header = header
        header_ptr = ctypes.addressof(self._put_header)
        ctypes.memmove(header_ptr, metadata, len(metadata))
        data_size = tensor.numel() * tensor.element_size()
        buffers = [
            (tensor, tensor.data_ptr(), data_size),
        ]
        with self._registered_io(buffers) as confirm:
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
