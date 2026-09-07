# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin Mooncake Store client for embedding objects."""

from __future__ import annotations

import copy
import ctypes
import json
import os
import struct
from dataclasses import dataclass
from typing import Any

import regex as re

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    EMBEDDING_PROTOCOL_VERSION,
    EMBEDDING_TENSOR_LAYOUT,
    MOONCAKE_TENSOR_METADATA_NBYTES,
    EmbeddingPoolKey,
    TensorMeta,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.keys import (
    make_embedding_data_key,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024
_MOONCAKE_TENSOR_OBJECT_MAGIC = 0x4D4F4F4E
_MOONCAKE_TENSOR_OBJECT_VERSION = 1
_MOONCAKE_TENSOR_HEADER_FORMAT = "<IHHiiIIQQ"
_MOONCAKE_TENSOR_HEADER_NBYTES = struct.calcsize(_MOONCAKE_TENSOR_HEADER_FORMAT)
_MOONCAKE_TENSOR_LOCAL_SHAPE_OFFSET = _MOONCAKE_TENSOR_HEADER_NBYTES + 64
_MOONCAKE_DTYPE_TO_TORCH_DTYPE = {
    0: "torch.float32",
    1: "torch.float64",
    2: "torch.int8",
    3: "torch.uint8",
    4: "torch.int16",
    5: "torch.uint16",
    6: "torch.int32",
    7: "torch.uint32",
    8: "torch.int64",
    9: "torch.uint64",
    10: "torch.bool",
    11: "torch.float16",
    12: "torch.bfloat16",
    13: "torch.float8_e4m3fn",
    14: "torch.float8_e5m2",
}
_TORCH_DTYPE_TO_MOONCAKE_DTYPE = {
    "torch.float32": 0,
    "torch.float64": 1,
    "torch.int8": 2,
    "torch.uint8": 3,
    "torch.int16": 4,
    "torch.uint16": 5,
    "torch.int32": 6,
    "torch.uint32": 7,
    "torch.int64": 8,
    "torch.uint64": 9,
    "torch.bool": 10,
    "torch.float16": 11,
    "torch.bfloat16": 12,
    "torch.float8_e4m3fn": 13,
    "torch.float8_e5m2": 14,
}
_SUPPORTED_EMBEDDING_TORCH_DTYPES = {
    "torch.float16",
    "torch.bfloat16",
    "torch.float32",
}


@dataclass
class MooncakeEmbeddingStoreConfig:
    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: str = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE

    @staticmethod
    def from_file(file_path: str) -> MooncakeEmbeddingStoreConfig:
        with open(file_path, encoding="utf-8") as file:
            config = json.load(file)
        mode = config.get("mode", "embedded")
        return MooncakeEmbeddingStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=mode,
            global_segment_size=_parse_size(
                config.get(
                    "global_segment_size",
                    0 if mode == "standalone-store" else DEFAULT_GLOBAL_SEGMENT_SIZE,
                )
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
        )

    @staticmethod
    def load_from_env() -> MooncakeEmbeddingStoreConfig:
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeEmbeddingStoreConfig.from_file(config_path)


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return int(value)

    cleaned = value.strip().lower()
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid size format: {value!r}")

    multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
        None: 1,
    }
    return int(float(match.group(1)) * multipliers[match.group(2)])


def create_mooncake_embedding_store_client() -> MooncakeEmbeddingStoreClient:
    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Please install mooncake to run vLLM with MooncakeStoreECConnector."
        ) from e

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils

    config = MooncakeEmbeddingStoreConfig.load_from_env()
    store = MooncakeDistributedStore()
    local_ip = get_ip()
    local_hostname = rdma_utils.get_requester_local_hostname(local_ip)
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
    )
    if ret != 0:
        raise RuntimeError("Initialize MooncakeDistributedStore failed.")

    logger.info(
        "Initialized embedding Mooncake store mode=%s global_segment_size=%d "
        "local_buffer_size=%d",
        config.mode,
        config.global_segment_size,
        config.local_buffer_size,
    )
    return MooncakeEmbeddingStoreClient(store, replicate_config=ReplicateConfig())


class EmbeddingStoreError(RuntimeError):
    pass


class EmbeddingStoreLoadError(EmbeddingStoreError):
    pass


class EmbeddingStoreSaveError(EmbeddingStoreError):
    pass


class MooncakeEmbeddingStoreClient:
    """Wraps Mooncake object and buffer APIs used by embedding transfer."""

    def __init__(self, store: Any, replicate_config: Any | None = None):
        self.store = store
        self.replicate_config = replicate_config

    def close(self) -> None:
        """Best-effort shutdown for Mooncake store implementations."""
        for method_name in ("close", "teardown", "disconnect", "finalize"):
            close_fn = getattr(self.store, method_name, None)
            if close_fn is None:
                continue
            try:
                close_fn()
            except Exception:
                logger.warning(
                    "failed to close embedding Mooncake store with %s()",
                    method_name,
                    exc_info=True,
                )
            return

    def exists(self, pool_key: EmbeddingPoolKey) -> bool:
        data_key = make_embedding_data_key(pool_key)
        states = self.store.batch_is_exist([data_key])
        return len(states) == 1 and states[0] == 1

    def batch_exists(self, pool_keys: list[EmbeddingPoolKey]) -> list[bool]:
        if not pool_keys:
            return []

        keys = [make_embedding_data_key(pool_key) for pool_key in pool_keys]
        states = self.store.batch_is_exist(keys)
        return [state == 1 for state in states]

    def get_tensor_meta(self, pool_key: EmbeddingPoolKey) -> TensorMeta | None:
        metadata = self._read_range(
            pool_key,
            src_offset=0,
            size=MOONCAKE_TENSOR_METADATA_NBYTES,
        )
        if metadata is None:
            return None
        try:
            return _decode_mooncake_tensor_metadata(pool_key, metadata)
        except EmbeddingStoreLoadError:
            logger.exception(
                "failed to decode embedding Mooncake tensor metadata for %s",
                pool_key.to_string(),
            )
            return None

    def get_tensor_metas(
        self,
        pool_keys: list[EmbeddingPoolKey],
    ) -> list[TensorMeta | None]:
        """Batch read tensor metadata for multiple pool keys.

        Reads all metadata headers into one contiguous host buffer with a
        single ``get_into_ranges`` call, then decodes them individually.
        A failed read or decode yields ``None`` for that key (matching the
        single-key ``get_tensor_meta`` semantics).
        """
        if not pool_keys:
            return []

        keys = [make_embedding_data_key(pool_key) for pool_key in pool_keys]
        num_keys = len(pool_keys)
        metadata_nbytes = MOONCAKE_TENSOR_METADATA_NBYTES
        buffer = (ctypes.c_ubyte * (num_keys * metadata_nbytes))()
        buffer_ptr = ctypes.addressof(buffer)
        self.register_tensor(buffer_ptr, len(buffer))
        try:
            results = self.store.get_into_ranges(
                [buffer_ptr],
                [keys],
                [[[i * metadata_nbytes for i in range(num_keys)]]],
                [[[0] * num_keys]],
                [[[metadata_nbytes] * num_keys]],
            )
        finally:
            self.unregister_tensor(buffer_ptr)

        tensor_metas: list[TensorMeta | None] = []
        for index, pool_key in enumerate(pool_keys):
            if _range_result(results, 0, index) != metadata_nbytes:
                tensor_metas.append(None)
                continue
            start = index * metadata_nbytes
            metadata = bytes(buffer[start:start + metadata_nbytes])
            try:
                tensor_metas.append(
                    _decode_mooncake_tensor_metadata(pool_key, metadata)
                )
            except EmbeddingStoreLoadError:
                logger.exception(
                    "failed to decode embedding Mooncake tensor metadata for %s",
                    pool_key.to_string(),
                )
                tensor_metas.append(None)
        return tensor_metas

    def put_tensor(
        self,
        pool_key: EmbeddingPoolKey,
        tensor: Any,
        *,
        with_soft_pin: bool = False,
    ) -> None:
        results = self.put_tensors(
            [pool_key],
            [tensor],
            with_soft_pin=with_soft_pin,
        )
        if not results[0]:
            raise EmbeddingStoreSaveError(
                f"failed to put embedding tensor for {pool_key.to_string()}"
            )

    def put_tensors(
        self,
        pool_keys: list[EmbeddingPoolKey],
        tensors: list[Any],
        *,
        with_soft_pin: bool = False,
    ) -> list[bool]:
        if len(pool_keys) != len(tensors):
            raise ValueError("pool_keys and tensors must have the same length")
        if not pool_keys:
            return []
        for tensor in tensors:
            _validate_supported_embedding_tensor_dtype(tensor)

        replicate_config = _make_embedding_replicate_config(
            self.replicate_config,
            with_soft_pin=with_soft_pin,
        )
        batch_put_from_multi_buffers = getattr(
            self.store,
            "batch_put_from_multi_buffers",
            None,
        )
        if batch_put_from_multi_buffers is not None:
            return self._put_tensors_from_buffers(
                pool_keys,
                tensors,
                replicate_config=replicate_config,
            )

        results = []
        for pool_key, tensor in zip(pool_keys, tensors, strict=True):
            key = make_embedding_data_key(pool_key)
            if replicate_config is None:
                put_fn = getattr(self.store, "put_tensor", None)
                if put_fn is None:
                    raise EmbeddingStoreSaveError(
                        "Mooncake Embedding Store requires put_tensor or pub_tensor "
                        "support for single-object embedding tensors."
                    )
                ret = put_fn(key, tensor)
            else:
                put_fn = getattr(self.store, "pub_tensor", None)
                if put_fn is None:
                    raise EmbeddingStoreSaveError(
                        "Mooncake Embedding Store requires pub_tensor support when "
                        "a ReplicateConfig is configured."
                    )
                ret = put_fn(key, tensor, replicate_config)
            results.append(ret == 0)
        return results

    def _put_tensors_from_buffers(
        self,
        pool_keys: list[EmbeddingPoolKey],
        tensors: list[Any],
        *,
        replicate_config: Any | None,
    ) -> list[bool]:
        keys = []
        buffer_ptrs = []
        buffer_sizes = []
        metadata_buffers = []
        for pool_key, tensor in zip(pool_keys, tensors, strict=True):
            if not tensor.is_contiguous():
                raise EmbeddingStoreSaveError(
                    "embedding tensor must be contiguous before batch buffer put"
                )
            data_size = tensor.numel() * tensor.element_size()
            metadata = _encode_mooncake_tensor_metadata(tensor)
            metadata_buffer = (ctypes.c_ubyte * len(metadata)).from_buffer_copy(
                metadata
            )
            metadata_buffers.append(metadata_buffer)
            metadata_ptr = ctypes.addressof(metadata_buffer)
            keys.append(make_embedding_data_key(pool_key))
            buffer_ptrs.append([metadata_ptr, tensor.data_ptr()])
            buffer_sizes.append([len(metadata), data_size])

        registered_addrs: list[int] = []
        registered_addr_set: set[int] = set()
        try:
            for ptrs, sizes in zip(buffer_ptrs, buffer_sizes, strict=True):
                for offset in (1, 0):
                    addr = ptrs[offset]
                    size = sizes[offset]
                    if addr in registered_addr_set:
                        continue
                    self.register_tensor(addr, size)
                    registered_addrs.append(addr)
                    registered_addr_set.add(addr)
            results = self.store.batch_put_from_multi_buffers(
                keys,
                buffer_ptrs,
                buffer_sizes,
                replicate_config,
            )
            if len(results) != len(keys):
                raise EmbeddingStoreSaveError(
                    "Mooncake batch put returned an unexpected number of results: "
                    f"expected {len(keys)}, got {len(results)}"
                )
            return [result >= 0 for result in results]
        finally:
            for addr in reversed(registered_addrs):
                self.unregister_tensor(addr)

    def register_tensor(self, addr: int, size: int) -> None:
        ret = self.store.register_buffer(addr, size)
        if ret != 0:
            raise EmbeddingStoreError(
                f"failed to register embedding buffer addr={addr:#x} size={size}: {ret}"
            )

    def unregister_tensor(self, addr: int) -> None:
        unregister_fn = getattr(self.store, "unregister_buffer", None)
        if unregister_fn is None:
            return
        try:
            ret = unregister_fn(addr)
        except Exception:
            logger.warning(
                "failed to unregister embedding buffer addr=%#x",
                addr,
                exc_info=True,
            )
            return
        if ret != 0:
            logger.warning(
                "unregister embedding buffer failed addr=%#x ret=%s",
                addr,
                ret,
            )

    def get_tensor_payload(
        self,
        pool_key: EmbeddingPoolKey,
        addr: int,
        size: int,
        src_offset: int,
    ) -> int:
        self.register_tensor(addr, size)
        try:
            key = make_embedding_data_key(pool_key)
            results = self.store.get_into_ranges(
                [addr],
                [[key]],
                [[[0]]],
                [[[src_offset]]],
                [[[size]]],
            )
            result = _single_range_result(results)
            if result != size:
                raise EmbeddingStoreLoadError(
                    "failed to get embedding tensor payload for "
                    f"{pool_key.to_string()}: {result}"
                )
            return result
        finally:
            self.unregister_tensor(addr)

    def get_tensor_payloads(
        self,
        pool_keys: list[EmbeddingPoolKey],
        addrs: list[int],
        sizes: list[int],
        data_offsets: list[int],
    ) -> None:
        """Batch read tensor payloads for multiple pool keys.

        Registers all target buffers, issues a single ``get_into_ranges``
        call for every key, then unregisters. Raises ``EmbeddingStoreLoadError``
        naming the failed keys if any range does not return the expected size.
        """
        if not pool_keys:
            return

        for addr, size in zip(addrs, sizes, strict=True):
            self.register_tensor(addr, size)
        try:
            keys = [make_embedding_data_key(pool_key) for pool_key in pool_keys]
            results = self.store.get_into_ranges(
                addrs,
                [[key] for key in keys],
                [[[0]] for _ in keys],
                [[[offset]] for offset in data_offsets],
                [[[size]] for size in sizes],
            )
        finally:
            for addr in addrs:
                self.unregister_tensor(addr)

        failed: list[tuple[EmbeddingPoolKey, int, int]] = []
        for index, (pool_key, size) in enumerate(
            zip(pool_keys, sizes, strict=True)
        ):
            got = _range_result(results, index)
            if got != size:
                failed.append((pool_key, size, got))
        if failed:
            raise EmbeddingStoreLoadError(
                "failed to get embedding tensor payloads: "
                + "; ".join(
                    f"{pool_key.to_string()} expected={expected} got={got}"
                    for pool_key, expected, got in failed[:3]
                )
            )

    def _read_range(
        self,
        pool_key: EmbeddingPoolKey,
        *,
        src_offset: int,
        size: int,
    ) -> bytes | None:
        buffer = (ctypes.c_ubyte * size)()
        buffer_ptr = ctypes.addressof(buffer)
        self.register_tensor(buffer_ptr, size)
        key = make_embedding_data_key(pool_key)
        try:
            results = self.store.get_into_ranges(
                [buffer_ptr],
                [[key]],
                [[[0]]],
                [[[src_offset]]],
                [[[size]]],
            )
        finally:
            self.unregister_tensor(buffer_ptr)
        if _single_range_result(results) != size:
            return None
        return bytes(buffer)


def _single_range_result(results: Any) -> int:
    return _range_result(results, 0, 0, 0)


def _range_result(
    results: Any,
    buffer_index: int,
    key_index: int = 0,
    range_index: int = 0,
) -> int:
    try:
        return int(results[buffer_index][key_index][range_index])
    except Exception:
        return -1


def _decode_mooncake_tensor_metadata(
    pool_key: EmbeddingPoolKey,
    metadata: bytes,
) -> TensorMeta:
    if len(metadata) < MOONCAKE_TENSOR_METADATA_NBYTES:
        raise EmbeddingStoreLoadError(
            f"embedding tensor metadata is too small: {len(metadata)}"
        )
    (
        magic,
        version,
        header_size,
        dtype,
        ndim,
        _layout_kind,
        _reserved_flags,
        data_offset,
        data_bytes,
    ) = struct.unpack_from(_MOONCAKE_TENSOR_HEADER_FORMAT, metadata, 0)
    if (
        magic != _MOONCAKE_TENSOR_OBJECT_MAGIC
        or version != _MOONCAKE_TENSOR_OBJECT_VERSION
        or header_size != MOONCAKE_TENSOR_METADATA_NBYTES
    ):
        raise EmbeddingStoreLoadError(
            f"invalid Mooncake tensor metadata header for {pool_key.to_string()}"
        )
    if ndim < 0 or ndim > 8:
        raise EmbeddingStoreLoadError(
            f"invalid embedding tensor ndim for {pool_key.to_string()}: {ndim}"
        )
    if dtype not in _MOONCAKE_DTYPE_TO_TORCH_DTYPE:
        raise EmbeddingStoreLoadError(
            f"unsupported Mooncake tensor dtype for {pool_key.to_string()}: {dtype}"
        )
    local_shape = struct.unpack_from(
        "<8q",
        metadata,
        _MOONCAKE_TENSOR_LOCAL_SHAPE_OFFSET,
    )
    shape = tuple(int(dim) for dim in local_shape[:ndim])
    if any(dim < 0 for dim in shape):
        raise EmbeddingStoreLoadError(
            f"invalid embedding tensor shape for {pool_key.to_string()}: {shape}"
        )
    return TensorMeta(
        pool_key=pool_key,
        protocol_version=EMBEDDING_PROTOCOL_VERSION,
        layout=EMBEDDING_TENSOR_LAYOUT,
        shape=shape,
        dtype=_MOONCAKE_DTYPE_TO_TORCH_DTYPE[dtype],
        nbytes=int(data_bytes),
        device_type="cpu",
        data_offset=int(data_offset),
    )


def _encode_mooncake_tensor_metadata(tensor: Any) -> bytes:
    dtype = str(tensor.dtype)
    _validate_supported_embedding_tensor_dtype(tensor)
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) > 8:
        raise EmbeddingStoreSaveError(
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
        raise EmbeddingStoreSaveError(
            f"invalid Mooncake tensor metadata size: {len(metadata)}"
        )
    return metadata


def _validate_supported_embedding_tensor_dtype(tensor: Any) -> None:
    dtype = str(tensor.dtype)
    if dtype not in _SUPPORTED_EMBEDDING_TORCH_DTYPES:
        raise EmbeddingStoreSaveError(f"unsupported embedding tensor dtype: {dtype}")


def _make_embedding_replicate_config(
    replicate_config: Any | None,
    *,
    with_soft_pin: bool,
) -> Any | None:
    if replicate_config is None:
        return None

    config = _clone_replicate_config(replicate_config)
    embedding_data_type = _get_embedding_object_data_type()
    if embedding_data_type is not None and hasattr(config, "data_type"):
        config.data_type = embedding_data_type
    if hasattr(config, "with_soft_pin"):
        config.with_soft_pin = bool(config.with_soft_pin) or with_soft_pin
    return config


def _clone_replicate_config(replicate_config: Any) -> Any:
    try:
        return copy.copy(replicate_config)
    except Exception:
        config = type(replicate_config)()
        for attr in (
            "replica_num",
            "nof_replica_num",
            "with_soft_pin",
            "with_hard_pin",
            "preferred_segments",
            "preferred_nof_segments",
            "preferred_segment",
            "prefer_alloc_in_same_node",
            "data_type",
            "group_ids",
        ):
            if hasattr(replicate_config, attr) and hasattr(config, attr):
                setattr(config, attr, getattr(replicate_config, attr))
        return config


def _get_embedding_object_data_type() -> Any | None:
    try:
        from mooncake.store import ObjectDataType  # type: ignore
    except Exception:
        return None
    embedding_type = getattr(ObjectDataType, "EMBEDDING", None)
    if embedding_type is not None:
        return embedding_type
    return getattr(ObjectDataType, "TENSOR", None)
