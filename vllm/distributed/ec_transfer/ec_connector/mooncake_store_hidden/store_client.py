# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin Mooncake Store client for hidden-state objects."""

from __future__ import annotations

import copy
import ctypes
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_LAYOUT_VERSION,
    HIDDEN_PROTOCOL_VERSION,
    MOONCAKE_TENSOR_METADATA_NBYTES,
    HiddenPoolKey,
    TensorMeta,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.keys import (
    make_hidden_data_key,
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


@dataclass
class MooncakeHiddenStoreConfig:
    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: str = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE

    @staticmethod
    def from_file(file_path: str) -> MooncakeHiddenStoreConfig:
        with open(file_path, encoding="utf-8") as file:
            config = json.load(file)
        mode = config.get("mode", "embedded")
        return MooncakeHiddenStoreConfig(
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
    def load_from_env() -> MooncakeHiddenStoreConfig:
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeHiddenStoreConfig.from_file(config_path)


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


def create_mooncake_hidden_store_client() -> MooncakeHiddenStoreClient:
    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except ImportError as e:
        raise ImportError(
            "Please install mooncake to run vLLM with " "MooncakeStoreECConnector."
        ) from e

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils

    config = MooncakeHiddenStoreConfig.load_from_env()
    config.device_name = rdma_utils.get_configured_worker_rnic(
        protocol=config.protocol,
        configured_device=config.device_name,
    )

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
        "Initialized hidden Mooncake store mode=%s global_segment_size=%d "
        "local_buffer_size=%d",
        config.mode,
        config.global_segment_size,
        config.local_buffer_size,
    )
    return MooncakeHiddenStoreClient(store, replicate_config=ReplicateConfig())


class HiddenStoreError(RuntimeError):
    pass


class HiddenStoreLoadError(HiddenStoreError):
    pass


class HiddenStoreSaveError(HiddenStoreError):
    pass


class MooncakeHiddenStoreClient:
    """Wraps Mooncake object and buffer APIs used by hidden transfer."""

    def __init__(self, store: Any, replicate_config: Any | None = None):
        self.store = store
        self.replicate_config = replicate_config

    def exists(self, pool_key: HiddenPoolKey) -> bool:
        data_key = make_hidden_data_key(pool_key)
        states = self.store.batch_is_exist([data_key])
        return len(states) == 1 and states[0] == 1

    def batch_exists(self, pool_keys: list[HiddenPoolKey]) -> list[bool]:
        if not pool_keys:
            return []

        keys = [make_hidden_data_key(pool_key) for pool_key in pool_keys]
        states = self.store.batch_is_exist(keys)
        return [state == 1 for state in states]

    def get_tensor_meta(self, pool_key: HiddenPoolKey) -> TensorMeta | None:
        metadata = self._read_range(
            pool_key,
            src_offset=0,
            size=MOONCAKE_TENSOR_METADATA_NBYTES,
        )
        if metadata is None:
            return None
        try:
            return _decode_mooncake_tensor_metadata(pool_key, metadata)
        except HiddenStoreLoadError:
            logger.exception(
                "failed to decode hidden Mooncake tensor metadata for %s",
                pool_key.to_string(),
            )
            return None

    def put_tensor(
        self,
        pool_key: HiddenPoolKey,
        tensor: Any,
        *,
        with_soft_pin: bool = False,
    ) -> None:
        key = make_hidden_data_key(pool_key)
        replicate_config = _make_hidden_replicate_config(
            self.replicate_config,
            with_soft_pin=with_soft_pin,
        )
        batch_put_from_multi_buffers = getattr(
            self.store,
            "batch_put_from_multi_buffers",
            None,
        )
        if batch_put_from_multi_buffers is not None:
            self._put_tensor_from_buffers(
                pool_key,
                tensor,
                replicate_config=replicate_config,
            )
            return

        if replicate_config is None:
            put_fn = getattr(self.store, "put_tensor", None)
            if put_fn is None:
                raise HiddenStoreSaveError(
                    "Mooncake Hidden Store requires put_tensor or pub_tensor "
                    "support for single-object hidden tensors."
                )
            ret = put_fn(key, tensor)
        else:
            put_fn = getattr(self.store, "pub_tensor", None)
            if put_fn is None:
                raise HiddenStoreSaveError(
                    "Mooncake Hidden Store requires pub_tensor support when "
                    "a ReplicateConfig is configured."
                )
            ret = put_fn(key, tensor, replicate_config)
        if ret != 0:
            raise HiddenStoreSaveError(
                f"failed to put hidden tensor for {pool_key.to_string()}: {ret}"
            )

    def _put_tensor_from_buffers(
        self,
        pool_key: HiddenPoolKey,
        tensor: Any,
        *,
        replicate_config: Any | None,
    ) -> None:
        if not tensor.is_contiguous():
            raise HiddenStoreSaveError(
                "hidden tensor must be contiguous before batch buffer put"
            )
        data_size = tensor.numel() * tensor.element_size()
        metadata = _encode_mooncake_tensor_metadata(tensor)
        metadata_buffer = (ctypes.c_ubyte * len(metadata)).from_buffer_copy(metadata)
        metadata_ptr = ctypes.addressof(metadata_buffer)
        self.register_tensor(tensor.data_ptr(), data_size)
        self.register_tensor(metadata_ptr, len(metadata))

        key = make_hidden_data_key(pool_key)
        results = self.store.batch_put_from_multi_buffers(
            [key],
            [[metadata_ptr, tensor.data_ptr()]],
            [[len(metadata), data_size]],
            replicate_config,
        )
        unregister_fn = getattr(self.store, "unregister_buffer", None)
        if unregister_fn is not None:
            unregister_fn(metadata_ptr)
        failed = [result for result in results if result < 0]
        if failed:
            raise HiddenStoreSaveError(
                "failed to put hidden tensor for " f"{pool_key.to_string()}: {failed}"
            )

    def register_tensor(self, addr: int, size: int) -> None:
        ret = self.store.register_buffer(addr, size)
        if ret != 0:
            raise HiddenStoreError(
                f"failed to register hidden buffer addr={addr:#x} size={size}: {ret}"
            )

    def get_tensor_payload(
        self,
        pool_key: HiddenPoolKey,
        addr: int,
        size: int,
        src_offset: int,
    ) -> int:
        self.register_tensor(addr, size)
        key = make_hidden_data_key(pool_key)
        results = self.store.get_into_ranges(
            [addr],
            [[key]],
            [[[0]]],
            [[[src_offset]]],
            [[[size]]],
        )
        result = _single_range_result(results)
        if result != size:
            raise HiddenStoreLoadError(
                "failed to get hidden tensor payload for "
                f"{pool_key.to_string()}: {result}"
            )
        return result

    def _read_range(
        self,
        pool_key: HiddenPoolKey,
        *,
        src_offset: int,
        size: int,
    ) -> bytes | None:
        buffer = (ctypes.c_ubyte * size)()
        buffer_ptr = ctypes.addressof(buffer)
        self.register_tensor(buffer_ptr, size)
        key = make_hidden_data_key(pool_key)
        try:
            results = self.store.get_into_ranges(
                [buffer_ptr],
                [[key]],
                [[[0]]],
                [[[src_offset]]],
                [[[size]]],
            )
        finally:
            unregister_fn = getattr(self.store, "unregister_buffer", None)
            if unregister_fn is not None:
                unregister_fn(buffer_ptr)
        if _single_range_result(results) != size:
            return None
        return bytes(buffer)


def _single_range_result(results: Any) -> int:
    try:
        return int(results[0][0][0])
    except Exception:
        return -1


def _decode_mooncake_tensor_metadata(
    pool_key: HiddenPoolKey,
    metadata: bytes,
) -> TensorMeta:
    if len(metadata) < MOONCAKE_TENSOR_METADATA_NBYTES:
        raise HiddenStoreLoadError(
            f"hidden tensor metadata is too small: {len(metadata)}"
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
        raise HiddenStoreLoadError(
            "invalid Mooncake tensor metadata header for " f"{pool_key.to_string()}"
        )
    if ndim < 0 or ndim > 8:
        raise HiddenStoreLoadError(
            f"invalid hidden tensor ndim for {pool_key.to_string()}: {ndim}"
        )
    if dtype not in _MOONCAKE_DTYPE_TO_TORCH_DTYPE:
        raise HiddenStoreLoadError(
            f"unsupported Mooncake tensor dtype for {pool_key.to_string()}: {dtype}"
        )
    local_shape = struct.unpack_from(
        "<8q",
        metadata,
        _MOONCAKE_TENSOR_LOCAL_SHAPE_OFFSET,
    )
    shape = tuple(int(dim) for dim in local_shape[:ndim])
    if any(dim < 0 for dim in shape):
        raise HiddenStoreLoadError(
            f"invalid hidden tensor shape for {pool_key.to_string()}: {shape}"
        )
    return TensorMeta(
        pool_key=pool_key,
        protocol_version=HIDDEN_PROTOCOL_VERSION,
        layout=HIDDEN_LAYOUT_VERSION,
        shape=shape,
        dtype=_MOONCAKE_DTYPE_TO_TORCH_DTYPE[dtype],
        nbytes=int(data_bytes),
        device_type="cpu",
        data_offset=int(data_offset),
    )


def _encode_mooncake_tensor_metadata(tensor: Any) -> bytes:
    dtype = str(tensor.dtype)
    if dtype not in _TORCH_DTYPE_TO_MOONCAKE_DTYPE:
        raise HiddenStoreSaveError(f"unsupported hidden tensor dtype: {dtype}")
    shape = tuple(int(dim) for dim in tensor.shape)
    if len(shape) > 8:
        raise HiddenStoreSaveError(
            f"hidden tensor has too many dimensions: {len(shape)}"
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
        raise HiddenStoreSaveError(
            f"invalid Mooncake tensor metadata size: {len(metadata)}"
        )
    return metadata


def _make_hidden_replicate_config(
    replicate_config: Any | None,
    *,
    with_soft_pin: bool,
) -> Any | None:
    if replicate_config is None:
        return None

    config = _clone_replicate_config(replicate_config)
    hidden_state_data_type = _get_hidden_state_object_data_type()
    if hidden_state_data_type is not None and hasattr(config, "data_type"):
        config.data_type = hidden_state_data_type
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


def _get_hidden_state_object_data_type() -> Any | None:
    try:
        from mooncake.store import ObjectDataType  # type: ignore
    except Exception:
        return None
    hidden_state_type = getattr(ObjectDataType, "HIDDEN_STATE", None)
    if hidden_state_type is not None:
        return hidden_state_type
    return getattr(ObjectDataType, "TENSOR", None)
