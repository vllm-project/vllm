# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable auxiliary-output blocks in a deployment-managed Mooncake Store."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from math import prod
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast
from uuid import uuid4

import msgspec
import numpy as np
from numpy.typing import NDArray
from safetensors import SafetensorError
from safetensors.numpy import load, save
from typing_extensions import Buffer

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.v1.request import Request

from vllm.distributed.aux_output_connector.store import (
    BlockObject,
    BlockObjectStoreError,
)

RoutingArray: TypeAlias = NDArray[np.uint8] | NDArray[np.uint16]
RoutingArrayDtype: TypeAlias = np.dtype[np.uint8] | np.dtype[np.uint16]

MAX_OBJECT_BYTES = 64 * 1024**2
# Native batches are sized from raw R3 bytes; reserve room for the variable
# Safetensors header, including metadata for a uint64 token position.
R3_HEADER_RESERVE = 256
OUTPUT_KEY_PREFIX = "vllm-aux/v1/output"


class _Metadata(msgspec.Struct):
    version: str
    token_start: str


class _Header(msgspec.Struct):
    metadata: _Metadata = msgspec.field(name="__metadata__")


@dataclass(frozen=True)
class AuxBlock:
    token_start: int
    routed_experts: RoutingArray

    @property
    def num_rows(self) -> int:
        return len(self.routed_experts)

    @property
    def fields(self) -> dict[str, RoutingArray]:
        return {"routed_experts": self.routed_experts}

    def _validate(self) -> None:
        rows = self.routed_experts
        if (
            rows.ndim != 3
            or rows.dtype.kind != "u"
            or rows.dtype.itemsize not in (1, 2)
        ):
            raise ValueError("Routed experts must be a uint8 or uint16 [T,L,E] array")
        if not all(rows.shape):
            raise ValueError("Routed-experts dimensions must be positive")
        if not 0 <= self.token_start < self.token_start + len(rows) < 2**64:
            raise ValueError("Invalid auxiliary token range")

    def encode(self) -> bytes:
        self._validate()
        if self.routed_experts.nbytes + R3_HEADER_RESERVE > MAX_OBJECT_BYTES:
            raise ValueError("Auxiliary object exceeds size limit")
        rows = np.ascontiguousarray(
            self.routed_experts, dtype=self.routed_experts.dtype.newbyteorder("<")
        )
        return save(
            {"routed_experts": rows},
            metadata={"version": "1", "token_start": str(self.token_start)},
        )

    @classmethod
    def decode(cls, raw: bytes) -> "AuxBlock":
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError("Auxiliary object exceeds size limit")
        try:
            tensors = load(raw)
            header_size = int.from_bytes(raw[:8], "little")
            header = msgspec.json.decode(raw[8 : 8 + header_size], type=_Header)
        except (SafetensorError, msgspec.DecodeError) as error:
            raise ValueError("Invalid auxiliary Safetensors object") from error
        if header.metadata.version != "1":
            raise ValueError("Unsupported auxiliary schema version")
        if tensors.keys() != {"routed_experts"}:
            raise ValueError(
                "Unsupported auxiliary fields; only routed experts are enabled"
            )
        rows = cast(RoutingArray, tensors["routed_experts"])
        block = cls(int(header.metadata.token_start), rows)
        block._validate()
        rows.flags.writeable = False
        return block


class MooncakeClient(Protocol):
    def put_batch(self, keys: list[str], values: list[bytes]) -> int: ...
    def batch_get_buffer(self, keys: list[str]) -> Sequence[Buffer | None]: ...
    def close(self) -> None: ...


class MooncakeBlockObjectStore:
    """Publish immutable objects; retention belongs to the deployment."""

    def __init__(
        self, store: MooncakeClient, *, object_nbytes: int, max_batch_bytes: int
    ) -> None:
        encoded_nbytes = R3_HEADER_RESERVE + object_nbytes
        if object_nbytes <= 0 or encoded_nbytes > min(
            max_batch_bytes, MAX_OBJECT_BYTES
        ):
            raise ValueError(
                "Mooncake batch must fit at least one encoded auxiliary block"
            )
        self._store = store
        self.object_nbytes = object_nbytes
        self._encoded_nbytes = encoded_nbytes
        self._batch_size = max_batch_bytes // encoded_nbytes

    def put(
        self,
        objects: Sequence[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        unique = {obj.key: obj.payload for obj in objects}
        for payload in unique.values():
            if len(payload) > self._encoded_nbytes:
                raise ValueError("Mooncake object exceeds auxiliary block capacity")
        keys = list(unique)
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            result = self._store.put_batch(
                batch_keys, [unique[key] for key in batch_keys]
            )
            if result != 0:
                raise BlockObjectStoreError(f"Mooncake publication failed: {result}")

    def get_concatenated(
        self,
        keys: list[str],
        *,
        shape_per_token: tuple[int, ...] | None = None,
        dtype: RoutingArrayDtype | None = None,
        token_start: int | None = None,
    ) -> bytes:
        chunks: list[bytes] = []
        cursor = token_start
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            buffers = self._store.batch_get_buffer(batch_keys)
            for key, buffer in zip(batch_keys, buffers, strict=True):
                if buffer is None:
                    raise BlockObjectStoreError(f"Mooncake object unavailable: {key}")
                view = memoryview(buffer)
                if view.nbytes > self._encoded_nbytes:
                    raise ValueError("Mooncake object exceeds auxiliary block capacity")
                block = AuxBlock.decode(bytes(view))
                del view
                rows = block.routed_experts
                if shape_per_token is None:
                    shape_per_token = rows.shape[1:]
                if dtype is None:
                    dtype = rows.dtype
                if rows.shape[1:] != shape_per_token or rows.dtype != dtype:
                    raise ValueError("Mooncake routing profile does not match")
                if cursor is not None and block.token_start != cursor:
                    raise ValueError(
                        "Mooncake auxiliary token ranges are not contiguous"
                    )
                cursor = block.token_start + block.num_rows
                chunks.append(rows.tobytes())
            del buffer, buffers
        return b"".join(chunks)

    def close(self) -> None:
        self._store.close()


@dataclass(frozen=True)
class _RoutingFragment:
    token_start: int
    payload: bytes


@dataclass
class _RequestOutput:
    chunks: list[str | _RoutingFragment] = field(default_factory=list)
    completion_id: str | None = None


class MooncakeOutputPublisher:
    """Finalize accepted boundaries without reading complete output blocks."""

    def __init__(self, output: "AuxRequestOutput", block_size: int) -> None:
        self._block_size = block_size
        self._shape = output.rows.shape[1:]
        self._dtype = output.rows.dtype
        self._row_nbytes = prod(self._shape) * self._dtype.itemsize
        self._store = create_mooncake_block_store(
            object_nbytes=block_size * self._row_nbytes
        )
        self._requests: dict[str, _RequestOutput] = {}

    def take_output(
        self, request: "Request", output: "AuxRequestOutput"
    ) -> list[str] | None:
        start, end = int(output.token_start), request.num_tokens - 1
        if end < start:
            assert not request.is_finished()
            return None
        keys = output.block_keys
        assert keys is not None
        block_size = self._block_size
        stored_start = start // block_size * block_size
        stored_end = stored_start + len(keys) * block_size if keys else start
        assert end <= stored_end + len(output.rows)
        pending = self._requests.setdefault(request.request_id, _RequestOutput())
        chunks = pending.chunks
        for index, key in enumerate(keys):
            block_start = stored_start + index * block_size
            lo, hi = max(start, block_start), min(end, block_start + block_size)
            if hi <= lo:
                break
            if lo == block_start and hi == block_start + block_size:
                chunks.append(key)
            else:
                payload = self._store.get_concatenated(
                    [key],
                    shape_per_token=self._shape,
                    dtype=self._dtype,
                    token_start=block_start,
                )
                chunks.append(
                    _RoutingFragment(
                        lo,
                        payload[
                            (lo - block_start) * self._row_nbytes : (hi - block_start)
                            * self._row_nbytes
                        ],
                    )
                )
        if end > stored_end:
            chunks.append(
                _RoutingFragment(stored_end, output.rows[: end - stored_end].tobytes())
            )
        # String stops need keys before the API process aborts the core request.
        assert request.sampling_params is not None
        if not request.is_finished() and not request.sampling_params.stop:
            return None
        pending.chunks = []
        if pending.completion_id is None:
            pending.completion_id = uuid4().hex
        result: list[str] = []
        objects: list[BlockObject] = []
        for is_key, group in groupby(chunks, lambda chunk: isinstance(chunk, str)):
            if is_key:
                result.extend(cast(Iterable[str], group))
            else:
                encoded = self._encode_fragments(
                    list(cast(Iterable[_RoutingFragment], group)),
                    pending.completion_id,
                )
                objects.extend(encoded)
                result.extend(obj.key for obj in encoded)
        self._store.put(objects)
        if request.is_finished():
            self.discard(request.request_id)
        return result

    def _encode_fragments(
        self, fragments: list[_RoutingFragment], completion_id: str
    ) -> list[BlockObject]:
        position = fragments[0].token_start
        payload = b"".join(fragment.payload for fragment in fragments)
        rows = cast(
            RoutingArray,
            np.frombuffer(payload, dtype=self._dtype).reshape(-1, *self._shape),
        )
        objects: list[BlockObject] = []
        offset = 0
        while offset < len(rows):
            count = min(
                self._block_size - position % self._block_size, len(rows) - offset
            )
            key = f"{OUTPUT_KEY_PREFIX}/{completion_id}/{position}"
            objects.append(
                BlockObject(
                    key, AuxBlock(position, rows[offset : offset + count]).encode()
                )
            )
            offset += count
            position += count
        return objects

    def discard(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close(self) -> None:
        self._requests.clear()
        self._store.close()


def create_mooncake_block_store(
    *, object_nbytes: int, max_batch_bytes: int = 64 * 1024**2
) -> MooncakeBlockObjectStore:
    from mooncake.store import (
        MooncakeDistributedStore,  # type: ignore[import-not-found]
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
    from vllm.distributed.mooncake_store import (
        MooncakeStoreConfig,
        setup_mooncake_store,
    )
    from vllm.utils.network_utils import get_ip

    # Pool capacity is user-sized, not derived from the GPU KV cache. Each
    # embedded client contributes its configured segment, including the
    # EngineCore publisher.
    config = MooncakeStoreConfig.load_from_config(
        "VLLM_AUX_OUTPUT_MOONCAKE_CONFIG_PATH"
    )
    store = MooncakeDistributedStore()
    try:
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        return MooncakeBlockObjectStore(
            store,
            object_nbytes=object_nbytes,
            max_batch_bytes=min(max_batch_bytes, config.local_buffer_size),
        )
    except Exception:
        store.close()
        raise
