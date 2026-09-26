# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable auxiliary-output blocks in a deployment-managed Mooncake Store."""

from collections.abc import Iterable
from itertools import groupby
from math import prod
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.v1.request import Request

from vllm.distributed.aux_output_connector.store import (
    BlockObject,
    BlockObjectStoreError,
)


class MooncakeBlockObjectStore:
    """Use native owned-buffer I/O; publication runs on the shared writer thread.

    Retention belongs to the deployment, not individual vLLM requests. Missing
    objects fail closed; releasing a request never deletes external objects.
    """

    def __init__(self, store: Any, *, object_nbytes: int, max_batch_bytes: int) -> None:
        if not 0 < object_nbytes <= max_batch_bytes:
            raise ValueError("Mooncake batch must fit at least one auxiliary block")
        self._store = store
        self.object_nbytes = object_nbytes
        self._batch_size = max_batch_bytes // object_nbytes

    def put(
        self,
        objects: list[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        unique = {obj.key: obj.payload for obj in objects}
        keys = list(unique)
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            values = [unique[key] for key in batch_keys]
            if any(not 0 < len(value) <= self.object_nbytes for value in values):
                raise ValueError("Mooncake object exceeds auxiliary block capacity")
            result = self._store.put_batch(batch_keys, values)
            if result != 0:
                raise BlockObjectStoreError(f"Mooncake publication failed: {result}")

    def get_concatenated(self, keys: list[str]) -> bytes:
        chunks = []
        for start in range(0, len(keys), self._batch_size):
            batch_keys = keys[start : start + self._batch_size]
            buffers = self._store.batch_get_buffer(batch_keys)
            for key, buffer in zip(batch_keys, buffers, strict=True):
                if buffer is None:
                    raise BlockObjectStoreError(f"Mooncake object unavailable: {key}")
                chunks.append(bytes(buffer))
            # Release native handles before the next batch uses the local pool.
            del buffer, buffers
        return b"".join(chunks)

    def close(self) -> None:
        self._store.close()


class MooncakeOutputPublisher:
    """Finalize accepted boundary bytes without reading complete output blocks."""

    def __init__(self, output: "AuxRequestOutput", block_size: int) -> None:
        self._block_size = block_size
        self._row_nbytes = prod(output.rows.shape[1:]) * output.rows.dtype.itemsize
        self._store = create_mooncake_block_store(
            object_nbytes=block_size * self._row_nbytes
        )
        self._chunks: dict[str, list[str | bytes]] = {}

    def take_output(
        self, request: "Request", output: "AuxRequestOutput"
    ) -> list[str] | None:
        start, end = output.token_start, request.num_tokens - 1
        if end < start:
            assert not request.is_finished()
            return None
        keys = output.block_keys
        assert keys is not None
        block_size = self._block_size
        stored_start = start // block_size * block_size
        stored_end = stored_start + len(keys) * block_size if keys else start
        assert end <= stored_end + len(output.rows)
        chunks = self._chunks.setdefault(request.request_id, [])
        for index, key in enumerate(keys):
            block_start = stored_start + index * block_size
            lo, hi = max(start, block_start), min(end, block_start + block_size)
            if hi <= lo:
                break
            if lo == block_start and hi == block_start + block_size:
                chunks.append(key)
            else:
                payload = self._store.get_concatenated([key])
                chunks.append(
                    payload[
                        (lo - block_start) * self._row_nbytes : (hi - block_start)
                        * self._row_nbytes
                    ]
                )
        if end > stored_end:
            chunks.append(output.rows[: end - stored_end].tobytes())
        # String stops are detected in the API process, which needs these keys
        # before it can finish the request and abort the EngineCore request.
        assert request.sampling_params is not None
        if not request.is_finished() and not request.sampling_params.stop:
            return None
        del self._chunks[request.request_id]
        result: list[str] = []
        objects = []
        for is_key, group in groupby(chunks, lambda chunk: isinstance(chunk, str)):
            if is_key:
                result.extend(cast(Iterable[str], group))
            else:
                payload = b"".join(cast(Iterable[bytes], group))
                object_nbytes = self._store.object_nbytes
                for offset in range(0, len(payload), object_nbytes):
                    key = f"r3-tail:{uuid4().hex}"
                    result.append(key)
                    objects.append(
                        BlockObject(key, payload[offset : offset + object_nbytes])
                    )
        self._store.put(objects)
        return result

    def discard(self, request_id: str) -> None:
        self._chunks.pop(request_id, None)

    def close(self) -> None:
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
