# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write auxiliary outputs of session requests to a shared object store."""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping, Sequence

import numpy as np

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStore,
    AuxObjectStoreError,
)
from vllm.distributed.aux_output_connector.objects import encode_aux_object
from vllm.logger import init_logger

logger = init_logger(__name__)

ROUTED_EXPERTS_FIELD = "routed_experts"


def aux_object_key(
    key_prefix: str, session: str, block_hash_hex: str, identity: str
) -> str:
    """Key of the object that ends with the hash block `block_hash_hex` and
    whose rows come from the KV computation `identity`."""
    return f"{key_prefix}/{session}/{block_hash_hex}@{identity}"


class SessionObjectWriter:
    """Assemble full objects from hash blocks and write them in the background.

    An object holds `object_size` positions and is named by the hash of its
    last hash block and the identity of the KV computation that produced it,
    which the scheduler assigns. Each key is written once. A failed write is
    logged; readers of its keys then time out.
    """

    def __init__(
        self,
        store: AuxObjectStore,
        *,
        key_prefix: str,
        object_size: int,
        hash_block_size: int,
        max_pending_batches: int,
    ) -> None:
        if object_size % hash_block_size:
            raise ValueError(
                f"object size {object_size} is not a multiple of the hash block "
                f"size {hash_block_size}"
            )
        self._store = store
        self._key_prefix = key_prefix
        self.object_size = object_size
        self._hash_block_size = hash_block_size
        self._pieces_per_object = object_size // hash_block_size
        self._staged: dict[str, dict[int, dict[int, np.ndarray]]] = {}
        self._written: dict[str, set[str]] = {}
        self._queue: queue.Queue[tuple[list[str], list[bytes]] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def _run(self) -> None:
        while (batch := self._queue.get()) is not None:
            try:
                self._store.put_batch(*batch)
            except AuxObjectStoreError:
                logger.exception(
                    "Failed to write %d auxiliary output objects; reading their "
                    "keys will time out.",
                    len(batch[0]),
                )
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def add_blocks(
        self,
        request_id: str,
        session: str,
        block_hash_hexes: Sequence[str],
        blocks: Sequence[tuple[int, np.ndarray]],
        object_ids: Mapping[int, str],
    ) -> None:
        """Stage computed hash blocks; write the objects they complete.

        `blocks` are `(start, rows)` pairs of full hash blocks,
        `block_hash_hexes` are the hash-block hashes of the request so far and
        `object_ids` are the identities of its objects by index. An object
        without an identity is never listed, so it is not written.
        """
        staged = self._staged.setdefault(request_id, {})
        written = self._written.setdefault(session, set())
        keys: list[str] = []
        values: list[bytes] = []
        for start, rows in blocks:
            index, offset = divmod(start, self.object_size)
            pieces = staged.setdefault(index, {})
            pieces[offset // self._hash_block_size] = np.array(rows)
            if len(pieces) < self._pieces_per_object:
                continue
            del staged[index]
            if index not in object_ids:
                continue
            last = (index + 1) * self._pieces_per_object - 1
            key = aux_object_key(
                self._key_prefix, session, block_hash_hexes[last], object_ids[index]
            )
            if key in written:
                continue
            written.add(key)
            keys.append(key)
            values.append(
                encode_aux_object(
                    {
                        ROUTED_EXPERTS_FIELD: np.concatenate(
                            [pieces[i] for i in range(self._pieces_per_object)]
                        )
                    }
                )
            )
        if keys:
            self._queue.put((keys, values))

    def discard(self, request_id: str) -> None:
        self._staged.pop(request_id, None)

    def forget_session(self, session: str) -> None:
        """Drop what was written for `session`; later blocks are written again."""
        self._written.pop(session, None)

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        self._store.close()
