# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared object stores that deliver auxiliary outputs to other processes."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from safetensors.numpy import load

from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


class MooncakeAuxObjectStore:
    """Immutable, hard-pinned objects in a standalone Mooncake store. Writing
    a key that exists succeeds; one of the objects written under a key is
    kept, never a mix."""

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(cls) -> MooncakeAuxObjectStore:
        """Connect with the configuration at `MOONCAKE_CONFIG_PATH`."""
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
            rdma_utils,
        )
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        config = MooncakeStoreConfig.load_from_config()
        if config.mode != "standalone-store":
            raise ValueError(
                "Auxiliary-output objects must outlive the clients that write "
                "and read them: set mode to 'standalone-store' in the Mooncake "
                "configuration."
            )
        store = MooncakeDistributedStore()
        setup_mooncake_store(
            store, config, rdma_utils.get_requester_local_hostname(get_ip())
        )
        replicate_config = ReplicateConfig()
        replicate_config.with_hard_pin = True
        return cls(store, replicate_config)

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        """Write every object, or raise `BlockObjectStoreError`."""
        code = self._store.put_batch(list(keys), list(values), self._replicate_config)
        if code:
            raise BlockObjectStoreError(
                f"Mooncake failed to write {len(keys)} auxiliary-output "
                f"objects from {keys[0]!r}: {code}"
            )

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""
        return self._store.get(key) or None

    def close(self) -> None:
        self._store.close()


def aux_object_names(keys: Sequence[str], pipeline_parallel_size: int = 1) -> list[str]:
    """Names of the stored objects of the keys a response lists: one per
    pipeline-parallel stage."""
    if pipeline_parallel_size == 1:
        return list(keys)
    return [
        f"{key}/pp{stage}" for key in keys for stage in range(pipeline_parallel_size)
    ]


def aux_session_pattern(key_prefix: str, session: str) -> str:
    """Pattern for `remove_by_regex` that matches every object of a session."""
    return "^" + re.escape(f"{key_prefix}/{session}/")


def read_aux_rows(
    store: MooncakeAuxObjectStore,
    keys: Sequence[str],
    field: str,
    pipeline_parallel_size: int = 1,
    timeout_s: float = 60.0,
) -> np.ndarray:
    """Concatenate `field` of the objects `keys`; wait up to `timeout_s` in
    total for objects that are still being written. With pipeline
    parallelism each key names one object per stage, whose layers are
    concatenated in stage order; a field that only some stages hold, such as
    the hidden states of the last stage, is read from those."""
    deadline = time.monotonic() + timeout_s
    stages = []
    for name in aux_object_names(keys, pipeline_parallel_size):
        while (data := store.get(name)) is None:
            if time.monotonic() >= deadline:
                raise BlockObjectStoreError(
                    f"auxiliary-output object {name!r} is missing after {timeout_s}s"
                )
            time.sleep(0.05)
        stages.append(load(data).get(field))
    parts = []
    for key, start in zip(keys, range(0, len(stages), pipeline_parallel_size)):
        held = [
            rows
            for rows in stages[start : start + pipeline_parallel_size]
            if rows is not None
        ]
        if not held:
            raise BlockObjectStoreError(
                f"auxiliary-output object {key!r} has no field {field!r}"
            )
        parts.append(np.concatenate(held, axis=1))
    return np.concatenate(parts) if parts else np.empty(0)
