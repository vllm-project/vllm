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

_PUT_NO_MEMORY = -200


class AuxObjectStoreFullError(BlockObjectStoreError):
    """The store had no memory for the objects; a later write may fit."""


class MooncakeAuxObjectStore:
    """Immutable objects in a standalone Mooncake store. Writing a key that
    exists succeeds; one of the objects written under a key is kept, never a
    mix."""

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls, config_path: str | None = None, hard_pin: bool = False
    ) -> MooncakeAuxObjectStore:
        """Connect with the Mooncake configuration file `config_path`, or the
        one at `MOONCAKE_CONFIG_PATH` if it is `None`. Objects are written
        hard-pinned if `hard_pin`."""
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
            rdma_utils,
        )
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        config = (
            MooncakeStoreConfig.load_from_config()
            if config_path is None
            else MooncakeStoreConfig.from_file(config_path)
        )
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
        replicate_config.with_hard_pin = hard_pin
        return cls(store, replicate_config)

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        """Write every object, or raise `BlockObjectStoreError`, which is
        `AuxObjectStoreFullError` if the store has no memory for them."""
        code = self._store.put_batch(list(keys), list(values), self._replicate_config)
        if not code:
            return
        error = (
            AuxObjectStoreFullError if code == _PUT_NO_MEMORY else BlockObjectStoreError
        )
        raise error(
            f"Mooncake failed to write {len(keys)} auxiliary-output "
            f"objects from {keys[0]!r}: {code}"
        )

    def get(self, key: str) -> bytes | None:
        """Return the object, or `None` if it does not exist."""
        return self._store.get(key) or None

    def close(self) -> None:
        self._store.close()


_REGEX_SYNTAX = re.compile(r"[\\^$.*+?()[\]{}|]")


def _escape(text: str) -> str:
    return _REGEX_SYNTAX.sub(r"\\\g<0>", text)


def aux_prefix_pattern(key_prefix: str) -> str:
    """Mooncake regex of every object under `key_prefix`."""
    return f"^{_escape(key_prefix)}/"


def aux_session_pattern(key_prefix: str, session: str) -> str:
    """Mooncake regex of every object of a session."""
    return f"^{_escape(key_prefix)}/{_escape(session)}/"


def read_aux_rows(
    store: MooncakeAuxObjectStore,
    keys: Sequence[str],
    field: str,
    timeout_s: float = 600.0,
) -> np.ndarray:
    """Concatenate `field` of the objects `keys`; wait up to `timeout_s` in
    total for objects that are still being written."""
    deadline = time.monotonic() + timeout_s
    parts = []
    for key in keys:
        while (data := store.get(key)) is None:
            if time.monotonic() >= deadline:
                raise BlockObjectStoreError(
                    f"auxiliary-output object {key!r} is missing after {timeout_s}s"
                )
            time.sleep(0.05)
        rows = load(data)
        if field not in rows:
            raise BlockObjectStoreError(
                f"auxiliary-output object {key!r} has no field {field!r}"
            )
        parts.append(rows[field])
    return np.concatenate(parts) if parts else np.empty(0)
