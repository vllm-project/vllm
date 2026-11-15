# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake backend of the auxiliary-output object store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import regex as re

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStore,
    AuxObjectStoreError,
)

if TYPE_CHECKING:
    from vllm.distributed.mooncake_store import MooncakeStoreConfig


class MooncakeAuxObjectStore(AuxObjectStore):
    """Objects in a Mooncake distributed store.

    Objects are hard-pinned: the store never evicts them, so a consumer can
    read every key it was given until it removes them. A full pool fails the
    write instead of dropping objects that were already delivered.
    """

    def __init__(self, store: Any, replicate_config: Any) -> None:
        self._store = store
        self._replicate_config = replicate_config

    @classmethod
    def create(
        cls,
        config: MooncakeStoreConfig | None = None,
        local_hostname: str | None = None,
    ) -> MooncakeAuxObjectStore:
        """Connect with `config`, by default read from `MOONCAKE_CONFIG_PATH`."""
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/"
                "en/build.md to deliver auxiliary outputs through Mooncake."
            ) from e
        from vllm.distributed.mooncake_store import (
            MooncakeStoreConfig,
            setup_mooncake_store,
        )
        from vllm.utils.network_utils import get_ip

        store = MooncakeDistributedStore()
        setup_mooncake_store(
            store,
            config or MooncakeStoreConfig.load_from_config(),
            local_hostname or get_ip(),
        )
        replicate_config = ReplicateConfig()
        replicate_config.with_hard_pin = True
        return cls(store, replicate_config)

    def put_batch(self, keys: Sequence[str], values: Sequence[bytes]) -> None:
        if len(keys) != len(values):
            raise ValueError("key and value counts differ")
        if not keys:
            return
        if self._store.put_batch(list(keys), list(values), self._replicate_config):
            missing = [key for key, ok in zip(keys, self.exists(keys)) if not ok]
            if missing:
                raise AuxObjectStoreError(
                    f"Mooncake failed to write {len(missing)} of {len(keys)} "
                    f"auxiliary-output objects, first {missing[0]!r}"
                )

    def get(self, key: str) -> bytes | None:
        return self._store.get(key) or None

    def exists(self, keys: Sequence[str]) -> list[bool]:
        if not keys:
            return []
        return [code == 1 for code in self._store.batch_is_exist(list(keys))]

    def remove(self, keys: Sequence[str]) -> None:
        if keys:
            self._store.batch_remove(list(keys), True)

    def remove_prefix(self, prefix: str) -> int:
        removed = self._store.remove_by_regex("^" + re.escape(prefix), True)
        if removed < 0:
            raise AuxObjectStoreError(
                f"Mooncake failed to remove objects under {prefix!r}: {removed}"
            )
        return removed

    def close(self) -> None:
        self._store.close()
