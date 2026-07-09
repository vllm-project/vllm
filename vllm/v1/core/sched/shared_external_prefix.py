# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bookkeeping for coalesced asynchronous external-prefix loads."""

from collections.abc import Hashable
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True)
class SharedExternalPrefixKey:
    """Identity of one exact, block-aligned external-prefix load."""

    connector_namespace: Hashable
    hash_block_size: int
    num_local_tokens: int
    num_external_tokens: int
    local_end_hash: BlockHash | None
    external_end_hash: BlockHash


@dataclass
class PendingExternalPrefixLoad:
    """An owner load and requests waiting for its APC publication."""

    key: SharedExternalPrefixKey
    owner_request_id: str
    follower_request_ids: set[str] = field(default_factory=set)
    accepting_followers: bool = True


class SharedExternalPrefixLoadManager:
    """Indexes pending exact-prefix loads without owning any KV blocks."""

    def __init__(self) -> None:
        self._loads_by_key: dict[
            SharedExternalPrefixKey, PendingExternalPrefixLoad
        ] = {}
        self._loads_by_owner: dict[str, PendingExternalPrefixLoad] = {}
        self._follower_to_owner: dict[str, str] = {}

    def find(self, key: SharedExternalPrefixKey) -> PendingExternalPrefixLoad | None:
        load = self._loads_by_key.get(key)
        return load if load is not None and load.accepting_followers else None

    def register_owner(
        self, key: SharedExternalPrefixKey, request_id: str
    ) -> PendingExternalPrefixLoad:
        assert key not in self._loads_by_key
        assert request_id not in self._loads_by_owner
        assert request_id not in self._follower_to_owner
        load = PendingExternalPrefixLoad(key=key, owner_request_id=request_id)
        self._loads_by_key[key] = load
        self._loads_by_owner[request_id] = load
        return load

    def add_follower(self, load: PendingExternalPrefixLoad, request_id: str) -> None:
        assert self._loads_by_key.get(load.key) is load
        assert request_id not in self._loads_by_owner
        assert request_id not in self._follower_to_owner
        load.follower_request_ids.add(request_id)
        self._follower_to_owner[request_id] = load.owner_request_id

    def get_by_owner(self, request_id: str) -> PendingExternalPrefixLoad | None:
        return self._loads_by_owner.get(request_id)

    def is_follower(self, request_id: str) -> bool:
        return request_id in self._follower_to_owner

    def stop_accepting_followers(self, request_id: str) -> None:
        load = self.get_by_owner(request_id)
        if load is not None:
            load.accepting_followers = False
            if self._loads_by_key.get(load.key) is load:
                del self._loads_by_key[load.key]

    def detach_follower(self, request_id: str) -> bool:
        owner_id = self._follower_to_owner.pop(request_id, None)
        if owner_id is None:
            return False
        load = self.get_by_owner(owner_id)
        if load is not None:
            load.follower_request_ids.discard(request_id)
        return True

    def pop_owner(self, request_id: str) -> PendingExternalPrefixLoad | None:
        load = self._loads_by_owner.pop(request_id, None)
        if load is None:
            return None
        if self._loads_by_key.get(load.key) is load:
            del self._loads_by_key[load.key]
        for follower_id in load.follower_request_ids:
            self._follower_to_owner.pop(follower_id, None)
        return load

    def has_unresolved_loads(self) -> bool:
        return bool(self._loads_by_owner)
