#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping


class MooncakeSessionTracker:
    """Track chunk-spanning Mooncake sessions owned by this Worker."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_load_entries: dict[str, dict[str, int]] = {}
        self._pending_put_owners: dict[str, dict[str, int]] = {}
        self._load_key_owners: dict[str, set[str]] = {}

    @staticmethod
    def _replace_block_entry(entries: dict[str, int], key: str, block_index: int) -> None:
        for previous_key, previous_index in list(entries.items()):
            if previous_index == block_index and previous_key != key:
                del entries[previous_key]
        entries[key] = block_index

    def register_put_keys(
        self,
        req_id: str,
        entries: Iterable[tuple[str, int]],
    ) -> None:
        """Remember which requests should consume a key after PutEnd succeeds."""
        with self._lock:
            for key, block_index in entries:
                self._pending_put_owners.setdefault(key, {})[req_id] = block_index

    def commit_put_keys(self, keys: Iterable[str]) -> None:
        """Promote successfully committed keys into each owner's future loads."""
        with self._lock:
            for key in keys:
                request_entries = self._pending_put_owners.pop(key, {})
                for req_id, block_index in request_entries.items():
                    entries = self._request_load_entries.setdefault(req_id, {})
                    self._replace_block_entry(entries, key, block_index)

    def revoke_put_keys(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                self._pending_put_owners.pop(key, None)

    def prepare_load_entries(
        self,
        req_id: str,
        current_entries: Iterable[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        """Merge current hits with keys completed by earlier chunks."""
        with self._lock:
            entries = self._request_load_entries.setdefault(req_id, {})
            for key, block_index in current_entries:
                self._replace_block_entry(entries, key, block_index)
            return list(entries.items())

    def record_get_result(
        self,
        key: str,
        req_ids: Iterable[str],
        *,
        succeeded: bool,
    ) -> None:
        """Update active owners after a per-chunk BatchGetStart result."""
        with self._lock:
            if succeeded:
                self._load_key_owners.setdefault(key, set()).update(req_ids)
            else:
                # RealClient erases its process-local session when renewal
                # fails, so no previous owner still has an active session.
                self._load_key_owners.pop(key, None)

    def release_failed_get_attempts(
        self,
        request_ids_by_key: Mapping[str, Iterable[str]],
    ) -> list[str]:
        """Release only owners involved in a failed BatchGetStart attempt."""
        with self._lock:
            keys_to_end: list[str] = []
            for key, req_ids in request_ids_by_key.items():
                owners = self._load_key_owners.get(key)
                if owners is None:
                    # The failed call may have opened a Client session before
                    # returning a malformed result. No request owns it.
                    keys_to_end.append(key)
                    continue
                owners.difference_update(req_ids)
                if not owners:
                    keys_to_end.append(key)
                    del self._load_key_owners[key]
            return keys_to_end

    def _release_active_owners_locked(
        self,
        req_ids: set[str],
    ) -> list[str]:
        keys_to_end: list[str] = []
        for key, owners in list(self._load_key_owners.items()):
            owners.difference_update(req_ids)
            if not owners:
                keys_to_end.append(key)
                del self._load_key_owners[key]
        return keys_to_end

    def release_for_retry(self, req_ids: set[str]) -> list[str]:
        """Release active owners while retaining state needed by a retry."""
        if not req_ids:
            return []
        with self._lock:
            return self._release_active_owners_locked(req_ids)

    def release_terminal(self, req_ids: set[str]) -> list[str]:
        """Release active owners and delete all terminal request state."""
        if not req_ids:
            return []
        with self._lock:
            keys_to_end = self._release_active_owners_locked(req_ids)
            for req_id in req_ids:
                self._request_load_entries.pop(req_id, None)
            for key, owners in list(self._pending_put_owners.items()):
                for req_id in req_ids:
                    owners.pop(req_id, None)
                if not owners:
                    del self._pending_put_owners[key]
            return keys_to_end
