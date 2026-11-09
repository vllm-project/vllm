# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EC file probes using the native KV asynchronous lookup lifecycle."""

from collections.abc import Iterable
from pathlib import Path
from stat import S_ISREG

from vllm.distributed.ec_transfer.ec_connector.tiering.layout import get_ec_entry_path
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager


class ECFileLookup(AsyncLookupManager):
    def __init__(self, root_dir: Path, namespace_digest: str) -> None:
        self._root_dir = root_dir
        self._namespace_digest = namespace_digest
        super().__init__(tier_type="ec_fs")

    @staticmethod
    def key(cache_key: str, file_size_bytes: int) -> OffloadKey:
        return OffloadKey(
            cache_key.encode("ascii") + file_size_bytes.to_bytes(8, "big")
        )

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        for key in keys:
            path = get_ec_entry_path(
                self._root_dir, self._namespace_digest, key[:-8].decode("ascii")
            )
            try:
                info = path.lstat()
                yield S_ISREG(info.st_mode) and info.st_size == int.from_bytes(
                    key[-8:], "big"
                )
            except OSError:
                yield False
