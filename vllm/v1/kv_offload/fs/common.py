# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey


class FSLoadStoreSpec(LoadStoreSpec):
    """Spec for loading/storing KV blocks from a filesystem tier.

    Carries content-addressed OffloadKeys that the worker resolves to
    file paths via a FileMapper.
    """

    def __init__(self, keys: list[OffloadKey]):
        self.keys = keys
