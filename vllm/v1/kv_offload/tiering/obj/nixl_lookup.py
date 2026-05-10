# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S3 block existence check via NIXL query_memory."""

import hashlib
from typing import TYPE_CHECKING

from nixl._api import nixl_agent, nixl_agent_config

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.obj.obj_store_config import ObjStoreConfig


def obj_key_to_dev_id(obj_key: str) -> int:
    """Deterministic device ID derived from the S3 key."""
    return int(hashlib.md5(obj_key.encode()).hexdigest(), 16) % (2**31)


class NixlLookup:
    """Checks whether an S3 object exists using NIXL query_memory."""

    def __init__(self, obj_config: "ObjStoreConfig"):
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjNixlLookup", agent_config)
        self._agent.create_backend("OBJ", obj_config.to_nixl_params())

    def exists(self, s3_key: str) -> bool:
        results = self._agent.query_memory(
            [(0, 1, obj_key_to_dev_id(s3_key), s3_key)], "OBJ", "OBJ"
        )
        return results[0] is not None
