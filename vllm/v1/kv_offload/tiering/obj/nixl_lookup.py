# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S3 block existence check via NIXL query_memory."""

from nixl._api import nixl_agent, nixl_agent_config

from vllm.v1.kv_offload.tiering.obj.obj_store_config import ObjStoreConfig


class NixlLookup:
    """Checks whether an S3 object exists using NIXL query_memory."""

    def __init__(self, obj_config: ObjStoreConfig):
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjNixlLookup", agent_config)
        self._agent.create_backend("OBJ", obj_config.to_nixl_params())

    def exists(self, s3_key: str) -> bool:
        results = self._agent.query_memory(
            [(0, 1, 0, s3_key)], "OBJ", "OBJ"
        )
        return results[0] is not None
