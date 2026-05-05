# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S3 block existence check via NIXL query_memory."""

import hashlib

from nixl._api import nixl_agent, nixl_agent_config


def obj_key_to_dev_id(obj_key: str) -> int:
    """Deterministic device ID derived from the S3 key."""
    return int(hashlib.md5(obj_key.encode()).hexdigest(), 16) % (2**31)


class NixlLookup:
    """Checks whether an S3 object exists using NIXL query_memory."""

    def __init__(
        self,
        bucket: str,
        endpoint_override: str,
        access_key: str,
        secret_key: str,
        scheme: str = "http",
        ca_bundle: str = "",
    ):
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjNixlLookup", agent_config)
        params: dict[str, str] = {
            "bucket": bucket,
            "endpoint_override": endpoint_override,
            "scheme": scheme,
            "access_key": access_key,
            "secret_key": secret_key,
        }
        if ca_bundle:
            params["ca_bundle"] = ca_bundle
        self._agent.create_backend("OBJ", params)

    def exists(self, s3_key: str) -> bool:
        results = self._agent.query_memory(
            [(0, 1, obj_key_to_dev_id(s3_key), s3_key)], "OBJ", "OBJ"
        )
        return results[0] is not None
