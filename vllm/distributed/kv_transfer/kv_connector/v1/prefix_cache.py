# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache protocol selection for EAGLE engines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
    )


def is_eagle_prefix_cache_hashing_enabled(
    vllm_config: "VllmConfig",
    kv_connector: "KVConnectorBase_V1 | None" = None,
) -> bool:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_eagle():
        return False
    if vllm_config.kv_transfer_config is not None:
        return bool(
            kv_connector is not None
            and kv_connector.supports_eagle_prefix_cache_hashing
        )
    return vllm_config.cache_config.enable_prefix_caching
