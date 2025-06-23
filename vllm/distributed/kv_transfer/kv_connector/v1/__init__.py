# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole)

__all__ = ["KVConnectorRole", "KVConnectorBase_V1"]


def _lazy_import_safe_connector():
    """Lazy import to avoid CUDA initialization during worker setup."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.safe_lmcache_connector import (  # noqa: E501
            SafeLMCacheConnectorV1)
        return SafeLMCacheConnectorV1
    except ImportError:
        return None


# Only add to __all__ if we can import it (check at access time)
def __getattr__(name):
    if name == "SafeLMCacheConnectorV1":
        connector_class = _lazy_import_safe_connector()
        if connector_class is None:
            raise ImportError("SafeLMCacheConnectorV1 not available")
        return connector_class
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
