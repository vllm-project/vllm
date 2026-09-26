# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-exports from ``kv_transfer_state`` are resolved lazily.

Eager import pulls in the connector factory and ``vllm.v1.engine``, which
cycles back into ``vllm.v1.metrics.stats`` for any leaf module under here.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_transfer_state import (
        KVConnectorBaseType,
        ensure_kv_transfer_initialized,
        ensure_kv_transfer_shutdown,
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )

__all__ = [
    "get_kv_transfer_group",
    "has_kv_transfer_group",
    "is_v1_kv_transfer_group",
    "ensure_kv_transfer_initialized",
    "ensure_kv_transfer_shutdown",
    "KVConnectorBaseType",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from vllm.distributed.kv_transfer import kv_transfer_state

        return getattr(kv_transfer_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
