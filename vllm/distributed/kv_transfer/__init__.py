# SPDX-License-Identifier: Apache-2.0
from typing import Union

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized, get_kv_transfer_group,
    has_kv_transfer_group, is_v1_kv_transfer_group)

KVConnectorBaseType = Union[KVConnectorBase, KVConnectorBase_V1]

__all__ = [
    "get_kv_transfer_group", "has_kv_transfer_group",
    "is_v1_kv_transfer_group", "ensure_kv_transfer_initialized",
    "KVConnectorBaseType"
]
