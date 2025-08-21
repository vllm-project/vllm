# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Defines the base type for KV cache connectors."""

from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          kv_connector_manager)

KVConnectorBase = KVConnectorBase_V1
KVConnectorBaseType = KVConnectorBase_V1

__all__ = ["KVConnectorBase", "KVConnectorBaseType", "kv_connector_manager"]
