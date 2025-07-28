# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Used to define the KVConnectorBase Class for Distributed KV Cache & Hidden
State communication in V0.
"""

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1

KVConnectorBaseType = KVConnectorBase_V1
