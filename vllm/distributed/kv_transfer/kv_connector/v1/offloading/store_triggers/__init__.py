# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobSink,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_compute import (  # noqa: E501
    OnComputeStoreHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_handoff import (  # noqa: E501
    OnHandoffStoreHandler,
)
from vllm.v1.kv_offload.base import StoreTrigger

# The handler of each store trigger.
HANDLER_FOR_TRIGGER: dict[StoreTrigger, type[StoreTriggerHandler]] = {
    StoreTrigger.ON_COMPUTE: OnComputeStoreHandler,
    StoreTrigger.ON_HANDOFF: OnHandoffStoreHandler,
}

__all__ = [
    "HANDLER_FOR_TRIGGER",
    "OnComputeStoreHandler",
    "OnHandoffStoreHandler",
    "StoreTriggerHandler",
    "StoreJobSink",
]
