# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_compute import (  # noqa: E501
    OnComputeStoreHandler,
)
from vllm.v1.kv_offload.base import StoreTrigger

# The handler that makes the store jobs of each store trigger.
HANDLER_FOR_TRIGGER: dict[StoreTrigger, type[StoreTriggerHandler]] = {
    StoreTrigger.ON_COMPUTE: OnComputeStoreHandler,
}
