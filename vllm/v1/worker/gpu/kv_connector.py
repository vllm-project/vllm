# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import logging
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def post_forward(
        self, scheduler_output: "SchedulerOutput", wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_dict: dict[str, torch.Tensor],
        excluded_layer_names: set[str] | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()

        # Filter out draft model KV caches (e.g., EAGLE draft attention
        # layers) before registering with the connector. This prevents
        # P/D mismatches when speculative decoding adds extra layers on
        # the decode side.
        if excluded_layer_names:
            found = sorted(excluded_layer_names & kv_caches_dict.keys())
            if found:
                logger.info(
                    "Excluding draft KV caches from transfer: %s",
                    found,
                )
            kv_caches_dict = {
                k: v for k, v in kv_caches_dict.items() if k not in excluded_layer_names
            }

        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        self.kv_connector.handle_preemptions(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self,
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        clear_metadata: bool = True,
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )

        if clear_metadata:
            self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        kv_connector_output = self.post_forward(scheduler_output, wait_for_save=False)
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig,
    kv_caches_dict: dict[str, torch.Tensor],
    excluded_layer_names: set[str] | None = None,
) -> KVConnector:
    if not has_kv_transfer_group():
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict, excluded_layer_names)
