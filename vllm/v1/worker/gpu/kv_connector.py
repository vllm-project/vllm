# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.v1.attention.backends.mla.hisparse import (
    invalidate_blocks,
    release_pinned_state,
    take_hisparse_stats,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def __init__(self, hisparse_block_size: int | None = None) -> None:
        self.hisparse_block_size = hisparse_block_size

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self.hisparse_block_size is None:
            return
        block_ids = [
            block_id
            for request in scheduler_output.scheduled_new_reqs
            for block_id in request.block_ids[0]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                block_ids.extend(new_block_ids[0])

        invalidate_blocks(block_ids, self.hisparse_block_size)

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self.hisparse_block_size is None:
            return None
        stats = take_hisparse_stats()
        return KVConnectorOutput(hisparse_stats=stats) if stats is not None else None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        self.pre_forward(scheduler_output)
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass

    def shutdown(self) -> None:
        if self.hisparse_block_size is None:
            return
        release_pinned_state()


class ActiveKVConnector(KVConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_dict: dict[str, torch.Tensor],
        hisparse_block_size: int | None = None,
    ):
        super().__init__(hisparse_block_size)
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        super().pre_forward(scheduler_output)
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return super().post_forward(finished_req_ids, wait_for_save)

        output = super().post_forward(finished_req_ids, wait_for_save)
        if output is None:
            output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig,
    kv_caches_dict: dict[str, torch.Tensor],
    kv_cache_config: "KVCacheConfig",
) -> KVConnector:
    hisparse_block_size = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        if vllm_config.attention_config.hisparse_config is not None
        else None
    )
    if has_kv_transfer_group():
        return ActiveKVConnector(vllm_config, kv_caches_dict, hisparse_block_size)
    if hisparse_block_size is not None:
        return KVConnector(hisparse_block_size)
    return NO_OP_KV_CONNECTOR
