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
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.metrics.stats import HiSparseStats


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        batch_request_indices: torch.Tensor | None = None,
        batch_request_ids: list[str] | None = None,
    ) -> None:
        pass

    def finish_forward(self) -> None:
        pass

    def stage_host_mirror_mapping(
        self, slot_mappings: dict[str, torch.Tensor], num_tokens: int
    ) -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def finish_step(self) -> "HiSparseStats | None":
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass

    def reset_capture_state(self) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_caches_dict: dict[str, torch.Tensor],
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._pending_load_start = False
        self._pending_request_state_indices: torch.Tensor | None = None
        self._pending_request_ids: list[str] | None = None
        self._disabled = False

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        batch_request_indices: torch.Tensor | None = None,
        batch_request_ids: list[str] | None = None,
    ) -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        if scheduler_output.has_sync_kv_loads:
            # Sync loads need to run before this step's forward.
            self._start_load_kv(batch_request_indices, batch_request_ids)
        else:
            # Start any async loads in post-forward instead, keeping
            # their host-side submission cost off the critical path.
            self._pending_load_start = True
            self._pending_request_state_indices = batch_request_indices
            self._pending_request_ids = batch_request_ids

    def _start_load_kv(
        self,
        batch_request_indices: torch.Tensor | None = None,
        batch_request_ids: list[str] | None = None,
    ) -> None:
        self._pending_load_start = False
        if batch_request_indices is None:
            batch_request_indices = self._pending_request_state_indices
        if batch_request_ids is None:
            batch_request_ids = self._pending_request_ids
        self._pending_request_state_indices = None
        self._pending_request_ids = None
        # TODO: sort out KV Connectors' use of forward_context
        worker_kwargs = {
            "request_state_indices": batch_request_indices,
            "request_ids": batch_request_ids,
        }
        if is_forward_context_available():
            self.kv_connector.start_load_kv(
                get_forward_context(),
                **worker_kwargs,
            )
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(
                    get_forward_context(),
                    **worker_kwargs,
                )

    def finish_forward(self) -> None:
        if not self._disabled:
            self.kv_connector.finish_forward()

    def stage_host_mirror_mapping(
        self, slot_mappings: dict[str, torch.Tensor], num_tokens: int
    ) -> None:
        if not self._disabled:
            self.kv_connector.stage_host_mirror_mapping(slot_mappings, num_tokens)

    def reset_capture_state(self) -> None:
        self.kv_connector.reset_capture_state()

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        if self._pending_load_start:
            self._start_load_kv()

        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        transfer_results = self.kv_connector.get_transfer_results(finished_req_ids)
        output.finished_sending = transfer_results.finished_sending or None
        output.finished_recving = transfer_results.finished_recving or None
        output.failed_recving = transfer_results.failed_recving
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self.kv_connector.clear_connector_metadata()
        return output

    def finish_step(self) -> "HiSparseStats | None":
        if self._disabled:
            return None
        return self.kv_connector.finish_step()

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        self.finish_forward()
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
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
