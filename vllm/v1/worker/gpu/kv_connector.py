# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_checksum.worker import create_kv_checksum_worker
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.utils.torch_utils import current_stream
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.async_utils import (
    AsyncKVChecksumOutput,
    AsyncOutput,
    AsyncPoolingOutput,
    KVChecksumCopy,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def finish_forward(self) -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], defer_finalize: bool = False
    ) -> KVConnectorOutput | None:
        return None

    def finalize_step(self) -> None:
        pass

    def add_kv_checksums(
        self, output: ModelRunnerOutput | AsyncModelRunnerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """``output`` with the KV checksums of the last ``post_forward``."""
        return output

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
        kv_cache_config: "KVCacheConfig",
        kv_caches_dict: dict[str, torch.Tensor],
        output_copy_stream: torch.cuda.Stream,
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
        self.kv_checksum_worker = create_kv_checksum_worker(
            vllm_config, kv_cache_config, kv_caches_dict
        )

        self._pending_load_kwargs: dict[str, Any] | None = None
        self._output_copy_stream = output_copy_stream
        self._kv_checksums: KVChecksumCopy | None = None
        self._disabled = False

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        self._pending_load_kwargs = self._build_load_kwargs(
            scheduler_output, input_batch, attn_metadata
        )
        if self.kv_checksum_worker is not None:
            self.kv_checksum_worker.bind(scheduler_output.kv_checksum_scheduled)

        if scheduler_output.has_sync_kv_loads:
            # Sync loads need to run before this step's forward.
            self._start_load_kv()
        # Otherwise start the async load after forward to keep submission
        # off the critical path.

    @staticmethod
    def _build_load_kwargs(
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None",
        attn_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keyword args for the connector's start_load_kv()."""
        if input_batch is None:
            # No forward this step.
            return {}
        load_kwargs: dict[str, Any] = dict(
            scheduler_output=scheduler_output,
            request_state_indices=input_batch.idx_mapping,
            request_ids=input_batch.req_ids,
            num_tokens=input_batch.num_tokens,
        )
        if attn_metadata is not None:
            load_kwargs["attn_metadata"] = attn_metadata
        return load_kwargs

    def _start_load_kv(self) -> None:
        load_kwargs = self._pending_load_kwargs
        assert load_kwargs is not None
        self._pending_load_kwargs = None
        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)

    def finish_forward(self) -> None:
        if not self._disabled:
            self.kv_connector.finish_forward()

    def reset_capture_state(self) -> None:
        self.kv_connector.reset_capture_state()

    def post_forward(
        self, finished_req_ids: set[str], defer_finalize: bool = False
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        if self._pending_load_kwargs is not None:
            self._start_load_kv()

        output = KVConnectorOutput()
        if not defer_finalize:
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
        if self.kv_checksum_worker is not None and (
            pending := self.kv_checksum_worker.post_forward(
                transfer_results.finished_recving,
                transfer_results.failed_recving,
                finished_req_ids,
                transfer_results.kv_checksums,
            )
        ):
            # Every post_forward's checksums must reach an output.
            assert self._kv_checksums is None
            # Copied on their own event, so other outputs do not wait for them.
            self._kv_checksums = KVChecksumCopy(
                pending, current_stream(), self._output_copy_stream
            )
        if not defer_finalize:
            self.kv_connector.clear_connector_metadata()
        return output

    def finalize_step(self) -> None:
        if self._disabled:
            return
        self.kv_connector.wait_for_save()
        self.kv_connector.clear_connector_metadata()

    def add_kv_checksums(
        self, output: ModelRunnerOutput | AsyncModelRunnerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        kv_checksums, self._kv_checksums = self._kv_checksums, None
        if kv_checksums is None:
            return output
        if isinstance(output, ModelRunnerOutput):
            return AsyncKVChecksumOutput(output, kv_checksums)
        assert isinstance(output, (AsyncOutput, AsyncPoolingOutput))
        output.kv_checksums = kv_checksums
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        self.finish_forward()
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig",
    kv_caches_dict: dict[str, torch.Tensor],
    output_copy_stream: torch.cuda.Stream,
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(
        vllm_config, kv_cache_config, kv_caches_dict, output_copy_stream
    )
