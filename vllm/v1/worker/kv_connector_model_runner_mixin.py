# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""
import copy
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Generator  # noqa: UP035
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (ensure_kv_transfer_shutdown,
                                          get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    EMPTY_KV_TRANSFER_STATS, KVTransferStats)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput,
                             ModelRunnerOutput)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:

    def _init_kv_connector_runner(self):
        # This buffer is used to aggregate stats across iterations. This is
        # needed because telemetry is usually independent of the scheduler in
        # disaggregated setups (eg async kv transfers). Hence we aggregate
        # stats until the scheduler.update_from_output can forward them to the
        # logger, that is when `num_scheduled_tokens` are present.
        self._kv_transfer_stats_buffer = None

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def ensure_kv_transfer_shutdown() -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if has_kv_transfer_group and has_kv_transfer_group():
            ensure_kv_transfer_shutdown()

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    def kv_connector_no_forward(self, scheduler_output: "SchedulerOutput",
                                vllm_config: VllmConfig) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(
                None, vllm_config), self._get_kv_connector_output(
                    scheduler_output,
                    wait_for_save=False) as kv_connector_output:
            pass

        if (not kv_connector_output.finished_sending
                and not kv_connector_output.finished_recving):
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def maybe_get_kv_connector_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> AbstractContextManager[Optional[KVConnectorOutput]]:
        return self._get_kv_connector_output(
            scheduler_output) if has_kv_transfer_group() else nullcontext()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV conector lifecycle within execute_model
    @contextmanager
    def _get_kv_connector_output(self,
                                 scheduler_output: "SchedulerOutput",
                                 wait_for_save: bool = True
                                 ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if wait_for_save:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids))

            kv_transfer_stats = self.get_kv_transfer_stats()
            output.kv_transfer_stats = self.accumulate_kv_transfer_stats(
                scheduler_output, kv_transfer_stats)

    def accumulate_kv_transfer_stats(self, scheduler_output: "SchedulerOutput",
                                     kv_transfer_stats: KVTransferStats):
        # Accumulate stats until the scheduler can forward them.
        if scheduler_output.num_scheduled_tokens:
            if self._kv_transfer_stats_buffer is not None:
                # Stats ready to send, aggregate and reset buffer.
                assert isinstance(kv_transfer_stats,
                                  type(self._kv_transfer_stats_buffer))
                kv_transfer_stats = \
                    kv_transfer_stats.aggregate(self._kv_transfer_stats_buffer)
                self._kv_transfer_stats_buffer = None
            return kv_transfer_stats
        elif self._kv_transfer_stats_buffer is None:
            # Accumulate but do not send yet.
            self._kv_transfer_stats_buffer = kv_transfer_stats
        else:
            self._kv_transfer_stats_buffer = \
                self._kv_transfer_stats_buffer.aggregate(kv_transfer_stats)
        return None

    @staticmethod
    def get_kv_transfer_stats() -> KVTransferStats:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_kv_transfer_stats()
        return EMPTY_KV_TRANSFER_STATS
