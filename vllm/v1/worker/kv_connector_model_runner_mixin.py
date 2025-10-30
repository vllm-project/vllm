# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import (
    TYPE_CHECKING,  # noqa: UP035
)

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

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
    ) -> tuple[set[str] | None, set[str] | None]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids
            )
        return None, None

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output)
            if has_kv_transfer_group()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput", wait_for_save: bool = True
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

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
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            output.kv_connector_stats = (
                KVConnectorModelRunnerMixin.get_kv_connector_stats()
            )
            kv_connector.clear_connector_metadata()

    @staticmethod
    def get_kv_connector_stats() -> KVConnectorStats | None:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_kv_connector_stats()
        return None
