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
from vllm.distributed.ec_transfer import (get_ec_transfer,
                                          has_ec_transfer)
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput
import torch
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:

    @staticmethod
    def maybe_setup_ec_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_ec_transfer():
            ec_connector = get_ec_transfer()
            assert isinstance(ec_connector, ECConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None
            ec_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            ec_connector.start_load_caches()
    
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.info("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        logger.info("Start save caches")
        connector.save_caches(encoder_cache=encoder_cache,mm_hash=mm_hash)

    @staticmethod
    def maybe_wait_for_ec_save() -> None:
        if has_ec_transfer():
            get_ec_transfer().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_ec_transfer():
            return get_ec_transfer().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        **kwargs,
    ) -> AbstractContextManager[Optional[ECConnectorOutput]]:
        return ECConnectorModelRunnerMixin._get_ec_connector_output(
            scheduler_output, **kwargs) if has_ec_transfer() else nullcontext()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV conector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(
            scheduler_output.ec_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        ec_connector.start_load_caches(**kwargs)
        try:
            yield output
        finally:
            if wait_for_save:
                ec_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids))

            ec_connector.clear_connector_metadata()
