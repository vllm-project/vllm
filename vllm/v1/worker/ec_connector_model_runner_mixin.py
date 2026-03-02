# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[set[str] | None, set[str] | None]:
        if has_ec_transfer():
            return get_ec_transfer().get_finished(scheduler_output.finished_req_ids)
        return None, None

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # Load caches for consumer or both roles
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            ec_connector.clear_connector_metadata()
