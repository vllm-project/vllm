# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


class ECConnector:
    """EC connector interface used by the V2 GPU model runner."""

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        yield None


class ActiveECConnector(ECConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        encoder_cache: dict[str, torch.Tensor],
    ) -> None:
        self.encoder_cache = encoder_cache
        self.save_new_caches = vllm_config.is_ec_producer_only
        self.ec_connector = get_ec_transfer()
        assert isinstance(self.ec_connector, ECConnectorBase)

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        if scheduler_output.ec_connector_metadata is None:
            yield None
            return

        output = ECConnectorOutput()
        ec_connector = self.ec_connector
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        if ec_connector.is_consumer:
            ec_connector.start_load_caches(self.encoder_cache)

        cached_hashes = set(self.encoder_cache) if self.save_new_caches else None
        try:
            yield output
            if cached_hashes is not None:
                for mm_hash in self.encoder_cache.keys() - cached_hashes:
                    ec_connector.save_caches(
                        encoder_cache=self.encoder_cache, mm_hash=mm_hash
                    )
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
            ec_connector.clear_connector_metadata()


NO_OP_EC_CONNECTOR = ECConnector()


def get_ec_connector(
    vllm_config: VllmConfig,
    encoder_cache: "EncoderCache | None",
) -> ECConnector:
    if (
        not has_ec_transfer()
        or vllm_config.model_config.is_encoder_decoder
        or encoder_cache is None
    ):
        return NO_OP_EC_CONNECTOR

    return ActiveECConnector(vllm_config, encoder_cache.encoder_outputs)
