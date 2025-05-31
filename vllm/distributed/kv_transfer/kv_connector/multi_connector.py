# SPDX-License-Identifier: Apache-2.0
"""
MultiConnectorV0 - v0 implementation combining multiple KV connectors
"""
import copy
from typing import TYPE_CHECKING, Union

import torch

from vllm.config import KVTransferConfig, VllmConfig, logger
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class MultiConnectorV0(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self._connectors = []
        ktcs = config.kv_transfer_config.kv_connector_extra_config.get(
            "connectors", [])
        assert ktcs is not None
        for ktc in ktcs:
            temp_config = copy.copy(config)
            temp_config.kv_transfer_config = KVTransferConfig(**ktc)
            self._connectors.append(
                KVConnectorFactory.create_connector_v0(rank, local_rank,
                                                       temp_config))

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        for connector in self._connectors:
            if connector.get_config().kv_transfer_config.is_kv_consumer:
                hidden, bypass, new_input = (
                    connector.recv_kv_caches_and_hidden_states(
                        model_executable, model_input, kv_caches))
                # if bypass or model_input changed, return immediately
                if len(self._connectors
                       ) == 1 or bypass or new_input is not model_input:
                    return hidden, bypass, new_input
        return None, False, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        raise RuntimeError("Should not call this method in multi connector")

    def send_kv_caches_and_hidden_states_with_ori_input(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        model_input_before_recv: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        for connector in self._connectors:
            kv_transfer_config = connector.get_config().kv_transfer_config
            if kv_transfer_config.is_kv_producer:
                is_transfer = kv_transfer_config.kv_connector_extra_config.get(
                    "transfer", False)
                if is_transfer:
                    model_input = model_input_before_recv
                connector.send_kv_caches_and_hidden_states(
                    model_executable, model_input, kv_caches,
                    hidden_or_intermediate_states)
                logger.debug(
                    "sent to connector %s with mode transfer=%s",
                    connector.get_config().kv_transfer_config.kv_connector,
                    is_transfer)
            else:
                logger.debug(
                    "not sending to connector %s",
                    connector.get_config().kv_transfer_config.kv_connector)

    def close(self) -> None:
        for connector in self._connectors:
            connector.close()
