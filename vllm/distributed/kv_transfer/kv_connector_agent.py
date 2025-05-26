# SPDX-License-Identifier: Apache-2.0
"""A centralized entrypoint to perform distributed KV cache transfer.

This implementation is a shim wrapper on two APIs exposed by `kv_connector`:
1. `send_kv_caches_and_hidden_states`
2. `recv_kv_caches_and_hidden_states
"""
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    from vllm.config import VllmConfig

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class KVTransferAgent:
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):

        self.config = config

        if config.kv_transfer_config is None:
            raise ValueError("KVTransferConfig is not set in the VllmConfig,"
                             " cannot initialize KVConnector.")

        assert self.config.kv_transfer_config.is_kv_transfer_instance, "KV"\
            "TransferAgent should only be used when kv_connector is set."

        self.connector = KVConnectorFactory.create_connector_v0(
            rank, local_rank, config)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        self.connector.send_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches,
            hidden_or_intermediate_states)

    def close(self) -> None:
        self.connector.close()

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        return self.connector.recv_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches)
