# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A centralized entrypoint to perform distributed KV cache transfer.

This implementation is a shim wrapper on two APIs exposed by `kv_connector`:
1. `send_kv_caches_and_hidden_states`
2. `recv_kv_caches_and_hidden_states
"""
from typing import TYPE_CHECKING, List, Tuple, Union

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
        world_group,
    ):

        self.config = config

        if config.kv_transfer_config is None:
            raise ValueError("KVTransferConfig is not set in the VllmConfig,"
                             " cannot initialize KVConnector.")

        assert self.config.kv_transfer_config.is_kv_transfer_instance, "KV"\
            "TransferAgent should only be used when kv_connector is set."

        self.connector = KVConnectorFactory.create_connector(
            rank, local_rank, config, world_group)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
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
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        return self.connector.recv_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches)
