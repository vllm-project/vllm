# SPDX-License-Identifier: Apache-2.0
"""
KVConnectorBase Class for Distributed KV Cache & Hidden State communication

The class provides two primary abstract methods:
1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class KVConnectorBase(ABC):
    """
    Abstract base class for a KV connector.

    The class provides two primary abstract methods:
    1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
    2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
    """

    @abstractmethod
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the buffer and release resources.

        This method is responsible for cleaning up resources related to the 
        connector when it is no longer needed.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        """
        Send KV caches and hidden states to the connector.

        This method processes the input tokens, KV caches, and 
        hidden/intermediate states for a given model and sends the data to the 
        decode instance.

        Args:
            model_executable (torch.nn.Module): The model executable containing 
                start and end layer information.
            model_input (ModelInputForGPUWithSamplingMetadata): The input
                metadata from vLLM.
            kv_caches (List[torch.Tensor]): List of KV caches (keys and values) 
                for each layer.
            hidden_or_intermediate_states (Union[torch.Tensor, 
            IntermediateTensors]): 
                The hidden or intermediate states associated with the tokens.

        Returns:
            None

        """

        raise NotImplementedError

    @abstractmethod
    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        """
        Receive KV caches and hidden states from the connector.

        This method attempts to retrieve KV caches and hidden states for input
        tokens. If all required KV caches and hidden states are received, it
        will bypass model input, else it will fall back to normal vLLM model 
        forwarding.

        Args:
            model_executable (torch.nn.Module): 
                The model executable from vLLM modelrunner.
            model_input (ModelInputForGPUWithSamplingMetadata): 
                The model input from vLLM modelrunner.
            kv_caches (List[torch.Tensor]): 
                List of KV caches for each layer.

        Returns:
            - hidden_or_intermediate_states (torch.Tensor or
            IntermediateTensors): 
                Concatenated hidden states if all required data is retrieved, 
                otherwise `None`.
            - bypass_model_exec (bool): 
                Indicates whether the model execution can be skipped (True) or 
                needs to be redone (False).
            - model_input (ModelInputForGPUWithSamplingMetadata): 
                Optionally adjusted input metadata for re-execution when 
                `bypass_model_exec=False`.

        """

        raise NotImplementedError


KVConnectorBaseType = Union[KVConnectorBase, KVConnectorBase_V1]
