"""
KVConnectorBase Class for Distributed KV Cache & Hidden State communication

The class provides two primary abstract methods:
1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
"""

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class KVConnectorBase(ABC):
    """
    Abstract base class for a KV connector.

    The class provides two primary abstract methods:
    1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
    2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        mode_methods_dict = {
            "bulk_transfer": [
                "send_kv_caches_and_hidden_states",
                "recv_kv_caches_and_hidden_states",
            ],
            "layerwise_transfer": [
                "send_one_layer_kv_cache",
                "send_hidden_states",
                "recv_kv_caches_and_hidden_states",
            ],
        }

        supported_groups = [
            group_name for group_name, methods in mode_methods_dict.items()
            if all(method in cls.__dict__ for method in methods)
        ]

        if not supported_groups:
            error_msg = (
                f"{cls.__name__} must implement at least one of the groups:\n"
                + "\n".join(
                    f"- {group_name}: {', '.join(methods)}"
                    for group_name, methods in mode_methods_dict.items()))
            raise TypeError(error_msg)

        # Validate method signatures for all methods in supported groups
        signature_errors = []
        for group_name in supported_groups:
            for method in mode_methods_dict[group_name]:
                # Get base class method and subclass method
                base_method = getattr(__class__, method)
                subclass_method = getattr(cls, method)

                # Compare signatures
                base_sig = inspect.signature(base_method)
                subclass_sig = inspect.signature(subclass_method)
                if base_sig != subclass_sig:
                    signature_errors.append(
                        f"Signature mismatch in group '{group_name}': "
                        f"Method '{method}' expects {base_sig}, "
                        f"got {subclass_sig}")

        # Raise all signature errors at once
        if signature_errors:
            raise TypeError("\n".join(signature_errors))

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

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        **kwargs,
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

    def send_one_layer_kv_cache(self, layer_id: int,
                                input_token_hash: List[str],
                                kv_cache: torch.Tensor,
                                attn_metadata: "AttentionMetadata",
                                block_size: int) -> None:
        """
        Sends the KV cache of a single layer to the connector.

        This method transmits the KV cache for a specific transformer layer,
        along with metadata, allowing the connector to store or utilize it
        for future requests. The transmission is layer-specific.

        Args:
            layer_id (int): 
                The id of the transformer layer being sent.
            input_token_hash (List[str]): 
                Hashes of the input tokens associated with this KV cache.
            kv_cache (torch.Tensor): 
                The KV cache tensor for the specified layer.
            attn_metadata (AttentionMetadata): 
                Attention metadata for the current batch.
            block_size (int): 
                The number of tokens in one page of KV cache.

        Returns:
            None: This method does not return a value.
        """
        raise NotImplementedError

    def send_hidden_states(self, input_token_hash: List[str],
                           hidden_states: torch.Tensor,
                           attn_metadata: "AttentionMetadata") -> None:
        """
        Sends hidden states to the connector.

        Transmits computed hidden states along with attention metadata,
        enabling the connector to bypass the full model execution 
        using these cached states.

        Args:
            input_token_hash (List[str]): 
                Hash values of the input tokens.
            hidden_states (torch.Tensor): 
                The hidden states tensor computed by the model.
            attn_metadata (AttentionMetadata): 
                Attention metadata associated with these hidden states.

        Returns:
            None: This method does not return a value.
        """
        raise NotImplementedError
