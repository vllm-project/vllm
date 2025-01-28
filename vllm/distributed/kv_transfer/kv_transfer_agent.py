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

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata

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

        self.connector = KVConnectorFactory.create_connector(
            rank, local_rank, config)

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

    def send_one_layer_kv_cache(self, layer_id: int,
                                input_token_hash: List[str],
                                kv_cache: torch.Tensor,
                                attn_metadata: "AttentionMetadata",
                                block_size: int) -> None:
        self.connector.send_one_layer_kv_cache(layer_id, input_token_hash,
                                               kv_cache, attn_metadata,
                                               block_size)

    def send_hidden_states(self, input_token_hash: List[str],
                           hidden_states: torch.Tensor, attn_metadata) -> None:
        self.connector.send_hidden_states(input_token_hash, hidden_states,
                                          attn_metadata)

    def close(self) -> None:
        self.connector.close()

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        return self.connector.recv_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches, **kwargs)
