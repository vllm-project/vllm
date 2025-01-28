"""
Simple KV Cache Connector for Distributed Machine Learning Inference

LayerwiseConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe, 
layer bylayer.

The logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_dict_buffer import (
    SimpleDictBuffer)
from vllm.distributed.kv_transfer.kv_transfer_utils import (
    get_tensor_stable_hash)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class LayerwiseConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.config = config.kv_transfer_config

        if self.config.kv_connector != "LayerwisePyNcclConnector":
            raise ValueError(
                "LayerwiseConnector only supports LayerwisePyNcclConnector")

        from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
        logger.info("Initializing PyNcclConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleDictBuffer] = None
        self.consumer_buffer: Optional[SimpleDictBuffer] = None

        self.producer_data_pipe: PyNcclPipe
        self.consumer_data_pipe: PyNcclPipe
        self.producer_signal_pipe: PyNcclPipe
        self.consumer_signal_pipe: PyNcclPipe

        # 2 pipes for every rank in the world
        port_offset_base = 2 * rank

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:

            self.producer_data_pipe = PyNcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
            )
            self.producer_signal_pipe = PyNcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )

            self.producer_buffer = SimpleDictBuffer(self.producer_signal_pipe,
                                                    self.producer_data_pipe,
                                                    self.config.kv_buffer_size)

        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder

            self.consumer_data_pipe = PyNcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base,
            )
            self.consumer_signal_pipe = PyNcclPipe(
                local_rank=local_rank,
                config=self.config,
                port_offset=port_offset_base + 1,
                device="cpu",
            )

            self.consumer_buffer = SimpleDictBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    def select(self, key: str) -> Optional[torch.Tensor]:

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(key)

    def insert(self, key: str, value: torch.Tensor) -> None:

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(key, value)

    def _get_kv_cache_key(self, input_tokens_hash: str, layer: int) -> str:
        return f"{input_tokens_hash}_layer_{layer}"

    def _get_hs_cache_key(self, input_tokens_hash: str) -> str:
        return f"{input_tokens_hash}_hs"

    def send_one_layer_kv_cache(self, layer_id: int,
                                input_token_hash: List[str],
                                kv_cache: torch.Tensor,
                                attn_metadata: "AttentionMetadata",
                                block_size: int) -> None:
        seq_lens = attn_metadata.seq_lens
        slot_mapping_flat = attn_metadata.slot_mapping.flatten()

        assert len(input_token_hash) == len(seq_lens)

        for idx, slen in enumerate(seq_lens):
            kv_cache_key = self._get_kv_cache_key(input_token_hash[idx],
                                                  layer_id)

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

            page_index = [
                x // block_size for x in current_slot_mapping[::block_size]
            ]

            paged_kv_cache = kv_cache[:, page_index, ...]

            self.insert(kv_cache_key, paged_kv_cache)

    def send_hidden_states(self, input_token_hash: List[str],
                           hidden_states: torch.Tensor,
                           attn_metadata: "AttentionMetadata") -> None:
        seq_lens = attn_metadata.seq_lens
        assert len(input_token_hash) == len(seq_lens)

        for idx, slen in enumerate(seq_lens):
            hs_cache_key = self._get_hs_cache_key(input_token_hash[idx])

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            hs_cache = hidden_states[start_pos:end_pos]

            self.insert(hs_cache_key, hs_cache)

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        assert 'block_size' in kwargs, "block_size is required in kwargs"
        block_size = kwargs['block_size']

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        try:
            for idx, slen in enumerate(seq_lens):

                start_pos = sum(seq_lens[:idx])
                end_pos = start_pos + slen
                current_tokens = input_tokens_tensor[start_pos:end_pos]
                current_tokens_hash = get_tensor_stable_hash(current_tokens)
                current_slot_mapping = slot_mapping[start_pos:end_pos]
                num_tokens = slen

                for i in range(model_executable.model.start_layer,
                               model_executable.model.end_layer):

                    kv_cache = kv_caches[i -
                                         model_executable.model.start_layer]

                    kv_cache_key = self._get_kv_cache_key(
                        current_tokens_hash, i)
                    recv_kv_cache = self.select(kv_cache_key)
                    page_index = [
                        x // block_size
                        for x in current_slot_mapping[::block_size]
                    ]
                    kv_cache[:, page_index, ...] = recv_kv_cache

                hs_cache_key = self._get_hs_cache_key(current_tokens_hash)
                hidden_states = self.select(hs_cache_key)

                hidden_or_intermediate_states_for_one_req.append(hidden_states)

                assert num_tokens == hidden_states.shape[0]

        except Exception as e:
            import traceback
            traceback.print_stack()
            logger.error(
                "[rank%d]: Failed to receive all KVs and hidden states. "
                "Error: %s", torch.distributed.get_rank(), e)
            bypass_model_exec = False

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == "PyNcclConnector":
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass
