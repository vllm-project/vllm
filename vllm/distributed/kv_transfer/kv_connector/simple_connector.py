"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from vllm import _custom_ops as ops
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        # The following config is needed to rebuild the model input
        self.cache_config = config.cache_config

        if self.config.kv_connector == "PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            logger.info(
                "Initializing PyNcclConfig under kv_transfer_config %s",
                self.config)
        elif self.config.kv_connector == "MooncakeConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_distributed_pipe = os.getenv(
                'MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_distributed_pipe:
                raise ValueError(
                    "To use MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                    MooncakePipe)
                logger.info(
                    "Initializing MooncakeConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]

        # 2 pipes for every rank in the world
        port_offset_base = 2 * rank

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:

            if self.config.kv_connector == "PyNcclConnector":
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
            elif self.config.kv_connector == "MooncakeConnector":
                self.producer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                # We only need to initialize MooncakePipe once
                self.producer_signal_pipe = self.producer_data_pipe

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size)

        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder
            if self.config.kv_connector == "PyNcclConnector":
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
            elif self.config.kv_connector == "MooncakeConnector":
                self.consumer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.consumer_signal_pipe = self.consumer_data_pipe

            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(input_tokens, roi)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(input_tokens, roi, key, value, hidden)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            self.insert(current_tokens,
                        torch.ones_like(current_tokens,
                                        dtype=bool), keys, values,
                        hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], List[bool],
               "ModelInputForGPUWithSamplingMetadata"]:

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # When bypass_model_exec[i] is set to False, it means that for
        # request[i] its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states of request[i].
        bypass_model_exec = [True] * len(seq_lens)

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(current_tokens,
                              torch.ones_like(current_tokens, dtype=bool))
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec[idx] = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec[idx] = False
                continue

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # Avoid error when prefix is exactly the same as the retrieved
            if num_computed_tokens == num_tokens:
                num_computed_tokens -= 1
            num_computed_tokens_list.append(num_computed_tokens)

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    keys[i - model_executable.model.start_layer].to(
                        key_cache.device),
                    values[i - model_executable.model.start_layer].to(
                        value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        all_bypass_flag = True
        for idx, bypass_flag in enumerate(bypass_model_exec):
            if not bypass_flag:
                # Some of the KV cache of this request is not retrieved
                # Here we will fall back to normal model forwarding
                logger.debug(
                    "[rank%d]: Failed to receive request %d's"
                    " KVs and hidden states, "
                    "redo model forwarding.", torch.distributed.get_rank(),
                    idx)

                hidden_or_intermediate_states = torch.cat(
                    hidden_or_intermediate_states_for_one_req, dim=0)
                all_bypass_flag = False
        if all_bypass_flag:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        if not all(bypass_model_exec):
            rebuilt_model_input = self.build_partial_prefill_input(
                model_input, input_tokens_list, num_computed_tokens_list,
                start_pos_list, slot_mapping, kv_caches[0][0].device)
            logger.debug("Rebuilt the input!")
            return (hidden_or_intermediate_states, bypass_model_exec,
                    rebuilt_model_input)

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

    def build_partial_prefill_input(
            self, model_input: "ModelInputForGPUWithSamplingMetadata",
            full_tokens_list: List[torch.Tensor],
            num_computed_tokens_list: List[int], start_pos_list: List[int],
            slot_mapping_flat: torch.Tensor,
            device: torch.device) -> "ModelInputForGPUWithSamplingMetadata":
        """Helper function to rebuild the model input for the current request.
        """
        assert model_input.attn_metadata is not None
        assert isinstance(model_input.attn_metadata, FlashAttentionMetadata), \
            "Only FlashAttention backend is supported for now."
        assert model_input.attn_metadata.context_lens_tensor is not None
        assert model_input.attn_metadata.block_tables is not None
        assert model_input.attn_metadata.query_start_loc is not None
        assert model_input.input_positions is not None

        rebuilt_input_tokens = []
        rebuilt_input_positions = []
        rebuilt_num_prefills = 0
        rebuilt_num_prefill_tokens = 0
        rebuilt_slot_mapping = []
        rebuilt_max_query_len = 0

        rebuilt_block_tables = []

        rebuilt_query_start_loc = [0]
        rebuilt_context_lens_tensor = []

        last_query_start_loc = 0

        # recounting query and context lengths
        for idx in range(len(full_tokens_list)):
            token_tensor = full_tokens_list[idx]
            num_token = len(token_tensor)
            num_computed_token = num_computed_tokens_list[idx]
            start_pos = start_pos_list[idx]
            q_len = num_token - num_computed_token

            rebuilt_input_tokens.append(token_tensor[num_computed_token:])

            assert q_len > 0
            start_input_pos_idx = start_pos + num_computed_token
            end_input_pos_idx = start_input_pos_idx + q_len
            rebuilt_input_positions.append(
                model_input.
                input_positions[start_input_pos_idx:end_input_pos_idx])

            # Attn metadata-related
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
            start_slot_idx = start_pos + num_computed_token
            end_slot_idx = start_slot_idx + q_len
            new_slot_mapping = slot_mapping_flat[start_slot_idx:end_slot_idx]
            rebuilt_slot_mapping.append(new_slot_mapping)
            rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
            last_query_start_loc += q_len
            rebuilt_query_start_loc.append(last_query_start_loc)
            rebuilt_context_lens_tensor.append(num_computed_token)

            # recover `block_table`
            if len(model_input.attn_metadata.block_tables[idx]) > 0:
                rebuilt_block_tables.append(
                    model_input.attn_metadata.block_tables[idx])
            else:
                slot_mapping_req = slot_mapping_flat[start_pos:end_slot_idx]
                vllm_block_size = self.cache_config.block_size
                rebuilt_block_table = slot_mapping_req[::16].to(torch.int32) \
                    // vllm_block_size
                rebuilt_block_tables.append(rebuilt_block_table)

        # rebuilt attn_metadata
        rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
        rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
        rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
        rebuilt_attn_metadata.slot_mapping = torch.cat(
            rebuilt_slot_mapping).to(device)
        rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len
        rebuilt_attn_metadata.block_tables = pad_sequence(
            rebuilt_block_tables, batch_first=True).to(device)
        rebuilt_attn_metadata.query_start_loc = torch.tensor(
            rebuilt_query_start_loc,
            dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
        rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
            rebuilt_context_lens_tensor,
            dtype=model_input.attn_metadata.context_lens_tensor.dtype,
        ).to(device)
        rebuilt_attn_metadata._cached_prefill_metadata = None

        # import here to avoid circular import.
        from vllm.worker.model_runner import (
            ModelInputForGPUWithSamplingMetadata)
        rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
            input_tokens=torch.cat(rebuilt_input_tokens).to(device),
            input_positions=torch.cat(rebuilt_input_positions).to(device),
            seq_lens=model_input.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            attn_metadata=rebuilt_attn_metadata,
            prompt_adapter_mapping=model_input.prompt_adapter_mapping,
            prompt_adapter_requests=model_input.prompt_adapter_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
            finished_requests_ids=model_input.finished_requests_ids,
            virtual_engine=model_input.virtual_engine,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=model_input.is_prompt,
            async_callback=model_input.async_callback,
        )

        return rebuilt_model_input
