"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.

Currently supporting TP. The TP between prefill and decode instance needs to be 
the same.

Workflow (disaggregated prefill)
- In prefill instance
    - After prefill, vLLM `insert` its KV caches into a lookup buffer.
    - The prefill instance will also open up a thread that listens to 
      `drop_select` request.
- In decode instance
    - vLLM first runs `drop_select` to send input tokens and a mask on input 
      tokens (we call it roi, region of interest) to prefill instance
    - The prefill instance then respond to `drop_select` request by
        - Finding a match in current lookup buffer.
        - Clone and send the matched item out
        - Delete the matched item in the lookup buffer to free up GPU memory.
    - The decode vLLM then store the KV cache into paged memory.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

from copy import deepcopy

import torch
from torch.distributed import Backend

import vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer as sklb
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import (
    TorchDistributedPipe)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# check VLLM_DISTRIBUTERD_KV_ROLE and set corresponding flags
assert envs.VLLM_DISTRIBUTED_KV_ROLE in [None, "producer", "consumer", "both"],\
    "VLLM_DISTRIBUTERD_KV_ROLE can only be producer, consumer or both."
IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE
                                    in ["producer", "consumer", "both"])
IS_KV_PRODUCER: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE in ["producer", "both"])
IS_KV_CONSUMER: bool = (envs.VLLM_DISTRIBUTED_KV_ROLE in ["consumer", "both"])

# When the current instance is both KV producer and KV consumer,
# it is likely connected to a KV storage service on CPU/disk
# so the communication backend needs to be "gloo" for that case.
DISTRIBUTED_BACKEND: str = "gloo" if (IS_KV_PRODUCER
                                      and IS_KV_CONSUMER) else "nccl"
# corresponding device
DISTRIBUTED_DEVICE: str = "cpu" if (IS_KV_PRODUCER
                                    and IS_KV_CONSUMER) else "cuda"


class KV_transfer_agent:
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend] = DISTRIBUTED_BACKEND,
        # FIXME(Kuntai): remove this hardcoding
        lookup_buffer_size: int = int(1e10)):

        self.lookup_buffer_size = lookup_buffer_size

        self.send_buffer: Optional[KVLookupBufferBase] = None
        self.recv_buffer: Optional[KVLookupBufferBase] = None

        SimpleKVLookupBuffer = sklb.SimpleKVLookupBuffer

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        # In remote KV cache store, vLLM will use both send pipe and recv pipe
        # So we build both send pipe and recv pipe for simplicity.
        if IS_KV_PRODUCER:

            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                    self.send_pipe,
                                                    self.lookup_buffer_size)
            self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                    self.recv_pipe,
                                                    self.lookup_buffer_size)
            self.tensor_device = DISTRIBUTED_DEVICE
        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder

            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                    self.send_pipe,
                                                    self.lookup_buffer_size)
            self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                    self.recv_pipe,
                                                    self.lookup_buffer_size)
            self.tensor_device = DISTRIBUTED_DEVICE

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

                _, _, num_heads, head_size = kv_cache[0].shape

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            if self.send_buffer is not None:
                self.send_buffer.insert(
                    current_tokens, torch.ones_like(current_tokens,
                                                    dtype=bool), keys, values,
                    hidden_or_intermediate_states[start_pos:end_pos])

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def destroy(self) -> None:
        if self.send_buffer is not None:
            self.send_buffer.close()
        if self.recv_buffer is not None:
            self.recv_buffer.close()

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When this flag is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

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

            if self.recv_buffer is None:
                bypass_model_exec = False
                break

            ret = self.recv_buffer.drop_select(
                current_tokens, torch.ones_like(current_tokens, dtype=bool))
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

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

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # so we need to adjust model_input and redo the forwarding.
            logger.debug(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            rebuilt_model_input = build_partial_prefill_input(
                model_input,
                input_tokens_list,
                num_computed_tokens_list,
                start_pos_list,
                slot_mapping,
                device=input_tokens_tensor.device,
            )
            model_input = rebuilt_model_input
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input


def build_partial_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    input_tokens_list: List[torch.Tensor],
    num_computed_tokens_list: List[int],
    start_pos_list: List[int],
    slot_mapping_flat: torch.Tensor,
    device: torch.device,
) -> "ModelInputForGPUWithSamplingMetadata":
    """
    Helper function to rebuild the model input for the current request.
    Goal: avoid running redundant prefill on those tokens that already has KV 
    caches received.
    """
    rebuilt_input_tokens = []
    rebuilt_input_positions = []
    rebuilt_query_lens = []

    rebuilt_num_prefills = 0
    rebuilt_num_prefill_tokens = 0
    rebuilt_slot_mapping = []
    rebuilt_max_query_len = 0

    rebuilt_block_tables = []

    rebuilt_query_start_loc = [0]
    rebuilt_context_lens_tensor = []
    rebuilt_selected_token_indices = []

    # recounting query and context lengths
    for idx in range(len(input_tokens_list)):
        token_tensor = input_tokens_list[idx]
        num_token = len(token_tensor)
        num_computed_token = num_computed_tokens_list[idx]
        # currently attention kernel cannot handle the case where there is 0
        # query token.
        if num_computed_token == num_token:
            num_computed_token -= 1
        start_pos = start_pos_list[idx]

        rebuilt_input_tokens.append(token_tensor[num_computed_token:])
        # TODO(Jiayi): please check the correctness of next line
        rebuilt_input_positions.append(
            model_input.input_positions[start_pos +
                                        num_computed_token:start_pos +
                                        num_token])
        q_len = num_token - num_computed_token
        rebuilt_query_lens.append(q_len)

        # Attn metadata-related
        rebuilt_num_prefills += 1
        rebuilt_num_prefill_tokens += q_len
        new_slot_mapping = slot_mapping_flat[start_pos +
                                             num_computed_token:start_pos +
                                             num_token]
        rebuilt_slot_mapping.append(new_slot_mapping)
        rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
        # TODO(Jiayi): remove hard-code (block_size=16)
        blk_size = 16
        temp_block_table = [
            slot_mapping_flat[i] // blk_size
            for i in range(start_pos, start_pos + num_token, blk_size)
        ]
        rebuilt_block_tables.append(temp_block_table)
        rebuilt_query_start_loc.append(
            rebuilt_num_prefill_tokens)  #start with 0
        rebuilt_context_lens_tensor.append(num_computed_token)

        # Sampling metadata related
        #seq_groups (use rebuilt query lens)
        rebuilt_selected_token_indices.append(rebuilt_num_prefill_tokens - 1)

    # rebuilt attn_metadata
    rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
    rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
    rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
    rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
        device)
    rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

    rebuilt_attn_metadata.block_tables = torch.tensor(
        rebuilt_block_tables,
        dtype=model_input.attn_metadata.block_tables.dtype).to(device)

    rebuilt_attn_metadata.query_start_loc = torch.tensor(
        rebuilt_query_start_loc,
        dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
    rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
        rebuilt_context_lens_tensor,
        dtype=model_input.attn_metadata.context_lens_tensor.dtype,
    ).to(device)

    rebuilt_attn_metadata._cached_prefill_metadata = None

    # rebuilt sampling_metadata
    rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
    for idx, q_len in enumerate(rebuilt_query_lens):
        if rebuilt_sampling_metadata.seq_groups is not None:
            rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

    rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
        rebuilt_selected_token_indices,
        dtype=model_input.sampling_metadata.selected_token_indices.dtype,
    ).to(device)

    # import here to avoid circular import.
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.cat(rebuilt_input_tokens).to(device),
        input_positions=torch.cat(rebuilt_input_positions).to(device),
        seq_lens=model_input.seq_lens,
        query_lens=rebuilt_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=rebuilt_attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=rebuilt_sampling_metadata,
        is_prompt=model_input.is_prompt,
    )

    return rebuilt_model_input
