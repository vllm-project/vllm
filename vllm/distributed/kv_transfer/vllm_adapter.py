"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.

Currently supporting TP and PP.

Workflow:
- In prefill instance, KV cache sender *buffers* the KV cache send requests
- In decode instance
    - KV cache receiver sends the hash of input tokens to sender
    - KV cache sender executes send request
    - KV cache receiver receives the KV cache
"""
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from copy import deepcopy
import time
import threading

import torch
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.logger import init_logger
import vllm.distributed.parallel_state as ps
from vllm import _custom_ops as ops
from vllm.sequence import IntermediateTensors
from vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe import TorchDistributedPipe
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_kv_lookup_buffer import SimpleKVLookupBuffer

assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode", "lmc"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE in ["prefill", "decode"])
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")

'''Jiayi starts here'''
IS_LMC_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "lmc")
'''Jiayi ends here'''

# add a tag when sending/recving input hash
DISTRIBUTED_KV_GLOO_TAG = 24857323

logger = init_logger(__name__)

import logging


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
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool = False,
        use_custom_allreduce: bool = False,
        use_tpu_communicator: bool = True,
        use_message_queue_broadcaster: bool = False
    ):
        
        # init pipe
        self.pipe = TorchDistributedPipe(
            group_ranks,
            local_rank,
            torch_distributed_backend,
            use_pynccl,
            use_custom_allreduce,
            use_tpu_communicator,
            use_message_queue_broadcaster,
        )
        # init lookup buffer
        self.buffer = SimpleKVLookupBuffer(self.pipe)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:

        #input_tokens_tuple = tuple(model_input.input_tokens.tolist())
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            logger.debug(f"sending request {idx}")
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            
            keys, values = [], []
            
            
            for l in range(model_executable.model.start_layer,
                        model_executable.model.end_layer):
                logger.debug(f"sending layer {l}")
                kv_cache = kv_caches[l - model_executable.model.start_layer]

                _, _, num_heads, head_size = kv_cache[0].shape

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqeeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))
                
            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            self.buffer.insert(
                input_tokens_tensor[start_pos:end_pos], 
                None,
                keys, 
                values, 
                hidden_or_intermediate_states[start_pos:end_pos]
            )
            

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool]:

        bypass_model_exec = True

        # This is disagg decode instance, during prefill state
        # Need to receive KV from the prefill instance
        input_tokens_tuple = tuple(model_input.input_tokens.tolist())
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):

            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_input_tokens = input_tokens_tuple[start_pos:end_pos]
            num_tokens = slen

            ret = self.buffer.drop_select(current_input_tokens, None)
            if ret[0] is None:
                # didn't find any match.
                self.bypass_model_exec = False
                continue
            
            _, _, keys, values, hidden = ret

            # receive KV cache from disaggregated prefill instance
            for i in range(model_executable.model.start_layer,
                        model_executable.model.end_layer):

                # get kv cache
                kv_cache = kv_caches[i - model_executable.model.start_layer]
                # get corresponding layer
                layer = model_executable.model.layers[i]

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    keys[i],
                    values[i],
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
            # so we need to recompute the hidden state
            return [], bypass_model_exec

        # concatenate hidden states from different requests
        hidden_or_intermediate_states = torch.cat(
            hidden_or_intermediate_states_for_one_req, dim=0)

        logger.debug("[rank%d]: KV recv DONE.", torch.distributed.get_rank())
        return hidden_or_intermediate_states, bypass_model_exec