"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.logger import init_logger
import vllm.distributed.parallel_state as ps
from vllm import _custom_ops as ops

assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE = (envs.VLLM_DISAGG_PREFILL_ROLE is not None)
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")

logger = init_logger(__name__)


class DistributedKVCoordinator(GroupCoordinator):
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
        use_pynccl: bool = True,
        use_custom_allreduce: bool = False,
        use_tpu_communicator: bool = True,
        use_message_queue_broadcaster: bool = False,
        use_cpu_comm_for_sanity_check: bool = True,
    ):

        super().__init__(
            group_ranks,
            local_rank,
            torch_distributed_backend,
            use_pynccl,
            use_custom_allreduce,
            use_tpu_communicator,
            use_message_queue_broadcaster,
        )

        # if turned on, will use CPU-based communication to perform a series of sanity check.
        # but it adds ~5ms delay, so please turn it off in performance-demanding usecases (e.g. disaggregated prefill)
        self.use_cpu_comm_for_sanity_check = use_cpu_comm_for_sanity_check

        # use a threadpool to buffer send request in disaggregated prefill
        self.input_hash_to_kv_sending_requests = defaultdict(list)
        self.kv_sending_thread = None
        
        
        torch.set_default_device(self.device)
        

    def debug_send(self,
                   tensor: torch.Tensor,
                   dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """Will send several metadata. Useful for debugging."""
        """NOTE: `dst` is the local rank of the destination rank."""

        self.send_tensor_dict(
            {
                "tensor": tensor,
                "mean": tensor.float().mean(),
                "shape": tensor.shape
            }, dst)

    def debug_recv(self,
                   size: torch.Size,
                   dtype: torch.dtype,
                   src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the local rank of the destination rank."""

        result = self.recv_tensor_dict(src)
        tensor = result["tensor"]
        assert torch.allclose(result["mean"], tensor.float().mean())
        assert result["shape"] == tensor.shape
        assert result[
            "shape"] == size, f"The shape sent by sender is {result['shape']} but trying to receive {size}"
        return tensor

    def kv_cache_send(self,
                      input_hash: int,
                      tensor: torch.Tensor,
                      dst: Optional[int] = None) -> None:
        """Push the KV cache send request into the send buffer"""
        """NOTE: `dst` is the local rank of the destination rank."""

        if self.use_cpu_comm_for_sanity_check:
            send_func = self.debug_send
        else:
            send_func = self.send

        self.input_hash_to_kv_sending_requests[input_hash].append([
            send_func,
            # tensor needs to be cloned, if not the tensor may be freed
            tensor.clone(),
            dst
        ])

    def kv_cache_recv(self,
                      size: torch.Size,
                      dtype: torch.dtype,
                      src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank (blocking)."""
        """This API should be used together with `push`"""
        """NOTE: `src` is the local rank of the destination rank."""

        if self.use_cpu_comm_for_sanity_check:
            recv_func = self.debug_recv
        else:
            recv_func = self.recv

        tensor = recv_func(size, dtype, src)

        return tensor

    def recv_input_hash_and_send_kv(self):

        try:

            # receive the input hash that the decode instance requires
            logger.debug('Rank %d: Waiting for input hash from rank %d, my hashes are %s', 
                         torch.distributed.get_rank(), 
                         self.ranks[(self.rank_in_group - 1) % self.world_size],
                         list(self.input_hash_to_kv_sending_requests.keys()))
            # FIXME(Kuntai): debug_recv guarantees correctness but hurts perf
            input_hash_tensor = self.debug_recv(torch.Size([1]), torch.long)
            input_hash = input_hash_tensor.item()
            logger.debug('Successfully received input hash %d', input_hash)
            assert input_hash in self.input_hash_to_kv_sending_requests, \
                f"The KV cache of {input_hash} does not exist."
            logger.debug('Input hash %d exists, start sending', input_hash)

            # execute corresponding kv cache sending jobs in request queue
            for idx, request in enumerate(
                    self.input_hash_to_kv_sending_requests[input_hash]):
                request[0](*request[1:])
            logger.debug('Finish input hash %d, free memory...' % input_hash)
            # free GPU memory occupied by sending
            del self.input_hash_to_kv_sending_requests[input_hash]

        except Exception as e:
            import sys
            import traceback
            exc_info = traceback.format_exc()
            import time
            time.sleep(torch.distributed.get_rank())
            logger.error("An error occured: %s, stack trace: %s", e, exc_info)
                         

    def kv_cache_send_finish(self):

        if self.kv_sending_thread is None:
            self.kv_sending_thread = ThreadPoolExecutor(max_workers=1)

        job = self.kv_sending_thread.submit(self.recv_input_hash_and_send_kv)
        logger.debug(f'Submit job {job} into kv cache sending thread')

    def kv_cache_recv_start(self, input_hash: int):

        logger.debug('Rank %d: Sending input hash %d to rank %d',
                     torch.distributed.get_rank(),
                     input_hash, self.ranks[(self.rank_in_group + 1) % self.world_size])

        input_hash_tensor = torch.tensor([input_hash]).long().to(self.device)
        logger.error("Rank %d: input hash tensor on device %s",
                     torch.distributed.get_rank(),
                     input_hash_tensor.device)
        # notify the kv cache sender with the input hash id
        # FIXME(Kuntai): debug_send guarantees correctness but hurts perf.
        self.debug_send(input_hash_tensor)




def buffer_kv_caches_send_and_listen_for_input_hash(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    hidden_or_intermediate_states: torch.Tensor,
) -> None:
    
    _input_tokens_list = model_input.input_tokens.tolist()
    seq_lens = model_input.seq_lens
    query_lens = model_input.query_lens
    seq_lens = ps.get_tp_group().broadcast_object(seq_lens)
    query_lens = ps.get_tp_group().broadcast_object(query_lens)
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    
    logger.info("KV cache shape is %s", kv_caches[0].shape)
    
    # failed = False
    # reason = ""

    # if sum(query_lens) != sum(seq_lens):
    #     logger.error("Query len sum is %d but seq len sum is %d", sum(query_lens), sum(seq_lens))
    #     failed=True
    # if sum(query_lens) != len(_input_tokens_list):
    #     logger.error("Input tokens len is %d, doesn't match with query lens sum %d",
    #                  sum(query_lens),
    #                  len(_input_tokens_list))
    #     failed=True
    # if slot_mapping.shape[0] != len(_input_tokens_list):
    #     logger.error("Slot mapping shape is %s, mismatch with input shape %s",
    #                  slot_mapping.shape,
    #                  len(_input_tokens_list))
    #     failed=True
    # if failed:
    #     import subprocess 
    #     subprocess.run("ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9", shell=True)
        
    
    # query_lens contains new KV caches that are added to vLLM.
    # so we will send them to decode instance
    # FIXME(Kuntai): This assume that all requests are prefill. 
    for idx, qlen in enumerate(query_lens):


        start_pos = sum(query_lens[:idx])
        end_pos = start_pos + qlen
        input_hash = hash(tuple(_input_tokens_list[start_pos:end_pos]))
        
        for i in range(model_executable.model.start_layer,
                model_executable.model.end_layer):
            kv_cache = kv_caches[i - model_executable.model.start_layer]
            
            _, _, num_heads, head_size = kv_cache[0].shape
            
            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

            current_slot_mapping = slot_mapping[start_pos:end_pos]

            ps.get_disagg_group().kv_cache_send(
                input_hash,
                key_cache[current_slot_mapping])
            ps.get_disagg_group().kv_cache_send(
                input_hash,
                value_cache[current_slot_mapping])


        ps.get_disagg_group().kv_cache_send(
            input_hash, 
            hidden_or_intermediate_states[start_pos:end_pos])
        ps.get_disagg_group().kv_cache_send_finish()

    logger.error("\033[92mKV send DONE for rank %d\033[0m", torch.distributed.get_rank())
    

    
def send_input_hash_and_do_kv_caches_recv(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor]
) -> torch.Tensor:
    
    # This is disagg decode instance, during prefill state
    # Need to receive KV from the prefill instance
    # FIXME(Kuntai): This impl assumes that all requests are prefill. 

    _input_tokens_list = model_input.input_tokens.tolist()
    seq_lens = model_input.seq_lens
    query_lens = model_input.query_lens
    seq_lens = ps.get_tp_group().broadcast_object(seq_lens)
    query_lens = ps.get_tp_group().broadcast_object(query_lens)
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
    
    hidden_or_intermediate_states_for_one_req = []
        
    # enumerate different requests
    logger.debug("My query lens is %s, seq len is %s, rank is %s", 
                    str(query_lens),
                    str(seq_lens),
                    torch.distributed.get_rank())
    for idx, qlen in enumerate(query_lens):

        start_pos = sum(query_lens[:idx])
        end_pos = start_pos + qlen
        input_hash = hash(tuple(_input_tokens_list[start_pos:end_pos]))
        num_tokens = qlen
        
        # notify the prefill instance to start sending KVs associated with input_hash 
        ps.get_disagg_group().kv_cache_recv_start(input_hash)

        # receive KV cache from disaggregated prefill instance
        for i in range(model_executable.model.start_layer,
                    model_executable.model.end_layer):
            
            # get kv cache
            kv_cache = kv_caches[i - model_executable.model.start_layer]
            # get corresponding layer
            layer = model_executable.model.layers[i]
            
            # get kv cache shape (after sliced by tp)
            _, _, num_heads, head_size = kv_cache[0].shape
            key = ps.get_disagg_group().kv_cache_recv(
                torch.Size([num_tokens, num_heads, head_size]),
                kv_cache[0].dtype
            )
            value = ps.get_disagg_group().kv_cache_recv(
                torch.Size([num_tokens, num_heads, head_size]),
                kv_cache[0].dtype
            )
            
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping[start_pos:end_pos],
                layer.self_attn.attn.kv_cache_dtype,
                layer.self_attn.attn._k_scale,
                layer.self_attn.attn._v_scale,
            )


        hidden_or_intermediate_states_for_one_req.append(
            ps.get_disagg_group().kv_cache_recv(
                torch.Size([num_tokens, model_executable.config.hidden_size]),
                kv_cache[0].dtype
            )
        )

    # concatenate hidden states from different requests
    hidden_or_intermediate_states = torch.cat(hidden_or_intermediate_states_for_one_req, dim=0)

    logger.error("\033[92mKV receive DONE for rank %d\033[0m", torch.distributed.get_rank())
    return hidden_or_intermediate_states