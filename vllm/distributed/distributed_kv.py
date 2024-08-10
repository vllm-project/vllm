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

assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE is not None)
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")

# add a tag when sending/recving input hash
DISTRIBUTED_KV_GLOO_TAG = 24857323

logger = init_logger(__name__)

import logging


class RankFilter(logging.Filter):

    def filter(self, record):
        # Only log if rank is 4
        rank = 1
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            pass
        return rank % 4 == 0


for handler in logger.handlers:
    handler.addFilter(RankFilter())


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
        # DO NOT use pynccl here
        # Pynccl send is non-blocking
        # and it's possible that the memory is freed before the data being sent
        # which may happen at high qps
        use_pynccl: bool = False,
        use_custom_allreduce: bool = False,
        use_tpu_communicator: bool = True,
        use_message_queue_broadcaster: bool = False,
        use_cpu_comm_for_sanity_check: bool = False,
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
        self.input_hash_to_kv_sending_requests = defaultdict(deque)
        self.kv_sending_thread = None
        self.input_hash_to_kv_sending_requests_lock = Lock()
        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]

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
                      tensor: Union[torch.Tensor, IntermediateTensors],
                      is_hidden: bool = False,
                      dst: Optional[int] = None) -> None:
        """Push the KV cache send request into the send buffer"""
        """NOTE: `dst` is the local rank of the destination rank."""

        if self.use_cpu_comm_for_sanity_check:
            send_func = self.debug_send
        else:
            send_func = self.send

        if is_hidden and not ps.get_pp_group().is_last_rank:

            assert isinstance(tensor, IntermediateTensors)

            output = deepcopy(tensor.tensors)
            for key in output:
                output[key] = output[key].contiguous()

            self.input_hash_to_kv_sending_requests[input_hash].append(
                [self.send_tensor_dict, output, dst])

        else:

            assert isinstance(tensor, torch.Tensor)

            self.input_hash_to_kv_sending_requests[input_hash].append([
                send_func,
                # use clone to make sure the tensor is contiguous
                tensor.clone(),
                dst
            ])

    def kv_cache_recv(
            self,
            size: torch.Size,
            dtype: torch.dtype,
            is_hidden: bool = False,
            src: Optional[int] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Receives a tensor from the src rank (blocking)."""
        """This API should be used together with `push`"""
        """NOTE: `src` is the local rank of the destination rank."""

        if self.use_cpu_comm_for_sanity_check:
            recv_func = self.debug_recv
        else:
            recv_func = self.recv

        if is_hidden and not ps.get_pp_group().is_last_rank:
            tensor = IntermediateTensors(self.recv_tensor_dict(src))
        else:
            tensor = recv_func(size, dtype, src)

        return tensor

    def send_input_hash(self, input_hash: int) -> int:

        logger.debug('[rank%d]: Sending input hash %d to rank %d',
                     torch.distributed.get_rank(), input_hash,
                     self.target_rank_for_send)

        # KV cache send go through CPU, and the original `send` only use GPU.
        # So create a new group for sending input hash.
        input_hash_tensor = torch.tensor([input_hash], device="cpu").long()
        torch.distributed.send(input_hash_tensor,
                               self.target_rank_for_send,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(return_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return return_tensor.item()

    def recv_input_hash(self) -> Optional[int]:
        '''
            Receive an input hash, and check if it is already cached
        '''
        input_hash_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(input_hash_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        input_hash = input_hash_tensor.item()
        # a new input hash comes in, see if it is already cached
        self.input_hash_to_kv_sending_requests_lock.acquire()
        logger.debug('Successfully received input hash %d', input_hash)
        if input_hash not in self.input_hash_to_kv_sending_requests:
            logger.warning(
            f"The KV cache of {input_hash} does not exist. "\
            f"Existing input hash: {list(self.input_hash_to_kv_sending_requests.keys())}")
            
            # 0 for fail
            x = torch.tensor([0], device="cpu").long()
            torch.distributed.send(x,
                                    self.target_rank_for_send,
                                    self.cpu_group,
                                    tag=DISTRIBUTED_KV_GLOO_TAG)
            return None
        else:
            logger.debug('Input hash %d exists, start sending', input_hash)
            
            # 1 for success
            x = torch.tensor([1], device="cpu").long()
            torch.distributed.send(x,
                                   self.target_rank_for_send,
                                   self.cpu_group,
                                   tag=DISTRIBUTED_KV_GLOO_TAG)
            return input_hash

    def kv_cache_send_loop(self):
        
        while True:
            logger.debug(
                '[rank%d]: Waiting for input hash from rank %d, my keys are %s',
                torch.distributed.get_rank(),
                self.target_rank_for_recv,
                list(self.input_hash_to_kv_sending_requests.keys()),
            )
            # wait for a new input hash
            # this function will acquire the lock
            input_hash = self.recv_input_hash()
            if input_hash is None:
                self.input_hash_to_kv_sending_requests_lock.release()
                continue

            # execute corresponding kv cache sending jobs in request queue
            while True:
                request = self.input_hash_to_kv_sending_requests[
                    input_hash].popleft()
                # An empty request: the KV cahe of one request are all sent
                if request == []:
                    break

                request[0](*request[1:])

            if len(self.input_hash_to_kv_sending_requests[input_hash]) == 0:
                logger.debug('Finish input hash %d, free GPU memory...',
                             input_hash)
                del self.input_hash_to_kv_sending_requests[input_hash]
            else:
                logger.debug(
                    'The buffer for input hash %d is not empty, meaning that '\
                    'there are two jobs with identical input.',
                    input_hash)

            self.input_hash_to_kv_sending_requests_lock.release()


    def kv_cache_send_ready(self, input_hash: int):

        if self.kv_sending_thread is None:
            self.kv_sending_thread = threading.Thread(
                target=self.kv_cache_send_loop)
            self.kv_sending_thread.start()
        
        # append an empty list to separate requests
        # as there might be identical requests, that has the same input hash
        self.input_hash_to_kv_sending_requests[input_hash].append([])
        logger.debug(f'Buffered input hash {input_hash}')

    def kv_cache_recv_start(self, input_hash: int):
        # notify the kv cache sender with the input hash id
        return self.send_input_hash(input_hash)

    def block_if_buffer_full(self):
        
        # block vLLM if the KV cache sending buffer is full
        # TODO: allow using other policies to handle buffer full
        while True:
            self.input_hash_to_kv_sending_requests_lock.acquire()
            if len(self.input_hash_to_kv_sending_requests.keys()) > 55:
                self.input_hash_to_kv_sending_requests_lock.release()
                time.sleep(0.1)
            else:
                self.input_hash_to_kv_sending_requests_lock.release()
                break


def send_kv_caches_and_hidden_states(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
) -> None:

    input_tokens_tuple = tuple(model_input.input_tokens.tolist())
    seq_lens = model_input.attn_metadata.seq_lens
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    # Assumption: current batch is all-prefill requests
    assert torch.allclose(model_input.attn_metadata.query_start_loc,
                          model_input.attn_metadata.seq_start_loc)
    assert torch.all(model_input.attn_metadata.context_lens_tensor == 0)

    ps.get_disagg_group().input_hash_to_kv_sending_requests_lock.acquire()

    # query_lens contains new KV caches that are added to vLLM.
    # so we will send them to decode instance
    # FIXME(Kuntai): This assume that all requests are prefill.
    for idx, slen in enumerate(seq_lens):

        start_pos = sum(seq_lens[:idx])
        end_pos = start_pos + slen
        input_hash = hash(input_tokens_tuple[start_pos:end_pos])

        for i in range(model_executable.model.start_layer,
                       model_executable.model.end_layer):
            kv_cache = kv_caches[i - model_executable.model.start_layer]

            _, _, num_heads, head_size = kv_cache[0].shape

            key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
            value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

            current_slot_mapping = slot_mapping[start_pos:end_pos]

            ps.get_disagg_group().kv_cache_send(
                input_hash, key_cache[current_slot_mapping])
            ps.get_disagg_group().kv_cache_send(
                input_hash, value_cache[current_slot_mapping])

        ps.get_disagg_group().kv_cache_send(
            input_hash,
            hidden_or_intermediate_states[start_pos:end_pos],
            is_hidden=True)
        ps.get_disagg_group().kv_cache_send_ready(input_hash)

    ps.get_disagg_group().input_hash_to_kv_sending_requests_lock.release()

    ps.get_disagg_group().block_if_buffer_full()

    logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


def recv_kv_caches_and_hidden_states(
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

    # Assumption: current batch is all-prefill requests
    assert torch.allclose(model_input.attn_metadata.query_start_loc,
                          model_input.attn_metadata.seq_start_loc)
    assert torch.all(model_input.attn_metadata.context_lens_tensor == 0)

    hidden_or_intermediate_states_for_one_req = []

    # enumerate different requests
    # FIXME(Kuntai): This impl assumes that all requests are prefill.
    for idx, slen in enumerate(seq_lens):

        start_pos = sum(seq_lens[:idx])
        end_pos = start_pos + slen
        input_hash = hash(input_tokens_tuple[start_pos:end_pos])
        num_tokens = slen

        # notify the prefill instance to start sending KVs associated with input_hash
        contain = ps.get_disagg_group().kv_cache_recv_start(input_hash)
        
        # fail to find input_hash in prefill instance
        # this can occur but idk why...
        if contain == 0:
            bypass_model_exec = False
            continue

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
                kv_cache[0].dtype)
            value = ps.get_disagg_group().kv_cache_recv(
                torch.Size([num_tokens, num_heads, head_size]),
                kv_cache[0].dtype)

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
            ps.get_disagg_group().kv_cache_recv(torch.Size(
                [num_tokens, model_executable.config.hidden_size]),
                                                kv_cache[0].dtype,
                                                is_hidden=True))

    # concatenate hidden states from different requests
    if isinstance(hidden_or_intermediate_states_for_one_req[0], torch.Tensor):
        hidden_or_intermediate_states = torch.cat(
            hidden_or_intermediate_states_for_one_req, dim=0)
    else:
        # concat the IntermediateTensors
        keys = list(
            hidden_or_intermediate_states_for_one_req[0].tensors.keys())
        result_its = {}

        for key in keys:
            result_its[key] = []
            for its in hidden_or_intermediate_states_for_one_req:
                result_its[key].append(its[key])
            result_its[key] = torch.cat(result_its[key], dim=0)

        hidden_or_intermediate_states = IntermediateTensors(result_its)

    logger.debug("[rank%d]: KV recv DONE.", torch.distributed.get_rank())
    return hidden_or_intermediate_states, bypass_model_exec
