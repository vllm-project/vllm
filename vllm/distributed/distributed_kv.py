"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict, deque
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

    def send_input_hash(self, input_hash: int) -> None:

        # KV cache send go through CPU, and the original `send` only use GPU.
        # So create a new group for sending input hash.
        input_hash_tensor = torch.tensor([input_hash], device="cpu").long()
        torch.distributed.isend(input_hash_tensor, self.target_rank_for_send,
                                self.cpu_group)

    def recv_input_hash(self) -> int:
        input_hash_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.irecv(input_hash_tensor, self.target_rank_for_recv,
                                self.cpu_group).wait()
        return input_hash_tensor.item()

    def recv_input_hash_and_send_kv(self):

        try:

            # receive the input hash that the decode instance requires
            logger.debug(
                '[rank%d]: Waiting for input hash from rank %d',
                torch.distributed.get_rank(),
                self.target_rank_for_recv,
            )
            input_hash = self.recv_input_hash()
            logger.debug(
                'Successfully received input hash %d',
                input_hash)
            assert input_hash in self.input_hash_to_kv_sending_requests, \
                f"The KV cache of {input_hash} does not exist. "\
                f"Existing input hash: {list(self.input_hash_to_kv_sending_requests.keys())}"
            logger.debug('Input hash %d exists, start sending', input_hash)

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
                    'there are two jobs with identical input. Free GPU '\
                    'memory for one of the request.',
                    input_hash)

        except Exception as e:
            # This function is executed in ThreadPoolExecutor
            # and it will block all exceptions by default
            # so log the potential error message here.
            import traceback
            import time
            exc_info = traceback.format_exc()
            # avoid the output of different rank overlaps
            time.sleep(torch.distributed.get_rank())
            logger.error("An error occured: %s, stack trace: %s", e, exc_info)

    def kv_cache_send_finish(self, input_hash: int):

        if self.kv_sending_thread is None:
            self.kv_sending_thread = ThreadPoolExecutor(max_workers=1)

        # append an empty job to signal that this is the end of a request
        self.input_hash_to_kv_sending_requests[input_hash].append([])
        job = self.kv_sending_thread.submit(self.recv_input_hash_and_send_kv)
        logger.debug(f'Submit job {job} into kv cache sending thread')

    def kv_cache_recv_start(self, input_hash: int):

        logger.debug('[rank%d]: Sending input hash %d to rank %d',
                     torch.distributed.get_rank(), input_hash,
                     self.ranks[(self.rank_in_group + 1) % self.world_size])

        # notify the kv cache sender with the input hash id
        self.send_input_hash(input_hash)


def buffer_kv_caches_send_and_listen_for_input_hash(
    model_executable: torch.nn.Module,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    kv_caches: List[torch.Tensor],
    hidden_or_intermediate_states: torch.Tensor,
) -> None:

    input_tokens_tuple = tuple(model_input.input_tokens.tolist())
    seq_query_obj = {
        "seq_lens": model_input.seq_lens,
        "query_lens": model_input.query_lens,
    }
    seq_query_obj = ps.get_tp_group().broadcast_object(seq_query_obj)
    seq_lens = seq_query_obj["seq_lens"]
    query_lens = seq_query_obj["query_lens"]
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    logger.debug("My query lens is %s, seq len is %s, rank is %s",
                 str(query_lens), str(seq_lens), torch.distributed.get_rank())
                 
    # query_lens contains new KV caches that are added to vLLM.
    # so we will send them to decode instance
    # FIXME(Kuntai): This assume that all requests are prefill.
    for idx, qlen in enumerate(query_lens):

        start_pos = sum(query_lens[:idx])
        end_pos = start_pos + qlen
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
            input_hash, hidden_or_intermediate_states[start_pos:end_pos])
        ps.get_disagg_group().kv_cache_send_finish(input_hash)

    logger.error("\033[92mKV send DONE for rank %d\033[0m",
                 torch.distributed.get_rank())


def send_input_hash_and_do_kv_caches_recv(
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]) -> torch.Tensor:

    # This is disagg decode instance, during prefill state
    # Need to receive KV from the prefill instance
    # FIXME(Kuntai): This impl assumes that all requests are prefill.
    input_tokens_tuple = tuple(model_input.input_tokens.tolist())
    seq_query_obj = {
        "seq_lens": model_input.seq_lens,
        "query_lens": model_input.query_lens,
    }
    seq_query_obj = ps.get_tp_group().broadcast_object(seq_query_obj)
    seq_lens = seq_query_obj["seq_lens"]
    query_lens = seq_query_obj["query_lens"]
    slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

    hidden_or_intermediate_states_for_one_req = []

    # enumerate different requests
    logger.debug("My query lens is %s, seq len is %s, rank is %s",
                 str(query_lens), str(seq_lens), torch.distributed.get_rank())
    for idx, qlen in enumerate(query_lens):

        start_pos = sum(query_lens[:idx])
        end_pos = start_pos + qlen
        input_hash = hash(input_tokens_tuple[start_pos:end_pos])
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
            ps.get_disagg_group().kv_cache_recv(
                torch.Size([num_tokens, model_executable.config.hidden_size]),
                kv_cache[0].dtype))

    # concatenate hidden states from different requests
    hidden_or_intermediate_states = torch.cat(
        hidden_or_intermediate_states_for_one_req, dim=0)

    logger.error("\033[92mKV receive DONE for rank %d\033[0m",
                 torch.distributed.get_rank())
    return hidden_or_intermediate_states
