"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.distributed.group_coordinator import GroupCoordinator


assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE = (envs.VLLM_DISAGG_PREFILL_ROLE is not None)
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")


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
        use_cpu_verfication: bool = True,
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
        self.use_cpu_verfication = use_cpu_verfication

        # use a threadpool to buffer send request in disaggregated prefill
        self.input_hash_to_kv_sending_requests = defaultdict(list)
        self.kv_sending_thread = None

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
                      dst: Optional[int] = None,
                      enable_verification: bool = True) -> None:
        """Push the KV cache send request into the send buffer"""
        """NOTE: `dst` is the local rank of the destination rank."""

        if enable_verification:
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
                      src: Optional[int] = None,
                      enable_verification: bool = True) -> torch.Tensor:
        """Receives a tensor from the src rank (blocking)."""
        """This API should be used together with `push`"""
        """NOTE: `src` is the local rank of the destination rank."""

        if enable_verification:
            recv_func = self.debug_recv
        else:
            recv_func = self.recv

        tensor = recv_func(size, dtype, src)

        return tensor

    def recv_input_hash_and_send_kv(self):

        try:

            # receive the input hash that the decode instance requires
            logger.debug('Waiting for input hash ...')
            # FIXME(Kuntai): debug_recv guarantees correctness but hurts perf
            input_hash_tensor = self.debug_recv(torch.Size([1]), torch.long)
            input_hash = input_hash_tensor.item()
            logger.debug('Receiving input hash %d', input_hash)
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

        except Exception:
            import sys
            import traceback

    def kv_cache_send_finish(self):

        if self.kv_sending_thread is None:
            self.kv_sending_thread = ThreadPoolExecutor(max_workers=1)

        job = self.kv_sending_thread.submit(self.recv_input_hash_and_send_kv)
        logger.debug(f'Submit job {job} into kv cache sending thread')

    def kv_cache_recv_start(self, input_hash: int):

        logger.debug('Requesting KV cache transfer for input hash %d',
                     input_hash)

        input_hash_tensor = torch.tensor([input_hash]).long().to(self.device)
        # notify the kv cache sender with the input hash id
        # FIXME(Kuntai): debug_send guarantees correctness but hurts perf.
        self.debug_send(input_hash_tensor)
