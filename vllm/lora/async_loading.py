# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

from vllm import envs
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
    try:
        from cuda.bindings import driver as _cu
    except ImportError as e:
        raise ImportError(
            "cuda.bindings is not available. Async LoRA loading requires "
            "CUDA with cuda-python installed "
        ) from e

    # wait flag EQ value
    _CU_STREAM_WAIT_VALUE_EQ = 0x0

    def _get_cu_stream():
        return _cu.CUstream(torch.cuda.current_stream().cuda_stream)

    def _wait_for_lora_flag_impl(flag: torch.Tensor, expected_value: int) -> None:
        _cu.cuStreamWaitValue32(
            _get_cu_stream(), flag.data_ptr(), expected_value, _CU_STREAM_WAIT_VALUE_EQ
        )

    def _wait_for_lora_flag_fake(flag: torch.Tensor, expected_value: int) -> None:
        return

    try:
        direct_register_custom_op(
            op_name="wait_for_lora_flag",
            op_func=_wait_for_lora_flag_impl,
            mutates_args=["flag"],
            fake_impl=_wait_for_lora_flag_fake,
        )
        wait_for_lora_flag = torch.ops.vllm.wait_for_lora_flag
    except AttributeError:
        wait_for_lora_flag = _wait_for_lora_flag_impl


class AsyncLoadLoRAMixin:
    """Mixin for async LoRA loading with pipelined stream synchronization.

    Uses cuStreamWriteValue32 / cuStreamWaitValue32 for GPU-side,
    torch.compile-safe, CUDA-graph-capturable signal/wait between
    the loading stream and the compute stream.
    """

    def create_lora_flag(self, device: torch.device) -> None:
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            self.lora_ready = torch.ones(1, dtype=torch.uint32, device=device)

    def load_begin(self, stream: torch.cuda.Stream) -> None:
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            cu_stream = _cu.CUstream(stream.cuda_stream)
            _cu.cuStreamWriteValue32(cu_stream, self.lora_ready.data_ptr(), 0, 0)

    def load_commit(self, stream: torch.cuda.Stream) -> None:
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            cu_stream = _cu.CUstream(stream.cuda_stream)
            _cu.cuStreamWriteValue32(cu_stream, self.lora_ready.data_ptr(), 1, 0)

    def _sync_lora_loads(self) -> None:
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            wait_for_lora_flag(self.lora_ready, 1)
