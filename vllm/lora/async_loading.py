# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.utils.import_utils import has_nvshmem4py
from vllm.utils.torch_utils import direct_register_custom_op

if has_nvshmem4py:
    import nvshmem.bindings as bindings
    from nvshmem.core import ComparisonType
    from nvshmem.core.interop.torch import tensor as nvshmem_tensor

if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA and not has_nvshmem4py:
    raise ImportError(
        "pip install nvshmem4py-cu12 # Required for async LoRA loading with NVSHMEM"
    )


def _nvshmem_wait_for_lora_flag_impl(flag: torch.Tensor, expected_value: int) -> None:
    stream = torch.cuda.current_stream().cuda_stream
    bindings.signal_wait_until_on_stream(
        flag.data_ptr(), ComparisonType.CMP_EQ, expected_value, stream
    )


def _nvshmem_wait_for_lora_flag_fake(flag: torch.Tensor, expected_value: int) -> None:
    return


try:
    direct_register_custom_op(
        op_name="nvshmem_wait_for_lora_flag",
        op_func=_nvshmem_wait_for_lora_flag_impl,
        mutates_args=["flag"],
        fake_impl=_nvshmem_wait_for_lora_flag_fake,
    )
    nvshmem_wait_for_lora_flag = torch.ops.vllm.nvshmem_wait_for_lora_flag
except AttributeError:
    nvshmem_wait_for_lora_flag = _nvshmem_wait_for_lora_flag_impl


class AsyncLoadLoRAMixin:
    """Mixin for async LoRA loading with NVSHMEM synchronization."""

    def create_lora_flag(self, device: torch.device) -> None:
        """Create flag tensor (uint64 for NVSHMEM signal compatibility)."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            self.lora_ready = nvshmem_tensor(shape=(1,), dtype=torch.int64)
            self.lora_ready.fill_(1)

    def load_begin(self) -> None:
        """Signal that LoRA loading is starting."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            if not hasattr(self, "lora_ready") or self.lora_ready is None:
                raise RuntimeError("lora_ready not initialized")
            stream = torch.cuda.current_stream().cuda_stream
            self.lora_ready.fill_(0)
            bindings.quiet_on_stream(stream)

    def load_commit(self) -> None:
        """Signal that LoRA loading is complete and ensure visibility across GPUs."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            if not hasattr(self, "lora_ready") or self.lora_ready is None:
                raise RuntimeError("lora_ready not initialized")
            stream = torch.cuda.current_stream().cuda_stream
            self.lora_ready.fill_(1)
            bindings.quiet_on_stream(stream)

    def _sync_lora_loads(self) -> None:
        """Block until LoRA weights are loaded (flag equals expected value)."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            if not hasattr(self, "lora_ready") or self.lora_ready is None:
                raise RuntimeError("lora_ready not initialized")
            nvshmem_wait_for_lora_flag(self.lora_ready, 1)
