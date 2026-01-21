# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.lora import LoRAConfig
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase

import nvshmem.bindings as bindings
from nvshmem.core import ComparisonType
from nvshmem.core.interop.torch import tensor as nvshmem_tensor


# NVSHMEM implementation
def _nvshmem_wait_for_lora_flag_impl(flag: torch.Tensor, expected_value: int) -> None:
    """Wait using NVSHMEM device-side signal_wait_until."""
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


class BaseLayerWithLoRA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lora_ready: torch.Tensor

    def create_lora_flag(self, device: torch.device) -> None:
        """Create flag tensor (uint64 for NVSHMEM signal compatibility)."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            self.lora_ready = nvshmem_tensor(shape=(1,), dtype=torch.int64)
            self.lora_ready.zero_()

    def set_lora_flag(self, load_value) -> None:
        """Set flag value and ensure visibility across GPUs via NVSHMEM quiet."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            stream = torch.cuda.current_stream().cuda_stream
            self.lora_ready.fill_(load_value)
            bindings.quiet_on_stream(stream)

    def _sync_lora_loads(self) -> None:
        """Block until LoRA weights are loaded (flag equals expected value)."""
        if envs.VLLM_LORA_REQUEST_ASYNC_LOADING_CUDA:
            nvshmem_wait_for_lora_flag(self.lora_ready, 1)

    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError
