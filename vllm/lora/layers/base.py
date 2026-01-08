# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase

from vllm.triton_utils import triton

triton_language = triton.language


@triton.jit()
def _wait_for_lora_flag_kernel(
    flag_ptr,
    expected_value,
):
    """Triton kernel: busy-wait until flag equals expected value."""
    while triton_language.load(flag_ptr, volatile=True) != expected_value:
        pass


@torch.inference_mode()
def _wait_for_lora_flag_impl(
    flag: torch.Tensor,
    expected_value: int,
) -> None:
    """Wait for LoRA flag to be set"""
    _wait_for_lora_flag_kernel[(1,)](
        flag,
        expected_value,
        num_warps=1,
    )


def _wait_for_lora_flag_fake(
    flag: torch.Tensor,
    expected_value: int,
) -> None:
    """Fake implementation for tracing/compilation - no-op."""
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


class BaseLayerWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_ready: torch.Tensor | None = None

    def _sync_lora_loads(self):
        """Wait for LoRA loads to complete"""
        if self.lora_ready is None:
            return
        # maybe replace with nvshem
        wait_for_lora_flag(self.lora_ready, 1)

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
