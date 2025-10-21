# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

from .punica_base import PunicaWrapperBase


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)

        self.max_loras = kwargs["max_loras"]

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )

        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_batches, device=device
        )

    def update_metadata(
        self,
        mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        extra_vocab_size: int,
        **kwargs,
    ):
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(
            mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size
        )

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            scale,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        lora_expand(
            x,
            lora_b_stacked,
            y,
            *self.token_mapping_meta.meta_args(num_tokens),
            offset_start=offset_start,
            add_inputs=True,
        )

        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked,),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        r = lora_b_stacked[0].size(-1)
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty(
            (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
        )

        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs,
        )
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            output_slices,
            add_inputs=True,
            **kwargs,
        )

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty((x.size(0), r), dtype=torch.float32, device=x.device)

        lora_shrink(
            x,
            [lora_a_stacked],
            buffer.unsqueeze(dim=0),
            *self.prompt_mapping_meta.meta_args(x.size(0)),
            scale,
        )

        lora_expand(
            buffer.unsqueeze(dim=0),
            [lora_b_stacked],
            y,
            *self.prompt_mapping_meta.meta_args(buffer.size(0)),
            add_inputs=True,
        )
        y = y.view_as(y_org)
