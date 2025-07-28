# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import Optional, Union, final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.lora.ops.ipex_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink

from .punica_base import PunicaWrapperBase


@final
class PunicaWrapperXPU(PunicaWrapperBase):
    """
    PunicaWrapperXPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica ipex kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)
        torch._dynamo.mark_dynamic(self._token_lora_indices, 0)
        torch._dynamo.mark_dynamic(self._embeddings_indices, 1)
        torch._dynamo.mark_dynamic(self._sampler_indices_padded, 0)

    def update_metadata(self, mapping: LoRAMapping,
                        lora_index_to_id: list[Optional[int]], max_loras: int,
                        vocab_size: int, extra_vocab_size: int, **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size)

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.IntTensor:
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    def _apply_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x), scale)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        token_lora_indices = self._get_token_lora_indices(x)
        bgmv_expand_slice(x, w_t_all, y, token_lora_indices, y_offset,
                          y_slice_size, add_inputs)

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: tuple[torch.Tensor,
                                         ...], scale: float, **kwargs):
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
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx],
                               scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = self._get_token_lora_indices(y)
            self._apply_bias(token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)

        # TODO fuse these kernels
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_start,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_start += output_slices[slice_idx]
        y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
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
        token_lora_indices = self._get_token_lora_indices(x)
        bgmv_expand(x, lora_b_stacked, y, token_lora_indices, add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = self._get_token_lora_indices(y)
            y = self._apply_bias(token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=torch.float32,
                device=x.device,
            )
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs)
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            None,
            output_slices,
            add_inputs=True,
            **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
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
        if buffer is None:
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices, scale)
        bgmv_expand(buffer,
                    lora_b_stacked,
                    y,
                    self.sampler_indices,
                    add_inputs=True)
        return y.view_as(y_org)
