# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, final

import torch

import vllm.envs as envs
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (LoRAKernelMeta, lora_expand,
                                          lora_shrink)

from .punica_base import PunicaWrapperBase

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.models import LongContextLoRAContext


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        self.token_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                      max_num_batched_tokens,
                                                      device=device)

        # When cudagraph capture size is greater than max_num_seqs (max_batches,
        # here), V0 captures the graph as if max_num_seqs is set to
        # the capture size.
        # V1 doesn't have this problem and always respects max_num_seqs.
        max_num_prompts = (max_batches
                           if envs.VLLM_USE_V1 else max_num_batched_tokens)
        self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                       max_num_prompts,
                                                       device=device)

    def update_metadata(
            self,
            mapping: LoRAMapping,
            lora_index_to_id: List[Optional[int]],
            max_loras: int,
            vocab_size: int,
            extra_vocab_size: int,
            long_lora_context: Optional["LongContextLoRAContext"] = None,
            **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size,
                                   long_lora_context)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: Tuple[torch.Tensor,
                                         ...], scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
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

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: Tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                   output_slices: Tuple[int, ...],
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
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            self._apply_bias(token_lora_indices, y, output_slices,
                             lora_bias_stacked)

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

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked, ),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: Tuple[int, ...],
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
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
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

        lora_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                    *self.prompt_mapping_meta.meta_args(x.size(0)), scale)

        lora_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                    y,
                    *self.prompt_mapping_meta.meta_args(buffer.size(0)),
                    add_inputs=True)
        y = y.view_as(y_org)
