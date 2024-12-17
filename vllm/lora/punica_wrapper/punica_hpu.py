from typing import Optional, Tuple, Union, final

import torch
from vllm_hpu_extension.ops import (dispatch_bgmv_embedding,
                                    dispatch_bgmv_linear)

from .punica_base import PunicaWrapperBase


@final
class PunicaWrapperHPU(PunicaWrapperBase):

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        # Increasing max_num_batched_tokens by 3x to handle increase in
        # tensor size due to padding.
        PunicaWrapperBase.__init__(self, 3 * max_num_batched_tokens,
                                   max_batches, device)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        dispatch_bgmv_embedding(y, x, lora_b_stacked, 0)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: Tuple[int, ...],
                        *,
                        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> None:
        y_org = y
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        offset_left = 0

        for slice_idx in range(len(output_slices)):
            dispatch_bgmv_linear(
                y[:, offset_left:offset_left + output_slices[slice_idx]], x,
                lora_a_stacked[slice_idx], lora_b_stacked[slice_idx], 0, scale)
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        dispatch_bgmv_linear(y, x, lora_a_stacked, lora_b_stacked, 0, scale)
        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
        output_slices: Tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        raise NotImplementedError
