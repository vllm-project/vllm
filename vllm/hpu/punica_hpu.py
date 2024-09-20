###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
from vllm.lora.punica import PunicaWrapper
from vllm_hpu_extension.ops import dispatch_bgmv_linear, dispatch_bgmv_embedding

class GaudiPunicaWrapper(PunicaWrapper):
    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: str):
        super().__init__(max_num_batched_tokens, max_batches, device)

    def add_lora(self,
                 y: torch.Tensor,
                 x: torch.Tensor,
                 wa_t_all: torch.Tensor,
                 wb_t_all: torch.Tensor,
                 scale: float,
                 y_offset: Optional[int] = None,
                 y_slice_size: Optional[int] = None,
                 *,
                 buffer: Optional[torch.Tensor] = None) -> None:
        y_org = y
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        dispatch_bgmv_linear(y, x, wa_t_all, wb_t_all, 0, 1.0)
        y = y.view_as(y_org)

    def add_lora_packed_nslice(self, y: torch.Tensor, x: torch.Tensor,
                               lora_a_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               lora_b_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               scale: float,
                               output_slices: Tuple[int, ...]) -> None:
        y_org = y
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        offset_left = 0

        for slice_idx in range(len(output_slices)):
            dispatch_bgmv_linear(
                y[:, offset_left:offset_left + output_slices[slice_idx]],
                x, lora_a_stacked[slice_idx], lora_b_stacked[slice_idx], 0, 1.0)
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        wa_t_all: torch.Tensor,
                        wb_t_all: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None) -> None:
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        dispatch_bgmv_linear(y, x, wa_t_all, wb_t_all, 0, 1.0)
        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_input: bool = True,
    ):
        dispatch_bgmv_embedding(y, x, w_t_all, 0, 1.0)