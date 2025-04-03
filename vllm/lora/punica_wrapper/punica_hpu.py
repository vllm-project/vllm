# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, final

import torch
from vllm_hpu_extension.ops import (dispatch_bgmv_embedding,
                                    dispatch_bgmv_linear)

from .punica_base import PunicaWrapperBase
from .utils import convert_mapping

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.models import LongContextLoRAContext


@final
class PunicaWrapperHPU(PunicaWrapperBase):

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        # Increasing max_num_batched_tokens by 3x to handle increase in
        # tensor size due to padding.
        PunicaWrapperBase.__init__(self, 3 * max_num_batched_tokens,
                                   max_batches, device)

    def _update_base_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: List[Optional[int]],
        max_loras: int,
        vocab_size: int,
        extra_vocab_size: int,
        long_lora_context: Optional["LongContextLoRAContext"] = None,
    ):
        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            long_lora_offsets_tensor,
            indices_len,
        ) = convert_mapping(mapping, lora_index_to_id, max_loras, vocab_size,
                            extra_vocab_size, self.device, None)
        # Updating each element in `long_lora_offsets` with `lora_offset` slows
        # down perf in HPU due to a series of `strided_insert` ops during lazy
        # graph accumulation. Hence HPU appends `lora_offset` to a list and
        # converts it to a tensor only after it is ready.
        if long_lora_context:
            index_mapping_indices: List[int] = list(
                mapping.index_mapping).copy()
            long_lora_offsets: List[int] = []
            for i in range(len(index_mapping_indices)):
                lora_offset: int = long_lora_context.offsets_by_lora_id.get(
                    index_mapping_indices[i], 0)
                long_lora_offsets.append(lora_offset)
            long_lora_offsets_tensor = torch.tensor(long_lora_offsets,
                                                    device=self.device,
                                                    dtype=torch.long)
            indices_len[-1] = long_lora_offsets_tensor.shape[-1]

        self._token_lora_indices[:base_indices.shape[0]].copy_(base_indices)
        self._sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self._sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        self._embeddings_indices[:embeddings_indices.
                                 shape[0], :embeddings_indices.shape[1]].copy_(
                                     embeddings_indices)
        if long_lora_offsets_tensor is not None:
            self._long_lora_indices[:long_lora_offsets_tensor.shape[0]].copy_(
                long_lora_offsets_tensor)
        else:
            self._long_lora_indices.zero_()
        self.indices_len[:] = indices_len

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
