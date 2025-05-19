# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
import torch.nn.functional as F

from vllm.lora.ops.xla_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink

from .punica_base import PunicaWrapperBase


class PunicaWrapperTPU(PunicaWrapperBase):
    """
    PunicaWrapperTPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        # PunicaWrapperBase defines some tensors with dtype=torch.int64, which
        # isn't supported by the TPU. So convert those tensors to int32.
        # Not all of them are used by the TPU so only convert the useful ones.
        self._token_lora_indices = self._token_lora_indices.to(
            dtype=torch.int32)
        self._sampler_indices = self._sampler_indices.to(dtype=torch.int32)
        self._sampler_indices_padded = self._sampler_indices_padded.to(
            dtype=torch.int32)

        torch._dynamo.mark_dynamic(self._token_lora_indices, 0)
        torch._dynamo.mark_dynamic(self._embeddings_indices, 1)
        torch._dynamo.mark_dynamic(self._sampler_indices_padded, 0)

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.IntTensor:
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    @property
    def embeddings_indices(self) -> torch.Tensor:
        """
        This property provides access to the indices used for lora embeddings,
        specifically for VocabParallelEmbeddingWithLoRA.
        """
        return self._embeddings_indices[:]

    @property
    def sampler_indices_padded(self) -> torch.Tensor:
        """
        This property provides access to padded sampler indices.
        """
        return self._sampler_indices_padded[:]

    def shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        if self.no_lora:
            return y
        return bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x),
                           scale)

    def expand(self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor,
               add_inputs: bool):
        return bgmv_expand(x, w_t_all, y, self._get_token_lora_indices(x),
                           add_inputs)

    def expand_slice(self, y: torch.Tensor, x: torch.Tensor,
                     w_t_all: torch.Tensor, y_offset: int, y_slice_size: int,
                     y_total_size: int, add_inputs: bool) -> torch.Tensor:
        return bgmv_expand_slice(x, w_t_all, y,
                                 self._get_token_lora_indices(x), y_offset,
                                 y_slice_size, add_inputs)

    def add_shrink(self, y: Union[tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: tuple[torch.Tensor, ...],
                   scale: float, **kwargs) -> Optional[torch.Tensor]:
        """
        Performs GEMM for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        torch.ops.xla.dynamo_set_buffer_donor_(y, True)
        x = x.view(-1, x.shape[-1])

        for slice_idx in range(len(lora_a_stacked)):
            y_s = y[slice_idx]
            lora_s = lora_a_stacked[slice_idx]
            y_s = self.shrink(y_s, x, lora_s, scale)
            y[slice_idx, :, :] = y_s  # type: ignore[index]
        return y

    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> torch.Tensor:
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
            x (Union[tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]):
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = 0

        if lora_bias_stacked is not None:
            y = self._apply_bias(self._get_token_lora_indices(y), y,
                                 output_slices, lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            y = self.expand_slice(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                y_total_size=sum(output_slices),
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        return y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> torch.Tensor:
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

        # Embedding layer only needs the expand op
        return self.expand(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> torch.Tensor:
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
            y (torch.Tensor): Output tensor. Will not be changed in-place.
            x (torch.Tensor): Input tensor (T, E)
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self._get_token_lora_indices(y), y,
                                 output_slices, lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            T = x.size(0)
            buffer = torch.zeros(
                (len(output_slices), T, r),
                dtype=torch.float32,
                device=x.device,
            )
        buffer = self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        return self.add_expand(y,
                               buffer,
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
                        **kwargs) -> torch.Tensor:
        """
        Applies lora specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        if self.no_lora:
            return y

        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        buffer = bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices,
                             scale)
        y = bgmv_expand(buffer,
                        lora_b_stacked,
                        y,
                        self.sampler_indices,
                        add_inputs=True)
        return y.view_as(y_org)

    def _apply_bias(
        self,
        indices: torch.Tensor,
        output: torch.Tensor,
        output_slices: tuple[int, ...],
        lora_bias_stacked: tuple[Optional[torch.Tensor], ...],
    ):
        """Applies bias to output

        Input shapes:
            lora_bias_stacked:      3 element tuple of (num_loras, output_dim)
            indices:           (batch_size)
            output:            (batch_size, q_slice_size + 2*kv_slice_size)
            output_slices:     n-1 element tuple of (slice_size...),
                            where n is number of slices
        """
        org_output = output
        output = output.view(-1, output.shape[-1])
        indices = indices.view(-1)

        offset_left = 0
        for slice_idx, slice in enumerate(output_slices):
            bias = lora_bias_stacked[slice_idx]
            if bias is not None:
                bias = bias.view(-1, bias.shape[-1])
                bias = bias[indices]
                bias = torch.where(indices[:, None] == -1, 0, bias)

                bias = F.pad(bias, (offset_left, output.shape[1] -
                                    (offset_left + slice), 0, 0))

                output += bias
            offset_left += slice

        return output.view_as(org_output)

    def _update_prefill_metadata(self,
                                 token_lora_tensor: torch.Tensor) -> None:
        self.batch_size = 1
        self._lora_indices_per_batch[:self.batch_size].copy_(
            token_lora_tensor[:self.batch_size])
        # TODO: .item() is extremely inefficient on TPU, so find a way around it
        self.no_lora = torch.all(token_lora_tensor == -1).item()
