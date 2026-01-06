# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_xla

from vllm.lora.ops.xla_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink
from vllm.lora.punica_wrapper.utils import convert_mapping

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping

from .punica_base import PunicaWrapperBase


class PunicaWrapperTPU(PunicaWrapperBase):
    """
    PunicaWrapperTPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)

        # PunicaWrapperBase defines some tensors with dtype=torch.int64, which
        # isn't supported by the TPU. So convert those tensors to int32.
        # Not all of them are used by the TPU so only convert the useful ones.
        self._token_lora_indices = self._token_lora_indices.to(dtype=torch.int32)
        self._sampler_indices = self._sampler_indices.to(dtype=torch.int32)
        self._sampler_indices_padded = self._sampler_indices_padded.to(
            dtype=torch.int32
        )

        torch.ops.xla.dynamo_set_buffer_donor_(self._token_lora_indices, True)
        torch.ops.xla.dynamo_set_buffer_donor_(self._sampler_indices, True)
        torch.ops.xla.dynamo_set_buffer_donor_(self._sampler_indices_padded, True)
        torch.ops.xla.dynamo_set_buffer_donor_(self._embeddings_indices, True)
        torch.ops.xla.dynamo_set_buffer_donor_(self._lora_indices_per_batch, True)

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
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        return bgmv_shrink(x, w_t_all, self._get_token_lora_indices(x), scale)

    def expand(
        self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, add_inputs: bool
    ):
        return bgmv_expand(x, w_t_all, y, self._get_token_lora_indices(x), add_inputs)

    def expand_slice(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ) -> torch.Tensor:
        return bgmv_expand_slice(
            x,
            w_t_all,
            y,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> torch.Tensor | None:
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
            lora_s = lora_a_stacked[slice_idx]
            y_s = self.shrink(x, lora_s, scale)
            y[slice_idx, :, :] = y_s  # type: ignore[index]
        return y

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs GEMM for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = 0

        for slice_idx in range(len(lora_b_stacked)):
            y = self.expand_slice(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        return y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor:
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

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor:
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
            y (torch.Tensor): Output tensor. Will not be changed in-place.
            x (torch.Tensor): Input tensor (T, E)
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            T = x.size(0)
            buffer = torch.zeros(
                (len(output_slices), T, r),
                dtype=x.dtype,
                device=x.device,
            )
        buffer = self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        return self.add_expand(
            y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs
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
    ) -> torch.Tensor:
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
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        sampler_indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))
        buffer = bgmv_shrink(x, lora_a_stacked, sampler_indices, scale)
        y = bgmv_expand(buffer, lora_b_stacked, y, sampler_indices, add_inputs=True)
        return y.view_as(y_org)

    # This performs the same tensor ops as the base method, except it does them
    # on the CPU then transfers the results to the TPU
    def _update_base_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
    ):
        # Make sure we don't accidentally collect outside operations
        torch_xla.sync()

        # Pad the prompt mapping to avoid running into recompiles on the TPU
        # TODO: Should this happen inside mapping internally? If so how can we
        # avoid having backend specific LoRAMapping classes?
        mapping.prompt_mapping = self._pad_prompt_mapping(mapping.prompt_mapping)

        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            indices_len,
        ) = convert_mapping(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            0,  # extra_vocab_size
            "cpu",
        )
        self._token_lora_indices = self._pad_to_shape(
            base_indices, self._token_lora_indices.shape, dims=1
        ).to(self.device)
        self._sampler_indices = self._pad_to_shape(
            sampler_indices, self._sampler_indices.shape, dims=1
        ).to(self.device)
        self._sampler_indices_padded = self._pad_to_shape(
            sampler_indices_padded, self._sampler_indices_padded.shape, dims=1
        ).to(self.device)
        self._embeddings_indices = self._pad_to_shape(
            embeddings_indices, self._embeddings_indices.shape, dims=2
        ).to(self.device)
        self.indices_len[:] = indices_len

    def _update_prefill_metadata(self, token_lora_tensor: torch.Tensor) -> None:
        self.batch_size = 1
        self._lora_indices_per_batch[: self.batch_size] = token_lora_tensor[
            : self.batch_size
        ]

    def _pad_prompt_mapping(self, prompt_mapping: tuple[int, ...]) -> tuple[int, ...]:
        num_reqs = len(prompt_mapping)

        # From vllm/v1/worker/tpu_model_runner:51, but need to avoid a circular
        # import
        MIN_NUM_SEQS = 8

        padded_num_reqs = max(2 ** math.ceil(math.log2(num_reqs)), MIN_NUM_SEQS)
        pad_len = padded_num_reqs - num_reqs

        padding = [-1] * pad_len
        return tuple(list(prompt_mapping) + padding)

    def _pad_to_shape(self, src, target_shape, dims=1):
        if dims == 1:
            pad_len = target_shape[0] - src.shape[0]
            return F.pad(src, (0, pad_len), value=0).to(torch.int32)
        else:
            pad_rows = target_shape[0] - src.shape[0]
            pad_cols = target_shape[1] - src.shape[1]
            return F.pad(src, (0, pad_cols, 0, pad_rows), value=0).to(torch.int32)
