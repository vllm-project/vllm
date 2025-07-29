# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import torch

from .utils import compute_meta, convert_mapping

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperABC(ABC):
    """
    PunicaWrapper ABC.
    """

    @abstractmethod
    def update_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[Optional[int]],
        max_loras: int,
        vocab_size: int,
        extra_vocab_size: int,
        **kwargs,
    ) -> None:
        """
        Update the lora-related metadata
        """
        raise NotImplementedError

    @abstractmethod
    def add_shrink(
        self,
        y: Union[tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Performs GEMM  for multiple slices of lora_a.
        """

        raise NotImplementedError

    @abstractmethod
    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...],
        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
        """
        raise NotImplementedError

    @abstractmethod
    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA, 
        and this layer only requires the expand operation.
        """
        raise NotImplementedError

    @abstractmethod
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
                        **kwargs) -> Optional[torch.Tensor]:
        """
        Applicable to linear-related lora. 
        """

        raise NotImplementedError

    @abstractmethod
    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> Optional[torch.Tensor]:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        """
        raise NotImplementedError


class PunicaWrapperBase(PunicaWrapperABC):
    """
    PunicaWrapperBase is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        self._token_lora_indices = torch.empty(max_num_batched_tokens,
                                               dtype=torch.long,
                                               device=device)
        self._sampler_indices = torch.empty(max_num_batched_tokens,
                                            dtype=torch.long,
                                            device=device)
        self._sampler_indices_padded = torch.empty(max_num_batched_tokens,
                                                   dtype=torch.long,
                                                   device=device)
        self._embeddings_indices = torch.empty(2,
                                               max_num_batched_tokens,
                                               dtype=torch.long,
                                               device=device)

        # 4 is the number of indices tensors.
        # base_indices, sampler_indices, sampler_indices_padded,
        # embeddings_indices
        self.indices_len: list[Optional[int]] = [None] * 4
        # these attributes are the information required for sgmv kernel
        self._seq_start_locs = torch.empty(max_batches,
                                           dtype=torch.long,
                                           device=device)
        self._seq_lengths = torch.empty(max_batches,
                                        dtype=torch.long,
                                        device=device)
        self._lora_indices_per_batch = torch.empty(max_batches,
                                                   dtype=torch.long,
                                                   device=device)
        self.device: torch.device = device
        self.max_length: int = 0
        self.token_nums: int = 0
        self.batch_size: int = -1
        self.is_prefill = False
        self.no_lora = False

    def _update_base_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[Optional[int]],
        max_loras: int,
        vocab_size: int,
        extra_vocab_size: int,
    ):
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
            extra_vocab_size,
            self.device,
        )
        self._token_lora_indices[:base_indices.shape[0]].copy_(base_indices)
        self._sampler_indices[:sampler_indices.shape[0]].copy_(sampler_indices)
        self._sampler_indices_padded[:sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded)
        self._embeddings_indices[:embeddings_indices.
                                 shape[0], :embeddings_indices.shape[1]].copy_(
                                     embeddings_indices)

        self.indices_len[:] = indices_len

    def _update_prefill_metadata(self,
                                 token_lora_tensor: torch.Tensor) -> None:

        (b_seq_start_tensor, seq_length_tensor, lora_indices_tensor,
         batch_size, max_length, token_nums,
         no_lora) = compute_meta(token_lora_tensor)

        self._seq_start_locs[:b_seq_start_tensor.shape[0]].copy_(
            b_seq_start_tensor)
        self._seq_lengths[:seq_length_tensor.shape[0]].copy_(seq_length_tensor)
        self._lora_indices_per_batch[:lora_indices_tensor.shape[0]].copy_(
            lora_indices_tensor)
        self.batch_size = batch_size
        self.max_length = max_length
        self.token_nums = token_nums
        self.no_lora = no_lora

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
                bias[indices == -1] = 0
                output[:, offset_left:offset_left + slice] += bias
            offset_left += slice

        return output.view_as(org_output)

    @property
    def prefill_metadata(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """
        This property provides a convenient way to access the necessary 
        metadata for prefill-related  kernel computations.
            1. seq_start_locs: Tensor of sequence start positions.
            2. seq_lengths: Tensor of sequence lengths.
            3. lora_indices_per_batch: Tensor of lora indices, and an index of 
                -1 means no lora should be applied.
            4. batch_size: Batch size after clustering identical lora indices.
            5. max_length: The maximum sequence length in the batch.
            6. token_nums: The token numbers in the batch.
        """
        return (self._seq_start_locs[:self.batch_size],
                self._seq_lengths[:self.batch_size],
                self._lora_indices_per_batch[:self.batch_size],
                self.batch_size, self.max_length, self.token_nums)

    @property
    def token_lora_indices(self) -> torch.Tensor:
        """
        This property provides the lora indices corresponding to each token 
        in the batch. An index of -1 means no lora should be applied.
        """
        token_lora_len = self.indices_len[0]
        return self._token_lora_indices[:token_lora_len]

    @property
    def sampler_indices(self) -> torch.Tensor:
        """ 
        This property is used to access the lora indices specifically for 
        LogitsProcessorWithLoRA.
        """
        sampler_indices_len = self.indices_len[1]
        return self._sampler_indices[:sampler_indices_len]

    @property
    def sampler_indices_padded(self) -> torch.Tensor:
        """
        This property provides access to padded sampler indices.
        """
        indices_padded_len = self.indices_len[2]
        return self._sampler_indices_padded[:indices_padded_len]

    @property
    def embeddings_indices(self) -> torch.Tensor:
        """
        This property provides access to the indices used for lora embeddings, 
        specifically for VocabParallelEmbeddingWithLoRA.
        """
        embeddings_indices_len = self.indices_len[3]
        return self._embeddings_indices[:, :embeddings_indices_len]

    def update_metadata(self, mapping: "LoRAMapping",
                        lora_index_to_id: list[Optional[int]], max_loras: int,
                        vocab_size: int, extra_vocab_size: int, **kwargs):

        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size)

        if mapping.is_prefill:
            # Update metadata required for prefill-related operators.
            self._update_prefill_metadata(self.token_lora_indices)
            self.is_prefill = True
        else:
            self.is_prefill = False

    @abstractmethod
    def add_shrink(self, y: Union[tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: tuple[torch.Tensor, ...],
                   scale: float, **kwargs) -> Optional[torch.Tensor]:
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> Optional[torch.Tensor]:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            offset = offset_start
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
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> Optional[torch.Tensor]:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.
        and this layer only requires the expand operation.
        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
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
                        **kwargs) -> Optional[torch.Tensor]:
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
            buffer (Optional[tuple[torch.Tensor, ...]]): Defaults to None.
        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> Optional[torch.Tensor]:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
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
        # TODO: implement it based on torch ops
        raise NotImplementedError
