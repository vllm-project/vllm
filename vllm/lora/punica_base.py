"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm.lora.utils import compute_meta, convert_mapping

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.models import LongContextLoRAContext


class PunicaWrapperBase(ABC):
    """
    PunicaWrapper is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str]):
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
        self._long_lora_indices = torch.empty(max_num_batched_tokens,
                                              dtype=torch.long,
                                              device=device)

        # 5 is the number of indicies tensors.
        # base_indices, sampler_indices, sampler_indices_padded,
        # embeddings_indices,long_lora_indices
        self.indices_len: List[Optional[int]] = [None] * 5
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

    def update_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: List[Optional[int]],
        max_loras: int,
        vocab_size: int,
        extra_vocab_size: int,
        long_lora_context: Optional["LongContextLoRAContext"] = None,
    ):

        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size,
                                   long_lora_context)
        if mapping.is_prefill:
            # Update metadata required for prefill-related operators.
            self._update_prefill_metada(self.token_lora_indices)
            self.is_prefill = True
        else:
            self.is_prefill = False

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
        ) = convert_mapping(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            extra_vocab_size,
            self.device,
            long_lora_context,
        )
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

    def _update_prefill_metada(self, token_lora_tensor: torch.Tensor) -> None:

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

    @property
    def prefill_metadata(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
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

    @property
    def long_lora_indices(self) -> torch.Tensor:
        """ 
        This property provides access to the indices used for long context 
        lora, specifically for LinearScalingRotaryEmbeddingWithLora.
        """
        long_lora_len = self.indices_len[4]
        return self._long_lora_indices[:long_lora_len]

    # TODO: we also need to consider lora bias
    # @abstractmethod
    # def add_bias(self):
    #     raise NotImplementedError

    # @abstractmethod
    # def add_bias_slice(self):
    #     raise NotImplementedError

    @abstractmethod
    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   w_t_all: torch.Tensor, scale: float, **kwarg):
        raise NotImplementedError

    @abstractmethod
    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   w_t_all: torch.Tensor,
                   add_input: bool = True,
                   **kwarg):
        raise NotImplementedError

    @abstractmethod
    def add_expand_slice(self,
                         y: torch.Tensor,
                         x: torch.Tensor,
                         w_t_all: torch.Tensor,
                         y_offset: Optional[int],
                         y_slice_size: Optional[int],
                         add_input: bool = True,
                         **kwarg):
        raise NotImplementedError

    @abstractmethod
    def add_lora(self,
                 y: torch.Tensor,
                 x: torch.Tensor,
                 wa_t_all: torch.Tensor,
                 wb_t_all: torch.Tensor,
                 scale: float,
                 y_offset: Optional[int] = None,
                 y_slice_size: Optional[int] = None,
                 *,
                 buffer: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_lora_packed_nslice(self, y: torch.Tensor, x: torch.Tensor,
                               lora_a_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               lora_b_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               scale: float, output_slices: Tuple[int, ...],
                               **kwarg) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           w_t_all: torch.Tensor,
                           add_input: bool = True,
                           **kwarg):
        raise NotImplementedError

    @abstractmethod
    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        wa_t_all: torch.Tensor,
                        wb_t_all: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwarg) -> None:

        raise NotImplementedError
