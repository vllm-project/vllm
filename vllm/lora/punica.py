"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch

from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.bgmv_expand import bgmv_expand
    from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
    from vllm.lora.ops.bgmv_shrink import bgmv_shrink
    from vllm.lora.ops.sgmv_expand import sgmv_expand
    from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
    from vllm.lora.ops.sgmv_shrink import sgmv_shrink

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.models import LongContextLoRAContext


def compute_meta(
    token_lora_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]:
    """
    Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function
    will combine them into a single request, improving sgmv kernel inference
    performance.
    2. At the beginning of each prefill stage inference, recalculations are
    needed based on the input, but only once.
    """

    lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
        token_lora_tensor, return_counts=True)
    cum_result = torch.cumsum(seq_length_tensor, dim=0)
    b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
    b_seq_start_tensor[1:].copy_(cum_result[:-1])
    max_length = seq_length_tensor.max().item()
    token_nums = seq_length_tensor.sum().item()
    batch_size = lora_indices_tensor.size(0)
    no_lora = False
    # -1 means no lora should be applied. Use `no_lora` to determine whether
    # the current step requires LoRA. If LoRA is not needed, the prefill stage
    # does not need to launch the triton kernel, which can improve performance
    if batch_size == 1 and lora_indices_tensor == -1:
        no_lora = True
    return (b_seq_start_tensor, seq_length_tensor, lora_indices_tensor,
            batch_size, max_length, token_nums, no_lora)


# TODO see if this can be vectorized
def convert_mapping(
    mapping: "LoRAMapping",
    lora_index_to_id: List[Optional[int]],
    max_loras: int,
    vocab_size: int,
    extra_vocab_size: int,
    device: torch.device,
    long_lora_context: Optional["LongContextLoRAContext"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], List[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.
        long_lora_context: Passed if there are long context lora in a batch.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            long_lora_indices: Tensor of shape [batch_size] mapping
                requests to RoPE offsets and rot dims for long LoRAs.
                None if long context lora doesn't exist.
            indices_len: List of lengths of the above tensors. It contains
                (base_indices, sampler_indices, sampler_indices_padded,
                embeddings_indices, long_lora_indices).
    """
    index_mapping_indices: List[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()
    long_lora_offsets: Optional[torch.Tensor] = None
    if long_lora_context:
        long_lora_offsets = torch.zeros(len(index_mapping_indices),
                                        device=device,
                                        dtype=torch.long)
    prompt_mapping: List[int] = [
        lora_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        # TODO index can be slow. optimize
        lora_idx = (lora_index_to_id.index(index_mapping_indices[i])
                    if index_mapping_indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx
        if long_lora_context:
            assert long_lora_offsets is not None
            lora_offset: int = long_lora_context.offsets_by_lora_id.get(
                index_mapping_indices[i], 0)
            long_lora_offsets[i] = lora_offset

    indices_list: List[Union[List[int], torch.Tensor]] = [
        index_mapping_indices,
        lora_indices,
        embedding_indices,
    ]
    if long_lora_context:
        assert long_lora_offsets is not None
        indices_list.append(long_lora_offsets)
    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    prompt_mapping_tensor = torch.tensor(prompt_mapping,
                                         dtype=torch.long,
                                         device=device)
    embeddings_indices = torch.stack([
        indices[2] * extra_vocab_size,
        indices[2] * (vocab_size + extra_vocab_size),
    ])
    embeddings_indices[embeddings_indices == -1] = max_loras - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_loras - 1
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device=device, dtype=torch.long) + (
            sampler_indices_padded * len(sampler_indices_padded))
    long_lora_indices = None
    long_lora_indices_len: Optional[int] = None
    if long_lora_context:
        long_lora_indices = indices[3]
        long_lora_indices_len = long_lora_indices.shape[-1]
    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    ]
    if long_lora_indices_len is not None:
        indices_len.append(long_lora_indices_len)
    else:
        # If long_lora doesn't exist,append None
        indices_len.append(None)

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        long_lora_indices,
        indices_len,
    )


class PunicaWrapper:
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

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_input: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_input,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_input: bool,
    ):
        bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_input)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_input: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_input,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_input: bool,
    ):
        bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices, y_offset,
                          y_slice_size, add_input)

    def _apply_expand(self,
                      y: torch.Tensor,
                      x: torch.Tensor,
                      w_t_all: torch.Tensor,
                      y_offset: Optional[int],
                      y_slice_size: Optional[int],
                      add_input: bool = True):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all` 
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = (self._expand_slice_prefill
                                      if self.is_prefill else
                                      self._expand_slice_decode)
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_input)

    def _apply_bias(
        self,
        indices: torch.Tensor,
        output: torch.Tensor,
        output_slices: Tuple[int, ...],
        lora_bias_stacked: Tuple[Optional[torch.Tensor], ...],
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

    def _apply_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = (self._shrink_prefill
                                if self.is_prefill else self._shrink_decode)
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        scale: float,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
    """

        x = x.view(-1, x.shape[-1])
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx],
                               scale)

    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
        output_slices: Tuple[int, ...],
        offset_start: int = 0,
        add_input=True,
    ) -> None:
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
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_input (bool):  Defaults to True.
            """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices,
                             lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_input=add_input,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_input: bool = True,
    ):
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_input (bool): Default to True.
   
        """

        # Embedding layer only need expand op
        expand_fun: Callable = (self._expand_prefill
                                if self.is_prefill else self._expand_decode)
        expand_fun(y, x, lora_b_stacked, add_input)

    def add_lora_linear(
            self,
            y: torch.Tensor,
            x: torch.Tensor,
            lora_a_stacked: Tuple[torch.Tensor, ...],
            lora_b_stacked: Tuple[torch.Tensor, ...],
            lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
            scale: float,
            output_slices: Tuple[int, ...],
            *,
            buffer: Optional[Tuple[torch.Tensor, ...]] = None) -> None:
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
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self.token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = tuple(
                torch.zeros(
                    (x.size(0), r), dtype=torch.float32, device=x.device)
                for _ in range(len(output_slices)))
        self.add_shrink(buffer, x, lora_a_stacked, scale)
        self.add_expand(y,
                        buffer,
                        lora_b_stacked,
                        None,
                        output_slices,
                        add_input=True)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None) -> None:
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
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)
        # LogitsProcessorWithLoRA always using bgmv.
        bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices, scale)
        bgmv_expand(buffer,
                    lora_b_stacked,
                    y,
                    self.sampler_indices,
                    add_inputs=True)
        y = y.view_as(y_org)
