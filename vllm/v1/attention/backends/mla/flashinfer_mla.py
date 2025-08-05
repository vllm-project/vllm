# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch

from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)
from torch.nn.utils.rnn import pad_sequence

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

class FlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl


g_fi_workspace = torch.empty(
    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
    dtype=torch.uint8,
    device="cuda",
)

def pad_tensor_with_lengths(q: torch.Tensor, q_len: torch.Tensor) -> torch.Tensor:
    """
    Pads a tensor based on a corresponding length tensor.

    This function takes a tensor 'q', which is a flattened concatenation of
    sequences, and a 1D tensor 'q_len' containing the length of each
    sequence. It pads each sequence to the length of the longest sequence
    in the batch.

    Args:
        q (torch.Tensor): The input tensor of shape [X, a, b, c], where
                          X is the sum of all sequence lengths.
        q_len (torch.Tensor): A 1D tensor of shape [K,] containing the
                              lengths of the K sequences in the batch.
                              The sum of `q_len` must equal X.

    Returns:
        torch.Tensor: The padded tensor of shape [K, max(q_len), a, b, c].
    """
    # --- Input Validation ---
    if not isinstance(q, torch.Tensor) or not isinstance(q_len, torch.Tensor):
        raise TypeError("Inputs 'q' and 'q_len' must be PyTorch tensors.")
    if q.shape[0] != torch.sum(q_len):
        raise ValueError(
            f"Sum of lengths in q_len ({torch.sum(q_len)}) must be equal to "
            f"the first dimension of q ({q.shape[0]})."
        )

    # Split the flattened tensor `q` into a list of tensors.
    sequences = torch.split(q, q_len.tolist())

    # Pad the sequences to the length of the longest one.
    padded_q = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_q

def unpad_tensor_with_lengths(padded_q: torch.Tensor, q_len: torch.Tensor) -> torch.Tensor:
    """
    Unpads a tensor, reversing the padding operation.

    This function takes a padded tensor and the original sequence lengths,
    and returns the original flattened, non-padded tensor.

    Args:
        padded_q (torch.Tensor): The padded tensor of shape [K, max_len, a, b, c].
        q_len (torch.Tensor): A 1D tensor of shape [K,] containing the
                              original lengths of the K sequences.

    Returns:
        torch.Tensor: The unpadded (flattened) tensor of shape [X, a, b, c],
                      where X is the sum of `q_len`.
    """
    # --- "Magic Math Function" (Identity for now) ---
    # This is where you would apply any operation to the padded tensor.
    processed_q = padded_q # Identity function

    # --- Unpadding Logic ---
    K, max_len = processed_q.shape[0], processed_q.shape[1]

    # Create a boolean mask to identify the non-padded elements.
    # 1. Create a range tensor: [0, 1, 2, ..., max_len-1]
    arange_max_len = torch.arange(max_len, device=padded_q.device)
    # 2. Expand q_len to be comparable with the range tensor.
    #    q_len shape: [K] -> [K, 1]
    # 3. Broadcasting compares each element of q_len with the entire range,
    #    creating a mask of shape [K, max_len].
    mask = arange_max_len < q_len.unsqueeze(-1)

    # Apply the mask. Boolean indexing selects only the `True` elements
    # and returns them as a flattened tensor. The shape will automatically
    # become [X, a, b, c].
    unpadded_q = processed_q[mask]

    return unpadded_q

def slice_qlen_to_match_sum(x_shape_0: int, q_len: torch.Tensor) -> torch.Tensor:
    """
    Slices a length tensor `q_len` so that its sum matches a target value.

    This operation is performed efficiently on the tensor's device (e.g., GPU)
    without requiring a CPU sync. It's useful when you have a `q_len` tensor
    that is longer than necessary for a given batch.

    Args:
        x_shape_0 (int): The target sum, typically from a tensor's shape
                         (e.g., `q.shape[0]`).
        q_len (torch.Tensor): The 1D length tensor to be sliced.

    Returns:
        torch.Tensor: The sliced 1D length tensor whose elements sum to `x_shape_0`.

    Raises:
        ValueError: If the total sum of `q_len` is less than `x_shape_0`.
    """
    # Calculate the cumulative sum of the lengths. This is a key operation
    # that can be performed efficiently on the GPU.
    # Example: q_len = [2, 1, 2, 2, 100] -> cumsum = [2, 3, 5, 7, 107]
    cumulative_sums = torch.cumsum(q_len, dim=0)

    # Find the index of the first element in the cumulative sum that is
    # greater than or equal to the target sum `x_shape_0`.
    # `cumulative_sums >= x_shape_0` creates a boolean mask:
    # Example (x_shape_0=7): [F, F, F, T, T]
    # `torch.where` can be used to get the indices of the True values.
    # We take the first one, as that marks our boundary.
    possible_indices = torch.where(cumulative_sums >= x_shape_0)[0]

    # If no index is found, it means the total sum of q_len is smaller
    # than the required size, which is an invalid state.
    if len(possible_indices) == 0:
        raise ValueError(
            f"The sum of q_len ({torch.sum(q_len)}) is less than the "
            f"target sum ({x_shape_0})."
        )

    # The index we need is the first one where the cumulative sum met the target.
    # We need to slice up to and *including* this index, so we add 1.
    slice_end_index = possible_indices[0] + 1

    # Return the sliced tensor.
    return q_len[:slice_end_index]

class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashInferMLA V1 with FP8 KV cache not yet supported")
        
        self._workspace_buffer = g_fi_workspace

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        q = torch.cat([q_nope, q_pe], dim=-1) # (batch_size * q_len_per_request, num_heads, head_dim_qk)

        # BS 4, Qlen 2
        # [2, 1, 2, 2]
        # q -> [1, 2, 3, 4, 5, 6, 7]

        # q -> [[1, 2], [3, 0], [4, 5], [6, 7]]

        batch_size = attn_metadata.decode.block_table.shape[0]
        max_seq_len = attn_metadata.decode.seq_lens.max().item()

        needs_padding = q.shape[0] % batch_size != 0

        if needs_padding:
            qlen = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            if qlen.shape[0] != batch_size:
                qlen = slice_qlen_to_match_sum(q.shape[0], qlen)
            q = pad_tensor_with_lengths(q, qlen)
        else:
            q = q.reshape((batch_size, q.shape[0] // batch_size, *q.shape[1:])) # (batch_size, q_len_per_request, num_heads, head_dim_qk)
        # if needs_padding:
        #     breakpoint()
        #     raise ValueError(
        #         "Padding is not supported in FlashInferMLAImpl. "
        #         "Ensure that the input tensor q has a shape that is "
        #         "a multiple of the batch size."
        #     )

        # q = q.unsqueeze(0)
        # B = q.shape[0]

        # trtllm API extras extra dimension for MTP
        # o = torch.empty(B,
        #                 self.num_heads,
        #                 self.kv_lora_rank,
        #                 dtype=q.dtype,
        #                 device=q.device)
        

        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=self.scale,
        )

        if needs_padding:
            o = unpad_tensor_with_lengths(o, qlen)

        return self._v_up_proj(o)
