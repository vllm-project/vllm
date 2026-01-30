# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


class PCPManager:
    """
    Manager for Prefill Context Parallelism (PCP) metadata and buffers.

    This manager encapsulates all PCP-related buffers and logic so that the
    ModelRunner can access them via `self.pcp_manager`.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_num_reqs: int,
        max_padded_num_tokens: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.device = device
        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_padded_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )

        # Buffers for tracking padding (needed for logits_indices and slot_mapping)
        # Create tensors first, then get numpy views to ensure they stay in sync
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            max_num_reqs, device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask_cpu_tensor = torch.zeros(
            max_padded_num_tokens, device="cpu", dtype=torch.bool
        )
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()
        self.pcp_padded_slot_mapping = torch.empty(
            max_padded_num_tokens, dtype=torch.int64, device=device
        )

    def _get_cumsum_and_arange(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(
            cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
        )
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = arange_np[:total_num_tokens] - cumsums_offsets
        return cu_num_tokens, arange

    def compute_rank_indices(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        reorder_batch_threshold: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute which tokens this PCP rank processes and their indices
        into the original batch.

        When using Prefill Context Parallelism (PCP), each request's prefill
        sequence is split across multiple PCP ranks. The splitting strategy
        used here is the "DualChunkSwap" style: each request's (padded)
        sequence is split into 2 * pcp_world_size chunks and ranks are
        assigned chunks in an interleaved head/tail pattern to balance load.

        This method:
        - Computes how many tokens each request should be processed by the
          current PCP rank (pcp_num_scheduled).
        - Computes indices into the original batch for gathering real tokens.
        - Updates self.pcp_allgather_restore_idx: index array used to restore
          original ordering after per-rank allgather and interleaving.

        Args:
            num_scheduled_tokens: 1D numpy array of per-request token counts.
            arange_np: Pre-allocated arange buffer for efficient operations.
            reorder_batch_threshold: Threshold for decode vs prefill requests.

        Returns:
            Tuple (pcp_num_scheduled, local_token_indices):
            - pcp_num_scheduled: per-request token counts for this rank
            - local_token_indices: numpy array of indices into original batch
              for gathering positions (padding indices are clamped to 0)

        Example:
            Assume tokens = [1, 5, 8], pcp_world_size = 2:
            - pcp_rank=0 gets pcp_num_scheduled=[1, 4, 4],
              local_token_indices=[0,0,1,6,7,0,1,6,7]
            - pcp_rank=1 gets pcp_num_scheduled=[1, 4, 4],
              local_token_indices=[0,2,3,4,5,2,3,4,5]
            - pcp_allgather_restore_idx:
              [0,9,1,2,10,11,12,13,3,4,5,6,14,15,16,17,7,8]
        """
        assert reorder_batch_threshold is not None, (
            "PCP depends on reorder batch to split decode and prefill requests."
        )
        num_reqs = len(num_scheduled_tokens)
        num_decode_reqs = sum(num_scheduled_tokens <= reorder_batch_threshold)
        num_decode_tokens = sum(num_scheduled_tokens[:num_decode_reqs])

        # DualChunkSwap requires alignment to a multiple of (2 * pcp_world_size).
        # We first pad each request's token count up to that multiple.
        num_padded_scheduled_tokens = np.ceil(
            num_scheduled_tokens / (2 * self.pcp_world_size)
        ).astype(np.int32) * (2 * self.pcp_world_size)
        # PCP does not split decode requests. For decode requests, we instead
        # duplicate the scheduled tokens across the pcp_world_size ranks.
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            num_scheduled_tokens[:num_decode_reqs] * self.pcp_world_size
        )

        # Track padding per request (for logits_indices calculation)
        self.num_pcp_pads_cpu[:num_reqs] = (
            num_padded_scheduled_tokens - num_scheduled_tokens
        )

        # cu_padded_tokens: cumulative sum of padded token counts
        # pcp_padded_arange: per-request arange flattened for padded tokens
        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )

        # Build mask marking real (unpadded) tokens in the all-gather buffer
        # (pcp_unpad_mask_cpu is a view of pcp_unpad_mask_cpu_tensor)
        padded_total = pcp_padded_arange.shape[0]
        self.pcp_unpad_mask_cpu[:padded_total] = pcp_padded_arange < np.repeat(
            num_scheduled_tokens, num_padded_scheduled_tokens
        )

        # pcp_tokens: tokens per request for this rank after splitting
        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size

        # Compute per-request "chunk sizes" for the head/tail splitting.
        # For prefill requests, we further split the pcp_tokens into two chunks
        # (head and tail). For decode requests, the chunk equals pcp_tokens.
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        # Build arange-style helpers for pcp tokens and chunk sizes:
        # - pcp_arange gives indices repeated for each token in pcp_tokens
        # - pcp_chunk_arange gives indices repeated for each position inside chunks
        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(pcp_chunk_sizes, arange_np)

        # Mask that marks whether a position belongs to the head chunk (True)
        # or the tail chunk (False). For decode requests, tail chunk won't exist
        # and is handled specially below.
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(
            positions_start_loc: int | np.ndarray, rank: int
        ) -> np.ndarray:
            """
            Compute flattened positions for the given rank with a given start
            offset for each request (positions_start_loc).

            - For head chunks: start at positions_start_loc + rank * chunk_size.
            - For tail chunks: start at positions_start_loc +
              (2*pcp_world_size - rank - 1) * chunk_size.
            - For decode requests: no tail chunks; their positions are filled
              from the contiguous (unpadded) `tokens` arange instead.
            """
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = (
                positions_start_loc
                + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            )
            # Fill head positions using chunk arange offset by head_start_loc.
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
                head_start_loc, pcp_chunk_sizes
            )
            # Fill tail positions. Note decode requests do not have tail chunks,
            # so the tail filling is only for prefill positions.
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            )
            return positions

        # Get positions for this rank (position VALUES, not indices)
        positions = get_current_rank_positions(0, self.pcp_rank)
        # Decode tokens are duplicated only after allgather. But their positions
        # are the same without prefill context parallel.
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                num_scheduled_tokens[:num_decode_reqs], arange_np
            )[1]

        # Build the restore index used after allgather.
        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions_lst = [
            get_current_rank_positions(padded_pos_start_loc, rank_i)
            for rank_i in range(self.pcp_world_size)
        ]
        all_positions = np.concatenate(all_positions_lst)
        self.pcp_allgather_restore_idx.np[: all_positions.shape[0]] = (
            all_positions.argsort()
        )
        self.pcp_allgather_restore_idx.copy_to_gpu(all_positions.shape[0])

        # Now compute indices into original batch
        # positions[i] is the position VALUE (relative) for PCP token i
        # We need to find which original token index that corresponds to

        # Compute cumsum of original tokens for request start offsets
        cu_orig_tokens = np.cumsum(num_scheduled_tokens)
        orig_start_offsets = np.concatenate([[0], cu_orig_tokens[:-1]])

        pcp_total_tokens = int(pcp_tokens.sum())

        # padding_mask: True if position >= original seq len for that request
        orig_seq_lens_expanded = np.repeat(num_scheduled_tokens, pcp_tokens)
        padding_mask_np = positions[:pcp_total_tokens] >= orig_seq_lens_expanded

        # For real tokens, compute index into original batch
        # original_index = orig_start_offset[req] + position
        orig_start_expanded = np.repeat(orig_start_offsets, pcp_tokens)
        local_token_indices = orig_start_expanded + positions[:pcp_total_tokens]
        # Clamp padding indices to 0 (position won't matter for padding)
        local_token_indices = np.where(padding_mask_np, 0, local_token_indices)

        return pcp_tokens[:num_reqs], local_token_indices.astype(np.int64)

    def restore_hidden_states(
        self, hidden_states: torch.Tensor, num_tokens_unpadded: int
    ):
        # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
        # ignores the padding from CUDA Graph.
        hidden_states = get_pcp_group().all_gather(
            hidden_states[:num_tokens_unpadded],
            0,
        )
        restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
        hidden_states = torch.index_select(
            hidden_states,
            0,
            restore_idx,
        )
        # Remove padding tokens so logits_indices can use normal calculation
        unpad_mask = self.pcp_unpad_mask_cpu_tensor[: hidden_states.shape[0]]
        return hidden_states[unpad_mask]

    def pad_slot_mapping(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """
        Expand slot_mapping for the all-gathered KV cache.

        After KV all-gather, slot_mapping needs to account for padding.
        This places real slot values at unpadded positions and -1 at padding.
        """
        num_tokens = slot_mapping.shape[0]
        padded_size = num_tokens * self.pcp_world_size
        pcp_padded_slot_mapping = self.pcp_padded_slot_mapping[:padded_size]
        cp_unpad_mask = self.pcp_unpad_mask_cpu_tensor[:padded_size]
        pcp_padded_slot_mapping.fill_(-1)
        # During warmup, mask may not be set yet
        if cp_unpad_mask.sum().item() > 0:
            pcp_padded_slot_mapping[cp_unpad_mask] = slot_mapping
        return pcp_padded_slot_mapping


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer_impl.__class__.__name__} "
                    "does not return the softmax lse for decode."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
