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
    """Manager for Prefill Context Parallelism (PCP) buffers and partitioning."""

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
        self._pcp_unpad_mask_tensor = torch.zeros(
            max_padded_num_tokens, device="cpu", dtype=torch.bool
        )
        self.pcp_unpad_mask = self._pcp_unpad_mask_tensor.numpy()
        self.pcp_padded_slot_mapping = torch.empty(
            max_padded_num_tokens, dtype=torch.int64, device=device
        )
        # Cached values from partition_inputs
        self.local_num_scheduled: np.ndarray = np.array([], dtype=np.int32)
        self.local_total: int = 0

    def partition_inputs(
        self,
        positions_np: np.ndarray,
        req_indices: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        reorder_batch_threshold: int | None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Partition inputs for this PCP rank using DualChunkSwap splitting.

        Each request's prefill tokens are split across PCP ranks using a
        "DualChunkSwap" pattern: tokens are padded to a multiple of 2*world_size,
        then split into head/tail chunks assigned to ranks in an interleaved
        pattern to balance computation across the sequence.

        For decode requests (tokens <= reorder_batch_threshold), tokens are
        duplicated across all ranks instead of split.

        This method:
        1. Computes which tokens this rank processes
        2. Gathers local positions/req_indices from the global arrays
        3. Builds pcp_allgather_restore_idx for reordering after all-gather
        4. Builds pcp_unpad_mask for removing padding after restore

        Args:
            positions_np: Global position values, modified in-place to local
            req_indices: Global request indices, modified in-place to local
            num_scheduled_tokens: Per-request token counts
            arange_np: Pre-allocated arange buffer
            reorder_batch_threshold: Threshold distinguishing decode vs prefill

        Returns:
            (local_total, positions_np[:local_total], req_indices[:local_total])
        """
        assert reorder_batch_threshold is not None
        num_reqs = len(num_scheduled_tokens)
        num_decode_reqs = int((num_scheduled_tokens <= reorder_batch_threshold).sum())
        num_decode_tokens = int(num_scheduled_tokens[:num_decode_reqs].sum())
        ws = self.pcp_world_size

        # Pad to multiple of 2*ws; decode reqs are duplicated instead
        padded = np.ceil(num_scheduled_tokens / (2 * ws)).astype(np.int32) * (2 * ws)
        padded[:num_decode_reqs] = num_scheduled_tokens[:num_decode_reqs] * ws

        # Cumsum and arange for padded tokens
        cu_padded = np.cumsum(padded)
        padded_total = cu_padded[-1]
        padded_arange = arange_np[:padded_total] - np.repeat(cu_padded - padded, padded)

        # Unpad mask: True for real tokens
        self.pcp_unpad_mask[:padded_total] = padded_arange < np.repeat(
            num_scheduled_tokens, padded
        )

        # Tokens per request for this rank
        pcp_tokens = padded // ws
        chunk = (pcp_tokens // 2).clip(min=1)
        chunk[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        # Aranges for pcp_tokens and chunks
        cu_pcp = np.cumsum(pcp_tokens)
        pcp_arange = arange_np[: cu_pcp[-1]] - np.repeat(
            cu_pcp - pcp_tokens, pcp_tokens
        )
        cu_chunk = np.cumsum(chunk)
        chunk_arange = arange_np[: cu_chunk[-1]] - np.repeat(cu_chunk - chunk, chunk)

        # Head/tail mask
        head_mask = pcp_arange < np.repeat(chunk, pcp_tokens)

        def get_positions(start_loc: int | np.ndarray, rank: int) -> np.ndarray:
            """Get positions for given rank with DualChunkSwap interleaving."""
            pos = np.zeros(len(head_mask), dtype=np.int32)
            head_start = start_loc + rank * chunk
            tail_start = start_loc + (2 * ws - rank - 1) * chunk
            pos[head_mask] = chunk_arange + np.repeat(head_start, chunk)
            pos[~head_mask] = (
                chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start, chunk)[num_decode_tokens:]
            )
            return pos

        # Positions for this rank
        positions = get_positions(0, self.pcp_rank)
        if num_decode_reqs > 0:
            cu_dec = np.cumsum(num_scheduled_tokens[:num_decode_reqs])
            positions[:num_decode_tokens] = arange_np[:num_decode_tokens] - np.repeat(
                cu_dec - num_scheduled_tokens[:num_decode_reqs],
                num_scheduled_tokens[:num_decode_reqs],
            )

        # Build restore index for all-gather
        padded_start = np.concatenate([[0], cu_padded[:-1]])
        all_pos = np.concatenate([get_positions(padded_start, r) for r in range(ws)])
        self.pcp_allgather_restore_idx.np[: len(all_pos)] = all_pos.argsort()
        self.pcp_allgather_restore_idx.copy_to_gpu(len(all_pos))

        # Convert position values to indices into original batch
        cu_orig = np.cumsum(num_scheduled_tokens)
        orig_start = np.concatenate([[0], cu_orig[:-1]])
        pcp_total = int(pcp_tokens.sum())
        orig_lens = np.repeat(num_scheduled_tokens, pcp_tokens)
        orig_starts = np.repeat(orig_start, pcp_tokens)
        local_indices = np.where(
            positions[:pcp_total] >= orig_lens,
            0,  # Clamp padding to 0
            orig_starts + positions[:pcp_total],
        ).astype(np.int64)

        # Gather local values
        local_total = pcp_total
        positions_np[:local_total] = positions_np[local_indices]
        req_indices[:local_total] = req_indices[local_indices]

        self.local_num_scheduled = pcp_tokens[:num_reqs]
        self.local_total = local_total
        return local_total, positions_np[:local_total], req_indices[:local_total]

    def restore_hidden_states(
        self, hidden_states: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """All-gather hidden states, restore order, and remove padding."""
        hidden_states = get_pcp_group().all_gather(hidden_states[:num_tokens], 0)
        restore_idx = self.pcp_allgather_restore_idx.gpu[: hidden_states.shape[0]]
        hidden_states = hidden_states.index_select(0, restore_idx)
        return hidden_states[self._pcp_unpad_mask_tensor[: hidden_states.shape[0]]]

    def pad_slot_mapping(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """
        Expand slot_mapping for the all-gathered KV cache.

        After KV all-gather, slot_mapping needs to account for padding.
        This places real slot values at unpadded positions and -1 at padding.
        """
        padded_size = slot_mapping.shape[0] * self.pcp_world_size
        out = self.pcp_padded_slot_mapping[:padded_size]
        out.fill_(-1)
        mask = self._pcp_unpad_mask_tensor[:padded_size]
        if mask.sum().item() > 0:
            out[mask] = slot_mapping
        return out


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.dcp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with dcp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return the softmax lse "
                    f"for decode, but {layer_impl.__class__.__name__} does not."
                )
            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    f"PCP requires attention impl support, but "
                    f"{layer_impl.__class__.__name__} does not support PCP."
                )


def get_total_cp_world_size():
    """Get total context parallelism world size for KV cache sharding.

    Only DCP shards the KV cache. With PCP, K/V are gathered after prefill
    so each rank has the full sequence - no KV sharding.
    """
    try:
        return get_dcp_group().world_size
    except AssertionError:
        return 1
