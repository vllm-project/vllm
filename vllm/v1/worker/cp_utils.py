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
    """Manager for Prefill Context Parallelism (PCP) buffers and partitioning.

    PCP partitions prefill tokens across ranks using a DualChunkSwap pattern.
    The manager owns:
      - the `pcp_allgather_restore_idx` buffer used to reorder K/V after the
        cross-rank all-gather inside the attention kernel, and
      - the `pcp_padded_slot_mapping` used to write K/V padding slots back
        to the global cache as PAD (-1).
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
        self._pcp_unpad_mask_tensor = torch.zeros(
            max_padded_num_tokens, device="cpu", dtype=torch.bool
        )
        self.pcp_unpad_mask = self._pcp_unpad_mask_tensor.numpy()
        self.pcp_padded_slot_mapping = torch.empty(
            max_padded_num_tokens, dtype=torch.int64, device=device
        )
        self.global_num_scheduled_tokens = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int32, device=device, pin_memory=pin_memory
        )
        # Cached values from partition_inputs.
        self.local_num_scheduled: np.ndarray = np.array([], dtype=np.int32)
        self.local_total: int = 0
        self.padded_total: int = 0
        self.global_total: int = 0

    def partition_inputs(
        self,
        positions_np: np.ndarray,
        req_indices: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        arange_np: np.ndarray,
        reorder_batch_threshold: int | None,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Partition inputs for this PCP rank using DualChunkSwap splitting.

        Each request's prefill tokens are padded to a multiple of 2*world_size
        then split into head/tail chunks assigned to ranks in an interleaved
        pattern that balances causal-attention compute.

        Decode requests (tokens <= reorder_batch_threshold AND has context)
        are duplicated across all PCP ranks instead of being split.

        Side effects:
            - `positions_np` is overwritten in-place with this rank's positions
              (relative to each request's start, plus its `num_computed_tokens`).
            - `req_indices` is overwritten in-place with this rank's request indices.
            - `self.pcp_allgather_restore_idx`, `self.pcp_unpad_mask`,
              `self.global_num_scheduled_tokens` are populated.

        Returns:
            (local_total, positions_np[:local_total], req_indices[:local_total],
             gathered_positions[:local_total]) where ``gathered_positions`` is
            the per-token original-batch position used for input token indexing
            (with padding positions clamped to each request's start).
        """
        assert reorder_batch_threshold is not None
        num_reqs = len(num_scheduled_tokens)
        is_decode = (num_scheduled_tokens <= reorder_batch_threshold) & (
            num_computed_tokens > 0
        )
        num_decode_reqs = int(is_decode.sum())
        num_decode_tokens = int(num_scheduled_tokens[:num_decode_reqs].sum())
        ws = self.pcp_world_size

        padded = np.ceil(num_scheduled_tokens / (2 * ws)).astype(np.int32) * (2 * ws)
        padded[:num_decode_reqs] = num_scheduled_tokens[:num_decode_reqs] * ws

        cu_padded = np.cumsum(padded)
        padded_total = cu_padded[-1]
        padded_arange = arange_np[:padded_total] - np.repeat(cu_padded - padded, padded)

        self.pcp_unpad_mask[:padded_total] = padded_arange < np.repeat(
            num_scheduled_tokens, padded
        )

        pcp_tokens = padded // ws
        chunk = (pcp_tokens // 2).clip(min=1)
        chunk[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        cu_pcp = np.cumsum(pcp_tokens)
        pcp_arange = arange_np[: cu_pcp[-1]] - np.repeat(
            cu_pcp - pcp_tokens, pcp_tokens
        )
        cu_chunk = np.cumsum(chunk)
        chunk_arange = arange_np[: cu_chunk[-1]] - np.repeat(cu_chunk - chunk, chunk)

        head_mask = pcp_arange < np.repeat(chunk, pcp_tokens)

        def get_positions(start_loc, rank: int) -> np.ndarray:
            pos = np.zeros(len(head_mask), dtype=np.int32)
            head_start = start_loc + rank * chunk
            tail_start = start_loc + (2 * ws - rank - 1) * chunk
            pos[head_mask] = chunk_arange + np.repeat(head_start, chunk)
            pos[~head_mask] = (
                chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start, chunk)[num_decode_tokens:]
            )
            return pos

        positions = get_positions(0, self.pcp_rank)
        if num_decode_reqs > 0:
            cu_dec = np.cumsum(num_scheduled_tokens[:num_decode_reqs])
            positions[:num_decode_tokens] = arange_np[:num_decode_tokens] - np.repeat(
                cu_dec - num_scheduled_tokens[:num_decode_reqs],
                num_scheduled_tokens[:num_decode_reqs],
            )

        padded_start = np.concatenate([[0], cu_padded[:-1]])
        all_pos = np.concatenate([get_positions(padded_start, r) for r in range(ws)])
        self.pcp_allgather_restore_idx.np[: len(all_pos)] = all_pos.argsort()
        self.pcp_allgather_restore_idx.copy_to_gpu(len(all_pos))

        cu_orig = np.cumsum(num_scheduled_tokens)
        orig_start = np.concatenate([[0], cu_orig[:-1]])
        pcp_total = int(pcp_tokens.sum())
        orig_lens = np.repeat(num_scheduled_tokens, pcp_tokens)
        orig_starts = np.repeat(orig_start, pcp_tokens)
        # For padding positions (position >= seq_len), clamp to the request's
        # first token so we don't index into a neighbor's positions.
        local_indices = np.where(
            positions[:pcp_total] >= orig_lens,
            orig_starts,
            orig_starts + positions[:pcp_total],
        ).astype(np.int64)

        gathered_positions = positions_np[local_indices].copy()

        # Actual RoPE positions: padding clamped to 0 so RoPE matches the
        # cloned token content.
        is_padding = positions[:pcp_total] >= orig_lens
        clamped_positions = np.where(is_padding, 0, positions[:pcp_total])
        req_computed_tokens = np.repeat(num_computed_tokens, pcp_tokens)
        positions_np[:pcp_total] = req_computed_tokens[:pcp_total] + clamped_positions

        req_indices[:pcp_total] = req_indices[local_indices]

        self.local_num_scheduled = pcp_tokens[:num_reqs]
        self.local_total = pcp_total
        self.padded_total = int(padded_total)
        self.global_total = int(num_scheduled_tokens.sum())

        gns = self.global_num_scheduled_tokens
        gns.np[:num_reqs] = num_scheduled_tokens
        gns.np[num_reqs:].fill(0)
        gns.copy_to_gpu()

        return (
            pcp_total,
            positions_np[:pcp_total],
            req_indices[:pcp_total],
            gathered_positions,
        )

    def restore_hidden_states(
        self, hidden_states: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """All-gather hidden states across PCP, restore order, and unpad."""
        hidden_states = get_pcp_group().all_gather(hidden_states[:num_tokens], 0)
        restore_size = (
            self.padded_total if self.padded_total > 0 else hidden_states.shape[0]
        )
        restore_idx = self.pcp_allgather_restore_idx.gpu[:restore_size]
        hidden_states = hidden_states.index_select(0, restore_idx)
        mask = self._pcp_unpad_mask_tensor[:restore_size].to(
            hidden_states.device, non_blocking=True
        )
        return hidden_states[mask]

    def pad_slot_mapping(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """Expand slot_mapping for the all-gathered KV cache.

        After K/V all-gather, slot_mapping needs to account for per-request
        PCP padding. Real slots go at unpadded positions; padding positions
        get -1 (so reshape_and_cache writes are no-ops there).
        """
        if self.padded_total == 0:
            return slot_mapping

        out = self.pcp_padded_slot_mapping[: self.padded_total]
        out.fill_(-1)
        mask = self._pcp_unpad_mask_tensor[: self.padded_total].to(
            slot_mapping.device, non_blocking=True
        )
        out[mask] = slot_mapping
        return out


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
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
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
