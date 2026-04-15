# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class RoutedExpertsPendingCopy:
    """Pending non-blocking D2H copy for routed experts capture.

    Both ``cpu_buffer`` and ``slot_mapping`` are copied from GPU with
    ``non_blocking=True`` on the default stream.  Call :meth:`finalize`
    only after the stream has been synchronized (or an event recorded on
    that stream has been waited on).

    NOTE: ``cpu_buffer`` holds a **reference** to the shared pinned
    buffer on GPUModelRunner, not a copy.  ``finalize()`` must be called
    before the next ``execute_model()`` overwrites the buffer.  This is
    guaranteed when ``batch_queue_size <= 1`` (sync scheduling).
    """

    cpu_buffer: torch.Tensor
    total: int
    slot_mapping: torch.Tensor  # D2H in-flight; call .numpy() after sync

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays after D2H synchronization."""
        return self.cpu_buffer[: self.total].numpy(), self.slot_mapping.numpy()


def get_num_experts(hf_config) -> int:
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(hf_config, key, None)
        if val is not None:
            return val
    raise ValueError(
        "Could not resolve num_experts from model config. "
        "Expected one of 'num_experts', 'n_routed_experts', "
        "or 'num_local_experts'."
    )


class RoutedExpertsCapturer:
    """
    Capturer for routed experts with device buffer.

    This class captures expert routing decisions during model forward passes
    and stores them in a device buffer for later extraction via
    ModelRunnerOutput.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        dtype = torch.uint8 if num_experts <= 256 else torch.int32
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                hf_config.num_experts_per_tok,
            ),
            dtype=dtype,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated
                # before routing.
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens.
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch dim "
                    f"{n} (expected {total} or {token_num_per_dp} "
                    f"for dp_rank={self.dp_rank})"
                )

        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Get the device buffer for external extraction."""
        return self.device_buffer


class RoutedExpertsManager:
    """Slot-indexed buffer that stores and retrieves routed experts data.

    Each slot corresponds to block_id * block_size + offset_in_block, so
    data is tied to physical KV-cache blocks and survives preemption for
    prefix-cached blocks.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Find the attention group for block/slot mapping.
        self.attn_gid = next(
            (
                gid
                for gid, g in enumerate(kv_cache_config.kv_cache_groups)
                if isinstance(g.kv_cache_spec, AttentionSpec)
            ),
            0,
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # Routed experts indexed by KV-cache slot.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        dtype = np.uint8 if num_experts <= 256 else np.int32
        # Use the full block pool size: block IDs span [0, num_blocks)
        # regardless of how many kv_cache_groups exist, because all groups
        # share the same physical block pool.
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                hf_config.num_hidden_layers,
                hf_config.num_experts_per_tok,
            ),
            dtype=dtype,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Store a whole batch of routed experts using pre-computed slot mapping.

        Equivalent to the old shared-memory write:
            shared_memory[slot_mapping] = data
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def get(self, block_ids: list[int], num_tokens: int) -> np.ndarray:
        """Read routed experts data for a completed request.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of generated tokens (excluding the last).

        Returns:
            Array of shape (num_tokens, num_layers, num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        return self.routed_experts_by_slot[slot_mapping]
