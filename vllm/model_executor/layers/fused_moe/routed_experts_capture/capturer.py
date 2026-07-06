# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts_per_token,
)
from vllm.platforms import current_platform


class RoutedExpertsCapturer:
    """Worker-side GPU buffer for per-layer routed expert IDs."""

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        moe_top_k = get_num_experts_per_token(hf_config)
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                moe_top_k,
            ),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Write one layer's routing IDs into this DP rank's buffer slot.

        Under data parallelism, topk_ids may cover all DP ranks, this DP rank
        only, or a sequence-parallel shard of this rank. We slice out and keep
        just this rank's rows.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (num_tokens, moe_top_k).
        """

        forward_context = get_forward_context()
        if forward_context.dp_metadata is None:
            start_index = 0
            end_index = topk_ids.shape[0]
            num_tokens_for_dp_rank = topk_ids.shape[0]
        else:
            num_tokens_across_dp = forward_context.dp_metadata.num_tokens_across_dp_cpu
            num_tokens_for_dp_rank = int(num_tokens_across_dp[self.dp_rank].item())
            total_num_tokens = int(num_tokens_across_dp.sum().item())
            num_input_tokens = topk_ids.shape[0]

            if num_input_tokens == total_num_tokens:
                cumulative_num_tokens = torch.cumsum(num_tokens_across_dp, dim=0)
                end_index = int(cumulative_num_tokens[self.dp_rank].item())
                start_index = end_index - num_tokens_for_dp_rank
            elif num_input_tokens == num_tokens_for_dp_rank:
                start_index = 0
                end_index = num_tokens_for_dp_rank
            elif (
                self.tp_size > 1
                and num_input_tokens
                == (num_tokens_for_dp_rank + self.tp_size - 1) // self.tp_size
            ):
                # Sequence parallelism split this rank's tokens across TP
                # ranks. Gather the shards back, then drop the padding rows.
                topk_ids = get_tp_group().all_gather(topk_ids, dim=0)
                start_index = 0
                end_index = num_tokens_for_dp_rank
            else:
                sequence_parallel_shard_size = (
                    num_tokens_for_dp_rank + self.tp_size - 1
                ) // self.tp_size
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {num_input_tokens} (expected {total_num_tokens}, "
                    f"{num_tokens_for_dp_rank}, or "
                    f"{sequence_parallel_shard_size} "
                    f"for dp_rank={self.dp_rank}, "
                    f"tp_size={self.tp_size})"
                )

        if layer_id >= self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts capture layer_id {layer_id} exceeds buffer "
                f"layer dim {self.device_buffer.shape[1]} (num_hidden_layers "
                "mismatch); routing would be silently dropped"
            )

        self.device_buffer[:num_tokens_for_dp_rank, layer_id, :] = topk_ids[
            start_index:end_index, :
        ]

    def clear_buffer(self) -> None:
        """Zero the device buffer before each step."""
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the shared device buffer for the model runner D2H copy."""
        return self.device_buffer
