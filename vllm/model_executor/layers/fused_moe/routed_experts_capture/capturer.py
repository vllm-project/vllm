# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts_per_tok,
)
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer hooks call ``capture`` during the forward pass with the per-layer
    ``topk_ids``; the rows owned by this DP rank are written into a
    preallocated device buffer. ``GPUModelRunner`` then D2Hs the buffer and
    hands it to the scheduler via ``RoutedExpertsLists``.

    The transit buffer is ``torch.int32`` (not the narrow ``uint8``/``uint16``
    the scheduler slot buffer uses): it matches the router's native dtype,
    avoids casts on the SP all-gather path, is universally NCCL-supported, and
    costs only a few MB per worker. ``store_batch`` narrows on the way in.

    Invariants: one instance per worker; shape fixed at init for the
    worst-case step (``max_num_batched_tokens``); ``clear_buffer`` runs each
    step so unused slots stay zero.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        num_experts_per_tok = get_num_experts_per_tok(hf_config)
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                hf_config.num_hidden_layers,
                num_experts_per_tok,
            ),
            dtype=torch.int32,  # see class docstring for why int32, not narrow
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Write one layer's routing into this DP rank's buffer slot.

        Under data parallelism ``topk_ids`` arrives in one of three batch
        layouts (naive dispatch / modular-kernel / SP+modular-kernel); each
        branch below recovers this rank's rows. See the per-branch comments.

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
                # before routing. This rank owns tokens
                # [end_loc - token_num_per_dp, end_loc).
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens, take the whole tensor.
                start_loc = 0
                end_loc = token_num_per_dp
            elif (
                self.tp_size > 1
                and n != token_num_per_dp
                and n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
            ):
                # SP + modular-kernel: each TP rank holds a dim=0 shard, so
                # all-gather to rebuild this DP rank's full tensor and keep the
                # first token_num_per_dp rows (rest is SP ceil-div padding).
                # Every rank in the group hits this branch in lockstep, so the
                # bare all_gather can't deadlock; let it raise rather than skip
                # if the precondition is violated. The router's int32/int64
                # ids narrow into the int32 buffer on the setitem below.
                topk_ids = get_tp_group().all_gather(topk_ids, dim=0)
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                sp_expected = (
                    (token_num_per_dp + self.tp_size - 1) // self.tp_size
                    if self.tp_size > 0
                    else -1
                )
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {n} (expected {total}, {token_num_per_dp}, "
                    f"or {sp_expected} for dp_rank={self.dp_rank}, "
                    f"tp_size={self.tp_size})"
                )

        # Defensive: model may expose more layers than the capture buffer
        # was sized for (unusual, but guards against miss-config).
        if layer_id >= self.device_buffer.shape[1]:
            return

        self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step so
        slots belonging to finished / preempted tokens don't leak into
        the next step.
        """
        self.device_buffer.zero_()

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass runs
        ``clear_buffer``.
        """
        return self.device_buffer
