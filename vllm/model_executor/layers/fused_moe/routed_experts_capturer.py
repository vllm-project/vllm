# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


def _get_routed_experts_shape(vllm_config: VllmConfig) -> tuple[int, int, int]:
    model_config = vllm_config.model_config
    num_layers = model_config.get_total_num_hidden_layers()
    num_experts = model_config.get_num_experts()
    num_experts_per_tok = model_config.get_num_experts_per_tok()
    if num_layers <= 0 or num_experts <= 0 or num_experts_per_tok <= 0:
        raise ValueError(
            "Routed-experts capture requires positive layer, expert, and "
            "experts-per-token counts, got "
            f"{num_layers=}, {num_experts=}, {num_experts_per_tok=}."
        )
    return num_layers, num_experts, num_experts_per_tok


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` inside the forward pass. Routing
    rows owned by this DP rank are written into a preallocated device buffer.

    The capture buffer uses the narrowest unsigned dtype that can represent
    every logical expert ID. SP all-gather operates on the router's native
    dtype before writing into this buffer.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - Every routed layer overwrites the current step's token rows.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        num_layers, num_experts, num_experts_per_tok = _get_routed_experts_shape(
            vllm_config
        )
        dtype = torch.uint8 if num_experts <= 256 else torch.uint16
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, num_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s)",
            max_num_batched_tokens,
            num_layers,
            num_experts_per_tok,
            vllm_config.model_config.hf_text_config.model_type,
        )
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                num_layers,
                num_experts_per_tok,
            ),
            dtype=dtype,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have three different batch
        layouts depending on where the DP combine happens and whether
        Sequence Parallelism (SP) is active for the MoE layer:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.
          - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
            path): tokens were split along dim=0 across the TP group by
            ``_sequence_parallel_context``
            (``moe_runner_base.py:_sequence_parallel_context``), so each
            TP rank only sees its shard. We all-gather along dim=0 to
            reconstruct this DP rank's full routing tensor. SP pads with
            ceil-div (see ``_compute_sp_num_tokens`` in
            ``forward_context.py``), so the gathered tensor may contain a
            few trailing padding rows which are trimmed by the downstream
            ``[:token_num_per_dp]`` slice.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            local_topk_ids = topk_ids
        else:
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            num_local_tokens = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                end = int(num_tokens_dp[: self.dp_rank + 1].sum().item())
                local_topk_ids = topk_ids[end - num_local_tokens : end]
            elif n == num_local_tokens:
                local_topk_ids = topk_ids
            elif (
                self.tp_size > 1
                and n == (num_local_tokens + self.tp_size - 1) // self.tp_size
            ):
                # SP + modular-kernel path. All-gather across the TP
                # group along dim=0 to reconstruct the full per-DP-rank
                # tensor; keep only the first ``token_num_per_dp`` rows
                # (trailing rows are SP ceil-div padding). The TP group
                # is always initialized on real rollout workers, and
                # every rank in the group reaches this branch in
                # lockstep (bind is per-FusedMoEFactory layer, SP is a global
                # condition), so a bare all_gather here will not
                # deadlock -- let it raise if the precondition is
                # violated rather than skip silently.
                #
                # ``topk_ids`` is already whatever the router produced
                # (typically int32/int64, both supported by NCCL); the
                # downstream buffer assignment narrows to the capture dtype.
                local_topk_ids = get_tp_group().all_gather(topk_ids, dim=0)[
                    :num_local_tokens
                ]
            else:
                sp_expected = (num_local_tokens + self.tp_size - 1) // self.tp_size
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {n} (expected {total}, {num_local_tokens}, "
                    f"or {sp_expected} for dp_rank={self.dp_rank}, "
                    f"tp_size={self.tp_size})"
                )

        if not 0 <= layer_id < self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts layer {layer_id} exceeds capture buffer "
                f"layer count {self.device_buffer.shape[1]}"
            )

        self.device_buffer[: len(local_topk_ids), layer_id] = local_topk_ids

    def get_routing_data(self, num_tokens: int) -> torch.Tensor:
        """Return a stable snapshot of the current routing data."""
        return self.device_buffer[:num_tokens].clone()


def bind_routed_experts_capturer(
    model: torch.nn.Module,
    capturer: RoutedExpertsCapturer,
) -> None:
    """Attach capture callbacks to the target model's MoE routers."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    num_bound = 0
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        capture_fn = partial(capturer.capture, module.layer_id)
        quant_method = module._quant_method
        if quant_method.is_monolithic:
            moe_kernel = getattr(quant_method, "moe_kernel", None)
            impl = getattr(moe_kernel, "impl", None)
            fused_experts = getattr(impl, "fused_experts", None)
            if not (
                isinstance(fused_experts, FusedMoEExpertsMonolithic)
                and fused_experts.supports_routing_replay_capture()
            ):
                raise ValueError(
                    "Routed-experts capture is not supported with monolithic "
                    f"MoE kernel {type(fused_experts).__name__}."
                )
            fused_experts.set_capture_fn(capture_fn)
            num_bound += 1
        elif isinstance(module.router, BaseRouter):
            module.router.set_capture_fn(capture_fn)
            num_bound += 1
        else:
            raise ValueError(
                "Routed-experts capture is not supported with router "
                f"{type(module.router).__name__}."
            )

    if num_bound == 0:
        raise ValueError("No supported MoE router found for routed-experts capture.")
