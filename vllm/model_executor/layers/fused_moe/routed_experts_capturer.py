# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Protocol, runtime_checkable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

logger = logging.getLogger(__name__)


@runtime_checkable
class RoutedExpertsCaptureSource(Protocol):
    layer_id: int
    capture_fn: Callable[[torch.Tensor], None] | None


@runtime_checkable
class RoutedExpertsCaptureRowRestorer(Protocol):
    """Layer-owned restoration of routing rows to local token order."""

    def restore_routed_experts_rows(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Return local token-ordered logical IDs without modifying the input."""
        ...


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

    The device buffer uses ``int32``. Stable snapshots use the narrowest dtype
    that can represent every logical expert ID.

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
        self.shape_per_token = (num_layers, num_experts_per_tok)
        self.output_dtype_name = "uint8" if num_experts <= 256 else "uint16"
        self.output_dtype = getattr(torch, self.output_dtype_name)
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
            (max_num_batched_tokens, *self.shape_per_token),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.row_restorers: dict[int, RoutedExpertsCaptureRowRestorer] = {}

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if not 0 <= layer_id < self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts layer {layer_id} exceeds capture buffer "
                f"layer count {self.device_buffer.shape[1]}"
            )

        local_topk_ids = self.row_restorers[layer_id].restore_routed_experts_rows(
            topk_ids
        )
        if (
            local_topk_ids.ndim != 2
            or local_topk_ids.shape[1] != self.shape_per_token[1]
        ):
            raise ValueError("routed-experts rows must have shape (tokens, top_k)")
        offset = dbo_current_ubatch_token_offset()
        if offset + len(local_topk_ids) > len(self.device_buffer):
            raise ValueError("routed-experts rows exceed capture buffer capacity")
        self.device_buffer[offset : offset + len(local_topk_ids), layer_id] = (
            local_topk_ids
        )

    def snapshot_routing_data(self, num_tokens: int) -> torch.Tensor:
        """Return a stable snapshot of the current routing data."""
        return self.device_buffer[:num_tokens].to(self.output_dtype)


class RoutedExpertsSink:
    """Layer-owned buffer and callback for the expert ids a monolithic kernel
    routes to; it outlives kernel rebuilds on weight reload."""

    def __init__(
        self, moe_config: FusedMoEConfig, capture_fn: Callable[[torch.Tensor], None]
    ) -> None:
        # Sized for per-rank batches gathered across the DP or EP group.
        group_size = moe_config.ep_size if moe_config.use_ep else moe_config.dp_size
        self.buffer = torch.empty(
            (moe_config.max_num_tokens * group_size, moe_config.experts_per_token),
            dtype=torch.int16,
            device=moe_config.device,
        )
        self.capture_fn = capture_fn


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
        if not isinstance(module, (MoERunner, RoutedExpertsCaptureSource)):
            continue
        if not isinstance(module, RoutedExpertsCaptureRowRestorer):
            raise ValueError(
                f"{type(module).__name__} must implement restore_routed_experts_rows "
                "to support routed-experts capture."
            )
        if module.layer_id in capturer.row_restorers:
            raise ValueError(
                f"Duplicate capture row restorer for layer {module.layer_id}"
            )
        capturer.row_restorers[module.layer_id] = module
        if isinstance(module, RoutedExpertsCaptureSource):
            module.capture_fn = partial(capturer.capture, module.layer_id)
            num_bound += 1
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
            module.routed_experts.routing_sink = RoutedExpertsSink(
                fused_experts.moe_config, capture_fn
            )
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
