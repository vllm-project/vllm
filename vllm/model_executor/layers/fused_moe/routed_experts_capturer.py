# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, runtime_checkable

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_token_offset

logger = logging.getLogger(__name__)


@runtime_checkable
class RoutedExpertsCaptureSource(Protocol):
    layer_id: int
    capture_fn: Callable[[torch.Tensor], None] | None


@dataclass
class _CaptureBinding:
    """One capture callback this capturer installed on a model module."""

    target: Any
    attribute: str
    installed: Any
    setter: Callable[[Any], None] | None = None

    def install(self) -> None:
        self._set(self.installed)

    def _set(self, value: Any) -> None:
        if self.setter is None:
            setattr(self.target, self.attribute, value)
        else:
            self.setter(value)

    def release(self) -> None:
        """Remove the callback only if this binding still owns the slot."""
        if getattr(self.target, self.attribute, None) is not self.installed:
            return
        self._set(None)


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
    Under pipeline parallelism the buffer only covers this stage's layers.

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
        pp_group = get_pp_group()
        self.start_layer, end_layer = get_pp_indices(
            num_layers, pp_group.rank_in_group, pp_group.world_size
        )
        self.shape_per_token = (end_layer - self.start_layer, num_experts_per_tok)
        self.output_dtype_name = "uint8" if num_experts <= 256 else "uint16"
        self.output_dtype = getattr(torch, self.output_dtype_name)
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, num_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s)",
            max_num_batched_tokens,
            self.shape_per_token[0],
            num_experts_per_tok,
            vllm_config.model_config.hf_text_config.model_type,
        )
        self.device_buffer = torch.zeros(
            (max_num_batched_tokens, *self.shape_per_token),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._bindings: list[_CaptureBinding] = []
        self._closed = False

    def bind(self, bindings: list[_CaptureBinding]) -> None:
        """Record the callbacks installed by :func:`bind_routed_experts_capturer`."""
        if self._closed:
            raise ValueError("Cannot bind a closed routed-experts capturer")
        self._bindings = bindings

    def close(self) -> None:
        """Unbind this capturer's callbacks and release its device buffer."""
        if self._closed:
            return
        for binding in reversed(self._bindings):
            binding.release()
        self._bindings.clear()
        del self.device_buffer
        self._closed = True

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        local_layer_id = layer_id - self.start_layer
        if not 0 <= local_layer_id < self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts layer {layer_id} is outside the capture buffer "
                f"layers [{self.start_layer}, "
                f"{self.start_layer + self.device_buffer.shape[1]})"
            )

        ctx = get_forward_context()
        local_topk_ids = self._restore_local_topk_ids(topk_ids, ctx)
        offset = dbo_current_ubatch_token_offset()
        self.device_buffer[offset : offset + len(local_topk_ids), local_layer_id] = (
            local_topk_ids
        )

    def _restore_local_topk_ids(
        self, topk_ids: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Restore this DP rank's token rows, trimming trailing SP padding.

        Layout priority: cross-DP batch, local batch, gathered DP/TP shards,
        then a local TP shard. Gathered rows are DP-major/TP-minor; preserve
        this priority when layout sizes coincide. PCP is unsupported.

        Without DP metadata, use the pre-SP batch descriptor. Unknown counts
        pass through; known counts must match a full, sharded or padded batch.
        A full local batch avoids gathering, including zero/one-token batches
        where only the publishing TP0 rank needs the real rows.
        """
        n = topk_ids.shape[0]
        dp_metadata = ctx.dp_metadata
        if dp_metadata is None:
            if ctx.batch_descriptor is None:
                return topk_ids
            num_local_tokens = ctx.batch_descriptor.num_tokens
        else:
            num_tokens_dp = dp_metadata.num_tokens_across_dp_cpu
            num_local_tokens = int(num_tokens_dp[self.dp_rank].item())
            if n == int(num_tokens_dp.sum().item()):
                end = int(num_tokens_dp[: self.dp_rank + 1].sum().item())
                return topk_ids[end - num_local_tokens : end]

        if n == num_local_tokens:
            return topk_ids

        if dp_metadata is not None:
            shard_sizes = dp_metadata.local_sizes
            if shard_sizes is not None and n == sum(shard_sizes):
                num_dp_ranks = len(num_tokens_dp)
                if len(shard_sizes) % num_dp_ranks != 0:
                    raise ValueError(
                        "RoutedExpertsCapturer: gathered shard count "
                        f"{len(shard_sizes)} is not divisible by DP rank count "
                        f"{num_dp_ranks}"
                    )
                shards_per_dp_rank = len(shard_sizes) // num_dp_ranks
                first_shard = self.dp_rank * shards_per_dp_rank
                start = sum(shard_sizes[:first_shard])
                return topk_ids[start : start + num_local_tokens]

        shard_n = (num_local_tokens + self.tp_size - 1) // self.tp_size
        if n == shard_n:
            return get_tp_group().all_gather(topk_ids.contiguous(), dim=0)[
                :num_local_tokens
            ]
        if dp_metadata is None and n == shard_n * self.tp_size:
            return topk_ids[:num_local_tokens]

        raise AssertionError(
            f"RoutedExpertsCapturer: unexpected topk_ids batch dim {n} "
            f"(local_tokens={num_local_tokens}, "
            f"dp_rank={self.dp_rank}, tp_size={self.tp_size})"
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


def _get_routed_experts_bindings(
    model: torch.nn.Module, capturer: RoutedExpertsCapturer
) -> list[_CaptureBinding]:
    """Validate every capture target before installing any callback."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    bindings: list[_CaptureBinding] = []
    for module in model.modules():
        if isinstance(module, RoutedExpertsCaptureSource):
            if module.capture_fn is not None:
                raise ValueError(
                    f"Routed-experts source layer {module.layer_id} is already bound"
                )
            bindings.append(
                _CaptureBinding(
                    module, "capture_fn", partial(capturer.capture, module.layer_id)
                )
            )
            continue
        if not isinstance(module, MoERunner):
            continue
        if module._quant_method.is_monolithic:
            moe_kernel = getattr(module._quant_method, "moe_kernel", None)
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
            if module.routed_experts.routing_sink is not None:
                raise ValueError(
                    "Routed-experts monolithic layer "
                    f"{module.layer_id} is already bound"
                )
            bindings.append(
                _CaptureBinding(
                    module.routed_experts,
                    "routing_sink",
                    RoutedExpertsSink(
                        fused_experts.moe_config,
                        partial(capturer.capture, module.layer_id),
                    ),
                )
            )
        elif isinstance(module.router, BaseRouter):
            if module.router.capture_fn is not None:
                raise ValueError(
                    f"Routed-experts router layer {module.layer_id} is already bound"
                )
            bindings.append(
                _CaptureBinding(
                    module.router,
                    "capture_fn",
                    partial(capturer.capture, module.layer_id),
                    module.router.set_capture_fn,
                )
            )
        else:
            raise ValueError(
                "Routed-experts capture is not supported with router "
                f"{type(module.router).__name__}."
            )

    # A PP stage may contain only dense layers.
    if not bindings and get_pp_group().world_size == 1:
        raise ValueError("No supported MoE router found for routed-experts capture.")
    return bindings


def bind_routed_experts_capturer(
    model: torch.nn.Module,
    capturer: RoutedExpertsCapturer,
) -> None:
    """Attach capture callbacks to the target model's MoE routers.

    All targets are validated before any callback is installed. A failure while
    installing rolls back the callbacks already installed by this call, so a
    retry starts from an unbound model.
    """
    if capturer._closed:
        raise ValueError("Cannot bind a closed routed-experts capturer")
    if capturer._bindings:
        raise ValueError("Routed-experts capturer is already bound")

    bindings = _get_routed_experts_bindings(model, capturer)
    try:
        for binding in bindings:
            binding.install()
    except BaseException:
        for binding in reversed(bindings):
            binding.release()
        raise
    capturer.bind(bindings)
