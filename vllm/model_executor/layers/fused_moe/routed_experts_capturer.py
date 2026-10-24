# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


@runtime_checkable
class RoutedExpertsCaptureSource(Protocol):
    layer_id: int
    capture_fn: Callable[[torch.Tensor], None] | None


@dataclass(frozen=True)
class RoutedExpertsCaptureLayout:
    """Describe how to restore this DP rank's routing rows.

    Args:
        source_ranges: Ordered (start, length) spans to record. Offsets refer
            to the input rows, or to the rank-ordered concatenation after
            gathering. Spans may skip padding or rows owned by other DP ranks.
        gather_group: Optional device process group for gathering input rows.
            None means no communication. All group ranks must participate,
            even when their source_ranges are empty.
        shard_sizes: Physical input row counts, including padding, in group
            rank order. Required when gather_group is set; otherwise empty.
            All group ranks must agree on these sizes. The backend must support
            the supplied sizes, including uneven or empty shards when used.

    For shards of seven rows each, source_ranges=((0, 3), (7, 2)) selects
    three rows from the first shard and two from the second, in that order.

    Sizes and ranges must remain fixed for each captured graph's replays.

    """

    source_ranges: tuple[tuple[int, int], ...]
    gather_group: dist.ProcessGroup | None = None
    shard_sizes: tuple[int, ...] = ()


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
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        *,
        layout: RoutedExpertsCaptureLayout | None = None,
    ) -> None:
        """Record logical expert IDs, optionally restoring explicit row spans.

        The backend must provide the same collective plan on every group rank.
        Sizes and spans must remain stable across graph replay. When no layout
        is supplied, preserve the default row-restoration behavior.
        """
        if not 0 <= layer_id < self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts layer {layer_id} exceeds capture buffer "
                f"layer count {self.device_buffer.shape[1]}"
            )

        if layout is None:
            local_topk_ids = self._restore_local_topk_ids(
                topk_ids, get_forward_context()
            )
        else:
            local_topk_ids = self._capture_with_layout(topk_ids, layout)
            if len(local_topk_ids) > len(self.device_buffer):
                raise ValueError("routed-experts rows exceed capture buffer capacity")
        self.device_buffer[: len(local_topk_ids), layer_id] = local_topk_ids

    @staticmethod
    def _capture_with_layout(
        topk_ids: torch.Tensor, layout: RoutedExpertsCaptureLayout
    ) -> torch.Tensor:
        if layout.gather_group is not None:
            world_size = dist.get_world_size(layout.gather_group)
            if len(layout.shard_sizes) != world_size:
                raise ValueError("capture shard_sizes must match gather group size")
            if any(size < 0 for size in layout.shard_sizes):
                raise ValueError("capture shard_sizes must be non-negative")
            rank = dist.get_rank(layout.gather_group)
            if topk_ids.shape[0] != layout.shard_sizes[rank]:
                raise ValueError("capture local rows do not match shard_sizes")
            gathered = torch.empty(
                (sum(layout.shard_sizes), *topk_ids.shape[1:]),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )
            shards = list(torch.split(gathered, layout.shard_sizes, dim=0))
            dist.all_gather(shards, topk_ids.contiguous(), group=layout.gather_group)
            topk_ids = gathered
        elif layout.shard_sizes:
            raise ValueError("capture shard_sizes require a gather group")

        ranges = layout.source_ranges
        if not ranges:
            return topk_ids[:0]
        if any(
            start < 0 or length < 0 or start + length > len(topk_ids)
            for start, length in ranges
        ):
            raise ValueError("capture source range exceeds available rows")
        if len(ranges) == 1:
            start, length = ranges[0]
            return topk_ids.narrow(0, start, length)
        return torch.cat(
            [topk_ids.narrow(0, start, length) for start, length in ranges],
            dim=0,
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
        if isinstance(module, RoutedExpertsCaptureSource):
            module.capture_fn = partial(capturer.capture, module.layer_id)
            num_bound += 1
            continue
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
