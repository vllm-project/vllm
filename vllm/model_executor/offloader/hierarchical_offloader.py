# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical (Colibri-style) BaseOffloader implementation."""

from __future__ import annotations

from collections.abc import Generator

import torch.nn as nn

from vllm.config.offload import HierarchicalOffloadConfig
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.hierarchical.manager import (
    ExpertTierManager,
    set_tier_manager,
)
from vllm.model_executor.offloader.hierarchical.planner import (
    build_tier_plan,
    log_tier_plan,
)

logger = init_logger(__name__)


def _extract_layer_index(name: str) -> int:
    # Lazy import to avoid circular import via models.utils → model_loader → ...
    from vllm.model_executor.models.utils import extract_layer_index

    return extract_layer_index(name)


def _find_routed_experts(module: nn.Module) -> list[nn.Module]:
    """Find RoutedExperts-like modules that own MoE expert weight packs."""
    found: list[nn.Module] = []
    markers = (
        "w13_weight",
        "w2_weight",
        "w13_qweight",
        "w2_qweight",
    )
    for child in module.modules():
        params = dict(child.named_parameters(recurse=False))
        if any(m in params for m in markers):
            found.append(child)
    return found


class HierarchicalOffloader(BaseOffloader):
    """Colibri-style 3-tier MoE expert staging offloader.

    During ``wrap_modules``, discovers MoE expert modules inside each
    transformer block and registers them with ``ExpertTierManager``.
    ``post_init`` materializes the RAM/device/disk hierarchy.

    Optional dense prefetch for attention weights is applied when
    ``tier_dense_prefetch`` is enabled by composing with PrefetchOffloader
    on attention projection params.
    """

    def __init__(self, config: HierarchicalOffloadConfig, model_path: str | None = None):
        self.config = config
        self.manager = ExpertTierManager(config, model_path=model_path)
        set_tier_manager(self.manager)
        self._wrapped: list[nn.Module] = []
        # Startup plan stub (refined in post_init once shapes known).
        plan = build_tier_plan(
            config,
            num_moe_layers=0,
            num_local_experts=0,
            expert_row_bytes=0,
        )
        log_tier_plan(plan)
        if config.tier_dense_prefetch:
            logger.info(
                "tier_dense_prefetch enabled: wrapping non-expert Linear "
                "weights with PrefetchOffloader (group_size=8, num_in_group=2)"
            )
            from vllm.model_executor.offloader.prefetch import PrefetchOffloader

            self._dense_prefetch: PrefetchOffloader | None = PrefetchOffloader(
                group_size=8,
                num_in_group=2,
                prefetch_step=1,
                offload_params={"q_proj", "k_proj", "v_proj", "o_proj"},
                mode="cpu",
            )
        else:
            self._dense_prefetch = None

        if not config.tier_allow_cuda_graphs:
            logger.info_once(
                "Hierarchical expert staging defaults to eager execution; "
                "set --tier-allow-cuda-graphs to experiment with graphs"
            )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        # Optionally stage dense attention residuals via PrefetchOffloader.
        if self._dense_prefetch is not None:
            # Materialize once so both offloaders see the same modules.
            modules_list = list(modules_generator)

            def _gen():
                yield from modules_list

            modules_list = self._dense_prefetch.wrap_modules(_gen())
            modules_generator = (m for m in modules_list)

        modules: list[nn.Module] = []
        for module in modules_generator:
            modules.append(module)
            self._wrapped.append(module)
            try:
                prefix = (
                    getattr(module, "prefix", None)
                    or getattr(module, "layer_name", None)
                )
                if isinstance(prefix, str) and any(c.isdigit() for c in prefix):
                    layer_id = _extract_layer_index(prefix)
                else:
                    layer_id = len(modules) - 1
            except Exception:
                layer_id = len(modules) - 1

            for experts in _find_routed_experts(module):
                self.manager.register_moe_module(layer_id, experts)
                # Fill VRAM first; spill oldest MoE packs to host only when
                # device free memory drops below a reserve (avoids host-only
                # swap while the GPU sits nearly empty).
                self.manager.park_experts_on_host(experts)
                logger.debug(
                    "Registered MoE experts for hierarchical staging at layer %d",
                    layer_id,
                )
        return modules

    def post_init(self):
        # If wrap_modules didn't see nested experts (some models build MoE
        # lazily), scan wrapped blocks again.
        if not self.manager._pending_modules:
            for idx, module in enumerate(self._wrapped):
                for experts in _find_routed_experts(module):
                    try:
                        prefix = (
                            getattr(module, "prefix", None)
                            or getattr(module, "layer_name", None)
                        )
                        if isinstance(prefix, str) and any(
                            c.isdigit() for c in prefix
                        ):
                            layer_id = _extract_layer_index(prefix)
                        else:
                            layer_id = idx
                    except Exception:
                        layer_id = idx
                    self.manager.register_moe_module(layer_id, experts)

        self.manager.post_init()
        if self._dense_prefetch is not None:
            self._dense_prefetch.post_init()

    def sync_prev_onload(self) -> None:
        from vllm.platforms import current_platform

        current_platform.current_stream().wait_stream(self.manager.copy_stream)
        if self._dense_prefetch is not None:
            self._dense_prefetch.sync_prev_onload()

    def join_after_forward(self) -> None:
        self.sync_prev_onload()
        if self._dense_prefetch is not None:
            self._dense_prefetch.join_after_forward()

    def shutdown(self) -> None:
        self.manager.shutdown()
        set_tier_manager(None)
