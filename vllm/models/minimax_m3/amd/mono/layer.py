# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-local MonoKernel lifecycle adapter.

The initial kernel is deliberately a native-call shim. It exercises the final
attachment, routing, cache-rebind, and graph lifecycle without changing the
layer boundary or parameter tree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .routing import StepDecision, decide_step, tensor_identity
from .spec import MONO_SPEC, MiniMaxM3MonoSpec
from .weights import MonoLayerWeights

if TYPE_CHECKING:
    from vllm.models.minimax_m3.amd.model import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3Model,
    )

logger = init_logger(__name__)

CacheIdentity = tuple[
    tuple[int, int, torch.dtype, tuple[int, ...]],
    ...,
]


def _runtime_refusals(spec: MiniMaxM3MonoSpec) -> list[str]:
    reasons: list[str] = []
    if not current_platform.is_rocm():
        reasons.append("requires ROCm")
        return reasons
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401

        from vllm.platforms.rocm import get_cdna_version
    except ImportError as err:
        reasons.append(f"requires FlyDSL and AITER FlyDSL helpers ({err})")
        return reasons
    if get_cdna_version() != 4:
        reasons.append("requires CDNA4/gfx950")
    try:
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    except Exception as err:
        reasons.append(f"cannot determine compute-unit count ({err})")
    else:
        if cus != spec.blocks:
            reasons.append(f"requires {spec.blocks} compute units, found {cus}")
    return reasons


class MiniMaxM3MonoLayer:
    """Owns one rank's mono lifecycle and a single decision for each forward."""

    def __init__(self, model: MiniMaxM3Model, config: VllmConfig) -> None:
        self.model = model
        self.spec = MONO_SPEC
        reasons = self.spec.refusal_reasons(config) + _runtime_refusals(self.spec)
        if reasons:
            detail = "; ".join(reasons)
            raise ValueError(f"--monokernel unsupported: {detail}.")
        self.spec.configure_graph_widths(config)
        self.layers: list[MiniMaxM3DecoderLayer] = []
        self.active = False
        self.last_decision = StepDecision(False, "not started")
        self._cache_identities: dict[int, CacheIdentity] = {}
        self.weights: dict[int, MonoLayerWeights] = {}
        self._weights_attached = False

    def attach(self) -> None:
        for layer in self.model.layers:
            layer_id = getattr(layer, "layer_id", -1)
            if (
                self.spec.first_layer <= layer_id <= self.spec.last_layer
                and getattr(layer, "is_moe_layer", False)
                and hasattr(layer.self_attn, "indexer")
            ):
                layer.mono = self
                self.layers.append(layer)

    def _attach_after_load(self) -> None:
        if self._weights_attached:
            return
        for layer in self.layers:
            self.weights[layer.layer_id] = MonoLayerWeights.from_layer(layer)
        self._weights_attached = True

    @staticmethod
    def _cache_tensors(layer: MiniMaxM3DecoderLayer) -> tuple[torch.Tensor, ...]:
        attn = layer.self_attn
        tensors = [attn.kv_cache, attn.indexer.index_cache.kv_cache]
        for value in (
            getattr(attn, "_k_scale", None),
            getattr(attn, "_v_scale", None),
            getattr(attn, "sparse_bt_buffer", None),
            getattr(attn, "sparse_ctx_buffer", None),
        ):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        return tuple(tensors)

    def _bind_caches(self) -> None:
        changed: list[tuple[int, CacheIdentity]] = []
        for layer in self.layers:
            tensors = self._cache_tensors(layer)
            if any(t.numel() == 0 for t in tensors[:2]):
                raise RuntimeError("--monokernel caches are not bound")
            identity = tuple(tensor_identity(t) for t in tensors)
            if self._cache_identities.get(layer.layer_id) != identity:
                changed.append((layer.layer_id, identity))
        if not changed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "--monokernel cache first-bind/rebind reached inside graph capture"
            )
        self._cache_identities.update(changed)

    def begin_step(self, rows: int) -> None:
        self.active = False
        if not is_forward_context_available() or not self.layers:
            self.last_decision = StepDecision(False, "forward context missing")
            return
        context = get_forward_context()
        all_metadata = context.attn_metadata
        if not isinstance(all_metadata, dict):
            self.last_decision = StepDecision(False, "attention metadata missing")
            return
        layer_name = self.layers[0].self_attn.layer_name
        self.last_decision = decide_step(
            self.spec,
            rows,
            context.cudagraph_runtime_mode,
            all_metadata.get(layer_name),
        )
        if not self.last_decision.use_mono:
            return
        self._attach_after_load()
        self._bind_caches()
        self.active = True
        logger.info_once(
            "MiniMax-M3 MonoKernel lifecycle active for widths %s (native shim).",
            self.spec.widths,
        )

    def forward_layer(
        self,
        native: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor | None],
            tuple[torch.Tensor, torch.Tensor],
        ],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return native(positions, hidden_states, residual)
