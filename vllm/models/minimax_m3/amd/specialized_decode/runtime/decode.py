# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-mode specialized-decode lifecycle and per-forward plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available

from .caches import CacheBinding, TensorIdentity, checked_cache_bindings
from .guards import checked_weights
from .routing import decide_step
from .spec import DecodeConfig

LayerKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor],
]


class KernelExecutor(Protocol):
    def layer(self, layer_id: int) -> LayerKernel | None: ...

    def bind(self, layer_id: int, binding: CacheBinding) -> None: ...

    def health(self) -> Any: ...

    def after_step(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Step:
    first_layer: int
    kernels: tuple[LayerKernel | None, ...]

    def layer(self, layer_id: int) -> LayerKernel | None:
        offset = layer_id - self.first_layer
        if not 0 <= offset < len(self.kernels):
            return None
        return self.kernels[offset]


class SpecializedDecode:
    """Built after weight loading; never replaces stock modules or parameters."""

    def __init__(
        self,
        model,
        config: DecodeConfig,
        executor: KernelExecutor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.layers = tuple(
            layer
            for layer in model.layers
            if config.first_layer <= layer.layer_id <= config.last_layer
        )
        self.weights = checked_weights(model, config)
        self.executor = executor
        self._cache_identities: dict[int, dict[str, TensorIdentity]] = {}
        self._closed = False
        self.planned_steps = 0
        self.native_steps = 0

    def bind_caches(self) -> None:
        failure: Exception | None = None
        bindings: dict[int, CacheBinding] = {}
        try:
            bindings = checked_cache_bindings(self.layers)
            changed = {
                layer_id: binding
                for layer_id, binding in bindings.items()
                if self._cache_identities.get(layer_id) != binding.identity()
            }
            if changed and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "specialized-decode cache first-bind/rebind during graph capture"
                )
            if self.executor is not None:
                for layer_id, binding in changed.items():
                    self.executor.bind(layer_id, binding)
        except Exception as err:
            failure = err

        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        if not tp_agree(failure is None, get_tp_group().cpu_group):
            raise RuntimeError(
                "rank-uniform MiniMax specialized-decode cache bind failed"
            ) from failure
        if failure is not None:
            raise failure
        self._cache_identities = {
            layer_id: binding.identity() for layer_id, binding in bindings.items()
        }

    def plan(
        self,
        positions: torch.Tensor,
        intermediate_tensors: object | None,
    ) -> Step | None:
        if (
            intermediate_tensors is not None
            or self._closed
            or not is_forward_context_available()
        ):
            self.native_steps += 1
            return None
        context = get_forward_context()
        all_metadata = context.attn_metadata
        first = self.layers[0].self_attn.layer_name
        metadata = all_metadata.get(first) if isinstance(all_metadata, dict) else None
        decision = decide_step(
            self.config,
            positions.shape[0],
            context.cudagraph_runtime_mode,
            metadata,
        )
        if not decision.use_mono:
            self.native_steps += 1
            return None
        kernels = tuple(
            self.executor.layer(layer.layer_id) if self.executor is not None else None
            for layer in self.layers
        )
        self.planned_steps += 1
        return Step(self.config.first_layer, kernels)

    def health(self) -> Any:
        return None if self.executor is None else self.executor.health()

    def after_step(self) -> None:
        if self.executor is None:
            return
        self.executor.after_step()
        health = self.executor.health()
        failed = bool(getattr(health, "failed", health))
        if failed:
            raise RuntimeError(f"MiniMax specialized-decode kernel failed: {health}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.executor is not None:
            self.executor.close()
