# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernel specialized-decode lifecycle and per-forward plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .caches import CacheBinding, TensorIdentity, checked_cache_bindings
from .guards import model_refusals
from .routing import decide_step
from .spec import DecodeConfig

FAULT_MESSAGE = "MiniMax-M3 launch error word set; the step is discarded"

LayerKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor],
]


def tp_agree(ok: bool) -> bool:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

    return tp_agree(ok, get_tp_group().cpu_group)


@dataclass(frozen=True)
class Step:
    first_layer: int
    kernels: tuple[LayerKernel | None, ...]

    def layer(self, layer_id: int) -> LayerKernel | None:
        offset = layer_id - self.first_layer
        if not 0 <= offset < len(self.kernels):
            return None
        return self.kernels[offset]

    @property
    def launched(self) -> bool:
        return any(kernel is not None for kernel in self.kernels)


class SpecializedDecode:
    """Built after weight loading; never replaces stock modules or parameters."""

    def __init__(
        self,
        model,
        config: DecodeConfig,
        executor: Any,
    ) -> None:
        self.model = model
        self.config = config
        self.layers = tuple(
            layer
            for layer in model.layers
            if config.first_layer <= layer.layer_id <= config.last_layer
        )
        reasons = model_refusals(model, config)
        if not tp_agree(not reasons):
            raise ValueError(
                "MiniMax-M3 specialized decode model unsupported: "
                + ("; ".join(reasons) or "refused on another TP rank")
            )
        self.executor = executor(self) if callable(executor) else executor
        self._cache_identities: dict[int, dict[str, TensorIdentity]] = {}
        self._closed = False
        self.mono_steps = 0
        self.stock_steps = 0

    @classmethod
    def build(cls, model, config: DecodeConfig) -> SpecializedDecode:
        from .layer_executor import LayerExecutor

        return cls(
            model,
            config,
            lambda decode: LayerExecutor(decode.layers, config.widths, config.tp_size),
        )

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
            for layer_id, binding in changed.items():
                self.executor.bind(layer_id, binding)
        except Exception as err:
            failure = err

        if not tp_agree(failure is None):
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
        """Mono for a supported plain-decode width, eager or
        inside a FULL graph capture (padded rows are inert in the kernel);
        stock otherwise. Every predicate reads metadata every TP rank holds
        alike."""
        if (
            intermediate_tensors is not None
            or self._closed
            or not is_forward_context_available()
        ):
            self.stock_steps += 1
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
            self.stock_steps += 1
            return None
        kernels = tuple(self.executor.layer(layer.layer_id) for layer in self.layers)
        if any(kernel is None for kernel in kernels):
            raise RuntimeError(
                "MiniMax specialized-decode executor does not own every sparse layer"
            )
        assert isinstance(metadata, MiniMaxM3SparseMetadata)
        # the kernels take any negative slot for a padded row
        assert PAD_SLOT_ID < 0
        self.executor.begin_step(metadata.slot_mapping[: positions.shape[0]])
        self.mono_steps += 1
        return Step(self.config.first_layer, kernels)

    def health(self) -> Any:
        return self.executor.health()

    def after_step(self, step: Step | None, failure: Exception | None = None) -> None:
        """After every forward's layer loop (mono or stock, eager or captured
        into a FULL graph), before its logits and sampling: a device-side
        assert that the sticky error word is clear. A set word traps the GPU
        queue, so no later work of the step (sampling, the output copy) runs
        and the worker process aborts. A rank-local exception in the layer
        loop marks every rank's error word (their pending waits end and their
        asserts fire), then propagates here."""
        if failure is not None:
            if step is not None and step.launched:
                self.executor.abort()
            raise failure
        if not self._closed:
            torch._assert_async(self.executor.fault_word[0] == 0, FAULT_MESSAGE)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.close()
