# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hook a model calls: ``attach`` after loading, ``forward_layer`` per layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from . import guards
from .failstop import OUTPUT_RANK, HealthWatch
from .routing import Router, capturing
from .spec import LayerKernel, MonoContext, MonoSpec, MonoStep

if TYPE_CHECKING:
    from torch import nn

    from vllm.config import VllmConfig

logger = init_logger(__name__)


class MonoModel:
    """One rank's MonoKernel state for a model's layer loop."""

    def __init__(self, model: nn.Module, vllm_config: VllmConfig, spec: MonoSpec):
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
        from vllm.model_executor.models.utils import PPMissingLayer

        self.spec = spec
        hf = vllm_config.model_config.hf_text_config
        start, end = getattr(model, "start_layer", 0), getattr(model, "end_layer", None)
        end = len(model.layers) if end is None else end
        self.layers: dict[int, nn.Module] = {
            i: model.layers[i]
            for i in spec.layers(hf)
            if start <= i < end and not isinstance(model.layers[i], PPMissingLayer)
        }
        self.ctx = MonoContext(
            vllm_config=vllm_config,
            tp_size=get_tensor_model_parallel_world_size(),
            tp_rank=get_tensor_model_parallel_rank(),
            cpu_group=get_tp_group().cpu_group,
            device=torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )
        reasons = spec.refuse_model(model)
        if reasons:
            raise ValueError(
                f"{guards.SWITCH}=1 cannot run the {spec.name} MonoKernel on this "
                "model: " + "; ".join(reasons) + "."
            )
        self.packed = spec.pack(model, self.ctx)
        self.router = Router(
            spec, self.ctx, self.layers, spec.metadata_names(self.layers)
        )
        self.kernels: dict[int, dict[int, LayerKernel]] = {}
        self.watch: HealthWatch | None = None
        self.step: MonoStep | None = None
        self._fc: Any = None
        logger.info_once(
            "%s MonoKernel: %d layers, widths %s, options %s.",
            spec.name,
            len(self.layers),
            "/".join(map(str, spec.widths)),
            ", ".join(f"{k}={v}" for k, v in spec.options.items()),
        )

    def forward_layer(
        self, idx: int, layer: nn.Module, positions: torch.Tensor, *args: Any
    ) -> Any:
        fc = get_forward_context()
        if fc is not self._fc:
            self._fc = fc
            self.step = self.router.begin_step(fc, positions)
            if self.step is not None and not self.kernels:
                self._build()
        if self.step is None or idx not in self.layers:
            return layer(positions, *args)
        return self.kernels[self.step.width][idx].launch(self.step, *args)

    def _build(self) -> None:
        """Every width, before the first CUDA graph capture (vLLM runs each graph's
        batch eagerly first)."""
        if capturing():
            raise RuntimeError(
                f"{guards.SWITCH}=1: the {self.spec.name} MonoKernel was first reached "
                "inside CUDA graph capture, before an eager decode step."
            )
        for width in self.spec.widths:
            self.kernels[width] = {
                idx: self.spec.build(self.ctx, self.packed, layer, idx, width)
                for idx, layer in self.layers.items()
            }
        words, seen = [], set()
        for per_width in self.kernels.values():
            for kernel in per_width.values():
                w = kernel.health()
                if w.data_ptr() not in seen:
                    seen.add(w.data_ptr())
                    words.append(w)
        flags = self.ctx.state.get("flags") if self.ctx.tp_rank == OUTPUT_RANK else None
        self.watch = HealthWatch(
            self.spec.name, words, self.spec.stages, self.ctx.tp_rank, flags
        )


def attach(
    model: nn.Module, vllm_config: VllmConfig, spec_path: str
) -> MonoModel | None:
    """After weight loading: pack weights for this rank's layers. None when the
    switch is off."""
    if not guards.enabled():
        return None
    return MonoModel(model, vllm_config, guards.load_spec(spec_path))


def forward_layer(
    mono: MonoModel | None,
    idx: int,
    layer: nn.Module,
    positions: torch.Tensor,
    *args: Any,
) -> Any:
    if mono is None:
        return layer(positions, *args)
    return mono.forward_layer(idx, layer, positions, *args)


def health(mono: MonoModel | None) -> torch.Tensor | None:
    return None if mono is None or mono.watch is None else mono.watch.health()


def after_step(mono: MonoModel | None) -> None:
    if mono is not None and mono.watch is not None:
        mono.watch.after_step()
