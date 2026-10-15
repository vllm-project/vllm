# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step decision and cache re-binding.

The decision reads only what every rank builds alike from the scheduler output
(the forward context's metadata and graph mode), so all ranks in the kernel's
collectives agree. Under FULL graphs it is baked in at capture time.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext

from .failstop import counters
from .guards import spec_tokens
from .spec import MonoContext, MonoSpec, MonoStep

if TYPE_CHECKING:
    from torch import nn


def step_reason(
    fc: ForwardContext, tokens: int, spec: MonoSpec, metadata_names: list[str]
) -> str | None:
    """Why this forward runs vLLM's layers, or None for a kernel step."""
    if torch.compiler.is_compiling():
        return "torch.compile tracing"
    if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        return "piecewise CUDA graph"
    if tokens not in spec_tokens(spec):
        return f"{tokens} tokens"
    md = fc.attn_metadata
    if isinstance(md, list):
        return "micro-batched step"
    if not isinstance(md, dict) or any(name not in md for name in metadata_names):
        return "no attention metadata"
    for name in metadata_names:
        m = md[name]
        if getattr(m, "num_prefills", 0):
            return "prefill rows"
        if getattr(m, "max_query_len", 1) > 1:
            return "query longer than one token"
    return None


def capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def resolve(module: nn.Module, path: str) -> torch.Tensor:
    attrs, _, index = path.partition("[")
    value = operator.attrgetter(attrs)(module)
    if index:
        value = value[int(index.rstrip("]"))]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} is not a tensor")
    return value


class Router:
    """Decides each step for all kernel layers at once and keeps the kernels'
    cache bindings current."""

    def __init__(
        self,
        spec: MonoSpec,
        ctx: MonoContext,
        layers: dict[int, nn.Module],
        metadata_names: list[str],
    ) -> None:
        self.spec, self.ctx, self.layers = spec, ctx, layers
        self.metadata_names = metadata_names
        self._cache_key: tuple | None = None

    def begin_step(
        self, fc: ForwardContext, positions: torch.Tensor
    ) -> MonoStep | None:
        tokens = positions.shape[-1]
        reason = step_reason(fc, tokens, self.spec, self.metadata_names)
        step = None
        if reason is None:
            width = self.spec.width_for(tokens)
            assert width is not None
            slots: Any = fc.slot_mapping if isinstance(fc.slot_mapping, dict) else {}
            step = MonoStep(tokens, width, positions, fc.attn_metadata, slots)
            reason = self.spec.eligible(self.ctx, step)
        if reason is not None:
            counters[f"vllm_path:{reason}"] += 1
            return None
        self.bind()
        counters["mono_steps"] += 1
        return step

    def caches(self) -> dict[str, torch.Tensor]:
        out = {}
        for idx, layer in self.layers.items():
            for name, req in self.spec.caches.items():
                if req.when is None or req.when(layer):
                    out[f"{idx}.{name}"] = resolve(layer, req.path)
        return out

    def bind(self) -> None:
        """Re-check and re-bind when any cache tensor moved: memory profiling runs
        real steps on throwaway caches before the real ones exist."""
        caches = self.caches()
        key = tuple((k, t.data_ptr(), t.numel()) for k, t in caches.items())
        if key == self._cache_key:
            return
        if capturing():
            raise RuntimeError(
                f"{self.spec.name} MonoKernel caches moved inside CUDA graph capture"
            )
        reasons = []
        for k, t in caches.items():
            req = self.spec.caches[k.split(".", 1)[1]]
            if req.dtypes and t.dtype not in req.dtypes:
                reasons.append(f"{k}: dtype {t.dtype}")
            if req.contiguous_last_dim and t.dim() and t.stride(-1) != 1:
                reasons.append(f"{k}: last dim not contiguous")
        reasons += self.spec.bind(self.ctx, caches)
        if reasons:
            raise ValueError(
                f"{self.spec.name} MonoKernel cannot use these caches: "
                + "; ".join(reasons)
            )
        self._cache_key = key
