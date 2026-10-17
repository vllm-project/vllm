# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-step decision and cache re-binding.

``Mono.begin_step`` decides once a forward, for every kernel layer, from what
every rank sees alike: the batch's width and the attention metadata, never
rank-local state. Under a FULL CUDA graph the decision is baked in at capture,
so it is a function of the captured size: FULL graphs serve uniform decode
batches and every width is a capture size (``guards.set_capture_sizes``), so
no replay pads a request's rows. PIECEWISE steps run vLLM's layers.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.common.mono import failstop
from vllm.models.common.mono.guards import agree
from vllm.models.common.mono.spec import (
    LayerKernel,
    MonoContext,
    MonoSpec,
    MonoStep,
    MonoUnsupported,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class MonoWarmup(VllmJitKernel["MonoWarmup.CompileKey"]):
    """A spec's layer-kernel builds as vLLM JIT warm-up keys."""

    @dataclass(frozen=True)
    class CompileKey:
        spec: str
        build: Any  # a hashable build key its LayerKernel compiles

    def __init__(self, spec: str, owners: dict[Any, LayerKernel]):
        super().__init__()
        self.spec = spec
        self.owners = owners

    def get_warmup_keys(self) -> list["MonoWarmup.CompileKey"]:
        return [self.CompileKey(self.spec, b) for b in self.owners]

    def compile(self, compile_key: "MonoWarmup.CompileKey") -> None:
        self.owners[compile_key.build].compile(compile_key.build)
        self._compiled_cache[compile_key] = True


class Mono:
    """A model's MonoKernels on this rank: built once after load."""

    def __init__(self, spec: MonoSpec, model, vllm_config: "VllmConfig"):
        from vllm.distributed import get_tp_group

        tp = get_tp_group()
        self.spec = spec
        agree(
            lambda: spec.refuse_model(model),
            tp.cpu_group,
            f"MonoKernel {spec.name} refused for the loaded model",
        )
        self.ctx = MonoContext(
            vllm_config, tp.rank_in_group, next(model.parameters()).device
        )
        packed = spec.pack(model, self.ctx)
        self.widths = spec.widths(vllm_config)
        self.kernels: dict[tuple[int, int], LayerKernel] = {}
        for i in spec.layers(model.config):
            # PPMissingLayer and other ranks' layers never get a kernel
            if model.start_layer <= i < model.end_layer:
                for w in self.widths:
                    self.kernels[i, w] = spec.build(
                        self.ctx, packed, model.layers[i], w
                    )
        self.cache_layers = sorted(
            {n for k in self.kernels.values() for n in k.cache_layers}
        )
        self._register_warmup()
        self.step: MonoStep | None = None
        # the decision each captured FULL graph baked in, by size: a replay
        # runs no Python, so a step is counted from its size (``account``)
        self.baked: dict[int, str | None] = {}
        self.last: str | None = "no step"
        self._fc: Any = None
        self._bound: tuple | None = None
        logger.info(
            "MonoKernel %s: %d layers, widths %s",
            spec.name,
            len({i for i, _ in self.kernels}),
            list(self.widths),
        )

    def _register_warmup(self) -> None:
        """Every width's builds compile in vLLM's JIT warm-up, before capture."""
        owners: dict[Any, LayerKernel] = {}
        for k in self.kernels.values():
            for key in k.warmup_keys():
                owners.setdefault(key, k)
        if owners:
            MonoWarmup(self.spec.name, owners).register_warmup()

    def _bind(self) -> None:
        """Re-checks the caches whenever they moved (keyed on every tensor's
        data_ptr and numel): memory profiling runs real steps on throwaway
        caches before the real ones exist."""
        sfc = self.ctx.vllm_config.compilation_config.static_forward_context
        caches = {n: sfc[n].kv_cache for n in self.cache_layers}
        key = tuple(
            (t.data_ptr(), t.numel())
            for c in caches.values()
            for t in (c if isinstance(c, (list, tuple)) else (c,))
        )
        if key == self._bound:
            return
        reasons = self.spec.bind(self.ctx, caches)
        if reasons:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._bound = key

    def begin_step(self, positions: torch.Tensor) -> str | None:
        """None runs this step on the kernels (and sets ``step``), else the
        reason it runs vLLM's layers."""
        self.step = None
        why = self._decide(positions)
        # Model Runner V2 captures with the runtime mode NONE
        if (
            get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            or torch.cuda.is_current_stream_capturing()
        ):
            self.baked[positions.size(-1)] = why
        self.last = why
        return why

    def account(self, num_tokens: int, runtime_mode: CUDAGraphMode) -> None:
        """Counts one served step, after its forward."""
        if runtime_mode == CUDAGraphMode.FULL:
            why = self.baked.get(num_tokens, "not captured")
        else:
            why, self.last = self.last, "no step"
        failstop.count_step(self.spec.name, why)

    def _decide(self, positions: torch.Tensor) -> str | None:
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return "piecewise"
        n = positions.size(-1)
        if n not in self.widths:
            return "width"
        step = MonoStep(n, n, positions, fc.attn_metadata, fc.cudagraph_runtime_mode)
        why = self.spec.eligible(self.ctx, step)
        if why is None:
            self._bind()
            self.step = step
        return why

    def health(self) -> torch.Tensor:
        return next(iter(self.kernels.values())).health()
