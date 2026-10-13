# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The live side of a spec: one ``MonoRuntime`` a rank, built at attach, owning
this rank's layer kernels and making the per-step decision and cache re-binding.
It launches nothing itself.

``MonoRuntime.begin_step`` decides once a forward, for every kernel layer, from
what every rank sees alike: the batch's width and the attention metadata, never
rank-local state and never a collective. Under a FULL CUDA graph the decision
is baked in at capture, so it is a function of the captured size: FULL graphs
serve uniform decode batches and every width is a capture size
(``guards``), so no replay pads a request's rows. PIECEWISE
steps run vLLM's layers.

It also counts the steps it decided (``STEPS``, drained by ``drain`` and
exported as ``vllm:monokernel_steps``): steps on the kernels, and steps on
vLLM's path by reason.
"""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.common.mono.guards import (
    _agree_across_ranks,
    _refuse_model,
    tp_group,
)
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

STEPS: Counter[tuple[str, str]] = Counter()
"""Served steps by spec and path, where path is "mono" or the reason the step
ran vLLM's layers."""


def drain() -> dict[str, int] | None:
    """The counts since the last drain, keyed ``"steps:<spec>:<path>"``."""
    if not STEPS:
        return None
    out = {f"steps:{spec}:{path}": n for (spec, path), n in STEPS.items()}
    STEPS.clear()
    return out


class MonoWarmup(VllmJitKernel["MonoWarmup.CompileKey"]):
    """A spec's layer-kernel builds as vLLM JIT warm-up keys."""

    @dataclass(frozen=True)
    class CompileKey:
        spec: str
        build: Any

    def __init__(self, spec: str, owners: dict[Any, LayerKernel]):
        super().__init__()
        self.spec = spec
        self.owners = owners

    def get_warmup_keys(self) -> list["MonoWarmup.CompileKey"]:
        return [self.CompileKey(self.spec, b) for b in self.owners]

    def compile(self, compile_key: "MonoWarmup.CompileKey") -> None:
        self.owners[compile_key.build].compile(compile_key.build)
        self._compiled_cache[compile_key] = True


class MonoRuntime:
    """A model's MonoKernels on this rank: built once after load."""

    def __init__(self, spec: MonoSpec[Any], model, vllm_config: "VllmConfig"):
        rank, _, cpu_group = tp_group()
        self.spec = spec
        _agree_across_ranks(
            lambda: _refuse_model(spec, model),
            cpu_group,
            f"MonoKernel {spec.name} refused for the loaded model",
        )
        self.ctx = MonoContext(vllm_config, rank, next(model.parameters()).device)
        self.widths = spec.widths(vllm_config)
        self.kernels: dict[tuple[int, int], LayerKernel] = {}
        self.state: Any = None
        self.step: MonoStep | None = None
        self.baked: dict[int, str | None] = {}
        self.last: str | None = "no step"
        self._fc: Any = None
        self._bound: tuple | None = None

        indices = self._kernel_layers(model)
        if not indices:
            self.cache_layers: list[str] = []
            return
        self.state = spec.launch_state(model, self.ctx)
        for i in indices:
            for w in self.widths:
                self.kernels[i, w] = spec.build(
                    self.ctx, self.state, model.layers[i], w
                )
        self.cache_layers = sorted(
            {n for k in self.kernels.values() for n in k.cache_layers}
        )
        self._register_warmup()
        logger.info(
            "MonoKernel %s: %d layers, widths %s",
            spec.name,
            len(indices),
            list(self.widths),
        )

    def _kernel_layers(self, model) -> list[int]:
        start = getattr(model, "start_layer", 0)
        end = getattr(model, "end_layer", len(model.layers))
        out = []
        for i in self.spec.layers(model.config):
            if not start <= i < end:
                continue
            why = self.spec.skip_layer(self.ctx, model.layers[i])
            if why is None:
                out.append(i)
            else:
                logger.info_once(
                    "MonoKernel %s: layer %d on vLLM's path (%s)",
                    self.spec.name,
                    i,
                    why,
                )
        return out

    def _register_warmup(self) -> None:
        owners: dict[Any, LayerKernel] = {}
        for k in self.kernels.values():
            for key in k.warmup_keys():
                owners.setdefault(key, k)
        if owners:
            MonoWarmup(self.spec.name, owners).register_warmup()

    def reload_weights(self, model) -> None:
        """Writes ``model``'s current weights into the allocation
        ``launch_state`` returned, in place.

        Raises:
            MonoUnsupported: The spec declares no ``reload_weights``, so its
                weights cannot move after attach.

        """
        if self.spec.reload_weights is None:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name}: weights cannot move after attach"
            )
        self.spec.reload_weights(model, self.ctx, self.state)

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
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name}: caches moved during CUDA graph capture"
            )
        reasons = self.spec.bind(self.ctx, caches)
        if reasons:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._bound = key

    def begin_once(self, positions: torch.Tensor) -> None:
        """Decides this forward's step, on the first kernel layer to ask."""
        fc = get_forward_context()
        if fc is not self._fc:
            self._fc = fc
            self.begin_step(positions)

    def begin_step(self, positions: torch.Tensor) -> str | None:
        """None runs this step on the kernels (and sets ``step``), else the
        reason it runs vLLM's layers."""
        self.step = None
        why = self._decide(positions)
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            self.baked[positions.size(-1)] = why
        self.last = why
        return why

    def account(self, num_tokens: int, runtime_mode: CUDAGraphMode) -> None:
        """Counts one served step, after its forward."""
        if runtime_mode == CUDAGraphMode.FULL:
            why = self.baked.get(num_tokens, "not captured")
        else:
            why, self.last = self.last, "no step"
        STEPS[self.spec.name, why or "mono"] += 1

    def _decide(self, positions: torch.Tensor) -> str | None:
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return "piecewise"
        n = positions.size(-1)
        if n not in self.widths:
            return "width"
        step = MonoStep(n, n, positions, fc.attn_metadata, fc.cudagraph_runtime_mode)
        why = self.spec.skip_step(self.ctx, self.state, step)
        if why is None:
            self._bind()
            self.step = step
        return why
