# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One ``MonoRuntime`` a rank: the per-step decision and the cache re-checks,
behind the hooks a model calls (``attach``, then ``kernel_for_layer`` per layer).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.common.mono.guards import (
    _enabled,
    _load_specs,
    _refusal_message,
    _refuse_model,
    _select_spec,
    tp_agree,
    tp_group,
)
from vllm.models.common.mono.spec import (
    LayerKernel,
    MonoContext,
    MonoSpec,
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
        self.spec = spec
        reasons = _refuse_model(spec, model)
        agreed = tp_agree(not reasons, tp_group()[2])
        refused = f"MonoKernel {spec.name} refused for the loaded model"
        if reasons:
            raise MonoUnsupported(f"{refused}: " + "; ".join(reasons))
        if not agreed:
            raise MonoUnsupported(f"{refused}: another TP rank refused")
        self.ctx = MonoContext(vllm_config, next(model.parameters()).device)
        self.widths = spec.widths(vllm_config)
        self.kernels: dict[tuple[int, int], LayerKernel] = {}
        self.state: Any = None
        self.width: int | None = None
        self.why: str | None = None
        self._fc: Any = None
        self._checked: tuple | None = None

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
        for i in range(start, end):
            why = self.spec.skip_layer(model.layers[i])
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

    def _refuse_caches(self) -> None:
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
        if key == self._checked:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name}: caches moved during CUDA graph capture"
            )
        reasons = self.spec.refuse_caches(self.ctx, caches)
        if reasons:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._checked = key

    def begin_step(self, positions: torch.Tensor) -> str | None:
        """This forward's decision, taken on the first kernel layer to ask and
        reused by the rest: None runs the step on the kernels (and sets
        ``width``), else the reason it runs vLLM's layers."""
        fc = get_forward_context()
        if fc is self._fc:
            return self.why
        n = positions.size(-1)
        self._fc, self.width = fc, None
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            self.why = "piecewise"
        elif n not in self.widths:
            self.why = "width"
        else:
            self.why = self.spec.skip_step(self.ctx, self.state, n)
            if self.why is None:
                self._refuse_caches()
                self.width = n
        return self.why


def attach(model, vllm_config: "VllmConfig", specs_path: str) -> MonoRuntime | None:
    """Picks the variant, builds the launch state and this rank's layer
    kernels. Collective over the TP group.

    Args:
        model: The loaded model, in ``process_weights_after_loading``.
        vllm_config: The engine's config.
        specs_path: ``"package.module:MONO_SPECS"``.

    Returns:
        This rank's ``MonoRuntime``, or None with the switch off or when no layer
        takes a kernel.

    Raises:
        MonoUnsupported: Every variant refused the config, or this rank (or
            another) refused the loaded model.

    """
    if not _enabled(vllm_config):
        return None
    spec, refusals = _select_spec(_load_specs(specs_path), vllm_config)
    if spec is None:
        raise MonoUnsupported("MonoKernel refused: " + _refusal_message(refusals))
    mono = MonoRuntime(spec, model, vllm_config)
    if not mono.kernels:
        logger.info("MonoKernel %s: no layer takes a kernel", spec.name)
        return None
    return mono


def kernel_for_layer(mono: MonoRuntime | None, i: int, positions) -> LayerKernel | None:
    """Layer ``i``'s kernel when this step runs on the kernels, else None.

    The first call under a forward context decides for every layer. The model
    calls ``launch`` where it suits the layer, and runs the vLLM layer
    unchanged when this returns None.
    """
    if mono is None:
        return None
    mono.begin_step(positions)
    if mono.width is None:
        return None
    return mono.kernels.get((i, mono.width))
