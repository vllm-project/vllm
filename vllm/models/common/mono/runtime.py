# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One ``MonoRuntime`` a rank: the per-step decision and the cache re-checks,
behind the hooks a model calls (``attach``, then ``kernel_for_layer`` per layer).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.common.mono.guards import _refuse_model, tp_agree, tp_group
from vllm.models.common.mono.spec import (
    LaunchStateT,
    LayerKernel,
    MonoSpec,
    MonoUnsupported,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

KernelT = TypeVar("KernelT", bound=LayerKernel)
"""The kernel type one spec's ``build`` returns. ``LayerKernel`` is only the
lower bound, so a model annotates its runtime with its own kernel class
(``MonoRuntime[MyLayerKernel, MyLaunchState]``) and reads what that class
adds."""


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


class MonoRuntime(Generic[KernelT, LaunchStateT]):
    """A model's MonoKernels on this rank: built once after load.

    ``launch_state`` is what the spec built at attach, and the one handle the
    model reads outside a launch: anything a model needs between steps (the
    runner the kernels share, its peer buffers, the error words) lives there
    rather than in a module global.
    """

    launch_state: LaunchStateT
    """What ``MonoSpec.launch_state`` built, set whenever a layer takes a
    kernel, which is the only case ``attach`` hands the runtime back."""

    def __init__(self, spec: MonoSpec[LaunchStateT], model, vllm_config: "VllmConfig"):
        self.spec = spec
        reasons = _refuse_model(spec, model)
        agreed = tp_agree(not reasons, tp_group()[2])
        refused = f"MonoKernel {spec.name} refused for the loaded model"
        if reasons:
            raise MonoUnsupported(f"{refused}: " + "; ".join(reasons))
        if not agreed:
            raise MonoUnsupported(f"{refused}: another TP rank refused")
        self.vllm_config = vllm_config
        self.widths = spec.widths(vllm_config)
        self.kernels: dict[tuple[int, int], KernelT] = {}
        self._why: str | None = None
        self._fc: Any = None
        self._checked: tuple | None = None

        indices = self._kernel_layers(model)
        if not indices:
            self.cache_layers: list[str] = []
            return
        self.launch_state = spec.launch_state(model, vllm_config, tuple(indices))
        for i in indices:
            built = spec.build(
                vllm_config, self.launch_state, model.layers[i], self.widths
            )
            missing = set(self.widths) - set(built)
            if missing:
                raise MonoUnsupported(
                    f"MonoKernel {spec.name}: layer {i} has no kernel for widths "
                    f"{sorted(missing)}"
                )
            for w in self.widths:
                self.kernels[i, w] = cast(KernelT, built[w])
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
        sfc = self.vllm_config.compilation_config.static_forward_context
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
        reasons = self.spec.refuse_caches(self.vllm_config, self.launch_state, caches)
        if reasons:
            raise MonoUnsupported(
                f"MonoKernel {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._checked = key

    def skip_reason(self, positions: torch.Tensor) -> str | None:
        """Why this forward runs vLLM's layers, None when it runs on the
        kernels. Decided on the first kernel layer to ask and reused by the
        rest, so every layer of one forward is answered alike.

        Raises:
            MonoUnsupported: The caches the kernels would read moved to
                something this spec cannot serve.

        """
        fc = get_forward_context()
        if fc is self._fc:
            return self._why
        n = positions.size(-1)
        why: str | None
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            why = "piecewise"
        elif n not in self.widths:
            why = "width"
        else:
            why = self.spec.skip_step(self.vllm_config, self.launch_state, n)
            if why is None:
                self._refuse_caches()
        self._fc, self._why = fc, why
        return why


def attach(
    model, vllm_config: "VllmConfig", spec: MonoSpec[LaunchStateT] | None
) -> MonoRuntime[Any, LaunchStateT] | None:
    """Builds the launch state and this rank's layer kernels. Collective over
    the TP group.

    Args:
        model: The loaded model, in ``process_weights_after_loading``.
        vllm_config: The engine's config.
        spec: The variant ``configure`` picked, None with the switch off.

    Returns:
        This rank's ``MonoRuntime``, or None with the switch off or when no layer
        takes a kernel. Annotate it with the kernel class the spec's ``build``
        returns (which is what ``kernel_for_layer`` then hands back) and the
        spec's launch state.

    Raises:
        MonoUnsupported: This rank, or another, refused the loaded model.

    """
    if spec is None:
        return None
    mono: MonoRuntime[Any, LaunchStateT] = MonoRuntime(spec, model, vllm_config)
    if not mono.kernels:
        logger.info("MonoKernel %s: no layer takes a kernel", spec.name)
        return None
    return mono


def kernel_for_layer(
    mono: MonoRuntime[KernelT, Any] | None, i: int, positions
) -> KernelT | None:
    """Layer ``i``'s kernel when this step runs on the kernels, else None.

    The first call under a forward context decides for every layer. The model
    calls ``launch`` where it suits the layer, and runs the vLLM layer
    unchanged when this returns None.
    """
    if mono is None or mono.skip_reason(positions) is not None:
        return None
    return mono.kernels.get((i, positions.size(-1)))
