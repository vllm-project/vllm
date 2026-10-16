# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatcher for specialized decoder kernels.

Performs validation, jit registration, planning, and dispatching of specialized decode
kernels for a model based on a given specification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.common.specialized_decode.guards import (
    _refuse_model,
    tp_agree,
    tp_group,
)
from vllm.models.common.specialized_decode.spec import (
    DecodeSpec,
    DecodeUnsupported,
    DecodeWarmable,
    KernelT,
    Step,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class DecodeWarmup(VllmJitKernel["DecodeWarmup.CompileKey"]):
    """A spec's kernel builds as vLLM JIT warm-up keys."""

    @dataclass(frozen=True)
    class CompileKey:
        spec: str
        build: Any

    def __init__(self, spec: str, kernel: DecodeWarmable):
        super().__init__()
        self.spec = spec
        self.kernel = kernel

    def get_warmup_keys(self) -> list["DecodeWarmup.CompileKey"]:
        return [self.CompileKey(self.spec, b) for b in self.kernel.warmup_keys()]

    def compile(self, compile_key: "DecodeWarmup.CompileKey") -> None:
        self.kernel.compile(compile_key.build)
        self._compiled_cache[compile_key] = True


class SpecializedDecode(Generic[KernelT]):
    """A model's specialized decode on this rank.

    It reads the spec and builds the kernel, registers it for warm-up if it
    compiles from keys, checks the caches whenever the runner binds them, and
    decides each step
    between the kernel and vLLM's own forward. The kernel it builds is held in
    the ``kernel`` attribute.
    """

    def __init__(self, model, vllm_config: "VllmConfig", spec: DecodeSpec[KernelT]):
        """Builds this rank's kernel.

        Args:
            model: The loaded model, in ``process_weights_after_loading``.
            vllm_config: The engine's config.
            spec: The spec to build and run.

        """
        self.spec = spec
        reasons = _refuse_model(spec, model)
        agreed = tp_agree(not reasons, tp_group()[2])
        refused = f"Specialized decode {spec.name} refused for the loaded model"
        if reasons:
            raise DecodeUnsupported(f"{refused}: " + "; ".join(reasons))
        if not agreed:
            raise DecodeUnsupported(f"{refused}: another TP rank refused")
        self.vllm_config = vllm_config
        self.widths = spec.widths(vllm_config)
        self._bound = False
        self.kernel: KernelT = spec.build(vllm_config, model, self.widths)
        kernel = self.kernel
        if isinstance(kernel, DecodeWarmable) and kernel.warmup_keys():
            DecodeWarmup(spec.name, kernel).register_warmup()
        logger.info("Specialized decode %s: widths %s", spec.name, list(self.widths))

    def bind_caches(self) -> None:
        """Validate and bind caches to the kernel.

        The underlying kernel only ever sees caches this spec has accepted.
        """
        sfc = self.vllm_config.compilation_config.static_forward_context
        caches = {n: sfc[n].kv_cache for n in self.kernel.cache_layers}
        reasons = self.spec.refuse_caches(self.vllm_config, self.kernel, caches)
        if reasons:
            raise DecodeUnsupported(
                f"Specialized decode {self.spec.name} caches: " + "; ".join(reasons)
            )
        self.kernel.bind_caches(caches)
        self._bound = True

    def plan(self, positions: torch.Tensor, intermediate_tensors=None) -> Step | None:
        """This step's plan, or None to run the model's own forward instead.

        A step can fall back to running the model's own forward when the kernel cannot
        serve it, as described by the spec.
        """
        width = positions.size(-1)
        why: str | None
        if intermediate_tensors is not None:
            why = "pipeline stage input"
        elif width not in self.widths:
            why = "width"
        elif not is_forward_context_available():
            why = "no forward context"
        elif get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            why = "piecewise"
        elif not self._bound:
            why = "caches not bound"
        else:
            why = self.spec.skip_step(self.vllm_config, self.kernel, width)
        if why is not None:
            logger.info_once(
                "Specialized decode %s: step on vLLM's path (%s)", self.spec.name, why
            )
            return None
        return Step(width=width, positions=positions)

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        """Run the specialized kernel, returning what the model's own forward would
        have returned."""
        return self.kernel.run(step, input_ids, inputs_embeds, **kwargs)
