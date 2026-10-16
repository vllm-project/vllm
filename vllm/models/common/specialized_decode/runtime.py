# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One ``SpecializedDecode`` a rank: the per-step decision and the cache checks,
behind the two hooks a model's forward calls (``plan``, then ``run``).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    DecodeImpl,
    DecodeSpec,
    DecodeUnsupported,
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

    def __init__(self, spec: str, kernel: DecodeImpl):
        super().__init__()
        self.spec = spec
        self.kernel = kernel

    def get_warmup_keys(self) -> list["DecodeWarmup.CompileKey"]:
        return [self.CompileKey(self.spec, b) for b in self.kernel.warmup_keys()]

    def compile(self, compile_key: "DecodeWarmup.CompileKey") -> None:
        self.kernel.compile(compile_key.build)
        self._compiled_cache[compile_key] = True


class SpecializedDecode:
    """A model's specialized decode on this rank.

    The runtime reads the spec and builds the kernel, registers it for warm-up,
    checks the caches whenever the runner binds them, and decides each step
    between the kernel and vLLM's own forward. The kernel it builds is held in
    the ``kernel`` attribute.
    """

    def __init__(self, model, vllm_config: "VllmConfig", spec: DecodeSpec[Any]):
        """Builds this rank's kernel. Collective over the TP group.

        Args:
            model: The loaded model, in ``process_weights_after_loading``.
            vllm_config: The engine's config.
            spec: The variant ``select_decode_spec`` picked.

        Raises:
            DecodeUnsupported: This rank, or another, refused the loaded model.

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
        self.last_reason: str | None = None
        self._bound = False
        self.kernel: DecodeImpl = spec.build(vllm_config, model, self.widths)
        if self.kernel.warmup_keys():
            DecodeWarmup(spec.name, self.kernel).register_warmup()
        logger.info("Specialized decode %s: widths %s", spec.name, list(self.widths))

    def bind_caches(self) -> None:
        """Checks the caches the kernel will read, as the runner binds them.

        Called from ``ModelState.initialize_kv_cache``, which runs once for the
        throwaway cache memory profiling allocates and again for the real one.

        Raises:
            DecodeUnsupported: The caches are not what this spec serves.

        """
        sfc = self.vllm_config.compilation_config.static_forward_context
        caches = {n: sfc[n].kv_cache for n in self.kernel.cache_layers}
        reasons = self.spec.refuse_caches(self.vllm_config, self.kernel, caches)
        if reasons:
            raise DecodeUnsupported(
                f"Specialized decode {self.spec.name} caches: " + "; ".join(reasons)
            )
        self._bound = True

    def plan(self, positions: torch.Tensor, intermediate_tensors=None) -> Step | None:
        """This step's plan, or None to run the model's own forward instead.

        Steps that run the model's forward: a pipeline stage's inputs, a width
        off the spec's widths, a PIECEWISE step (keyed by token count alone, so
        it is replayed for mixed batches too), and whatever ``skip_step``
        refuses. ``last_reason`` holds why.

        Every predicate is the same on every TP rank: the kernel waits
        in-kernel on its peers, so the ranks must take the same path.
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
        self.last_reason = why
        if why is not None:
            logger.info_once(
                "Specialized decode %s: step on vLLM's path (%s)", self.spec.name, why
            )
            return None
        return Step(width=width, positions=positions)

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        """A planned step on the kernel, returning what the model's own forward
        would have returned."""
        return self.kernel.run(step, input_ids, inputs_embeds, **kwargs)
