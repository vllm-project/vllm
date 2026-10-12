# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a decode step runs: ``attach`` once after load, ``begin_step`` once a
forward, ``forward_layer`` for every decoder layer (RFC #60904's routing,
private to Kimi-K3 until ``vllm/models/common/mono/`` exists).

A step runs on the kernels only when it is a pure spec-verify step of a width
(``spec.widths``). The decision reads only what every TP rank sees alike (the
attention metadata), and under a FULL CUDA graph it is a function of the
captured size alone: FULL graphs serve uniform decode batches, and every width
is a capture size, so no replay pads a request's rows. PIECEWISE steps (mixed
batches under breakable graphs) run vLLM's layers.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.mono import spec
from vllm.models.kimi_k3.amd.mono.spec import MonoUnsupported

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def enabled() -> bool:
    return envs.VLLM_ROCM_MONO_DECODE


def configure(vllm_config: "VllmConfig") -> None:
    """Before load: raises with every reason the config is refused."""
    reasons = spec.refuse_config(vllm_config)
    if reasons:
        raise MonoUnsupported(
            "VLLM_ROCM_MONO_DECODE=1 is refused for Kimi-K3: " + "; ".join(reasons)
        )
    if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning(
            "Kimi-K3 mono decode under eager mode: for debugging only, "
            "serve with FULL CUDA graphs"
        )


class Mono:
    def __init__(self, model, vllm_config: "VllmConfig"):
        from vllm.models.kimi_k3.amd.mono.common.peer_memory import bind_agreed
        from vllm.models.kimi_k3.amd.mono.kernel import FfnLayer, WholeLayer
        from vllm.models.kimi_k3.amd.mono.runner import MonoRunner

        def refuse_model():
            reasons = spec.refuse_model(model)
            if reasons:
                raise MonoUnsupported(
                    "VLLM_ROCM_MONO_DECODE=1 is refused for the loaded Kimi-K3: "
                    + "; ".join(reasons)
                )

        from vllm.distributed import get_tp_group

        bind_agreed(refuse_model, get_tp_group().cpu_group)
        self.runner = MonoRunner(next(model.parameters()).device)
        self.runner.fault = _fault()
        assert vllm_config.speculative_config is not None
        num_spec = vllm_config.speculative_config.num_speculative_tokens
        self.widths = spec.widths(num_spec)
        self.kernels = {}
        for i, kind in spec.layers(model.config).items():
            if model.start_layer <= i < model.end_layer:
                cls = WholeLayer if kind == spec.WHOLE else FfnLayer
                self.kernels[i] = cls(model.layers[i], self.runner)
        self.whole = [k for k in self.kernels.values() if isinstance(k, WholeLayer)]
        self._bound: tuple | None = None
        self.active = False
        logger.info(
            "Kimi-K3 mono decode (%s): %d layers (%d whole, %d FFN), widths %s",
            spec.NAME,
            len(self.kernels),
            len(self.whole),
            len(self.kernels) - len(self.whole),
            list(self.widths),
        )

    def _bind(self) -> None:
        """Re-checks the KDA caches whenever they moved: memory profiling runs
        real steps on throwaway caches before the real ones exist."""
        caches = [k.caches() for k in self.whole]
        key = tuple((t.data_ptr(), t.numel()) for pair in caches for t in pair)
        if key == self._bound:
            return
        reasons = []
        for k, (conv_state, rstate) in zip(self.whole, caches):
            reasons += spec.refuse_caches(k.layer.layer_idx, conv_state, rstate)
        if reasons:
            raise MonoUnsupported("Kimi-K3 mono decode: " + "; ".join(reasons))
        self._bound = key

    def no_go(self, num_tokens: int) -> str | None:
        """Why this step runs vLLM's layers, or None to run the kernels."""
        fc = get_forward_context()
        if fc.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return "piecewise"
        if num_tokens not in self.widths:
            return "width"
        md = fc.attn_metadata
        if not isinstance(md, dict) or not self.whole:
            return "no metadata"
        m: Any = md.get(self.whole[0].layer.self_attn.prefix)
        if m is None or m.spec_sequence_masks is None:
            return "not spec-verify"
        if m.num_prefills or m.num_decodes or m.num_spec_decodes < 1:
            return "not pure decode"
        q = m.spec_state_indices_tensor.size(-1)
        if m.uniform_spec_sequence_length != q or m.num_spec_decodes * q != num_tokens:
            return "not uniform"
        return None

    def begin_step(self, num_tokens: int) -> None:
        self.active = self.no_go(num_tokens) is None
        if self.active:
            self._bind()
            self.runner.step_begin()

    def health(self) -> torch.Tensor:
        """The output rank's flags, a word a TP rank, nonzero once one of its
        waits gave up: the step's tokens must not be returned."""
        return self.runner.flags


def _fault() -> dict:
    """``VLLM_MONO_FAULT_INJECT=rank:epoch:us`` (tests only)."""
    v = envs.VLLM_MONO_FAULT_INJECT
    if not v:
        return {}
    rank, epoch, us = (int(x) for x in v.split(":"))
    logger.warning("Kimi-K3 mono decode fault injection: %s", v)
    # 100 MHz ticks
    return {"fault_rank": rank, "fault_epoch": epoch, "fault_ticks": us * 100}


def attach(model, vllm_config: "VllmConfig") -> Mono | None:
    """After load; None with the switch off."""
    if not enabled():
        return None
    return Mono(model, vllm_config)


def forward_layer(mono: Mono | None, i: int, layer, positions, hidden_states,
                  residual, prefix_delta):  # fmt: skip
    if mono is not None and mono.active and (k := mono.kernels.get(i)) is not None:
        return k.launch(positions, hidden_states, residual, prefix_delta)
    return layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
        prefix_delta=prefix_delta,
    )
