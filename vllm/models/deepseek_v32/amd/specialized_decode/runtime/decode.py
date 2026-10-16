# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.2 specialized decode executor (layer mode): one routing decision per step,
then a MonoKernel per MoE decoder layer inside the stock layer loop.

Built by the model state once the weights are loaded (``DeepseekV32ModelState``): reads
vLLM's loaded weights in place (``vllm_weights``), builds the kernels of every batch
size, the buffers and the IPC peers, before memory profiling and graph capture.
``prepare_kv`` binds it to every KV cache vLLM allocates (the CUDA-graph
memory-profiling cache, then the real one). Each step ``plan`` returns a ``Step`` or
None; ``Step.layer(i)`` is the kernel callable of mono layer i (None for vLLM's layers
0..2). Weight reloads are not supported."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import (
    DecodeConfig,
    guards,
)
from vllm.models.deepseek_v32.amd.specialized_decode.runtime.mono_layers import (
    MonoLayers,
    mla_cache,
)

logger = init_logger(__name__)

# (positions, hidden_states, residual) -> (hidden_states, residual)
LayerKernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class Step(NamedTuple):
    """A planned step: a batch of T rows padded to the built batch size S."""

    T: int
    S: int
    kernels: dict[int, LayerKernel]

    def layer(self, idx: int) -> LayerKernel | None:
        """The kernel replacing decoder layer idx in this step, or None (vLLM's)."""
        return self.kernels.get(idx)


def moe_refusal(backbone, cfg: DecodeConfig) -> str | None:
    """None if every mono layer's MoE weights have the layout the kernel reads in
    place (``vllm_weights.unsupported_moe``), else why not."""
    from vllm.models.deepseek_v32.amd.specialized_decode.runtime.vllm_weights import (
        unsupported_moe,
    )

    for L in cfg.layers:
        why = unsupported_moe(backbone.layers[L].mlp)
        if why is not None:
            return f"layer {L}: {why}"
    return None


class SpecializedDecode:
    """The executor: builds the mono layers, binds them to each KV cache and plans
    every step (layer mode)."""

    def __init__(self, backbone, cfg: DecodeConfig):
        why = moe_refusal(backbone, cfg)
        if why is not None:
            raise ValueError(
                "GLM-5.2 specialized decode does not support the loaded weights: "
                f"{why} (decode_backend=auto runs vLLM's decode)"
            )
        guards.check_rope_table(backbone, cfg)
        self.backbone, self.cfg = backbone, cfg
        self.mono = MonoLayers(backbone, cfg)
        self.kernels: dict[int, LayerKernel] = {
            L: partial(self.mono.run_layer, layer)
            for L, layer in self.mono.layers.items()
        }
        # (data_ptr, numel) of the first mono layer's KV cache prepare_kv bound
        self._kv: tuple[int, int] | None = None
        self.watch = self._poll_watch()
        logger.info(
            "GLM-5.2 specialized decode: layers %d..%d, batch sizes %s; larger batches "
            "run vLLM's decode",
            self.mono.first,
            self.mono.last,
            self.mono.batch_sizes,
        )

    def _poll_watch(self):
        """Fail-stop on expired kernel polls, advanced by ``after_step``;
        ``failstop="off"`` disables it, ``"warn"`` logs."""
        if self.cfg.failstop == "off":
            logger.warning("failstop=off: no poll-error fail-stop watch")
            return None
        return guards.PollErrorWatch(self.mono)

    def prepare_kv(self) -> None:
        """Bind the kernels to the KV caches vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one), checking them (``bind_caches``)."""
        self._kv = None
        kv = mla_cache(self.mono.layers[self.mono.first])
        if kv is None or kv.numel() == 0:
            return
        self.mono.bind_caches()
        self._kv = (kv.data_ptr(), kv.numel())

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        """This step's plan, or None to run vLLM's layers.

        Steps that run vLLM's layers:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps (captured without attention metadata);
        - steps with prefill rows, batches larger than MAX_BATCH_SIZE or top-k other
          than 2048 (``mono_layers.step_reason``) and, eager, a rank-uniform no-go vote.

        Eager steps (enforce_eager, or batch sizes FULL graphs did not capture) run
        the kernels too. Every predicate is the same on all TP ranks: the kernels'
        mailboxes need every rank on one path. Under FULL graphs this Python runs at
        capture only and the plan is baked in.
        """
        mono = self.mono
        mono.active = False
        if intermediate_tensors is not None or self._kv is None:
            return None
        if not is_forward_context_available():
            return None
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        kv = mla_cache(mono.layers[mono.first])
        if (kv.data_ptr(), kv.numel()) != self._kv:
            raise RuntimeError(
                "GLM-5.2 specialized decode: the KV cache was re-bound without "
                "prepare_kv (ModelState.initialize_kv_cache)"
            )
        S = mono.begin_step(positions)
        return None if S is None else Step(positions.shape[0], S, self.kernels)

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (runs eagerly on every rank, also
        under graph replay). Raises on expired polls."""
        if self.watch is not None and not torch.cuda.is_current_stream_capturing():
            self.watch.after_step()
