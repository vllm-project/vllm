# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.2 specialized decode executor (layer mode): one routing decision per step,
then a MonoKernel per MoE decoder layer inside the stock layer loop.

Built by the model state once the weights are loaded (`DeepseekV32ModelState`): reads
vLLM's loaded weights in place (`vllm_weights`), builds the kernels of every batch
size, the buffers and the IPC peers, before memory profiling and graph capture.
`prepare_kv` binds it to every KV cache vLLM allocates (the CUDA-graph
memory-profiling cache, then the real one). Each step `plan` returns a `Step` or
None; `Step.layer(i)` is the kernel callable of mono layer i (None for vLLM's layers
0..2). Weight reloads are not supported."""

from __future__ import annotations

import collections
import time
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

# plan() decisions on this rank: "mono_steps", and "vllm_path:<reason>" for steps
# that run vLLM's layers. Under FULL graphs plan() runs at capture only, so a replay
# is counted by the device step counter instead (SpecializedDecode.kernel_steps).
counters: collections.Counter[str] = collections.Counter()
# rank 0 logs the kernel step counters at most this often (a device read)
STEP_REPORT_S = 60.0

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
    place (`vllm_weights.unsupported_moe`), else why not."""
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
        self._next_report = time.monotonic() + STEP_REPORT_S
        logger.info(
            "GLM-5.2 specialized decode: layers %d..%d, batch sizes %s; larger batches "
            "run vLLM's decode",
            self.mono.first,
            self.mono.last,
            self.mono.batch_sizes,
        )

    def _poll_watch(self):
        """Fail-stop on expired kernel polls, advanced by `after_step`;
        `failstop="off"` disables it, `"warn"` logs."""
        if self.cfg.failstop == "off":
            logger.warning("failstop=off: no poll-error fail-stop watch")
            return None
        return guards.PollErrorWatch(self.mono)

    def prepare_kv(self) -> None:
        """Bind the kernels to the KV caches vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one), checking them (`bind_caches`)."""
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
          than 2048 (`mono_layers.step_reason`) and, eager, a rank-uniform no-go vote.

        Eager steps (enforce_eager, or batch sizes FULL graphs did not capture) run
        the kernels too. Every predicate is the same on all TP ranks: the kernels'
        mailboxes need every rank on one path. Under FULL graphs this Python runs at
        capture only and the plan is baked in.
        """
        mono = self.mono
        mono.active = False
        if intermediate_tensors is not None:
            return self._skip("pp_stage_input")
        if self._kv is None:
            return self._skip("no_kv_cache")
        if not is_forward_context_available():
            return self._skip("no_forward_context")
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return self._skip("piecewise_graph")
        kv = mla_cache(mono.layers[mono.first])
        if (kv.data_ptr(), kv.numel()) != self._kv:
            # The extensible KV cache re-binds every view after warm-up, at the same
            # addresses with the final block count, without a ModelState hook:
            # re-check and re-bind here. Never inside a capture (bind_caches copies
            # to the device), where a moved cache is a bug.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GLM-5.2 specialized decode: the KV cache moved inside a CUDA "
                    "graph capture"
                )
            self.prepare_kv()
            logger.info_once(
                "GLM-5.2 specialized decode: KV cache re-bound (%d MLA rows per layer)",
                kv.shape[0] * kv.shape[-2],
            )
        S = mono.begin_step(positions)
        if S is None:
            return self._skip(getattr(mono, "skip_reason", "") or "no_go")
        counters["mono_steps"] += 1
        return Step(positions.shape[0], S, self.kernels)

    @staticmethod
    def _skip(reason: str) -> None:
        key = f"vllm_path:{reason}"
        if not counters[key]:
            logger.debug("GLM-5.2 specialized decode: vLLM's layers for %s", reason)
        counters[key] += 1
        return None

    def kernel_steps(self) -> dict[int, int]:
        """Kernel steps run so far on this rank, per built batch size (graph replays
        included). Reads the device: not for the step path."""
        return self.mono.kernel_steps()

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (runs eagerly on every rank, also
        under graph replay). Raises on expired polls."""
        if self.watch is not None and not torch.cuda.is_current_stream_capturing():
            self.watch.after_step()
        self._report_steps()

    def _report_steps(self) -> None:
        """Rank 0, at most every STEP_REPORT_S: the kernel steps run per batch size
        (device counters, graph replays included) and the steps plan() sent to vLLM's
        layers, by reason (plan() runs for eager steps and at capture only)."""
        if self.mono.rank != 0 or torch.cuda.is_current_stream_capturing():
            return
        now = time.monotonic()
        if now < self._next_report:
            return
        self._next_report = now + STEP_REPORT_S
        prefix = "vllm_path:"
        logger.info(
            "GLM-5.2 specialized decode: kernel steps per batch size %s; vLLM-path "
            "plans %s",
            self.kernel_steps(),
            {k[len(prefix) :]: v for k, v in counters.items() if k.startswith(prefix)},
        )
