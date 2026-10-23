# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi-K3 specialized decode executor (layer mode): one routing decision per
step, then the layer kernels inside the stock layer loop.

Built by the model state once the weights are loaded (``KimiK3ModelState``): checks
the loaded weights, builds the per-rank launch state (``runner``) and the layer
kernels of every width, and compiles every build, before memory profiling and graph
capture. ``prepare_kv`` binds it to every KV cache vLLM allocates. Each step
``plan`` returns a ``Step`` or None; ``Step.layer(i)`` is the kernel replacing
decoder layer i (None for vLLM's layers). Weight reloads are not supported."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import FAILED
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    WHOLE,
    DecodeConfig,
    layer_kinds,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    weight_reasons,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import (
    FfnLayer,
    WholeLayer,
    compile_build,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import tp_agree
from vllm.models.kimi_k3.amd.specialized_decode.runtime.runner import MonoRunner

logger = init_logger(__name__)

Kernel = WholeLayer | FfnLayer


class Step(NamedTuple):
    """A planned step: ``width`` rows of whole uniform decode requests."""

    width: int
    attn_metadata: Mapping[str, Any]  # by layer name
    kernels: Mapping[int, Kernel]  # by decoder layer, at this width

    def layer(self, idx: int) -> Callable | None:
        """The kernel replacing decoder layer idx in this step, or None (vLLM's)."""
        kernel = self.kernels.get(idx)
        return None if kernel is None else partial(kernel.launch, self)


def step_reason(attn_metadata, kda_prefix: str, width: int, q: int) -> str | None:
    """None if a step's KDA metadata (identical on every TP rank) is a pure
    uniform regular-decode or spec-verify step filling ``width`` rows, else why
    not. ``q`` is the deployment's configured rows per request."""
    if not isinstance(attn_metadata, dict):
        return "no metadata"
    m: Any = attn_metadata.get(kda_prefix)
    if m is None:
        return "no metadata"
    if m.num_prefills:
        return "not pure decode"
    if m.num_spec_decodes:
        if m.num_decodes or m.spec_sequence_masks is None:
            return "not pure decode"
        actual_q = m.spec_state_indices_tensor.size(-1)
        if (
            actual_q != q
            or m.uniform_spec_sequence_length != actual_q
            or m.num_spec_decodes * actual_q != width
        ):
            return "not uniform"
        return None
    if q != 1:
        return "not spec-verify"
    if m.num_decodes < 1:
        return "not pure decode"
    state_indices = m.non_spec_state_indices_tensor
    if (
        state_indices is None
        or state_indices.ndim != 1
        or m.num_decodes != width
        or state_indices.size(0) != width
    ):
        return "not uniform"
    return None


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that refused
    alone would leave the others spinning in-kernel."""
    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"Kimi-K3 specialized decode does not support {what}: {why} "
            "(decode_backend=auto runs vLLM's decode)"
        )


class SpecializedDecode:
    """The executor: builds the layer kernels, binds them to each KV cache and
    plans every step."""

    def __init__(self, backbone, config: DecodeConfig):
        tp = get_tp_group()
        agree(weight_reasons(backbone), tp.cpu_group, "the loaded weights")
        self.backbone, self.config = backbone, config
        kinds = {
            i: kind
            for i, kind in layer_kinds(backbone.config).items()
            if backbone.start_layer <= i < backbone.end_layer
        }
        runner = self.runner = MonoRunner(next(backbone.parameters()).device)
        runner.num_spec = config.q - 1
        runner.first_layer = min(kinds)
        whole = [i for i, kind in kinds.items() if kind == WHOLE]
        runner.kda_prefix = backbone.layers[whole[0]].self_attn.prefix
        self.kernels: dict[int, dict[int, Kernel]] = {
            w: {
                i: (WholeLayer if kind == WHOLE else FfnLayer)(
                    backbone.layers[i], runner, w
                )
                for i, kind in kinds.items()
            }
            for w in config.widths
        }
        self.kda: list[WholeLayer] = [
            k for k in self.kernels[config.widths[0]].values()
            if isinstance(k, WholeLayer)
        ]  # fmt: skip
        builds = {
            b for by in self.kernels.values() for k in by.values() for b in k.builds()
        }
        for build in builds:
            compile_build(build)
        self.failstop = FailStop(
            runner.health[FAILED : FAILED + 1], runner.rank, tp.world_size, tp.cpu_group
        )
        # (data_ptr, numel) of the first KDA layer's state prepare_kv bound
        self._kv: tuple[int, int] | None = None
        logger.info(
            "Kimi-K3 specialized decode: %d layers, widths %s, %d builds; other "
            "steps run vLLM's decode",
            len(kinds),
            list(config.widths),
            len(builds),
        )

    def _kv_key(self) -> tuple[int, int] | None:
        state = self.kda[0].layer.self_attn.kv_cache
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            return None
        if state[1].numel() == 0:
            return None
        return state[1].data_ptr(), state[1].numel()

    def prepare_kv(self) -> None:
        """Checks and binds the KDA states vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one)."""
        self._kv = None
        if self._kv_key() is None:
            return
        caches = {k.layer.self_attn.prefix: k.caches() for k in self.kda}
        agree(cache_reasons(caches), get_tp_group().cpu_group, "the KV caches")
        self._kv = self._kv_key()

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        """This step's plan, or None to run vLLM's layers.

        Steps that run vLLM's layers:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps (captured without attention metadata);
        - widths without kernels (more than MAX_TOKENS rows);
        - steps that are not pure, uniform regular-decode or spec-verify steps
          (``step_reason``).

        Eager steps run the kernels too. Every predicate is the same on all TP
        ranks: the kernels' mailboxes need every rank on one path. Under FULL
        graphs this runs at capture only and the plan is baked in; a replay with
        fewer requests than its graph pads whole requests (NULL_BLOCK_ID state,
        one accepted token), which the kernels serve like real ones.
        """
        if intermediate_tensors is not None or self._kv is None:
            return None
        if not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        width = positions.size(-1)
        kernels = self.kernels.get(width)
        if kernels is None:
            return None
        md = context.attn_metadata
        if step_reason(md, self.runner.kda_prefix, width, self.config.q) is not None:
            return None
        assert isinstance(md, dict)
        if self._kv_key() != self._kv:
            raise RuntimeError(
                "Kimi-K3 specialized decode: the KV cache was re-bound without "
                "prepare_kv (ModelState.initialize_kv_cache)"
            )
        return Step(width, md, kernels)

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank, also
        under graph replay). Raises on every rank once a rank gave up an in-kernel
        wait."""
        self.failstop.after_step()
